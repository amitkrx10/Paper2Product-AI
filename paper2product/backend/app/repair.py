"""Stage F helpers - diagnose failures from logs and propose targeted, validated patches."""
from __future__ import annotations
import difflib
import re
from dataclasses import dataclass, asdict
from typing import Optional

from .security import safe_relpath, scan_python, redact


@dataclass
class Diagnosis:
    category: str            # name_error | import_error | syntax_error | assertion_failure | timeout | memory | other
    summary: str
    file: Optional[str] = None
    line: Optional[int] = None
    name: Optional[str] = None
    repairable_by_rule: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Patch:
    file: str
    new_content: str
    rationale: str
    source: str              # 'rule-based' | 'llm'

    def to_dict(self) -> dict:
        return {"file": self.file, "rationale": self.rationale, "source": self.source}


_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_ERR = re.compile(r"^(\w+(?:Error|Exception|Exit|Interrupt)):?\s*(.*)$", re.M)


def diagnose(stdout: str, stderr: str, files: dict[str, str], timed_out: bool = False) -> Diagnosis:
    log = stderr + "\n" + stdout
    if timed_out:
        return Diagnosis("timeout", "execution exceeded the time limit")
    errs = _ERR.findall(log)
    etype, emsg = (errs[-1] if errs else ("", ""))
    frames = [(p, int(n)) for p, n in _FRAME.findall(log)]
    mapped = []
    for path, line in frames:
        for rel in files:
            if path.endswith("/" + rel) or path == rel:
                mapped.append((rel, line))
    src_frames = [f for f in mapped if not f[0].startswith("tests/")]
    file, line = (src_frames[-1] if src_frames else (mapped[-1] if mapped else (None, None)))
    if etype == "NameError":
        m = re.search(r"name '(\w+)' is not defined", emsg)
        return Diagnosis("name_error", f"NameError: {emsg}", file, line, m.group(1) if m else None, bool(m and file))
    if etype in ("ModuleNotFoundError", "ImportError"):
        return Diagnosis("import_error", f"{etype}: {emsg}", file, line)
    if etype in ("SyntaxError", "IndentationError"):
        return Diagnosis("syntax_error", f"{etype}: {emsg}", file, line)
    if etype == "MemoryError":
        return Diagnosis("memory", "MemoryError: memory limit hit")
    if "AssertionError" in log or "FAILED (failures" in log:
        return Diagnosis("assertion_failure", f"test assertion failed: {emsg or 'see log'}", file, line)
    if re.search(r"(Killed|Cannot allocate memory)", log):
        return Diagnosis("memory", "process killed (likely memory limit)")
    return Diagnosis("other", f"{etype or 'error'}: {emsg}".strip(": ") or "unknown failure", file, line)


def rule_based_patch(diag: Diagnosis, files: dict[str, str]) -> Optional[Patch]:
    """Targeted fix for NameError typos: rename to the closest identifier defined in the same file."""
    if diag.category != "name_error" or not diag.file or not diag.name or not diag.line:
        return None
    src = files[diag.file]
    defined = set(re.findall(r"^\s*(\w+)\s*=", src, re.M)) | set(re.findall(r"def (\w+)\(", src)) \
        | set(re.findall(r"^\s*(?:from|import)\s+\S+(?:\s+import\s+(\w+))?", src, re.M)) | set(re.findall(r"\bimport (\w+)", src))
    defined.discard("")
    defined.discard(diag.name)
    match = difflib.get_close_matches(diag.name, sorted(defined), n=1, cutoff=0.6)
    if not match:
        return None
    lines = src.split("\n")
    idx = diag.line - 1
    if not (0 <= idx < len(lines)) or diag.name not in lines[idx]:
        return None
    lines[idx] = re.sub(rf"\b{re.escape(diag.name)}\b", match[0], lines[idx])
    return Patch(diag.file, "\n".join(lines),
                 f"`{diag.name}` is undefined at {diag.file}:{diag.line}; the closest defined name is `{match[0]}` (typo).",
                 "rule-based")


def llm_patch(diag: Diagnosis, files: dict[str, str], stderr: str, stdout: str, provider) -> Optional[Patch]:
    target = diag.file
    if not target:
        return None
    prompt = ("A generated project failed. Propose ONE minimal fix to ONE non-test file. Never edit tests or weaken assertions. "
              'Return JSON {"file": "...", "content": "<full new file content>", "rationale": "..."}.\n'
              f"DIAGNOSIS: {diag.to_dict()}\nLOG:\n{redact(stderr)[-3000:]}\n{redact(stdout)[-1500:]}\n"
              f"FILE {target}:\n{files[target]}")
    d = provider.generate_json(prompt)
    if not isinstance(d.get("file"), str) or not isinstance(d.get("content"), str):
        return None
    return Patch(d["file"], d["content"], str(d.get("rationale", ""))[:500], "llm")


class PatchRejected(Exception):
    pass


def validate_patch(patch: Patch, files: dict[str, str], max_changed_lines: int = 60) -> str:
    """Policy checks; returns unified diff. Raises PatchRejected on any violation."""
    try:
        rel = safe_relpath(patch.file)
    except Exception:
        raise PatchRejected("patch path is unsafe")
    if rel not in files:
        raise PatchRejected("patch targets a file that does not exist")
    if rel.startswith("tests/") or "/tests/" in rel:
        raise PatchRejected("repairs may not modify test files (prevents weakening tests)")
    old, new = files[rel], patch.new_content
    if old == new:
        raise PatchRejected("patch changes nothing")
    if rel.endswith(".py"):
        problems = scan_python(new, rel)
        if problems:
            raise PatchRejected("patched code violates security policy: " + problems[0])
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{rel}", f"b/{rel}", lineterm=""))
    changed = sum(1 for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---")))
    if changed > max_changed_lines:
        raise PatchRejected(f"patch too large ({changed} changed lines)")
    return "\n".join(diff)
