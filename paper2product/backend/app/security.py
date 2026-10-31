"""Security helpers: path safety, generated-code scanning, prompt-injection flags, secret redaction."""
from __future__ import annotations
import ast
import os
import re

MAX_FILE_BYTES = 200_000
MAX_FILES = 60
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".json", ".cfg", ".toml", ".yaml", ".yml", ".ini"}


class SecurityError(Exception):
    pass


def safe_relpath(path: str) -> str:
    """Validate a project-relative path. Rejects traversal, absolute paths, hidden/odd names."""
    if not isinstance(path, str) or not path or len(path) > 200:
        raise SecurityError("invalid path")
    if "\x00" in path or "\\" in path:
        raise SecurityError("invalid characters in path")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise SecurityError("absolute paths are not allowed")
    parts = path.split("/")
    for p in parts:
        if p in ("", ".", "..") or p.startswith("."):
            raise SecurityError(f"unsafe path segment: {p!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SecurityError(f"file type {ext or '(none)'} not allowed")
    return "/".join(parts)


def validate_file_map(files: dict[str, str]) -> dict[str, str]:
    if not isinstance(files, dict) or not files:
        raise SecurityError("no files provided")
    if len(files) > MAX_FILES:
        raise SecurityError("too many files")
    out = {}
    for p, c in files.items():
        p = safe_relpath(p)
        if not isinstance(c, str):
            raise SecurityError(f"content of {p} must be text")
        if len(c.encode("utf-8")) > MAX_FILE_BYTES:
            raise SecurityError(f"{p} exceeds size limit")
        out[p] = c
    return out


def resolve_inside(root: str, rel: str) -> str:
    """Resolve rel under root and guarantee the result stays inside root."""
    rel = safe_relpath(rel)
    full = os.path.realpath(os.path.join(root, rel))
    root_real = os.path.realpath(root)
    if os.path.commonpath([full, root_real]) != root_real:
        raise SecurityError("path escapes project root")
    return full


FORBIDDEN_IMPORTS = {"subprocess", "socket", "ctypes", "pty", "multiprocessing", "shutil",
                     "urllib", "http", "requests", "ftplib", "smtplib", "telnetlib", "pickle", "marshal"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "input"}
FORBIDDEN_ATTRS = {("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "rmdir"),
                   ("os", "unlink"), ("os", "fork"), ("os", "kill"), ("os", "environ"), ("os", "getenv")}


def scan_python(source: str, filename: str = "<generated>") -> list[str]:
    """Static policy scan of generated code. Returns a list of violations (empty = passes).

    This is defence in depth, NOT a sandbox: the real boundary is the execution sandbox.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [f"{filename}:{e.lineno}: syntax error: {e.msg}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    problems.append(f"{filename}:{node.lineno}: forbidden import '{a.name}'")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                problems.append(f"{filename}:{node.lineno}: forbidden import '{node.module}'")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                problems.append(f"{filename}:{node.lineno}: forbidden call '{f.id}()'")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and (node.value.id, node.attr) in FORBIDDEN_ATTRS:
                problems.append(f"{filename}:{node.lineno}: forbidden use of '{node.value.id}.{node.attr}'")
    return problems


INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above|system)",
    r"(reveal|print|show|output|leak) (your |the )?(system prompt|api[ _-]?key|secrets?|environment variables?|credentials)",
    r"you are now (a|an|in) ",
    r"\bnew instructions?\s*:",
    r"(run|execute) (this |the following )?(shell|bash|command)",
    r"curl\s+https?://",
    r"disable (the )?(sandbox|security|safety)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def scan_prompt_injection(text: str) -> list[str]:
    flags = []
    for line in text.splitlines():
        for rx in _INJ:
            if rx.search(line):
                flags.append(line.strip()[:200])
                break
    return flags


_SECRET_RX = [re.compile(p) for p in (
    r"AIza[0-9A-Za-z_\-]{20,}", r"ghp_[0-9A-Za-z]{20,}", r"github_pat_[0-9A-Za-z_]{20,}",
    r"hf_[0-9A-Za-z]{20,}", r"sk-[0-9A-Za-z]{20,}")]


def redact(text: str, extra_secrets: list[str] | None = None) -> str:
    for rx in _SECRET_RX:
        text = rx.sub("[REDACTED]", text)
    for s in extra_secrets or []:
        if s and len(s) > 6:
            text = text.replace(s, "[REDACTED]")
    return text
