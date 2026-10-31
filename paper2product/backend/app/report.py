"""Stage H - evidence-backed report, reproducibility manifest and project ZIP export."""
from __future__ import annotations
import hashlib
import io
import json
import time
import zipfile

from .security import redact

TOOL_VERSION = "1.0.0"
STATUS_LABEL = {"reproduced": "Successfully reproduced", "partially_reproduced": "Partially reproduced",
                "not_reproduced": "Not reproduced", "unable_to_verify": "Unable to verify"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(st: dict) -> dict:
    files = st.get("files", {})
    meta = st.get("run_meta") or {}
    return {
        "tool": "Paper2Product AI", "tool_version": TOOL_VERSION, "project_id": st.get("project_id"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "honesty": {
            "mode": st.get("mode"), "llm_used": st.get("mode") == "llm", "llm": st.get("llm_info"),
            "simulated_results": False,
            "note": ("DEMO MODE: no LLM reasoning was performed. Spec extraction used heuristic rules and code came from a "
                     "deterministic reviewed template. Test results and metrics are REAL outputs of code executed in the sandbox."
                     if st.get("mode") == "demo" else "LLM-assisted run; all model output was schema-validated."),
        },
        "paper": {"title": (st.get("spec") or {}).get("title"), "extraction_method": (st.get("spec") or {}).get("extraction_method")},
        "dataset": st.get("dataset"), "seed": meta.get("seed"), "test_size": meta.get("test_size"),
        "hyperparameters": (st.get("spec") or {}).get("hyperparameters"),
        "environment": meta.get("environment"),
        "sandbox": st.get("sandbox"), "budget": st.get("budget"),
        "tests": st.get("final_tests"), "runtime_s": meta.get("runtime_s"),
        "metrics": st.get("metrics"), "comparisons": st.get("comparisons"),
        "reproduction_status": st.get("reproduction_status"), "reproduction_note": st.get("reproduction_note"),
        "repair_attempts": len(st.get("attempts", [])), "usage": st.get("usage"),
        "approved_assumptions": st.get("approved_assumptions"),
        "file_sha256": {p: sha256(c) for p, c in sorted(files.items())},
        "rerun": ["pip install -r requirements.txt", "python -m unittest discover -s tests -t .", "python run_experiment.py"],
    }


def _fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def build_report_md(st: dict) -> str:
    spec, plan, ds = st.get("spec") or {}, st.get("plan") or {}, st.get("dataset") or {}
    m, sb = st.get("metrics") or {}, st.get("sandbox") or {}
    L: list[str] = []
    a = L.append
    status = st.get("reproduction_status", "unable_to_verify")
    a(f"# Reproduction Report: {spec.get('title', 'Untitled')}")
    a(f"\n**Reproduction status: {STATUS_LABEL.get(status, status)}**  \n{st.get('reproduction_note', '')}\n")
    a("> A passing unit test does not prove that a paper has been reproduced. Status is derived only from measured "
      "metrics compared with reported results under matching dataset/split conditions.\n")
    if st.get("mode") == "demo":
        a("> **DEMO MODE** - no LLM was used. Extraction was rule-based and code came from a deterministic template. "
          "Metrics below are real outputs from code executed in the sandbox, not simulated.\n")
    if sb and not sb.get("isolated"):
        a(f"> **Sandbox warning:** executed in `{sb.get('mode')}` mode - {sb.get('note')}\n")
    a("## 1. Paper summary")
    a(f"- **Authors:** {', '.join(spec.get('authors', [])) or 'n/a'}\n- **Abstract:** {spec.get('abstract') or 'n/a'}\n- **Problem:** {spec.get('problem') or 'n/a'}")
    a("\n## 2. Extracted research specification")
    a(f"- **Method:** {spec.get('method') or 'n/a'} _(page {', '.join(str(x) for x in spec.get('page_refs', {}).get('method', ['?']))})_")
    for e in spec.get("equations", []):
        a(f"  - `{e}`")
    a(f"- **Datasets:** {'; '.join(spec.get('datasets', [])) or 'n/a'}\n- **Preprocessing:** {'; '.join(spec.get('preprocessing', [])) or 'n/a'}")
    a(f"- **Hyperparameters:** `{json.dumps(spec.get('hyperparameters', {}))}`\n- **Metrics:** {', '.join(spec.get('metrics', [])) or 'n/a'}")
    a(f"- **Extraction method:** {spec.get('extraction_method')}")
    if spec.get("security_flags"):
        a("- **Security flags (treated as data, never obeyed):**")
        for f in spec["security_flags"]:
            a(f"  - `{f}`")
    a("\n## 3. Implementation plan")
    a(f"{plan.get('approach', 'n/a')}\n")
    for t in plan.get("tasks", []):
        a(f"- **{t['id']}** {t['title']} (depends on: {', '.join(t['depends_on']) or '-'})")
    a("\n## 4. Files created")
    for p, h in sorted((build_manifest(st)["file_sha256"]).items()):
        a(f"- `{p}` (sha256 `{h[:12]}`)")
    a("\n## 5. Dataset provenance")
    if ds:
        a(f"- **{ds.get('name')}** from {ds.get('source')} - version `{ds.get('version')}`\n- License: {ds.get('license')}\n- Source: {ds.get('url')}\n"
          f"- Rows/features/classes: {ds.get('n_rows')}/{ds.get('n_features')}/{ds.get('n_classes')}\n- sha256: `{ds.get('sha256', '')[:16]}`\n- Provenance: {ds.get('provenance')}")
        for w in ds.get("warnings", []):
            a(f"- Warning: {w}")
    else:
        a("- No dataset was resolved.")
    a("\n## 6. Execution history")
    for r in st.get("runs", []):
        a(f"- Run {r['n']} [{r['stage']}] exit={r['exit_code']} {'PASS' if r['ok'] else 'FAIL'} in {r['duration_s']}s - {r['summary']}")
    a("\n### Repair attempts")
    if not st.get("attempts"):
        a("- None needed.")
    for at in st.get("attempts", []):
        a(f"- **Attempt {at['n']}** ({at['stage']} failure): {at['diagnosis']['summary']}\n  - Patch ({at['patch']['source']}): {at['patch']['rationale']}\n"
          f"  - Changed: `{at['patch']['file']}` | outcome: {at.get('outcome', 'pending')}\n\n```diff\n{at['diff']}\n```")
    a("\n## 7. Tests")
    ft = st.get("final_tests") or {}
    a(f"- Final: {ft.get('summary', 'not run')} ({'PASS' if ft.get('ok') else 'FAIL'})")
    a("\n## 8. Measured metrics (computed independently from predictions)")
    if m.get("task") == "classification":
        a(f"- n={m['n']} accuracy={_fmt(m['accuracy'])} macro-precision={_fmt(m['precision'])} macro-recall={_fmt(m['recall'])} macro-F1={_fmt(m['f1'])}")
        a(f"- Confusion matrix (labels {m['labels']}): `{m['confusion_matrix']}`")
        if st.get("script_metric_check"):
            a(f"- Cross-check against the generated script's own accuracy: {st['script_metric_check']}")
    else:
        a("- No metrics were measured.")
    a("\n## 9. Comparison with reported results")
    if not st.get("comparisons"):
        a("- No reported numerical results were available to compare.")
    for c in st.get("comparisons", []):
        a(f"- **{c['metric']}**: reported {_fmt(c['reported'])} (page {c.get('page')}) vs measured {_fmt(c['measured'])} - "
          f"abs diff {_fmt(c['abs_diff'])}, rel diff {_fmt(c['rel_diff'])} -> **{STATUS_LABEL.get(c['status'])}**")
        for w in c["warnings"]:
            a(f"  - Warning: {w}")
    a("\n## 10. Differences, limitations and assumptions")
    for x in plan.get("assumptions", []) + spec.get("limitations", []):
        a(f"- {x}")
    a("\n## 11. Reproduction status")
    a(f"**{STATUS_LABEL.get(status, status)}** - {st.get('reproduction_note', '')}")
    a("\n## 12. Rerun instructions\n```bash\npip install -r requirements.txt\npython -m unittest discover -s tests -t .\npython run_experiment.py\n```")
    if st.get("usage"):
        a(f"\n_LLM usage:_ `{json.dumps(st['usage'])}`")
    return redact("\n".join(L)) + "\n"


def build_zip(files: dict[str, str], report_md: str, manifest: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p, c in sorted(files.items()):
            z.writestr(f"project/{p}", c)
        z.writestr("REPORT.md", report_md)
        z.writestr("reproducibility_manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue()
