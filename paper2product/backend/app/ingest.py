"""Stage A - research ingestion: validated PDF text extraction + specification extraction."""
from __future__ import annotations
import io
import re
from typing import Optional

from .schemas import ResearchSpec, ReportedResult
from .security import scan_prompt_injection


class IngestError(Exception):
    pass


def extract_pages(pdf_bytes: bytes, max_bytes: int = 20 * 1024 * 1024, max_pages: int = 60) -> list[str]:
    """Return text per page (index 0 = page 1).

    Validates magic bytes, size and page count. PyMuPDF is preferred; ANY failure inside it (corrupt or
    malicious PDF) falls back to pypdf, and every parser failure is surfaced as IngestError - never a raw
    third-party exception.
    """
    if not pdf_bytes or len(pdf_bytes) > max_bytes:
        raise IngestError(f"PDF is empty or larger than {max_bytes // (1024 * 1024)} MB")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise IngestError("file is not a PDF (bad magic bytes)")
    pages: list[str] | None = None
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.page_count > max_pages:
                raise IngestError(f"PDF has more than {max_pages} pages")
            pages = [p.get_text() for p in doc]
        except IngestError:
            raise
        except Exception:          # PyMuPDF parse error -> try the fallback parser
            pages = None
    if pages is None:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            if len(reader.pages) > max_pages:
                raise IngestError(f"PDF has more than {max_pages} pages")
            pages = [(p.extract_text() or "") for p in reader.pages]
        except IngestError:
            raise
        except Exception as e:
            raise IngestError(f"could not parse PDF ({type(e).__name__})") from e
    if not any(t.strip() for t in pages):
        raise IngestError("PDF contains no extractable text (scanned image? OCR is not supported)")
    return pages


SECTION_ALIASES = {
    "abstract": ["abstract"],
    "problem": ["research problem", "problem statement", "problem"],
    "method": ["proposed method", "method", "algorithm", "approach"],
    "equations": ["equations", "equation", "key equations"],
    "architecture": ["model architecture", "architecture"],
    "datasets": ["datasets", "dataset", "data"],
    "preprocessing": ["data preprocessing", "preprocessing"],
    "hyperparameters": ["hyperparameters", "hyper-parameters", "experimental settings"],
    "training": ["training procedure", "training"],
    "metrics": ["evaluation metrics", "metrics", "evaluation protocol"],
    "results": ["reported results", "results"],
    "hardware": ["hardware and software", "hardware", "software requirements"],
    "limitations": ["limitations"],
    "notes": ["notes", "additional notes"],
}
_ALIAS = {a: k for k, v in SECTION_ALIASES.items() for a in v}
_HEAD = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Za-z][A-Za-z \-]{2,40}?)\s*:?\s*$")
KNOWN_DATASETS = ["iris", "wine", "digits", "breast cancer", "diabetes", "mnist", "cifar-10", "cifar-100",
                  "imagenet", "california housing", "titanic"]


def _sections(pages: list[str]) -> tuple[dict[str, list[tuple[int, str]]], list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {}
    preamble: list[tuple[int, str]] = []
    current: Optional[str] = None
    for pno, text in enumerate(pages, start=1):
        for line in text.splitlines():
            if not line.strip():
                continue
            m = _HEAD.match(line)
            if m and m.group(1).strip().lower() in _ALIAS:
                current = _ALIAS[m.group(1).strip().lower()]
                sections.setdefault(current, [])
                continue
            (sections[current] if current else preamble).append((pno, line.strip()))
    return sections, preamble


def _txt(items: list[tuple[int, str]]) -> str:
    return " ".join(t for _, t in items).strip()


def _bullets(items: list[tuple[int, str]]) -> list[str]:
    return [re.sub(r"^[-•*\u2022]\s*", "", t).strip() for _, t in items if t.strip()]


def _num(v: str):
    v = v.strip().rstrip(".")
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


METRIC_WORDS = {"accuracy": "accuracy", "f1": "f1", "f1 score": "f1", "precision": "precision", "recall": "recall",
                "mae": "mae", "rmse": "rmse", "r2": "r2", "r-squared": "r2", "r^2": "r2"}


def _parse_results(items: list[tuple[int, str]]) -> list[ReportedResult]:
    out = []
    for pno, line in items:
        low = line.lower()
        for word, canon in METRIC_WORDS.items():
            m = re.search(rf"\b{re.escape(word)}\b\D{{0,40}}?(\d*\.\d+|\d+)", low)
            if m:
                ds = next((d for d in KNOWN_DATASETS if d in low), "")
                sp = re.search(r"(\d{2}\s*/\s*\d{2})", low)
                out.append(ReportedResult(canon, float(m.group(1)), ds, sp.group(1).replace(" ", "") if sp else "",
                                          pno, line[:200]))
                break
    return out


def infer_task(metrics: list[str]) -> str:
    ms = set(metrics)
    if ms & {"accuracy", "f1", "precision", "recall"}:
        return "classification"
    if ms & {"mae", "rmse", "r2"}:
        return "regression"
    return "unknown"


def heuristic_spec(pages: list[str]) -> ResearchSpec:
    """Deterministic, rule-based extraction. Works on papers with conventional section headings."""
    sec, pre = _sections(pages)
    lines = [t for _, t in pre]
    title = lines[0] if lines else ""
    authors = [a.strip() for a in re.split(r",| and ", lines[1])] if len(lines) > 1 and not lines[1].upper().startswith("NOTICE") else []
    hp: dict = {}
    for _, line in sec.get("hyperparameters", []):
        m = re.match(r"^[-•*\u2022]?\s*([A-Za-z_][\w \-/]*?)\s*[:=]\s*(.+?)\s*$", line)
        if m:
            hp[m.group(1).strip().lower().replace(" ", "_")] = _num(m.group(2))
    metrics_text = _txt(sec.get("metrics", [])).lower()
    metrics = sorted({c for w, c in METRIC_WORDS.items() if re.search(rf"\b{re.escape(w)}\b", metrics_text)})
    results = _parse_results(sec.get("results", []))
    if not metrics:
        metrics = sorted({r.metric for r in results})
    datasets = _bullets(sec.get("datasets", []))
    spec = ResearchSpec(
        title=title, authors=authors, abstract=_txt(sec.get("abstract", [])),
        problem=_txt(sec.get("problem", [])), method=_txt(sec.get("method", [])),
        equations=_bullets(sec.get("equations", [])), architecture=_txt(sec.get("architecture", [])),
        datasets=datasets, preprocessing=_bullets(sec.get("preprocessing", [])), hyperparameters=hp,
        training=_txt(sec.get("training", [])), metrics=metrics, reported_results=results,
        hardware=_txt(sec.get("hardware", [])), limitations=_bullets(sec.get("limitations", [])),
        task_type=infer_task(metrics),
        page_refs={k: sorted({p for p, _ in v}) for k, v in sec.items() if v},
        extraction_method="heuristic-sections (no LLM)",
    )
    missing = []
    if not spec.method:
        missing.append("No method/algorithm section found")
    if not spec.datasets:
        missing.append("No dataset specified")
    if not hp:
        missing.append("No hyperparameters found")
    if "random_seed" not in hp and "seed" not in hp:
        missing.append("Random seed not reported")
    if not spec.reported_results:
        missing.append("No numerical results found to compare against")
    spec.missing_info = missing
    return spec.validate()


def llm_spec(pages: list[str], provider) -> ResearchSpec:
    """LLM-based extraction. Document text is wrapped as untrusted data and the output is schema-validated."""
    body = "\n".join(f"[PAGE {i}]\n{t}" for i, t in enumerate(pages, 1))[:60_000]
    prompt = (
        "Extract a research specification from the paper below. Return JSON with keys: title, authors[], abstract, "
        "problem, method, equations[], architecture, datasets[], preprocessing[], hyperparameters{}, training, "
        "metrics[], reported_results[{metric,value,dataset,split,page}], hardware, limitations[], missing_info[], "
        "task_type (classification|regression|clustering|unknown), page_refs{field:[pages]}. "
        "Do not invent values; list anything absent in missing_info.\n"
        f"<untrusted_document>\n{body}\n</untrusted_document>")
    d = provider.generate_json(prompt)
    d["extraction_method"] = f"llm:{provider.name}"
    return ResearchSpec.from_dict(d)


def build_spec(pages: list[str], provider=None) -> ResearchSpec:
    full = "\n".join(pages)
    flags = scan_prompt_injection(full)
    spec = llm_spec(pages, provider) if provider is not None else heuristic_spec(pages)
    if not spec.title:
        raise IngestError("could not determine a paper title")
    spec.security_flags = flags
    return spec
