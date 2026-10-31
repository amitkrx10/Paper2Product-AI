"""Stage C - dataset discovery & validation with provenance."""
from __future__ import annotations
import csv
import hashlib
import os
from dataclasses import dataclass, field, asdict


class ResourceError(Exception):
    """Resource unavailable / needs credentials / failed validation. Message is user-safe."""


@dataclass
class DatasetInfo:
    id: str
    name: str
    source: str
    version: str
    license: str
    url: str
    n_rows: int
    n_features: int
    n_classes: int | None
    sha256: str
    validated: bool
    provenance: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


MAX_ROWS = 1_000_000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def resolve_sklearn_iris() -> DatasetInfo:
    try:
        import sklearn
        from sklearn.datasets import load_iris
    except ImportError as e:
        raise ResourceError("scikit-learn is required for the bundled Iris dataset") from e
    d = load_iris()
    if d.data.shape != (150, 4) or len(set(d.target.tolist())) != 3:
        raise ResourceError("bundled Iris dataset failed schema validation")
    digest = hashlib.sha256(d.data.tobytes() + d.target.tobytes()).hexdigest()
    return DatasetInfo(
        id="sklearn:iris", name="iris", source="scikit-learn bundled copy", version=f"scikit-learn {sklearn.__version__}",
        license="Per UCI ML Repository listing (CC BY 4.0) - verify at source before redistribution",
        url="https://archive.ics.uci.edu/dataset/53/iris", n_rows=150, n_features=4, n_classes=3, sha256=digest,
        validated=True, provenance="Fisher (1936), Annals of Eugenics 7(2):179-188; distributed via UCI and scikit-learn",
        warnings=["UCI's Iris copy is documented to have errors in two rows relative to Fisher's paper; scikit-learn's "
                  "copy corrects them (since v0.20). Other copies may differ slightly."])


def resolve_huggingface(dataset_id: str, token: str = "") -> DatasetInfo:
    if not dataset_id or len(dataset_id) > 120 or not all(c.isalnum() or c in "-_./" for c in dataset_id):
        raise ResourceError("invalid Hugging Face dataset id")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ResourceError("huggingface_hub is not installed: pip install huggingface_hub")
    try:
        info = HfApi(token=token or None).dataset_info(dataset_id)
    except Exception as e:
        name = type(e).__name__
        if "NotFound" in name:
            raise ResourceError(f"dataset '{dataset_id}' not found on Hugging Face")
        if "Gated" in name or "Unauthorized" in name or "401" in str(e):
            raise ResourceError("dataset requires authentication: set HF_TOKEN")
        raise ResourceError(f"Hugging Face lookup failed ({name})")
    card = getattr(info, "card_data", None)
    lic = (getattr(card, "license", None) if card else None) or "unknown"
    warnings = [] if lic != "unknown" else ["license not declared on dataset card"]
    return DatasetInfo(id=f"hf:{dataset_id}", name=dataset_id, source="Hugging Face Hub", version=info.sha or "unknown",
                       license=str(lic), url=f"https://huggingface.co/datasets/{dataset_id}", n_rows=-1, n_features=-1,
                       n_classes=None, sha256="", validated=False,
                       provenance="metadata only; size/schema must be validated before use", warnings=warnings)


def resolve_upload(path: str) -> DatasetInfo:
    if not os.path.isfile(path):
        raise ResourceError("uploaded dataset not found")
    size = os.path.getsize(path)
    if size > MAX_UPLOAD_BYTES:
        raise ResourceError("dataset larger than 50 MB limit")
    if not path.lower().endswith(".csv"):
        raise ResourceError("only .csv uploads are supported")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(path, newline="", encoding="utf-8", errors="strict") as f:
        rows = csv.reader(f)
        header = next(rows, None)
        if not header or len(header) < 2:
            raise ResourceError("CSV needs a header and at least 2 columns")
        n = 0
        for r in rows:
            if len(r) != len(header):
                raise ResourceError(f"row {n + 2} has {len(r)} fields, expected {len(header)}")
            n += 1
            if n > MAX_ROWS:
                raise ResourceError("dataset exceeds row limit")
    if n < 10:
        raise ResourceError("dataset has fewer than 10 rows")
    return DatasetInfo(id=f"upload:{os.path.basename(path)}", name=os.path.basename(path), source="user upload",
                       version=h.hexdigest()[:12], license="user-asserted", url="", n_rows=n,
                       n_features=len(header) - 1, n_classes=None, sha256=h.hexdigest(), validated=True,
                       provenance="uploaded by user; license/ownership not verified")


def resolve_for_spec(spec_datasets: list[str], hint: str = "", hf_token: str = "") -> DatasetInfo:
    """Pick a dataset source. Explicit hint wins; otherwise recognise bundled datasets by name."""
    if hint:
        if hint == "sklearn:iris":
            return resolve_sklearn_iris()
        if hint.startswith("hf:"):
            return resolve_huggingface(hint[3:], hf_token)
        if hint.startswith("upload:"):
            return resolve_upload(hint[7:])
        raise ResourceError(f"unsupported dataset source '{hint}' (use sklearn:iris, hf:<id> or upload:<path>)")
    if any("iris" in d.lower() for d in spec_datasets):
        return resolve_sklearn_iris()
    raise ResourceError("no supported dataset could be resolved from the paper; supply a dataset source "
                        "(sklearn:iris | hf:<id> | upload:<path>)")
