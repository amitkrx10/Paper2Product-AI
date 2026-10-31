"""Typed, validated data models shared by every agent node.

Plain dataclasses with explicit validation keep the core engine dependency-free
and testable; the HTTP layer wraps them with Pydantic request models.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

TASK_TYPES = {"classification", "regression", "clustering", "unknown"}
REPRO_STATUSES = {"reproduced", "partially_reproduced", "not_reproduced", "unable_to_verify"}


class SchemaError(ValueError):
    """Raised when data fails schema validation."""


def _str(d: dict, k: str, default: str = "") -> str:
    v = d.get(k, default)
    if v is None:
        return default
    if not isinstance(v, str):
        raise SchemaError(f"'{k}' must be a string")
    return v.strip()


def _strlist(d: dict, k: str) -> list[str]:
    v = d.get(k, [])
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise SchemaError(f"'{k}' must be a list of strings")
    return [x.strip() for x in v if x.strip()]


@dataclass
class ReportedResult:
    metric: str
    value: float
    dataset: str = ""
    split: str = ""
    page: Optional[int] = None
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ReportedResult":
        if not isinstance(d, dict):
            raise SchemaError("reported result must be an object")
        try:
            value = float(d["value"])
        except (KeyError, TypeError, ValueError):
            raise SchemaError("reported result needs a numeric 'value'")
        metric = _str(d, "metric").lower()
        if not metric:
            raise SchemaError("reported result needs a 'metric'")
        page = d.get("page")
        if page is not None and not isinstance(page, int):
            raise SchemaError("'page' must be an integer")
        return cls(metric, value, _str(d, "dataset"), _str(d, "split"), page, _str(d, "note"))


@dataclass
class ResearchSpec:
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    problem: str = ""
    method: str = ""
    equations: list[str] = field(default_factory=list)
    architecture: str = ""
    datasets: list[str] = field(default_factory=list)
    preprocessing: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training: str = ""
    metrics: list[str] = field(default_factory=list)
    reported_results: list[ReportedResult] = field(default_factory=list)
    hardware: str = ""
    limitations: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    page_refs: dict[str, list[int]] = field(default_factory=dict)
    security_flags: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"

    def validate(self) -> "ResearchSpec":
        if not self.title:
            raise SchemaError("title is required")
        if self.task_type not in TASK_TYPES:
            raise SchemaError(f"task_type must be one of {sorted(TASK_TYPES)}")
        if not isinstance(self.hyperparameters, dict):
            raise SchemaError("hyperparameters must be an object")
        for r in self.reported_results:
            if not isinstance(r, ReportedResult):
                raise SchemaError("reported_results must be ReportedResult items")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchSpec":
        if not isinstance(d, dict):
            raise SchemaError("spec must be an object")
        hp = d.get("hyperparameters", {}) or {}
        if not isinstance(hp, dict):
            raise SchemaError("'hyperparameters' must be an object")
        rr = d.get("reported_results", []) or []
        if not isinstance(rr, list):
            raise SchemaError("'reported_results' must be a list")
        refs = d.get("page_refs", {}) or {}
        if not isinstance(refs, dict):
            raise SchemaError("'page_refs' must be an object")
        spec = cls(
            title=_str(d, "title"), authors=_strlist(d, "authors"), abstract=_str(d, "abstract"),
            problem=_str(d, "problem"), method=_str(d, "method"), equations=_strlist(d, "equations"),
            architecture=_str(d, "architecture"), datasets=_strlist(d, "datasets"),
            preprocessing=_strlist(d, "preprocessing"), hyperparameters=dict(hp),
            training=_str(d, "training"), metrics=[m.lower() for m in _strlist(d, "metrics")],
            reported_results=[ReportedResult.from_dict(x) for x in rr],
            hardware=_str(d, "hardware"), limitations=_strlist(d, "limitations"),
            missing_info=_strlist(d, "missing_info"), task_type=_str(d, "task_type", "unknown") or "unknown",
            page_refs={str(k): [int(p) for p in v] for k, v in refs.items()},
            security_flags=_strlist(d, "security_flags"),
            extraction_method=_str(d, "extraction_method", "unknown"),
        )
        return spec.validate()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanTask:
    id: str
    title: str
    depends_on: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


@dataclass
class Plan:
    template: str                      # which implementation strategy: 'fisher_lda_iris' | 'llm_generic' | 'unsupported'
    approach: str
    tasks: list[PlanTask]
    files: list[str]
    packages: list[str]
    dataset_requirements: list[str]
    test_strategy: str
    metrics: list[str]
    expected_outputs: list[str]
    compute: str
    risks: list[str]
    assumptions: list[str]
    open_questions: list[str]
    feasible: bool = True
    infeasible_reason: str = ""

    def validate(self) -> "Plan":
        ids = {t.id for t in self.tasks}
        if len(ids) != len(self.tasks):
            raise SchemaError("duplicate task ids")
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in ids:
                    raise SchemaError(f"task {t.id} depends on unknown task {dep}")
        if self.feasible and not self.files:
            raise SchemaError("a feasible plan must list files")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Budget:
    max_repairs: int = 3
    exec_timeout_s: int = 120
    memory_mb: int = 2048
    max_output_kb: int = 256
    max_wall_s: int = 900
    max_tokens: int = 200_000

    def validate(self) -> "Budget":
        if not (0 <= self.max_repairs <= 10):
            raise SchemaError("max_repairs must be between 0 and 10")
        if not (1 <= self.exec_timeout_s <= 1800):
            raise SchemaError("exec_timeout_s must be between 1 and 1800")
        if not (128 <= self.memory_mb <= 16384):
            raise SchemaError("memory_mb must be between 128 and 16384")
        if not (1 <= self.max_output_kb <= 4096):
            raise SchemaError("max_output_kb must be between 1 and 4096")
        if not (10 <= self.max_wall_s <= 7200):
            raise SchemaError("max_wall_s must be between 10 and 7200")
        return self

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        d = d or {}
        allowed = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        try:
            b = cls(**{k: int(v) for k, v in allowed.items()})
        except (TypeError, ValueError):
            raise SchemaError("budget values must be integers")
        return b.validate()

    def to_dict(self) -> dict:
        return asdict(self)
