"""Explicit agent workflow: typed shared state, validated transitions, human gates, bounded repair loop.

Each node is a plain function over WorkflowState so it can be dropped into LangGraph unchanged
(LangGraph could not be installed in the offline build environment; the state machine here is self-contained).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import ingest, planner, codegen, repair, evaluation, report
from .config import Settings
from .integrations import datasets
from .llm.provider import LLMError, LLMProvider
from .sandbox import SandboxResult
from .schemas import Budget, Plan, PlanTask, ResearchSpec, SchemaError
from .security import SecurityError, redact, scan_python
from .storage import Storage


class S:
    CREATED = "created"; INGESTING = "ingesting"; AWAITING_SPEC = "awaiting_spec_approval"
    PLANNING = "planning"; AWAITING_PLAN = "awaiting_plan_approval"; RESOLVING = "resolving_data"
    GENERATING = "generating"; TESTING = "testing"; REPAIRING = "repairing"; EXECUTING = "executing"
    EVALUATING = "evaluating"; REPORTING = "reporting"; COMPLETED = "completed"
    FAILED = "failed"; BLOCKED = "blocked"; CANCELLED = "cancelled"


TERMINAL = {S.COMPLETED, S.FAILED, S.BLOCKED, S.CANCELLED}
_X = {S.CANCELLED}
TRANSITIONS: dict[str, set[str]] = {
    S.CREATED: {S.INGESTING} | _X,
    S.INGESTING: {S.AWAITING_SPEC, S.FAILED} | _X,
    S.AWAITING_SPEC: {S.PLANNING} | _X,
    S.PLANNING: {S.AWAITING_PLAN, S.BLOCKED, S.FAILED} | _X,
    S.AWAITING_PLAN: {S.RESOLVING} | _X,
    S.RESOLVING: {S.GENERATING, S.BLOCKED, S.FAILED} | _X,
    S.GENERATING: {S.TESTING, S.BLOCKED, S.FAILED} | _X,
    S.TESTING: {S.REPAIRING, S.EXECUTING, S.BLOCKED, S.FAILED} | _X,
    S.REPAIRING: {S.TESTING, S.BLOCKED, S.FAILED} | _X,
    S.EXECUTING: {S.EVALUATING, S.REPAIRING, S.BLOCKED, S.FAILED} | _X,
    S.EVALUATING: {S.REPORTING, S.FAILED} | _X,
    S.REPORTING: {S.COMPLETED, S.FAILED},
    S.COMPLETED: set(), S.FAILED: set(), S.BLOCKED: set(), S.CANCELLED: set(),
}


class InvalidTransition(Exception):
    pass


class Cancelled(Exception):
    pass


class GateError(Exception):
    """An action was attempted in the wrong state / without required approval."""


def check_transition(cur: str, new: str) -> None:
    if new not in TRANSITIONS.get(cur, set()):
        raise InvalidTransition(f"{cur} -> {new} is not allowed")


@dataclass
class WorkflowState:
    project_id: str = ""
    mode: str = "demo"                     # demo | llm
    llm_info: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    dataset_hint: str = ""
    inject_fault: bool = False
    spec: Optional[dict] = None
    approved_assumptions: list = field(default_factory=list)
    plan: Optional[dict] = None
    dataset: Optional[dict] = None
    files: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    final_tests: Optional[dict] = None
    run_meta: Optional[dict] = None
    metrics: Optional[dict] = None
    script_metric_check: str = ""
    comparisons: list = field(default_factory=list)
    reproduction_status: str = "unable_to_verify"
    reproduction_note: str = ""
    sandbox: Optional[dict] = None
    usage: dict = field(default_factory=dict)
    started_at: float = 0.0
    blocked_reason: str = ""
    report_md: str = ""
    manifest: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def summarize_unittest(res: SandboxResult) -> str:
    m = re.search(r"Ran (\d+) tests? in", res.stderr)
    ran = m.group(1) if m else "?"
    if res.timed_out:
        return f"timed out after {res.duration_s}s"
    if res.ok and "OK" in res.stderr:
        return f"{ran} tests passed"
    f = re.search(r"FAILED \(([^)]*)\)", res.stderr)
    return f"{ran} tests ran, FAILED ({f.group(1)})" if f else f"failed (exit {res.exit_code})"


class Engine:
    def __init__(self, settings: Settings, storage: Storage, sandbox, provider: Optional[LLMProvider]):
        self.s, self.db, self.sandbox, self.llm = settings, storage, sandbox, provider
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ---- plumbing --------------------------------------------------------
    def project_dir(self, pid: str) -> str:
        return os.path.join(self.s.data_dir, "projects", pid)

    def work_dir(self, pid: str) -> str:
        return os.path.join(self.project_dir(pid), "work")

    def cancel_event(self, pid: str) -> threading.Event:
        with self._lock:
            return self._cancel.setdefault(pid, threading.Event())

    def request_cancel(self, pid: str) -> None:
        self.cancel_event(pid).set()
        p = self.db.get_project(pid)
        if p and p["status"] in (S.AWAITING_SPEC, S.AWAITING_PLAN, S.CREATED):
            self._set_status(pid, S.CANCELLED)  # not running: cancel immediately
            self.db.add_event(pid, "cancel", "Cancelled by user", "warn")
        self.db.audit("cancel", "user", pid, {})

    def load(self, pid: str) -> WorkflowState:
        p = self.db.get_project(pid)
        if not p:
            raise KeyError(pid)
        return WorkflowState.from_dict(p["data"].get("state", {"project_id": pid}))

    def save(self, st: WorkflowState) -> None:
        if self.llm:
            st.usage = self.llm.usage.to_dict()
        self.db.update_data(st.project_id, state=st.to_dict())

    def status(self, pid: str) -> str:
        return self.db.get_project(pid)["status"]

    def _set_status(self, pid: str, new: str) -> None:
        cur = self.status(pid)
        check_transition(cur, new)
        self.db.set_status(pid, new)
        self.db.add_event(pid, "state", f"{cur} -> {new}", "info", {"from": cur, "to": new})

    def _log(self, st: WorkflowState, stage: str, msg: str, level: str = "info", data: dict | None = None) -> None:
        self.db.add_event(st.project_id, stage, msg, level, data)

    def _guard(self, st: WorkflowState) -> None:
        if self.cancel_event(st.project_id).is_set():
            raise Cancelled()
        b = st.budget
        if time.time() - st.started_at > b.get("max_wall_s", 900):
            raise TimeoutError("wall-clock budget exhausted")
        if self.llm and self.llm.usage.total > b.get("max_tokens", 200_000):
            raise TimeoutError("token budget exhausted")

    # ---- creation ----------------------------------------------------------
    def create(self, name: str, pdf_bytes: bytes, budget: Budget, dataset_hint: str = "", inject_fault: bool = False) -> str:
        ingest.extract_pages(pdf_bytes, self.s.max_upload_mb * 1024 * 1024, self.s.max_pdf_pages)  # validate early
        pid = self.db.create_project(name[:120], {})
        os.makedirs(self.project_dir(pid), exist_ok=True)
        with open(os.path.join(self.project_dir(pid), "input.pdf"), "wb") as f:
            f.write(pdf_bytes)
        st = WorkflowState(project_id=pid, mode="llm" if self.llm else "demo",
                           llm_info=self.llm.info() if self.llm else {"provider": None, "mock": False,
                                                                       "note": "no LLM configured - demo mode"},
                           budget=budget.to_dict(), dataset_hint=dataset_hint, inject_fault=inject_fault,
                           started_at=time.time())
        self.save(st)
        self.db.add_event(pid, "create", f"Project created in {st.mode.upper()} mode", "info",
                          {"inject_fault": inject_fault})
        return pid

    # ---- Stage A -----------------------------------------------------------
    def run_ingest(self, pid: str) -> None:
        st = self.load(pid)
        try:
            self._set_status(pid, S.INGESTING)
            self._guard(st)
            with open(os.path.join(self.project_dir(pid), "input.pdf"), "rb") as fh:
                pdf = fh.read()
            pages = ingest.extract_pages(pdf, self.s.max_upload_mb * 1024 * 1024, self.s.max_pdf_pages)
            self._log(st, "ingest", f"Extracted text from {len(pages)} pages (tool: pdf extractor)")
            spec = ingest.build_spec(pages, self.llm)
            st.spec = spec.to_dict()
            self._log(st, "ingest", f"Specification extracted via {spec.extraction_method}", data={"missing": spec.missing_info})
            if spec.security_flags:
                self._log(st, "security", f"{len(spec.security_flags)} prompt-injection pattern(s) found in the document; "
                          "content is treated as data only", "warn", {"flags": spec.security_flags})
            self.save(st)
            self._set_status(pid, S.AWAITING_SPEC)
        except Cancelled:
            self._cancelled(st)
        except Exception as e:
            self._fail(st, "ingest", e)

    # ---- gate 1 -> Stage B ------------------------------------------------------
    def approve_spec(self, pid: str, edits: dict | None, actor: str = "user") -> None:
        if self.status(pid) != S.AWAITING_SPEC:
            raise GateError("project is not awaiting spec approval")
        st = self.load(pid)
        if edits:
            merged = {**st.spec, **edits}
            st.spec = ResearchSpec.from_dict(merged).to_dict()   # raises SchemaError on invalid edits
            self.db.audit("spec_edited", actor, pid, {"fields": sorted(edits)})
            self._log(st, "approval", f"User corrected spec fields: {sorted(edits)}")
        self.db.audit("spec_approved", actor, pid, {})
        self._log(st, "approval", "Specification approved by user")
        self.save(st)

    def run_plan(self, pid: str) -> None:
        st = self.load(pid)
        try:
            self._set_status(pid, S.PLANNING)
            self._guard(st)
            spec = ResearchSpec.from_dict(st.spec)
            plan = planner.make_plan(spec, self.llm)
            st.plan = plan.to_dict()
            self.save(st)
            if not plan.feasible:
                st.blocked_reason = plan.infeasible_reason
                self.save(st)
                self._log(st, "plan", plan.infeasible_reason, "warn")
                self._set_status(pid, S.BLOCKED)
                return
            self._log(st, "plan", f"Plan with {len(plan.tasks)} tasks ({plan.template}); {len(plan.assumptions)} assumption(s) need approval")
            self._set_status(pid, S.AWAITING_PLAN)
        except Cancelled:
            self._cancelled(st)
        except Exception as e:
            self._fail(st, "plan", e)

    # ---- gate 2 -----------------------------------------------------------------------
    def approve_plan(self, pid: str, accept_assumptions: bool, actor: str = "user") -> None:
        if self.status(pid) != S.AWAITING_PLAN:
            raise GateError("project is not awaiting plan approval")
        st = self.load(pid)
        plan = st.plan or {}
        if plan.get("assumptions") and not accept_assumptions:
            raise GateError("the plan contains assumptions that must be explicitly accepted")
        st.approved_assumptions = list(plan.get("assumptions", []))
        self.db.audit("plan_approved", actor, pid, {"assumptions": st.approved_assumptions})
        self._log(st, "approval", "Plan and assumptions approved; execution permission granted")
        st.started_at = time.time()
        self.save(st)

    # ---- Stages C-H ---------------------------------------------------------------------------
    def run_execution(self, pid: str) -> None:
        st = self.load(pid)
        try:
            plan = Plan(**{**st.plan, "tasks": [PlanTask(**t) for t in st.plan["tasks"]]})
            spec = ResearchSpec.from_dict(st.spec)
            # Stage C
            self._set_status(pid, S.RESOLVING)
            self._guard(st)
            try:
                ds = datasets.resolve_for_spec(spec.datasets, st.dataset_hint, self.s.hf_token)
            except datasets.ResourceError as e:
                return self._block(st, f"Dataset unavailable: {e}")
            st.dataset = ds.to_dict()
            self._log(st, "dataset", f"Resolved dataset {ds.id} ({ds.n_rows} rows, license: {ds.license})", data={"warnings": ds.warnings})
            if not ds.validated:
                return self._block(st, f"Dataset {ds.id} is metadata-only and not yet validated for use in the sandbox.")
            # Stage D
            self._set_status(pid, S.GENERATING)
            self._guard(st)
            if plan.template == "fisher_lda_iris":
                files = codegen.generate_template_project(spec, plan, st.inject_fault)
                self._log(st, "codegen", f"Generated {len(files)} files from deterministic template (no LLM)",
                          data={"files": sorted(files)})
                if st.inject_fault:
                    self._log(st, "codegen", "DEMO: a deliberate bug was injected into lda_repro/lda.py to exercise the repair loop", "warn")
            else:
                if not self.llm:
                    return self._block(st, "Code generation for this paper requires an LLM (set GEMINI_API_KEY).")
                files = codegen.generate_llm_project(spec, plan, self.llm)
                self._log(st, "codegen", f"LLM generated {len(files)} files; passed path/size/static-security validation")
            wd = self.work_dir(pid)
            shutil.rmtree(wd, ignore_errors=True)
            codegen.write_project(wd, files)
            st.files = files
            st.sandbox = {"mode": self.sandbox.mode, "isolated": self.sandbox.isolated, "note": self.sandbox.description}
            self.save(st)
            ok = self._test_and_run(st)
            if not ok:
                return
            self._evaluate(st, spec)
        except Cancelled:
            self._cancelled(st)
        except TimeoutError as e:
            self._block(st, f"Budget exhausted: {e}")
        except Exception as e:
            self._fail(st, "execution", e)

    # ---- Stage E/F loop -----------------------------------------------------------
    def _run(self, st: WorkflowState, stage: str, args: list[str]) -> SandboxResult:
        b = st.budget
        res = self.sandbox.run_python(self.work_dir(st.project_id), args, b["exec_timeout_s"], b["memory_mb"], b["max_output_kb"])
        n = len(st.runs) + 1
        summary = summarize_unittest(res) if stage == "tests" else ("completed" if res.ok else "failed")
        st.runs.append({"n": n, "stage": stage, "ok": res.ok, "exit_code": res.exit_code, "duration_s": res.duration_s,
                        "summary": summary, "stdout": redact(res.stdout)[-4000:], "stderr": redact(res.stderr)[-4000:],
                        "mode": res.mode, "isolated": res.isolated, "timed_out": res.timed_out})
        self._log(st, "exec", f"[{stage}] sandbox({res.mode}) exit={res.exit_code} {summary}", "info" if res.ok else "error",
                  {"run": n})
        return res

    def _test_and_run(self, st: WorkflowState) -> bool:
        pid, b = st.project_id, st.budget
        wd = self.work_dir(pid)
        while True:
            self._guard(st)
            self._set_status(pid, S.TESTING)
            problems = [p for path, src in st.files.items() if path.endswith(".py") for p in scan_python(src, path)]
            if problems:
                self._log(st, "static", f"Static policy check failed: {problems[0]}", "error")
                res = SandboxResult(["static"], 1, "", "SecurityError: " + problems[0], 0.0)
                stage = "static"
            else:
                self._log(st, "static", "Static checks passed (syntax + security policy)")
                res = self._run(st, "tests", ["-m", "unittest", "discover", "-s", "tests", "-t", "."])
                stage = "tests"
                if res.ok:
                    self._set_status(pid, S.EXECUTING)
                    res = self._run(st, "experiment", ["run_experiment.py"])
                    stage = "experiment"
                    if res.ok:
                        if st.attempts:
                            st.attempts[-1]["outcome"] = "fixed: tests and experiment pass"
                        st.final_tests = next(r for r in reversed(st.runs) if r["stage"] == "tests")
                        self.save(st)
                        return True
            if st.attempts:
                st.attempts[-1]["outcome"] = f"still failing at {stage}"
            # failure -> bounded repair
            if len(st.attempts) >= b["max_repairs"]:
                st.final_tests = next((r for r in reversed(st.runs) if r["stage"] == "tests"), None)
                self._block(st, f"Repair budget exhausted after {len(st.attempts)} attempt(s); last failure at {stage}.")
                return False
            t0 = time.time()
            diag = repair.diagnose(res.stdout, res.stderr, st.files, res.timed_out)
            self._log(st, "diagnose", f"{diag.category}: {diag.summary}", "warn", diag.to_dict())
            patch, tool_calls = None, []
            if diag.repairable_by_rule:
                patch = repair.rule_based_patch(diag, st.files)
                tool_calls.append("rule_based_patch")
            if patch is None and self.llm and diag.file and diag.category not in ("timeout", "memory"):
                try:
                    patch = repair.llm_patch(diag, st.files, res.stderr, res.stdout, self.llm)
                    tool_calls.append("llm_patch")
                except LLMError as e:
                    self._log(st, "repair", f"LLM repair call failed: {e}", "error")
            if patch is None:
                st.final_tests = next((r for r in reversed(st.runs) if r["stage"] == "tests"), None)
                self._block(st, f"No safe automated repair available for: {diag.summary}")
                return False
            try:
                diff = repair.validate_patch(patch, st.files)
            except repair.PatchRejected as e:
                self._block(st, f"Proposed patch rejected by policy: {e}")
                return False
            self._set_status(pid, S.REPAIRING)
            codegen.write_project(wd, {patch.file: patch.new_content})
            st.files[patch.file] = patch.new_content
            attempt = {"n": len(st.attempts) + 1, "stage": stage, "diagnosis": diag.to_dict(), "patch": patch.to_dict(),
                       "diff": diff, "changed_files": [patch.file], "tool_calls": tool_calls,
                       "elapsed_s": round(time.time() - t0, 3), "outcome": "pending"}
            st.attempts.append(attempt)
            self._log(st, "repair", f"Attempt {attempt['n']}/{b['max_repairs']}: patched {patch.file} ({patch.source}) - {patch.rationale}",
                      "info", {"diff": diff})
            self.save(st)

    # ---- Stage G ---------------------------------------------------------------------------
    def _evaluate(self, st: WorkflowState, spec: ResearchSpec) -> None:
        pid = st.project_id
        self._set_status(pid, S.EVALUATING)
        self._guard(st)
        wd = self.work_dir(pid)
        try:
            with open(os.path.join(wd, "artifacts", "predictions.json")) as fh:
                preds = json.load(fh)
            with open(os.path.join(wd, "artifacts", "run_meta.json")) as fh:
                meta = json.load(fh)
            m = evaluation.classification_metrics(preds["y_true"], preds["y_pred"], preds.get("labels"))
        except Exception as e:
            return self._fail(st, "evaluate", e)
        st.metrics, st.run_meta = m, meta
        reported_acc = meta.get("script_reported_accuracy")
        if reported_acc is not None:
            ok = abs(reported_acc - m["accuracy"]) < 1e-9
            st.script_metric_check = ("agrees with independent recomputation" if ok else
                                      f"MISMATCH: script said {reported_acc:.6f}, evaluator computed {m['accuracy']:.6f}")
            if not ok:
                self._log(st, "evaluate", st.script_metric_check, "error")
        self._log(st, "evaluate", f"Computed metrics independently from predictions: accuracy={m['accuracy']:.4f} "
                  f"macro-F1={m['f1']:.4f} (n={m['n']})")
        run = {"dataset_name": (st.dataset or {}).get("name", ""), "test_size": meta.get("test_size")}
        measured = {k: m[k] for k in ("accuracy", "precision", "recall", "f1")}
        comps = evaluation.compare_results(spec.reported_results, measured, run)
        st.comparisons = [c.to_dict() for c in comps]
        tests_ok = bool(st.final_tests and st.final_tests["ok"])
        st.reproduction_status, st.reproduction_note = evaluation.overall_status(comps, tests_ok, True)
        if st.script_metric_check.startswith("MISMATCH"):
            st.reproduction_status, st.reproduction_note = "unable_to_verify", "Script-reported and independently computed metrics disagree."
        self._log(st, "evaluate", f"Reproduction status: {st.reproduction_status}. {st.reproduction_note}")
        self.save(st)
        self._set_status(pid, S.REPORTING)
        self._make_report(st)
        self._set_status(pid, S.COMPLETED)

    # ---- Stage H ---------------------------------------------------------------------------
    def _make_report(self, st: WorkflowState) -> None:
        d = st.to_dict()
        st.manifest = report.build_manifest(d)
        st.report_md = report.build_report_md(d)
        self.save(st)
        self._log(st, "report", "Report and reproducibility manifest generated")

    # ---- terminal helpers -----------------------------------------------------------------------
    def _block(self, st: WorkflowState, reason: str) -> None:
        st.blocked_reason = reason
        st.reproduction_status, st.reproduction_note = "unable_to_verify", f"Run blocked: {reason}"
        self._log(st, "blocked", reason, "error")
        self.save(st)
        self._make_report(st)
        self._set_status(st.project_id, S.BLOCKED)

    def _fail(self, st: WorkflowState, stage: str, e: Exception) -> None:
        # full details go to server logs only; users get a safe message
        import logging
        logging.getLogger("paper2product").exception("stage %s failed", stage)
        msg = redact(f"{type(e).__name__}: {str(e)[:300]}", [self.s.llm_api_key, self.s.github_token])
        st.blocked_reason = msg
        self._log(st, stage, f"Stage failed: {msg}", "error")
        self.save(st)
        cur = self.status(st.project_id)
        if cur not in TERMINAL:
            self._set_status(st.project_id, S.FAILED)

    def _cancelled(self, st: WorkflowState) -> None:
        self._log(st, "cancel", "Run cancelled", "warn")
        cur = self.status(st.project_id)
        if cur not in TERMINAL:
            self._set_status(st.project_id, S.CANCELLED)
