"""Application service layer: everything the HTTP API does, framework-free so it is unit-testable."""
from __future__ import annotations
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from . import report as report_mod
from .config import Settings
from .graph import Engine, S, TERMINAL, GateError
from .integrations.github import GitHubClient, GitHubError
from .llm.provider import get_provider, LLMConfigError
from .sandbox import get_sandbox
from .schemas import Budget, SchemaError
from .security import safe_relpath, SecurityError
from .storage import Storage

RUNNING_STATES = {S.INGESTING, S.PLANNING, S.RESOLVING, S.GENERATING, S.TESTING, S.REPAIRING, S.EXECUTING,
                  S.EVALUATING, S.REPORTING}
SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers", "fisher_lda_iris_fixture.pdf")


class RateLimited(Exception):
    pass


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute, self._hits = per_minute, defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise RateLimited("rate limit exceeded; retry later")
            q.append(now)


class JobRunner:
    def __init__(self, sync: bool = False, workers: int = 2):
        self.sync = sync
        self._pool = None if sync else ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p2p-job")

    def submit(self, fn: Callable, *args) -> None:
        if self.sync:
            fn(*args)
        else:
            self._pool.submit(fn, *args)


class NotFound(Exception):
    pass


class Service:
    def __init__(self, settings: Settings, sync_jobs: bool = False, storage: Storage | None = None,
                 sandbox=None, provider="auto", github: GitHubClient | None = None):
        self.s = settings
        self.db = storage or Storage(settings.db_path)
        self.llm_error: str | None = None
        if provider == "auto":
            try:
                provider = get_provider(settings)
            except LLMConfigError as e:
                self.llm_error, provider = str(e), None
        self.llm = provider
        self.sandbox = sandbox or get_sandbox(settings.sandbox_mode, settings.sandbox_image)
        self.engine = Engine(settings, self.db, self.sandbox, self.llm)
        self.jobs = JobRunner(sync_jobs)
        self.limiter = RateLimiter(settings.rate_limit_per_min)
        self.github = github or GitHubClient(settings.github_token)
        self.recover()

    def recover(self) -> None:
        """Projects left in a running state by a crash/restart cannot continue; mark them failed honestly."""
        for p in self.db.list_projects(1000):
            if p["status"] in RUNNING_STATES:
                self.db.add_event(p["id"], "recover", "Interrupted by server restart", "error")
                self.db.set_status(p["id"], S.FAILED)

    # ---- health -------------------------------------------------------------------
    def health(self) -> dict:
        return {"status": "ok"}

    def ready(self) -> dict:
        checks = {"database": False, "sandbox": self.sandbox.mode}
        try:
            checks["database"] = self.db.ping()
        except Exception:
            pass
        return {"ready": checks["database"], **checks, "sandbox_isolated": self.sandbox.isolated}

    def integrations(self) -> dict:
        return {
            "llm": ({**self.llm.info(), "configured": True} if self.llm else
                    {"configured": False, "mode": "demo",
                     "detail": self.llm_error or "GEMINI_API_KEY not set - running in DEMO MODE (deterministic template, no LLM reasoning)"}),
            "sandbox": {"mode": self.sandbox.mode, "isolated": self.sandbox.isolated, "detail": self.sandbox.description},
            "github": self.github.status(),
            "datasets": {"builtin": ["sklearn:iris"], "huggingface": "configured" if self.s.hf_token else
                         "public datasets only (set HF_TOKEN for gated ones; needs `huggingface_hub`)"},
        }

    # ---- projects -------------------------------------------------------------------
    def _rate(self, client: str) -> None:
        self.limiter.check(client)

    def create_project(self, filename: str, data: bytes, budget: dict | None, dataset_hint: str = "",
                       inject_fault: bool = False, client: str = "anon") -> str:
        self._rate(client)
        if not filename.lower().endswith(".pdf"):
            raise SchemaError("only .pdf files are accepted")
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise SchemaError(f"file larger than {self.s.max_upload_mb} MB")
        b = Budget.from_dict(budget)
        if dataset_hint and not re.match(r"^(sklearn:[a-z_]+|hf:[\w\-./]+|upload:[\w\-./]+)$", dataset_hint):
            raise SchemaError("dataset source must look like sklearn:iris, hf:<id> or upload:<name>")
        from .ingest import IngestError
        try:
            pid = self.engine.create(os.path.splitext(os.path.basename(filename))[0][:80], data, b, dataset_hint, inject_fault)
        except IngestError as e:
            raise SchemaError(str(e))
        self.db.audit("project_created", "user", pid, {"filename": os.path.basename(filename), "inject_fault": inject_fault})
        self.jobs.submit(self.engine.run_ingest, pid)
        return pid

    def create_demo(self, inject_fault: bool = True, client: str = "anon") -> str:
        with open(SAMPLE_PDF, "rb") as f:
            return self.create_project("fisher_lda_iris_fixture.pdf", f.read(), {}, "", inject_fault, client)

    def _get(self, pid: str) -> dict:
        p = self.db.get_project(pid)
        if not p:
            raise NotFound(pid)
        return p

    def list_projects(self) -> list[dict]:
        out = []
        for p in self.db.list_projects():
            st = p["data"].get("state", {})
            out.append({"id": p["id"], "name": p["name"], "status": p["status"], "created": p["created"],
                        "updated": p["updated"], "mode": st.get("mode"),
                        "reproduction_status": st.get("reproduction_status") if p["status"] in TERMINAL else None,
                        "repair_attempts": len(st.get("attempts", []))})
        return out

    def get_project(self, pid: str) -> dict:
        p = self._get(pid)
        st = dict(p["data"].get("state", {}))
        st.pop("report_md", None)
        st["files"] = sorted(st.get("files", {}))     # file list only; content via /files
        return {"id": p["id"], "name": p["name"], "status": p["status"], "created": p["created"], "state": st}

    def events(self, pid: str, after: int = 0) -> list[dict]:
        self._get(pid)
        return self.db.events(pid, after)

    def approve_spec(self, pid: str, edits: dict | None, client: str = "anon") -> None:
        self._get(pid); self._rate(client)
        self.engine.approve_spec(pid, edits)
        self.jobs.submit(self.engine.run_plan, pid)

    def approve_plan(self, pid: str, accept: bool, client: str = "anon") -> None:
        self._get(pid); self._rate(client)
        self.engine.approve_plan(pid, accept)
        self.jobs.submit(self.engine.run_execution, pid)

    def cancel(self, pid: str) -> None:
        self._get(pid)
        self.engine.request_cancel(pid)

    # ---- artifacts --------------------------------------------------------------------------
    def files(self, pid: str) -> dict:
        st = self.engine.load(pid)
        return {"files": sorted(st.files), "diffs": [{"attempt": a["n"], "file": a["patch"]["file"], "diff": a["diff"]} for a in st.attempts]}

    def file(self, pid: str, path: str) -> str:
        rel = safe_relpath(path)
        st = self.engine.load(pid)
        if rel not in st.files:
            raise NotFound(path)
        return st.files[rel]

    def report_md(self, pid: str) -> str:
        st = self.engine.load(pid)
        if not st.report_md:
            raise NotFound("report not generated yet")
        return st.report_md

    def manifest(self, pid: str) -> dict:
        st = self.engine.load(pid)
        if not st.manifest:
            raise NotFound("manifest not generated yet")
        return st.manifest

    def zip_bytes(self, pid: str) -> bytes:
        st = self.engine.load(pid)
        if not st.files or not st.manifest:
            raise NotFound("project has no generated files yet")
        return report_mod.build_zip(st.files, st.report_md, st.manifest)

    # ---- GitHub ---------------------------------------------------------------------------------
    def github_inspect(self, repo: str, client: str = "anon") -> dict:
        self._rate(client)
        out = self.github.inspect_repo(repo)
        self.db.audit("github_inspect", "user", None, {"repo": repo})
        return out

    def github_export(self, pid: str, name: str, confirm: bool, private: bool = True, client: str = "anon") -> dict:
        self._rate(client)
        st = self.engine.load(pid)
        if not st.files:
            raise NotFound("nothing to export")
        files = dict(st.files)
        files["REPORT.md"] = st.report_md
        files["reproducibility_manifest.json"] = json.dumps(st.manifest, indent=2)
        self.db.audit("github_export_requested", "user", pid, {"name": name, "confirm": confirm, "private": private})
        out = self.github.export_project(name, files, confirm, private)   # raises unless confirm is True
        self.db.audit("github_export_done", "user", pid, out)
        return out
