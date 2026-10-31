"""FastAPI application. All behaviour lives in app.services; this file is HTTP glue only.

NOTE: this module could not be executed in the offline build sandbox (FastAPI was not installable there).
It is syntax-checked, and the service layer it wraps is covered by the test-suite. Run `pytest` after
`pip install -r requirements.txt` to exercise tests/test_api.py, which uses FastAPI's TestClient.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings
from .graph import GateError, TERMINAL
from .integrations.github import GitHubError
from .schemas import SchemaError
from .security import SecurityError
from .services import NotFound, RateLimited, Service

log = logging.getLogger("paper2product")


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(r.created)), "level": r.levelname,
             "logger": r.name, "msg": r.getMessage(), "request_id": getattr(r, "request_id", None)}
        return json.dumps(d)


def _setup_logging() -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root = logging.getLogger("paper2product")
    root.handlers[:] = [h]
    root.setLevel(logging.INFO)


class SpecEdits(BaseModel):
    edits: Optional[dict] = None


class PlanApproval(BaseModel):
    accept_assumptions: bool = False


class GithubInspect(BaseModel):
    repo: str = Field(min_length=3, max_length=140)


class GithubExport(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    confirm: bool = False
    private: bool = True


class DemoRequest(BaseModel):
    inject_fault: bool = True


def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    _setup_logging()
    settings = settings or Settings.from_env()
    svc = service or Service(settings)
    app = FastAPI(title="Paper2Product AI", version="1.0.0")
    app.state.svc = svc
    frontend = os.environ.get("P2P_FRONTEND_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.rid = rid
        open_paths = ("/healthz", "/readyz", "/", "/docs", "/openapi.json")
        if settings.api_key and request.url.path not in open_paths and request.headers.get("x-api-key") != settings.api_key:
            return JSONResponse({"error": "unauthorized", "request_id": rid}, status_code=401)
        t0 = time.time()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, resp.status_code, (time.time() - t0) * 1000,
                 extra={"request_id": rid})
        return resp

    def err(request: Request, status: int, msg: str) -> JSONResponse:
        return JSONResponse({"error": msg, "request_id": getattr(request.state, "rid", None)}, status_code=status)

    @app.exception_handler(SchemaError)
    async def _schema(request, e):
        return err(request, 400, str(e))

    @app.exception_handler(SecurityError)
    async def _sec(request, e):
        return err(request, 400, f"rejected by security policy: {e}")

    @app.exception_handler(NotFound)
    async def _nf(request, e):
        return err(request, 404, "not found")

    @app.exception_handler(GateError)
    async def _gate(request, e):
        return err(request, 409, str(e))

    @app.exception_handler(RateLimited)
    async def _rl(request, e):
        return err(request, 429, str(e))

    @app.exception_handler(GitHubError)
    async def _gh(request, e):
        return err(request, 502, str(e))

    @app.exception_handler(Exception)
    async def _any(request, e):
        log.exception("unhandled error", extra={"request_id": getattr(request.state, "rid", None)})
        return err(request, 500, "internal error")   # never leak stack traces or env details

    def client(request: Request) -> str:
        return request.client.host if request.client else "anon"

    # ---- health ----
    @app.get("/healthz")
    def healthz():
        return svc.health()

    @app.get("/readyz")
    def readyz():
        r = svc.ready()
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/api/integrations")
    def integrations():
        return svc.integrations()

    # ---- projects ----
    @app.get("/api/projects")
    def list_projects():
        return svc.list_projects()

    @app.post("/api/projects", status_code=202)
    async def create_project(request: Request, file: UploadFile = File(...), dataset_source: str = Form(""),
                             max_repairs: int = Form(3), exec_timeout_s: int = Form(120), inject_fault: bool = Form(False)):
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        pid = svc.create_project(file.filename or "paper.pdf", data,
                                 {"max_repairs": max_repairs, "exec_timeout_s": exec_timeout_s}, dataset_source,
                                 inject_fault, client(request))
        return {"id": pid}

    @app.post("/api/demo", status_code=202)
    def demo(request: Request, body: DemoRequest = DemoRequest()):
        return {"id": svc.create_demo(body.inject_fault, client(request))}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return svc.get_project(pid)

    @app.get("/api/projects/{pid}/events")
    def events(pid: str, after: int = 0):
        return svc.events(pid, after)

    @app.get("/api/projects/{pid}/stream")
    async def stream(pid: str, after: int = 0):
        svc.get_project(pid)

        async def gen():
            last = after
            while True:
                evs = svc.events(pid, last)
                for e in evs:
                    last = e["id"]
                    yield f"id: {e['id']}\ndata: {json.dumps(e)}\n\n"
                if svc.get_project(pid)["status"] in TERMINAL and not evs:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/projects/{pid}/approve-spec", status_code=202)
    def approve_spec(pid: str, body: SpecEdits, request: Request):
        svc.approve_spec(pid, body.edits, client(request))
        return {"ok": True}

    @app.post("/api/projects/{pid}/approve-plan", status_code=202)
    def approve_plan(pid: str, body: PlanApproval, request: Request):
        svc.approve_plan(pid, body.accept_assumptions, client(request))
        return {"ok": True}

    @app.post("/api/projects/{pid}/cancel")
    def cancel(pid: str):
        svc.cancel(pid)
        return {"ok": True}

    # ---- artifacts ----
    @app.get("/api/projects/{pid}/files")
    def files(pid: str):
        return svc.files(pid)

    @app.get("/api/projects/{pid}/file")
    def file(pid: str, path: str):
        return PlainTextResponse(svc.file(pid, path))

    @app.get("/api/projects/{pid}/report")
    def report(pid: str):
        return PlainTextResponse(svc.report_md(pid), headers={"Content-Disposition": f'attachment; filename="REPORT-{pid}.md"'})

    @app.get("/api/projects/{pid}/manifest")
    def manifest(pid: str):
        return svc.manifest(pid)

    @app.get("/api/projects/{pid}/download")
    def download(pid: str):
        return Response(svc.zip_bytes(pid), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="paper2product-{pid}.zip"'})

    @app.get("/api/projects/{pid}/audit")
    def audit(pid: str):
        return svc.db.audit_log(pid)

    # ---- GitHub ----
    @app.post("/api/github/inspect")
    def gh_inspect(body: GithubInspect, request: Request):
        return svc.github_inspect(body.repo, client(request))

    @app.post("/api/projects/{pid}/github-export")
    def gh_export(pid: str, body: GithubExport, request: Request):
        return svc.github_export(pid, body.name, body.confirm, body.private, client(request))

    @app.get("/")
    def index():
        return FileResponse(os.path.join(frontend, "index.html"))

    return app


app = create_app()
