#!/usr/bin/env python3
"""Run ON YOUR MACHINE to verify the parts that could not be verified at build time.

    python scripts/verify_setup.py            # checks everything it can
    GEMINI_API_KEY=... GITHUB_TOKEN=... python scripts/verify_setup.py

Prints PASS / FAIL / SKIP per check (SKIP = not configured, never counted as a pass). Exit code 1 if any FAIL.
"""
import importlib, importlib.util, os, sys, tempfile, textwrap, unittest
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))
results = []


def check(name):
    def deco(fn):
        try:
            r = fn()
            results.append((name, "SKIP" if r and r.startswith("skip:") else "PASS", (r or "")[5:] if r and r.startswith("skip:") else (r or "")))
        except Exception as e:
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:200]}"))
        return fn
    return deco


@check("python dependencies (fastapi, pydantic, uvicorn, multipart, numpy, scikit-learn, pypdf)")
def _():
    missing = [m for m in ("fastapi", "pydantic", "uvicorn", "multipart", "numpy", "sklearn", "pypdf") if not importlib.util.find_spec(m)]
    assert not missing, f"missing: {missing} -> pip install -r backend/requirements.txt"
    return "PyMuPDF " + ("present" if importlib.util.find_spec("fitz") else "absent (pypdf fallback)")


@check("full unit suite incl. API tests (backend/tests)")
def _():
    suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "backend", "tests"), top_level_dir=os.path.join(ROOT, "backend"))
    res = unittest.TextTestRunner(stream=open(os.devnull, "w")).run(suite)
    assert res.wasSuccessful(), f"{len(res.failures)} failures, {len(res.errors)} errors"
    return f"{res.testsRun} run, {len(res.skipped)} skipped"


@check("FastAPI app boots and serves /healthz, /readyz, frontend")
def _():
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app
    from app.services import Service
    d = tempfile.mkdtemp()
    s = Settings(data_dir=d, db_path=os.path.join(d, "v.db"))
    c = TestClient(create_app(s, Service(s, sync_jobs=True)))
    assert c.get("/healthz").status_code == 200 and c.get("/readyz").status_code == 200
    assert "Paper2Product" in c.get("/").text
    r = c.post("/api/demo", json={"inject_fault": True}); assert r.status_code == 202
    pid = r.json()["id"]
    assert c.post(f"/api/projects/{pid}/approve-spec", json={}).status_code == 202
    assert c.post(f"/api/projects/{pid}/approve-plan", json={"accept_assumptions": True}).status_code == 202
    st = c.get(f"/api/projects/{pid}").json()["status"]
    assert st == "completed", f"status={st}"
    assert c.get(f"/api/projects/{pid}/download").status_code == 200
    return "demo flow completed over HTTP"


@check("Docker sandbox: real isolation (no network, read-only fs, no secrets)")
def _():
    from app.sandbox import DockerSandbox
    if not DockerSandbox.available():
        return "skip:Docker not available/running"
    os.environ["P2P_VERIFY_SECRET"] = "leakme"
    d = tempfile.mkdtemp(); os.chmod(d, 0o777)
    open(os.path.join(d, "probe.py"), "w").write(textwrap.dedent('''
        import os, socket
        try:
            socket.create_connection(("1.1.1.1", 53), 2); print("NETWORK:OPEN")
        except OSError: print("NETWORK:BLOCKED")
        try:
            open("/etc/pwned", "w"); print("ROOTFS:WRITABLE")
        except OSError: print("ROOTFS:READONLY")
        print("SECRET:" + str(os.environ.get("P2P_VERIFY_SECRET")))
    '''))
    r = DockerSandbox(os.environ.get("P2P_SANDBOX_IMAGE", "python:3.12-slim")).run_python(d, ["probe.py"], 60, 256, 64)
    out = r.stdout
    assert "NETWORK:BLOCKED" in out and "ROOTFS:READONLY" in out and "SECRET:None" in out, f"isolation probe output: {out!r} {r.stderr[-200:]}"
    return "network blocked, rootfs read-only, secrets not visible"


@check("Gemini live structured call")
def _():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "skip:GEMINI_API_KEY not set"
    from app.config import Settings
    from app.llm.provider import get_provider
    p = get_provider(Settings.from_env())
    assert p.generate_json('Return the JSON object {"ok": true}')["ok"] is True
    return f"model={p.info().get('model')} usage={p.usage.to_dict()}"


@check("Gemini end-to-end: LLM extraction of the sample paper")
def _():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "skip:GEMINI_API_KEY not set"
    from app.config import Settings
    from app.llm.provider import get_provider
    from app import ingest
    pages = ingest.extract_pages(open(os.path.join(ROOT, "sample_papers", "fisher_lda_iris_fixture.pdf"), "rb").read())
    spec = ingest.build_spec(pages, get_provider(Settings.from_env()))
    assert spec.title and spec.extraction_method.startswith("llm:")
    return f"title={spec.title!r}, hyperparameters={spec.hyperparameters}"


@check("GitHub live read-only inspection")
def _():
    from app.integrations.github import GitHubClient, GitHubError
    try:
        r = GitHubClient(os.environ.get("GITHUB_TOKEN", "")).inspect_repo("scikit-learn/scikit-learn")
    except GitHubError as e:
        if "Could not reach" in str(e):
            return "skip:no network access to api.github.com"
        raise
    assert r["full_name"].lower() == "scikit-learn/scikit-learn"
    return f"license={r['license']}"


@check("Hugging Face dataset metadata (live)")
def _():
    if not importlib.util.find_spec("huggingface_hub"):
        return "skip:huggingface_hub not installed"
    from app.integrations.datasets import resolve_huggingface
    d = resolve_huggingface("scikit-learn/iris", os.environ.get("HF_TOKEN", ""))
    return f"license={d.license}"


print(f"\n{'CHECK':<78} RESULT")
for n, s, d in results:
    print(f"{n[:76]:<78} {s}  {d}")
fails = [r for r in results if r[1] == "FAIL"]
print(f"\n{len([r for r in results if r[1]=='PASS'])} passed, {len(fails)} failed, {len([r for r in results if r[1]=='SKIP'])} skipped")
sys.exit(1 if fails else 0)
