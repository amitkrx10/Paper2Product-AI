import os, tempfile, shutil
from app.config import Settings
from app.storage import Storage
from app.sandbox import LocalSandbox
from app.graph import Engine
from app.schemas import Budget

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers", "fisher_lda_iris_fixture.pdf")


def sample_pdf() -> bytes:
    with open(SAMPLE, "rb") as f:
        return f.read()


def tmp_settings(**kw) -> Settings:
    d = tempfile.mkdtemp(prefix="p2p_test_")
    return Settings(data_dir=d, db_path=os.path.join(d, "t.db"), **kw)


class StubProvider:
    """TEST DOUBLE ONLY - returns canned dicts. Never used by the product; marks usage so tests can assert calls."""
    name, is_mock = "stub", True

    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0
        from app.llm.provider import Usage
        self.usage = Usage()

    def generate_json(self, prompt, system=""):
        self.calls += 1
        return self.responses.pop(0)

    def info(self):
        return {"provider": "stub", "mock": True}


def make_engine(provider=None, **kw):
    s = tmp_settings(**kw)
    db = Storage(s.db_path)
    return Engine(s, db, LocalSandbox(), provider), s, db


def drive(engine, pid, accept=True):
    engine.run_ingest(pid)
    engine.approve_spec(pid, None)
    engine.run_plan(pid)
    engine.approve_plan(pid, accept)
    engine.run_execution(pid)
