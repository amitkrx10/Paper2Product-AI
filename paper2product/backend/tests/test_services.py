import io, json, os, unittest, zipfile
from app.services import Service, RateLimited, NotFound, RateLimiter
from app.graph import S, GateError
from app.schemas import SchemaError
from app.storage import Storage
from app.sandbox import LocalSandbox
from app.integrations.github import GitHubClient, GitHubError
from app.security import SecurityError
from .helpers import tmp_settings, sample_pdf


def svc(**kw):
    s = tmp_settings(**kw)
    return Service(s, sync_jobs=True, sandbox=LocalSandbox(), provider=None), s


def run_all(sv, pid):
    sv.approve_spec(pid, None); sv.approve_plan(pid, True)


class TestRequestValidation(unittest.TestCase):
    def test_upload_validation(self):
        sv, s = svc()
        for name, data in [("paper.txt", sample_pdf()), ("a.pdf", b"not pdf"), ("a.pdf", b"")]:
            with self.assertRaises(SchemaError, msg=name): sv.create_project(name, data, {})
        with self.assertRaises(SchemaError): sv.create_project("a.pdf", sample_pdf(), {"max_repairs": 99})
        with self.assertRaises(SchemaError): sv.create_project("a.pdf", sample_pdf(), {}, dataset_hint="file:///etc/passwd")
        big = svc(max_upload_mb=0)[0]
        with self.assertRaises(SchemaError): big.create_project("a.pdf", sample_pdf(), {})

    def test_unknown_project_and_gate_errors(self):
        sv, s = svc()
        with self.assertRaises(NotFound): sv.get_project("nope")
        pid = sv.create_demo()
        with self.assertRaises(GateError): sv.approve_plan(pid, True)

    def test_file_endpoint_rejects_traversal(self):
        sv, s = svc(); pid = sv.create_demo(); run_all(sv, pid)
        for bad in ["../../etc/passwd", "/etc/passwd", ".env"]:
            with self.assertRaises(SecurityError): sv.file(pid, bad)
        with self.assertRaises(NotFound): sv.file(pid, "missing.py")
        self.assertIn("class FisherLDA", sv.file(pid, "lda_repro/lda.py"))

    def test_rate_limiting(self):
        r = RateLimiter(2); r.check("a"); r.check("a")
        with self.assertRaises(RateLimited): r.check("a")
        r.check("b")
        sv, s = svc(rate_limit_per_min=1)
        sv.create_demo(client="x")
        with self.assertRaises(RateLimited): sv.create_demo(client="x")


class TestPersistenceAndHistory(unittest.TestCase):
    def test_run_history_survives_restart(self):
        sv, s = svc(); pid = sv.create_demo(); run_all(sv, pid)
        sv2 = Service(s, sync_jobs=True, sandbox=LocalSandbox(), provider=None)    # "restart"
        lst = sv2.list_projects()
        self.assertEqual(len(lst), 1); self.assertEqual(lst[0]["status"], "completed")
        self.assertEqual(lst[0]["repair_attempts"], 1)
        self.assertGreater(len(sv2.events(pid)), 10)
        self.assertEqual(sv2.get_project(pid)["state"]["mode"], "demo")

    def test_interrupted_jobs_marked_failed_on_startup(self):
        s = tmp_settings(); db = Storage(s.db_path)
        pid = db.create_project("x", {}); db.set_status(pid, "testing")
        sv = Service(s, sync_jobs=True, storage=db, sandbox=LocalSandbox(), provider=None)
        self.assertEqual(db.get_project(pid)["status"], "failed")

    def test_migrations_idempotent(self):
        s = tmp_settings(); a = Storage(s.db_path); b = Storage(s.db_path)
        self.assertEqual(b.migrate(), 2)

    def test_project_listing_hides_report_body_and_file_contents(self):
        sv, s = svc(); pid = sv.create_demo(); run_all(sv, pid)
        d = sv.get_project(pid)
        self.assertNotIn("report_md", d["state"]); self.assertIsInstance(d["state"]["files"], list)


class TestReportAndExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sv, cls.s = svc(); cls.pid = cls.sv.create_demo(); run_all(cls.sv, cls.pid)

    def test_report_is_honest_and_complete(self):
        md = self.sv.report_md(self.pid)
        for must in ["Reproduction status", "DEMO MODE", "Repair attempts", "```diff", "Comparison with reported results",
                     "does not prove", "Rerun instructions", "Sandbox warning", "Dataset provenance", "sha256"]:
            self.assertIn(must, md)

    def test_demo_report_never_claims_llm_reasoning(self):
        m = self.sv.manifest(self.pid)
        self.assertFalse(m["honesty"]["llm_used"]); self.assertFalse(m["honesty"]["simulated_results"])
        self.assertIn("DEMO MODE", m["honesty"]["note"])

    def test_manifest_contains_reproducibility_fields(self):
        m = self.sv.manifest(self.pid)
        self.assertEqual((m["seed"], m["test_size"], m["dataset"]["id"]), (42, 0.3, "sklearn:iris"))
        self.assertIn("scikit_learn", m["environment"]); self.assertEqual(m["repair_attempts"], 1)
        self.assertFalse(m["sandbox"]["isolated"])                    # honest: local mode is labelled
        self.assertEqual(len(m["file_sha256"]), 10); self.assertTrue(all(len(h) == 64 for h in m["file_sha256"].values()))
        self.assertEqual(m["tests"]["ok"], True)

    def test_zip_download_contains_project_report_manifest(self):
        z = zipfile.ZipFile(io.BytesIO(self.sv.zip_bytes(self.pid)))
        names = z.namelist()
        for n in ["project/lda_repro/lda.py", "project/tests/test_lda.py", "REPORT.md", "reproducibility_manifest.json"]:
            self.assertIn(n, names)
        self.assertTrue(all(".." not in n and not n.startswith("/") for n in names))
        json.loads(z.read("reproducibility_manifest.json"))

    def test_downloaded_project_really_runs_standalone(self):
        import subprocess, sys, tempfile
        d = tempfile.mkdtemp(); zipfile.ZipFile(io.BytesIO(self.sv.zip_bytes(self.pid))).extractall(d)
        p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=os.path.join(d, "project"),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        p = subprocess.run([sys.executable, "run_experiment.py"], cwd=os.path.join(d, "project"), capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[-500:])

    def test_no_secrets_in_outputs(self):
        os.environ["GEMINI_API_KEY"] = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE12345"
        try:
            blob = self.sv.report_md(self.pid) + json.dumps(self.sv.manifest(self.pid)) + json.dumps(self.sv.get_project(self.pid))
            self.assertNotIn("AIzaSy", blob)
        finally:
            del os.environ["GEMINI_API_KEY"]

    def test_github_export_never_happens_without_explicit_confirmation(self):
        calls = []
        class Fake(GitHubClient):
            def _request(self, *a, **k): calls.append(a); return {"full_name": "me/x", "html_url": "u"}
        self.sv.github = Fake("tok")
        with self.assertRaises(GitHubError): self.sv.github_export(self.pid, "x", confirm=False)
        self.assertEqual(calls, [])
        out = self.sv.github_export(self.pid, "x", confirm=True)
        self.assertEqual(out["full_name"], "me/x"); self.assertTrue(calls)
        actions = [a["action"] for a in self.sv.db.audit_log(self.pid)]
        self.assertIn("github_export_requested", actions); self.assertIn("github_export_done", actions)

    def test_audit_log_records_approvals(self):
        actions = [a["action"] for a in self.sv.db.audit_log(self.pid)]
        for a in ["project_created", "spec_approved", "plan_approved"]: self.assertIn(a, actions)


class TestIntegrationStatusAndHealth(unittest.TestCase):
    def test_status_reports_demo_mode_and_unconfigured(self):
        sv, s = svc(); i = sv.integrations()
        self.assertFalse(i["llm"]["configured"]); self.assertEqual(i["llm"]["mode"], "demo")
        self.assertFalse(i["github"]["configured"]); self.assertFalse(i["sandbox"]["isolated"])
        self.assertTrue(sv.ready()["ready"]); self.assertEqual(sv.health()["status"], "ok")

    def test_missing_llm_key_with_gemini_provider_degrades_with_message(self):
        sv = Service(tmp_settings(llm_provider="gemini", llm_api_key=""), sync_jobs=True, sandbox=LocalSandbox())
        self.assertIsNone(sv.llm); self.assertIn("GEMINI_API_KEY", sv.integrations()["llm"]["detail"])

    def test_wall_clock_budget_blocks(self):
        sv, s = svc(); pid = sv.create_project("a.pdf", sample_pdf(), {"max_wall_s": 10})
        sv.approve_spec(pid, None)
        import app.graph as g, time
        st = sv.engine.load(pid); sv.engine.approve_plan(pid, True)
        st = sv.engine.load(pid); st.started_at = time.time() - 9999; sv.engine.save(st)
        sv.engine.run_execution(pid)
        self.assertEqual(sv.db.get_project(pid)["status"], "blocked")
        self.assertIn("Budget exhausted", sv.engine.load(pid).blocked_reason)
