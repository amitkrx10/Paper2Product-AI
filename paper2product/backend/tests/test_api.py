"""HTTP-level tests. Require fastapi + httpx (pip install -r requirements.txt); skipped (NOT passed) otherwise."""
import io, unittest
try:
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

from .helpers import tmp_settings, sample_pdf


@unittest.skipUnless(HAVE, "fastapi/httpx not installed in this environment")
class TestAPI(unittest.TestCase):
    def setUp(self):
        from app.main import create_app
        from app.services import Service
        from app.sandbox import LocalSandbox
        s = tmp_settings()
        self.client = TestClient(create_app(s, Service(s, sync_jobs=True, sandbox=LocalSandbox(), provider=None)))

    def test_health_and_ready(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        self.assertTrue(self.client.get("/readyz").json()["ready"])

    def test_request_id_header_and_error_shape(self):
        r = self.client.get("/api/projects/nope")
        self.assertEqual(r.status_code, 404); self.assertIn("request_id", r.json()); self.assertIn("X-Request-ID", r.headers)
        self.assertNotIn("Traceback", r.text)

    def test_upload_validation(self):
        r = self.client.post("/api/projects", files={"file": ("x.txt", b"hello", "text/plain")})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/projects", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
        self.assertEqual(r.status_code, 400)

    def test_full_flow_over_http(self):
        r = self.client.post("/api/projects", files={"file": ("p.pdf", sample_pdf(), "application/pdf")}, data={"inject_fault": "true"})
        self.assertEqual(r.status_code, 202); pid = r.json()["id"]
        self.assertEqual(self.client.post(f"/api/projects/{pid}/approve-plan", json={"accept_assumptions": True}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/projects/{pid}/approve-spec", json={}).status_code, 202)
        self.assertEqual(self.client.post(f"/api/projects/{pid}/approve-plan", json={"accept_assumptions": False}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/projects/{pid}/approve-plan", json={"accept_assumptions": True}).status_code, 202)
        p = self.client.get(f"/api/projects/{pid}").json()
        self.assertEqual(p["status"], "completed")
        self.assertEqual(self.client.get(f"/api/projects/{pid}/download").headers["content-type"], "application/zip")
        self.assertEqual(self.client.get(f"/api/projects/{pid}/file", params={"path": "../../etc/passwd"}).status_code, 400)
        self.assertEqual(self.client.post(f"/api/projects/{pid}/github-export", json={"name": "x", "confirm": False}).status_code, 502)

    def test_api_key_enforced_when_configured(self):
        from app.main import create_app
        from app.services import Service
        from app.sandbox import LocalSandbox
        s = tmp_settings(api_key="k123")
        c = TestClient(create_app(s, Service(s, sync_jobs=True, sandbox=LocalSandbox(), provider=None)))
        self.assertEqual(c.get("/api/projects").status_code, 401)
        self.assertEqual(c.get("/api/projects", headers={"X-API-Key": "k123"}).status_code, 200)
        self.assertEqual(c.get("/healthz").status_code, 200)
