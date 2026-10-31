import csv, io, json, os, tempfile, unittest, urllib.error
from app.integrations import datasets as ds
from app.integrations.github import GitHubClient, GitHubError
from app.llm.provider import (get_provider, LLMConfigError, LLMOutputError, extract_json, GeminiProvider, Usage)
from app.config import Settings


class FakeResp:
    def __init__(self, body): self.body = json.dumps(body).encode()
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b"{}"))


class TestDatasets(unittest.TestCase):
    def test_iris_validated_with_provenance(self):
        d = ds.resolve_sklearn_iris()
        self.assertEqual((d.n_rows, d.n_features, d.n_classes, d.validated), (150, 4, 3, True))
        self.assertEqual(len(d.sha256), 64); self.assertIn("Fisher", d.provenance); self.assertIn("uci", d.url)

    def test_resolution_rules(self):
        self.assertEqual(ds.resolve_for_spec(["Iris dataset (150 samples)"]).id, "sklearn:iris")
        with self.assertRaises(ds.ResourceError) as cm: ds.resolve_for_spec(["ImageNet"])
        self.assertIn("supply a dataset source", str(cm.exception))
        with self.assertRaises(ds.ResourceError): ds.resolve_for_spec([], "ftp://evil")

    def _csv(self, rows, header=("a", "b", "y"), name="d.csv"):
        p = os.path.join(tempfile.mkdtemp(), name)
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        return p

    def test_upload_validation(self):
        ok = ds.resolve_upload(self._csv([[i, i * 2, i % 2] for i in range(20)]))
        self.assertEqual((ok.n_rows, ok.n_features, ok.validated), (20, 2, True))
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2, 3]] * 3))               # too few rows
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2]] * 20))                 # ragged
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2, 3]] * 20, name="d.exe"))  # type
        with self.assertRaises(ds.ResourceError): ds.resolve_upload("/nonexistent.csv")

    def test_huggingface_input_validation_and_missing_lib(self):
        with self.assertRaises(ds.ResourceError): ds.resolve_huggingface("a; rm -rf /")
        try:
            import huggingface_hub  # noqa
        except ImportError:
            with self.assertRaises(ds.ResourceError) as cm: ds.resolve_huggingface("scikit-learn/iris")
            self.assertIn("huggingface_hub", str(cm.exception))


class TestGitHub(unittest.TestCase):
    def test_status_without_token(self):
        s = GitHubClient("").status()
        self.assertFalse(s["configured"]); self.assertIn("GITHUB_TOKEN", s["detail"])

    def test_inspect_success_parses_metadata(self):
        calls = []
        def opener(req, timeout=0):
            calls.append(req.full_url)
            if req.full_url.endswith("/contents"): return FakeResp([{"name": "README.md", "type": "file", "size": 5}])
            return FakeResp({"full_name": "o/r", "stargazers_count": 3, "license": {"spdx_id": "MIT"}, "default_branch": "main"})
        r = GitHubClient("tok", opener).inspect_repo("o/r")
        self.assertEqual((r["license"], r["top_level_files"][0]["name"]), ("MIT", "README.md"))
        self.assertIn("never executed", r["note"])

    def test_http_errors_become_safe_messages(self):
        for code, text in [(401, "rejected the token"), (403, "rate limit"), (404, "not found"), (500, "error 500")]:
            def opener(req, timeout=0, c=code): raise http_error(c)
            with self.assertRaises(GitHubError) as cm: GitHubClient("secret-token-123", opener).inspect_repo("o/r")
            self.assertIn(text, str(cm.exception)); self.assertNotIn("secret-token-123", str(cm.exception))

    def test_network_failure(self):
        def opener(req, timeout=0): raise urllib.error.URLError("dns")
        with self.assertRaises(GitHubError) as cm: GitHubClient("", opener).inspect_repo("o/r")
        self.assertIn("Could not reach", str(cm.exception))

    def test_repo_name_validation(self):
        for bad in ["", "no-slash", "a/b/c", "../../etc", "o/r; rm", "o/" + "x" * 200]:
            with self.assertRaises(GitHubError, msg=bad): GitHubClient("t", lambda *a, **k: None).inspect_repo(bad)

    def test_export_requires_confirmation_and_token_and_makes_no_calls(self):
        called = []
        def opener(req, timeout=0): called.append(req); return FakeResp({})
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("n", {"a.py": "x"}, confirm=False)
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("n", {"a.py": "x"}, confirm="yes")
        with self.assertRaises(GitHubError): GitHubClient("", opener).export_project("n", {"a.py": "x"}, confirm=True)
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("bad name!", {"a.py": "x"}, confirm=True)
        self.assertEqual(called, [])

    def test_confirmed_export_creates_private_repo_then_uploads(self):
        seen = []
        def opener(req, timeout=0):
            seen.append((req.get_method(), req.full_url, json.loads(req.data) if req.data else None))
            return FakeResp({"full_name": "me/n", "html_url": "https://github.com/me/n"})
        r = GitHubClient("tok", opener).export_project("n", {"a.py": "x = 1"}, confirm=True)
        self.assertEqual(seen[0][0:2], ("POST", "https://api.github.com/user/repos")); self.assertTrue(seen[0][2]["private"])
        self.assertEqual(seen[1][0], "PUT"); self.assertEqual(r["files_uploaded"], 1)


class TestLLMProvider(unittest.TestCase):
    def test_missing_credentials_message(self):
        with self.assertRaises(LLMConfigError) as cm: get_provider(Settings(llm_provider="gemini", llm_api_key=""))
        self.assertIn("GEMINI_API_KEY", str(cm.exception))

    def test_auto_without_key_is_demo_mode(self):
        self.assertIsNone(get_provider(Settings(llm_provider="auto", llm_api_key="")))
        self.assertIsNone(get_provider(Settings(llm_provider="demo", llm_api_key="k")))

    def test_unknown_provider(self):
        with self.assertRaises(LLMConfigError): get_provider(Settings(llm_provider="skynet"))

    def test_gemini_needs_sdk(self):
        try:
            import google.genai  # noqa
            self.skipTest("google-genai installed")
        except ImportError:
            with self.assertRaises(LLMConfigError) as cm: GeminiProvider(Settings(llm_api_key="k"))
            self.assertIn("google-genai", str(cm.exception))

    def test_json_extraction_and_validation(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! {"a": 2} done'), {"a": 2})
        for bad in ["no json", "[1,2]", '{"a": ']:
            with self.assertRaises(LLMOutputError): extract_json(bad)

    def test_usage_and_cost_tracking(self):
        u = Usage(); u.add(1000, 500, 0.002); u.add(500, 0, 0.002)
        self.assertEqual((u.total, u.calls), (2000, 2)); self.assertAlmostEqual(u.est_cost_usd, 0.004)


@unittest.skipUnless(os.environ.get("P2P_LIVE_TESTS") == "1" and os.environ.get("GEMINI_API_KEY"), "live test: set P2P_LIVE_TESTS=1 and GEMINI_API_KEY")
class TestLiveGemini(unittest.TestCase):
    def test_real_structured_call(self):
        p = get_provider(Settings.from_env())
        self.assertEqual(p.generate_json('Return JSON {"ok": true}')["ok"], True)


@unittest.skipUnless(os.environ.get("P2P_LIVE_TESTS") == "1" and os.environ.get("GITHUB_TOKEN"), "live test: set P2P_LIVE_TESTS=1 and GITHUB_TOKEN")
class TestLiveGitHub(unittest.TestCase):
    def test_inspect_public_repo(self):
        self.assertEqual(GitHubClient(os.environ["GITHUB_TOKEN"]).inspect_repo("scikit-learn/scikit-learn")["full_name"], "scikit-learn/scikit-learn")
