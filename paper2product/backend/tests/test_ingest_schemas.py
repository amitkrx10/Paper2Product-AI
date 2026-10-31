import unittest
from app import ingest
from app.schemas import ResearchSpec, SchemaError, Budget, Plan, PlanTask, ReportedResult
from .helpers import sample_pdf


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.pages = ingest.extract_pages(sample_pdf())
        self.spec = ingest.build_spec(self.pages)

    def test_pages_and_title(self):
        self.assertEqual(len(self.pages), 2)
        self.assertIn("Linear Discriminant", self.spec.title)

    def test_extracts_core_fields_with_page_refs(self):
        s = self.spec
        self.assertEqual(s.hyperparameters["test_size"], 0.3)
        self.assertEqual(s.hyperparameters["random_seed"], 42)
        self.assertEqual(s.task_type, "classification")
        self.assertEqual(s.page_refs["method"], [1]); self.assertEqual(s.page_refs["hyperparameters"], [2])
        self.assertEqual(len(s.equations), 2)
        r = s.reported_results[0]
        self.assertEqual((r.metric, r.value, r.dataset, r.split, r.page), ("accuracy", 0.98, "iris", "70/30", 2))

    def test_prompt_injection_in_pdf_is_flagged_not_obeyed(self):
        self.assertEqual(len(self.spec.security_flags), 1)
        self.assertIn("API key", self.spec.security_flags[0])
        self.assertNotIn("api key", self.spec.method.lower())

    def test_rejects_non_pdf_and_oversize(self):
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"not a pdf")
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"%PDF-" + b"0" * 100, max_bytes=50)
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"")
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"%PDF-1.4 garbage")

    def test_page_limit(self):
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(sample_pdf(), max_pages=1)

    def test_missing_info_reported_instead_of_invented(self):
        spec = ingest.heuristic_spec(["Some Title\nAbstract\nHello"])
        self.assertIn("No method/algorithm section found", spec.missing_info)
        self.assertIn("Random seed not reported", spec.missing_info)
        self.assertEqual(spec.hyperparameters, {})


class TestPyMuPDFFailureHandling(unittest.TestCase):
    """Regression: a PyMuPDF exception on a corrupt PDF must become IngestError (found in external review)."""
    def _install_fake_fitz(self, behaviour):
        import sys, types
        m = types.ModuleType("fitz")
        m.open = behaviour
        self._old = sys.modules.get("fitz")
        sys.modules["fitz"] = m
        self.addCleanup(self._restore)

    def _restore(self):
        import sys
        if self._old is None:
            sys.modules.pop("fitz", None)
        else:
            sys.modules["fitz"] = self._old

    def test_corrupt_pdf_with_pymupdf_raising_becomes_ingest_error(self):
        def boom(*a, **k): raise RuntimeError("FileDataError: cannot open broken document")
        self._install_fake_fitz(boom)
        with self.assertRaises(ingest.IngestError):
            ingest.extract_pages(b"%PDF-1.4 garbage")

    def test_pymupdf_failure_falls_back_to_pypdf_for_valid_pdf(self):
        def boom(*a, **k): raise RuntimeError("pymupdf broke")
        self._install_fake_fitz(boom)
        self.assertEqual(len(ingest.extract_pages(sample_pdf())), 2)

    def test_pymupdf_page_limit_is_ingest_error(self):
        class Doc:
            page_count = 999
        self._install_fake_fitz(lambda *a, **k: Doc())
        with self.assertRaises(ingest.IngestError):
            ingest.extract_pages(sample_pdf(), max_pages=5)

    def test_pdf_without_text_is_rejected_cleanly(self):
        from reportlab.pdfgen import canvas
        import io
        buf = io.BytesIO(); c = canvas.Canvas(buf); c.showPage(); c.save()
        with self.assertRaises(ingest.IngestError) as cm:
            ingest.extract_pages(buf.getvalue())
        self.assertIn("no extractable text", str(cm.exception))


class TestLLMExtractionValidation(unittest.TestCase):
    class P:
        name = "stub"
        def __init__(self, d): self.d = d
        def generate_json(self, prompt, system=""):
            assert "<untrusted_document>" in prompt
            return dict(self.d)

    def test_valid_llm_output_accepted(self):
        s = ingest.build_spec(["x"], self.P({"title": "T", "task_type": "regression", "metrics": ["RMSE"]}))
        self.assertEqual((s.title, s.metrics, s.extraction_method), ("T", ["rmse"], "llm:stub"))

    def test_invalid_llm_output_rejected(self):
        for bad in ({"title": ""}, {"title": "T", "task_type": "magic"}, {"title": "T", "hyperparameters": [1]},
                    {"title": "T", "reported_results": [{"metric": "acc", "value": "abc"}]}, {"title": "T", "authors": "nope"}):
            with self.assertRaises(SchemaError, msg=str(bad)):
                ingest.build_spec(["x"], self.P(bad))


class TestSchemas(unittest.TestCase):
    def test_budget_bounds(self):
        self.assertEqual(Budget.from_dict(None).max_repairs, 3)
        for bad in ({"max_repairs": 11}, {"max_repairs": -1}, {"exec_timeout_s": 0}, {"memory_mb": 1}, {"max_repairs": "x"}):
            with self.assertRaises(SchemaError, msg=str(bad)): Budget.from_dict(bad)

    def test_plan_dependency_validation(self):
        with self.assertRaises(SchemaError):
            Plan("t", "a", [PlanTask("T1", "x", ["T9"])], ["f.py"], [], [], "", [], [], "", [], [], []).validate()
        with self.assertRaises(SchemaError):
            Plan("t", "a", [PlanTask("T1", "x"), PlanTask("T1", "y")], ["f.py"], [], [], "", [], [], "", [], [], []).validate()

    def test_spec_roundtrip(self):
        s = ingest.build_spec(ingest.extract_pages(sample_pdf()))
        self.assertEqual(ResearchSpec.from_dict(s.to_dict()).to_dict(), s.to_dict())
