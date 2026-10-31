import json, os, unittest
from app import graph, codegen, repair
from app.graph import S, GateError, InvalidTransition, check_transition
from app.schemas import Budget, SchemaError
from app.storage import Storage
from .helpers import make_engine, sample_pdf, drive, StubProvider


class TestStateMachine(unittest.TestCase):
    def test_valid_and_invalid_transitions(self):
        check_transition(S.CREATED, S.INGESTING)
        check_transition(S.TESTING, S.REPAIRING)
        for a, b in [(S.CREATED, S.EXECUTING), (S.COMPLETED, S.TESTING), (S.AWAITING_PLAN, S.EXECUTING),
                     (S.REPAIRING, S.EXECUTING), (S.BLOCKED, S.TESTING), (S.AWAITING_SPEC, S.GENERATING)]:
            with self.assertRaises(InvalidTransition, msg=f"{a}->{b}"):
                check_transition(a, b)

    def test_terminal_states_have_no_exits(self):
        for t in graph.TERMINAL:
            self.assertEqual(graph.TRANSITIONS[t], set())

    def test_execution_cannot_skip_approvals(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget())
        e.run_ingest(pid)
        with self.assertRaises(GateError): e.approve_plan(pid, True)          # spec not approved yet
        e.approve_spec(pid, None); e.run_plan(pid)
        with self.assertRaises(GateError): e.approve_plan(pid, False)         # assumptions not accepted
        self.assertEqual(e.status(pid), S.AWAITING_PLAN)
        with self.assertRaises(GateError): e.approve_spec(pid, None)          # wrong state


class TestHappyPathAndRepair(unittest.TestCase):
    """One real fault-injected run is shared by the read-only assertions (keeps the suite fast; still real execution)."""
    @classmethod
    def setUpClass(cls):
        cls.e, cls.s, cls.db = make_engine()
        cls.pid = cls.e.create("demo", sample_pdf(), Budget(), inject_fault=True)
        drive(cls.e, cls.pid)
        cls.st = cls.e.load(cls.pid)

    def test_fault_injection_is_diagnosed_repaired_and_verified(self):
        st, e, pid = self.st, self.e, self.pid
        self.assertEqual(e.status(pid), S.COMPLETED)
        self.assertEqual(len(st.attempts), 1)
        a = st.attempts[0]
        self.assertEqual(a["diagnosis"]["category"], "name_error"); self.assertEqual(a["diagnosis"]["name"], "Sw_inverse")
        self.assertEqual(a["patch"]["file"], "lda_repro/lda.py"); self.assertIn("-        self.coef_ = self.means_ @ Sw_inverse", a["diff"])
        self.assertTrue(a["outcome"].startswith("fixed"))
        stages = [(r["stage"], r["ok"]) for r in st.runs]
        self.assertEqual(stages, [("tests", False), ("tests", True), ("experiment", True)])   # real executions, in order
        self.assertTrue(st.final_tests["ok"]); self.assertIn("passed", st.final_tests["summary"])
        self.assertNotIn("Sw_inverse", st.files["lda_repro/lda.py"])

    def test_no_fault_needs_no_repair(self):
        e, s, db = make_engine()
        pid = e.create("demo", sample_pdf(), Budget(), inject_fault=False)
        drive(e, pid)
        self.assertEqual(len(e.load(pid).attempts), 0); self.assertEqual(e.status(pid), S.COMPLETED)

    def test_metrics_are_recomputed_independently_and_match_artifacts(self):
        st = self.st
        with open(os.path.join(self.e.work_dir(self.pid), "artifacts", "predictions.json")) as fh:
            preds = json.load(fh)
        acc = sum(t == p for t, p in zip(preds["y_true"], preds["y_pred"])) / len(preds["y_true"])
        self.assertAlmostEqual(st.metrics["accuracy"], acc)
        self.assertEqual(st.metrics["n"], 45)
        self.assertIn("agrees", st.script_metric_check)
        self.assertEqual(st.dataset["id"], "sklearn:iris")

    def test_reproduction_status_not_inflated_by_passing_tests(self):
        c = self.st.comparisons[0]
        self.assertEqual(c["metric"], "accuracy"); self.assertEqual(c["reported"], 0.98)
        expected = "reproduced" if c["rel_diff"] <= 0.02 else "partially_reproduced" if c["rel_diff"] <= 0.10 else "not_reproduced"
        self.assertEqual(self.st.reproduction_status, expected)

    def test_events_are_persisted_in_order(self):
        evs = self.db.events(self.pid)
        self.assertEqual([x["id"] for x in evs], sorted(x["id"] for x in evs))
        states = [x["data"]["to"] for x in evs if x["stage"] == "state"]
        self.assertEqual(states[:3], ["ingesting", "awaiting_spec_approval", "planning"])
        self.assertIn("repairing", states); self.assertEqual(states[-1], "completed")
        self.assertEqual(len(self.db.events(self.pid, after=evs[2]["id"])), len(evs) - 3)

    def test_repair_never_touches_tests(self):
        st = self.st
        self.assertEqual(st.files["tests/test_lda.py"], codegen.TEST_PY)
        self.assertTrue(all(not f.startswith("tests/") for a in st.attempts for f in a["changed_files"]))


class TestBoundedRepair(unittest.TestCase):
    def test_retry_limit_prevents_endless_loops(self):
        e, s, db = make_engine()
        n = {"i": 0}
        orig = repair.rule_based_patch

        def always_new_typo(diag, files):          # a "repair" that never fixes anything
            n["i"] += 1
            src = files[diag.file]
            return repair.Patch(diag.file, src.replace(diag.name, f"Broken{n['i']}"), "bad repair", "rule-based")
        repair.rule_based_patch = always_new_typo
        try:
            pid = e.create("demo", sample_pdf(), Budget(max_repairs=2), inject_fault=True)
            drive(e, pid)
        finally:
            repair.rule_based_patch = orig
        st = e.load(pid)
        self.assertEqual(e.status(pid), S.BLOCKED)
        self.assertEqual(len(st.attempts), 2)
        self.assertIn("budget exhausted", st.blocked_reason)
        self.assertEqual(st.reproduction_status, "unable_to_verify")
        self.assertTrue(st.report_md)                       # honest report is still produced
        self.assertIn("Unable to verify", st.report_md)

    def test_zero_repairs_budget(self):
        e, s, db = make_engine()
        pid = e.create("demo", sample_pdf(), Budget(max_repairs=0), inject_fault=True)
        drive(e, pid)
        self.assertEqual(e.status(pid), S.BLOCKED); self.assertEqual(len(e.load(pid).attempts), 0)

    def test_assertion_failures_are_not_auto_patched(self):
        e, s, db = make_engine()
        orig = codegen.generate_template_project

        def wrong_logic(spec, plan, fault=False):
            f = orig(spec, plan, False)
            f["lda_repro/lda.py"] = f["lda_repro/lda.py"].replace("return self.classes_[np.argmax(self.decision_function(X), axis=1)]",
                                                                    "return np.full(len(X), self.classes_[0])")
            return f
        codegen.generate_template_project = wrong_logic
        try:
            pid = e.create("demo", sample_pdf(), Budget())
            drive(e, pid)
        finally:
            codegen.generate_template_project = orig
        st = e.load(pid)
        self.assertEqual(e.status(pid), S.BLOCKED); self.assertEqual(len(st.attempts), 0)
        self.assertIn("No safe automated repair", st.blocked_reason)
        self.assertFalse(st.final_tests["ok"])                    # failing tests are NOT suppressed



class TestCancelAndBlocking(unittest.TestCase):
    def test_cancel_while_awaiting_gate(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget()); e.run_ingest(pid)
        e.request_cancel(pid)
        self.assertEqual(e.status(pid), S.CANCELLED)
        with self.assertRaises(GateError): e.approve_spec(pid, None)

    def test_cancel_signal_stops_running_job(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget()); e.run_ingest(pid); e.approve_spec(pid, None); e.run_plan(pid); e.approve_plan(pid, True)
        e.cancel_event(pid).set()
        e.run_execution(pid)
        self.assertEqual(e.status(pid), S.CANCELLED); self.assertEqual(e.load(pid).runs, [])

    def test_unsupported_paper_blocks_honestly_in_demo_mode(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget()); e.run_ingest(pid)
        e.approve_spec(pid, {"method": "A transformer with attention.", "datasets": ["mnist"]})
        e.run_plan(pid)
        self.assertEqual(e.status(pid), S.BLOCKED)
        self.assertIn("Demo mode", e.load(pid).blocked_reason)

    def test_dataset_unavailable_blocks_with_instructions(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget(), dataset_hint="upload:/nonexistent.csv"); e.run_ingest(pid)
        e.approve_spec(pid, None); e.run_plan(pid); e.approve_plan(pid, True); e.run_execution(pid)
        self.assertEqual(e.status(pid), S.BLOCKED); self.assertIn("Dataset unavailable", e.load(pid).blocked_reason)

    def test_invalid_spec_edit_rejected(self):
        e, s, db = make_engine()
        pid = e.create("t", sample_pdf(), Budget()); e.run_ingest(pid)
        with self.assertRaises(SchemaError): e.approve_spec(pid, {"task_type": "magic"})
        with self.assertRaises(SchemaError): e.approve_spec(pid, {"hyperparameters": [1]})
        self.assertEqual(e.status(pid), S.AWAITING_SPEC)

    def test_corrupt_pdf_rejected_at_creation(self):
        e, s, db = make_engine()
        from app.ingest import IngestError
        with self.assertRaises(IngestError): e.create("t", b"nope", Budget())


class TestLLMModePath(unittest.TestCase):
    """Uses a STUB provider (test double) to exercise the LLM-mode branches without network calls."""
    def test_llm_generated_unsafe_code_is_rejected(self):
        from app.security import SecurityError
        from app.schemas import ResearchSpec
        from app.planner import make_plan
        spec = ResearchSpec(title="T", method="x", datasets=["d"])
        stub = StubProvider([{"files": {"pkg/a.py": "import subprocess\n", "tests/test_a.py": "x=1\n"}}])
        plan = make_plan(ResearchSpec(title="T", method="discriminant", datasets=["iris"]))
        with self.assertRaises(SecurityError): codegen.generate_llm_project(spec, plan, stub)

    def test_llm_project_without_tests_rejected(self):
        from app.security import SecurityError
        from app.schemas import ResearchSpec
        from app.planner import make_plan
        plan = make_plan(ResearchSpec(title="T", method="discriminant", datasets=["iris"]))
        stub = StubProvider([{"files": {"pkg/a.py": "x = 1\n"}}])
        with self.assertRaises(SecurityError): codegen.generate_llm_project(ResearchSpec(title="T"), plan, stub)

    def test_llm_repair_path_used_when_rules_cannot_fix(self):
        e, s, db = make_engine(provider=StubProvider([]))
        orig = codegen.generate_template_project

        def broken(spec, plan, fault=False):
            f = orig(spec, plan, False)
            f["lda_repro/lda.py"] = f["lda_repro/lda.py"].replace("np.linalg.inv(Sw)", "np.linalg.inv(Sw) * undefined_scale_xyz")
            return f
        codegen.generate_template_project = broken
        try:
            good = codegen._lda_py(False)
            e.llm.responses = [{"file": "lda_repro/lda.py", "content": good, "rationale": "remove undefined factor"}]
            pid = e.create("t", sample_pdf(), Budget())
            # spec extraction would call the LLM in llm-mode; provide spec JSON first
            e.llm.responses.insert(0, {"title": "Linear Discriminant Iris", "method": "linear discriminant", "datasets": ["Iris dataset"],
                                       "hyperparameters": {"test_size": 0.3, "random_seed": 42}, "metrics": ["accuracy"],
                                       "reported_results": [{"metric": "accuracy", "value": 0.98, "dataset": "iris", "split": "70/30", "page": 2}]})
            e.run_ingest(pid)
            e.approve_spec(pid, None)
            # make_plan uses the template for iris/discriminant specs, so no LLM plan call is consumed
            e.run_plan(pid); e.approve_plan(pid, True); e.run_execution(pid)
        finally:
            codegen.generate_template_project = orig
        st = e.load(pid)
        self.assertEqual(e.status(pid), S.COMPLETED, st.blocked_reason)
        self.assertEqual(st.attempts[0]["patch"]["source"], "llm")
        self.assertEqual(st.mode, "llm")
