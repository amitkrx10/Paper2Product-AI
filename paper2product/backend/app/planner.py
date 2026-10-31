"""Stage B - research planner. Deterministic for the supported template, LLM-assisted otherwise (schema-validated)."""
from __future__ import annotations
from .schemas import Plan, PlanTask, ResearchSpec, SchemaError
from .codegen import supports_template


def make_plan(spec: ResearchSpec, provider=None) -> Plan:
    assumptions = list(spec.missing_info and [f"Missing in paper: {m}" for m in spec.missing_info] or [])
    hp = spec.hyperparameters
    if "random_seed" not in hp and "seed" not in hp:
        assumptions.append("Random seed not given in the paper; default 42 will be used and recorded.")
    open_q = [m for m in spec.missing_info]
    if supports_template(spec):
        assumptions.append("Train/test split is re-created with a seeded stratified split; the paper's exact split is unknown, "
                           "so exact numerical agreement is not expected.")
        files = ["README.md", "requirements.txt", "config.json", "experiment_metadata.json", "lda_repro/__init__.py",
                 "lda_repro/lda.py", "lda_repro/data.py", "run_experiment.py", "tests/__init__.py", "tests/test_lda.py"]
        tasks = [
            PlanTask("T1", "Load Iris and build deterministic stratified split", [], ["lda_repro/data.py"]),
            PlanTask("T2", "Implement Fisher/Gaussian LDA from the paper's equations", ["T1"], ["lda_repro/lda.py"]),
            PlanTask("T3", "Write unit tests incl. cross-check against scikit-learn", ["T2"], ["tests/test_lda.py"]),
            PlanTask("T4", "Write experiment runner producing predictions + metadata", ["T2"], ["run_experiment.py", "config.json"]),
            PlanTask("T5", "Evaluate independently and compare with reported results", ["T3", "T4"], []),
        ]
        return Plan(
            template="fisher_lda_iris",
            approach="Implement the paper's linear discriminant classifier from scratch in NumPy, run it on the Iris data "
                     "with the paper's 70/30 stratified split and seed, and recompute metrics independently of the script.",
            tasks=tasks, files=files, packages=["numpy", "scikit-learn (dataset + reference cross-check only)"],
            dataset_requirements=["Iris (bundled with scikit-learn, no network required)"],
            test_strategy="Shape/prior invariants, separable-blob sanity test, agreement with scikit-learn reference, "
                          "split determinism/stratification, accuracy floor, invalid-input rejection.",
            metrics=spec.metrics or ["accuracy"], expected_outputs=["artifacts/predictions.json", "artifacts/run_meta.json"],
            compute="CPU only, seconds, < 100 MB RAM.",
            risks=["Original split unknown -> numeric differences expected", "Single small dataset -> high variance (45 test samples)",
                   "Document contained prompt-injection text (treated as data)" if spec.security_flags else "None identified in document text"],
            assumptions=assumptions, open_questions=open_q).validate()
    if provider is None:
        return Plan(template="unsupported", approach="", tasks=[], files=[], packages=[], dataset_requirements=[],
                    test_strategy="", metrics=spec.metrics, expected_outputs=[], compute="", risks=[],
                    assumptions=assumptions, open_questions=open_q, feasible=False,
                    infeasible_reason="Demo mode (no LLM configured) only implements the bundled Fisher-LDA/Iris template. "
                                      "Set GEMINI_API_KEY to let the LLM plan and generate code for other papers.")
    prompt = ("Create an implementation plan as JSON with keys approach, tasks[{id,title,depends_on[],files[]}], files[], "
              "packages[], dataset_requirements[], test_strategy, metrics[], expected_outputs[], compute, risks[], "
              "assumptions[], open_questions[]. Only plan what the specification supports; list gaps in open_questions.\n"
              f"<untrusted_document>{spec.to_dict()}</untrusted_document>")
    d = provider.generate_json(prompt)
    try:
        tasks = [PlanTask(str(t["id"]), str(t["title"]), list(t.get("depends_on", [])), list(t.get("files", [])))
                 for t in d.get("tasks", [])]
        plan = Plan(template="llm_generic", approach=str(d.get("approach", "")), tasks=tasks,
                    files=[str(f) for f in d.get("files", [])], packages=[str(x) for x in d.get("packages", [])],
                    dataset_requirements=[str(x) for x in d.get("dataset_requirements", [])],
                    test_strategy=str(d.get("test_strategy", "")), metrics=[str(x) for x in d.get("metrics", [])],
                    expected_outputs=[str(x) for x in d.get("expected_outputs", [])], compute=str(d.get("compute", "")),
                    risks=[str(x) for x in d.get("risks", [])], assumptions=assumptions + [str(x) for x in d.get("assumptions", [])],
                    open_questions=open_q + [str(x) for x in d.get("open_questions", [])])
        return plan.validate()
    except (KeyError, TypeError, SchemaError) as e:
        raise SchemaError(f"LLM plan failed validation: {e}")
