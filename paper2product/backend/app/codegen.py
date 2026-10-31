"""Stage D - code generation.

Two strategies:
  * 'fisher_lda_iris' : deterministic, reviewed template (works with no LLM; used for the demo).
  * 'llm_generic'     : LLM-generated project, validated (paths, sizes, static security scan) before it is accepted.
"""
from __future__ import annotations
import json
import os

from .schemas import ResearchSpec, Plan
from .security import validate_file_map, scan_python, SecurityError, resolve_inside

FAULT_MARKER = "Sw_inverse"   # deliberately introduced NameError (demo of the repair loop)


def _lda_py(fault: bool) -> str:
    ref = "Sw_inverse" if fault else "Sw_inv"
    return f'''"""Fisher / Gaussian linear discriminant classifier implemented from scratch with NumPy.

Implements the method described in the specification:
  S       = sum_k sum_(i in k) (x_i - mu_k)(x_i - mu_k)^T / (N - K)      (pooled within-class covariance)
  delta_k = x^T S^-1 mu_k - 0.5 mu_k^T S^-1 mu_k + log(pi_k)             (linear discriminant score)
"""
from __future__ import annotations
import numpy as np


class FisherLDA:
    def __init__(self, regularization: float = 1e-9) -> None:
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        self.regularization = regularization

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherLDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and match y in length")
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        if n <= k:
            raise ValueError("need more samples than classes")
        means, priors = [], []
        Sw = np.zeros((d, d))
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            D = Xc - mu
            Sw += D.T @ D
            means.append(mu)
            priors.append(len(Xc) / n)
        Sw = Sw / (n - k) + self.regularization * np.eye(d)
        Sw_inv = np.linalg.inv(Sw)
        self.means_ = np.array(means)
        self.priors_ = np.array(priors)
        self.coef_ = self.means_ @ {ref}
        self.intercept_ = -0.5 * np.sum(self.coef_ * self.means_, axis=1) + np.log(self.priors_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
'''


DATA_PY = '''"""Dataset loading and a deterministic, stratified train/test split (no external split helpers)."""
from __future__ import annotations
import numpy as np


def load_iris_split(test_size: float, seed: int):
    from sklearn.datasets import load_iris  # Iris ships inside scikit-learn: no network needed
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    d = load_iris()
    X, y = d.data, d.target
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx, test_idx = sorted(train_idx), sorted(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], test_idx, [str(n) for n in d.target_names]
'''

RUN_PY = '''"""Experiment entry point: trains, predicts, writes artifacts/. Run: python run_experiment.py"""
from __future__ import annotations
import json
import os
import platform
import time

import numpy as np

from lda_repro.data import load_iris_split
from lda_repro.lda import FisherLDA


def main() -> None:
    cfg = json.load(open("config.json"))
    t0 = time.time()
    Xtr, Xte, ytr, yte, test_idx, names = load_iris_split(cfg["test_size"], cfg["seed"])
    model = FisherLDA(cfg["regularization"]).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = float(np.mean(pred == yte))
    os.makedirs("artifacts", exist_ok=True)
    json.dump({"y_true": yte.tolist(), "y_pred": pred.tolist(), "labels": [0, 1, 2], "label_names": names,
               "test_indices": test_idx, "n_train": int(len(ytr)), "n_test": int(len(yte))},
              open("artifacts/predictions.json", "w"))
    import sklearn
    json.dump({"script_reported_accuracy": acc, "seed": cfg["seed"], "test_size": cfg["test_size"],
               "dataset": cfg["dataset"], "runtime_s": round(time.time() - t0, 4),
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scikit_learn": sklearn.__version__, "platform": platform.platform()}},
              open("artifacts/run_meta.json", "w"), indent=2)
    print(f"test accuracy (script-reported, re-computed independently by the evaluator): {acc:.4f}")


if __name__ == "__main__":
    main()
'''

TEST_PY = '''import unittest
import numpy as np

from lda_repro.lda import FisherLDA
from lda_repro.data import load_iris_split


class TestFisherLDA(unittest.TestCase):
    def test_fit_produces_expected_shapes(self):
        Xtr, _, ytr, _, _, _ = load_iris_split(0.3, 42)
        m = FisherLDA().fit(Xtr, ytr)
        self.assertEqual(m.coef_.shape, (3, 4))
        self.assertEqual(m.intercept_.shape, (3,))
        self.assertAlmostEqual(float(m.priors_.sum()), 1.0)

    def test_separable_blobs_are_classified(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(8, 1, (60, 2))])
        y = np.array([0] * 60 + [1] * 60)
        self.assertGreater(float(np.mean(FisherLDA().fit(X, y).predict(X) == y)), 0.99)

    def test_agrees_with_scikit_learn_reference(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        Xtr, Xte, ytr, _, _, _ = load_iris_split(0.3, 42)
        ours = FisherLDA().fit(Xtr, ytr).predict(Xte)
        ref = LinearDiscriminantAnalysis(solver="lsqr").fit(Xtr, ytr).predict(Xte)
        self.assertGreaterEqual(float(np.mean(ours == ref)), 0.95)

    def test_split_is_stratified_and_deterministic(self):
        a = load_iris_split(0.3, 7)
        b = load_iris_split(0.3, 7)
        self.assertEqual(a[4], b[4])
        self.assertEqual(sorted(np.bincount(a[3]).tolist()), [15, 15, 15])

    def test_accuracy_floor_on_heldout_data(self):
        Xtr, Xte, ytr, yte, _, _ = load_iris_split(0.3, 42)
        self.assertGreater(float(np.mean(FisherLDA().fit(Xtr, ytr).predict(Xte) == yte)), 0.9)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            FisherLDA().fit(np.zeros((3, 2)), np.array([0, 1]))
        with self.assertRaises(ValueError):
            FisherLDA(regularization=-1)


if __name__ == "__main__":
    unittest.main()
'''


def _readme(spec: ResearchSpec, cfg: dict, fault: bool) -> str:
    return f"""# Reproduction: {spec.title}

Generated by Paper2Product AI (template `fisher_lda_iris`, deterministic - no LLM wrote this code).
{"**Demo fault injection was enabled for this build** (see execution history for the automated repair)." if fault else ""}

## Rerun
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
python run_experiment.py          # writes artifacts/predictions.json and artifacts/run_meta.json
```

## Configuration (`config.json`)
```json
{json.dumps(cfg, indent=2)}
```
A passing unit test does not prove the paper was reproduced; see the generated report for the evidence-based status.
"""


def supports_template(spec: ResearchSpec) -> bool:
    text = (spec.method + " " + spec.title).lower()
    return "discriminant" in text and any("iris" in d.lower() for d in spec.datasets)


def generate_template_project(spec: ResearchSpec, plan: Plan, inject_fault: bool = False) -> dict[str, str]:
    hp = spec.hyperparameters
    cfg = {"dataset": "sklearn:iris", "test_size": float(hp.get("test_size", 0.3)),
           "seed": int(hp.get("random_seed", hp.get("seed", 42))),
           "regularization": float(hp.get("regularization", 1e-9))}
    meta = {"paper_title": spec.title, "template": "fisher_lda_iris", "generator": "deterministic-template",
            "llm_used": False, "fault_injected": inject_fault, "assumptions": plan.assumptions}
    files = {
        "README.md": _readme(spec, cfg, inject_fault),
        "requirements.txt": "numpy>=1.24\nscikit-learn>=1.3\n",
        "config.json": json.dumps(cfg, indent=2) + "\n",
        "experiment_metadata.json": json.dumps(meta, indent=2) + "\n",
        "lda_repro/__init__.py": "",
        "lda_repro/lda.py": _lda_py(inject_fault),
        "lda_repro/data.py": DATA_PY,
        "run_experiment.py": RUN_PY,
        "tests/__init__.py": "",
        "tests/test_lda.py": TEST_PY,
    }
    return validate_file_map(files)


def generate_llm_project(spec: ResearchSpec, plan: Plan, provider) -> dict[str, str]:
    prompt = (
        "Generate a complete Python project implementing the method below faithfully (do not substitute an easier "
        "algorithm). Return JSON {\"files\": {\"relative/path.py\": \"content\"}, \"assumptions\": [...]}. "
        "Requirements: package directory + tests/ using unittest, config.json, requirements.txt, README.md, "
        "run_experiment.py that writes artifacts/predictions.json with keys y_true, y_pred, labels, "
        "artifacts/run_meta.json with dataset, seed, test_size. No network access, no subprocess, no eval/exec, "
        "no file access outside the project directory.\n"
        f"SPEC: {json.dumps(spec.to_dict())[:12000]}\nPLAN: {json.dumps(plan.to_dict())[:4000]}")
    d = provider.generate_json(prompt)
    files = validate_file_map(d.get("files", {}))
    for path, src in files.items():
        if path.endswith(".py"):
            problems = scan_python(src, path)
            if problems:
                raise SecurityError("generated code rejected by policy: " + "; ".join(problems[:5]))
    if not any(p.startswith("tests/") for p in files):
        raise SecurityError("generated project has no tests")
    return files


def write_project(root: str, files: dict[str, str]) -> None:
    os.makedirs(root, exist_ok=True)
    for rel, content in validate_file_map(files).items():
        full = resolve_inside(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


def read_project(root: str) -> dict[str, str]:
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "artifacts") and not d.startswith(".")]
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if os.path.splitext(f)[1].lower() in (".py", ".md", ".txt", ".json"):
                out[rel] = open(p, encoding="utf-8").read()
    return dict(sorted(out.items()))
