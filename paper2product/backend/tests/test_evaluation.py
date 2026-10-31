import math, unittest
from app.evaluation import *
from app.schemas import ReportedResult as R


class TestMetrics(unittest.TestCase):
    def test_classification_known_values(self):
        m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(m["per_class"]["0"]["precision"], 1.0)
        self.assertAlmostEqual(m["per_class"]["0"]["recall"], 0.5)
        self.assertAlmostEqual(m["per_class"]["1"]["f1"], 0.8)
        self.assertAlmostEqual(m["f1"], (2 / 3 + 0.8) / 2)

    def test_regression_known_values(self):
        m = regression_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(m["mae"], 1 / 3); self.assertAlmostEqual(m["rmse"], math.sqrt(1 / 3)); self.assertAlmostEqual(m["r2"], 0.5)

    def test_bad_input(self):
        with self.assertRaises(ValueError): classification_metrics([], [])
        with self.assertRaises(ValueError): regression_metrics([1], [1, 2])


class TestComparison(unittest.TestCase):
    ctx = {"dataset_name": "iris", "test_size": 0.3}

    def cmp(self, rep, meas, run=None):
        return compare_results([rep], {"accuracy": meas}, run or self.ctx)[0]

    def test_within_tolerance_reproduced(self):
        c = self.cmp(R("accuracy", 0.98, "iris", "70/30"), 0.978)
        self.assertEqual(c.status, "reproduced"); self.assertTrue(c.comparable)

    def test_partial_and_not_reproduced(self):
        self.assertEqual(self.cmp(R("accuracy", 0.98, "iris", "70/30"), 0.9556).status, "partially_reproduced")
        self.assertEqual(self.cmp(R("accuracy", 0.98, "iris", "70/30"), 0.70).status, "not_reproduced")

    def test_dataset_mismatch_blocks_comparison(self):
        c = self.cmp(R("accuracy", 0.98, "mnist", "70/30"), 0.98)
        self.assertEqual(c.status, "unable_to_verify"); self.assertFalse(c.comparable)
        self.assertTrue(any("Dataset mismatch" in w for w in c.warnings))

    def test_split_mismatch_blocks_comparison(self):
        c = self.cmp(R("accuracy", 0.98, "iris", "80/20"), 0.98)
        self.assertEqual(c.status, "unable_to_verify"); self.assertTrue(any("Split mismatch" in w for w in c.warnings))

    def test_unstated_dataset_or_split_warns(self):
        self.assertFalse(self.cmp(R("accuracy", 0.98), 0.98).comparable)
        self.assertTrue(any("split/protocol not stated" in w for w in self.cmp(R("accuracy", 0.98, "iris"), 0.98).warnings))

    def test_unmeasured_metric(self):
        c = compare_results([R("rmse", 1.0, "iris")], {"accuracy": 1.0}, self.ctx)[0]
        self.assertEqual(c.status, "unable_to_verify")

    def test_overall_status_rules(self):
        ok = self.cmp(R("accuracy", 0.98, "iris", "70/30"), 0.98)
        self.assertEqual(overall_status([ok], True, True)[0], "reproduced")
        self.assertEqual(overall_status([ok], True, False)[0], "unable_to_verify")
        self.assertEqual(overall_status([], True, True)[0], "unable_to_verify")
        bad = self.cmp(R("accuracy", 0.98, "mnist", "70/30"), 0.98)
        self.assertEqual(overall_status([bad], True, True)[0], "unable_to_verify")
        self.assertIn("did not all pass", overall_status([ok], False, True)[1])
