"""Pure-Python tests of the ML services (no Django DB needed). Deterministic data, fixed seeds."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from automl.services import artifacts, datasets, drift, evaluation, monitoring, prediction, training


def make_cls(n=200, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0 + shift, 1, n)
    x2 = rng.normal(5, 2, n)
    color = rng.choice(["red", "green", "blue"], n)
    y = np.where(x1 + 0.3 * rng.normal(size=n) > 0, "pos", "neg")
    df = pd.DataFrame({"x1": x1, "x2": x2, "color": color, "label": y})
    df.loc[::17, "x2"] = np.nan
    df.loc[::23, "color"] = np.nan
    return df


def make_reg(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b"], n)
    y = 3 * x1 - 2 * x2 + (cat == "a") * 1.5 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


class CsvValidation(unittest.TestCase):
    def test_valid_csv(self):
        csv = make_cls().to_csv(index=False).encode()
        df = datasets.read_csv_bytes(csv, 1_000_000)
        self.assertEqual(df.shape, (200, 4))

    def test_rejects_empty_binary_big_and_malformed(self):
        with self.assertRaises(datasets.DatasetError): datasets.read_csv_bytes(b"", 100)
        with self.assertRaises(datasets.DatasetError): datasets.read_csv_bytes(b"a,b\n" + b"1,2\n" * 50, 10)
        with self.assertRaises(datasets.DatasetError): datasets.read_csv_bytes(b"\x89PNG\x00\x01\xff\xfe", 1000)
        with self.assertRaises(datasets.DatasetError): datasets.read_csv_bytes(b"a,b\n1,2\n", 1000)  # too few rows
        with self.assertRaises(datasets.DatasetError): datasets.read_csv_bytes(b"a,b,c\n1,2\n" + b"1,2,3,4,5\n" * 30, 10_000)

    def test_profile(self):
        p = datasets.profile_dataframe(make_cls())
        self.assertEqual(p["rows"], 200)
        self.assertIn("x1", p["numeric"]); self.assertIn("color", p["categorical"])
        self.assertGreater(p["missing_total"], 0)

    def test_target_validation(self):
        df = make_cls()
        with self.assertRaises(datasets.DatasetError): datasets.validate_target(df, "nope", "classification")
        with self.assertRaises(datasets.DatasetError): datasets.validate_target(df, "label", "regression")
        datasets.validate_target(df, "label", "classification")
        self.assertEqual(datasets.infer_problem_type(df["label"]), "classification")
        self.assertEqual(datasets.infer_problem_type(make_reg()["y"]), "regression")
        dist = datasets.target_distribution(df, "label", "classification")
        self.assertEqual(sum(dist["values"]), 200)


class Metrics(unittest.TestCase):
    def test_classification_known_values(self):
        yt = ["a", "a", "b", "b"]; yp = ["a", "b", "b", "b"]
        m, d = evaluation.classification_metrics(yt, yp, np.array([[.9, .1], [.4, .6], [.2, .8], [.1, .9]]), ["a", "b"])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        self.assertEqual(d["confusion_matrix"]["matrix"], [[1, 1], [0, 2]])

    def test_regression_known_values(self):
        m = evaluation.regression_metrics([1, 2, 3], [1, 2, 5])
        self.assertAlmostEqual(m["mae"], 2 / 3); self.assertAlmostEqual(m["mse"], 4 / 3)
        self.assertAlmostEqual(m["rmse"], (4 / 3) ** 0.5)

    def test_roc_auc_none_when_one_class(self):
        m, _ = evaluation.classification_metrics(["a", "a"], ["a", "a"], np.array([[.9, .1], [.8, .2]]), ["a", "b"])
        self.assertIsNone(m["roc_auc"])


class Training(unittest.TestCase):
    def test_all_classifiers_train_on_same_split(self):
        df = make_cls()
        prep = training.prepare_data(df, "label", "classification", 42)
        self.assertEqual(prep["split_info"]["train"] + prep["split_info"]["validation"] + prep["split_info"]["test"], 200)
        tests = []
        for key in training.CLASSIFIERS:
            r = training.fit_and_evaluate(prep, key)
            self.assertGreater(r["metrics"]["accuracy"], 0.7, key)
            self.assertGreater(r["latency_ms"], 0)
            tests.append(len(prep["y_test"]))
        self.assertEqual(len(set(tests)), 1)

    def test_all_regressors(self):
        prep = training.prepare_data(make_reg(), "y", "regression", 1)
        for key in training.REGRESSORS:
            r = training.fit_and_evaluate(prep, key)
            self.assertIn("rmse", r["metrics"])
        lin = training.fit_and_evaluate(prep, "linear_regression")
        self.assertGreater(lin["metrics"]["r2"], 0.95)

    def test_reproducible(self):
        a = training.fit_and_evaluate(training.prepare_data(make_cls(), "label", "classification", 7), "random_forest")
        b = training.fit_and_evaluate(training.prepare_data(make_cls(), "label", "classification", 7), "random_forest")
        self.assertEqual(a["metrics"], b["metrics"])

    def test_no_leakage_scaler_fit_on_train_only(self):
        prep = training.prepare_data(make_reg(), "y", "regression", 3)
        r = training.fit_and_evaluate(prep, "ridge")
        scaler = r["pipeline"].named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, prep["X_train"][prep["numeric"]].mean().values, rtol=1e-6)

    def test_class_too_small_rejected(self):
        df = make_cls(); df.loc[df.index[:3], "label"] = "rare"
        with self.assertRaises(datasets.DatasetError):
            training.prepare_data(df, "label", "classification")


class SaveLoadPredict(unittest.TestCase):
    def setUp(self):
        self.prep = training.prepare_data(make_cls(), "label", "classification", 42)
        self.res = training.fit_and_evaluate(self.prep, "logistic_regression")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_save_load_roundtrip_and_tamper_detection(self):
        name, sha, size = artifacts.save_pipeline(self.res["pipeline"], self.tmp.name, "m.joblib")
        pipe = artifacts.load_pipeline(self.tmp.name, name, sha)
        np.testing.assert_array_equal(pipe.predict(self.prep["X_test"]), self.res["pipeline"].predict(self.prep["X_test"]))
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_pipeline(self.tmp.name, name, "0" * 64)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_pipeline(self.tmp.name, "../etc/passwd", sha)
        with open(f"{self.tmp.name}/m.joblib", "ab") as fh: fh.write(b"x")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_pipeline(self.tmp.name, name, sha)

    def test_prediction_and_validation(self):
        schema = self.prep["schema"]
        clean, errs = prediction.validate_input(schema, {"x1": "1.2", "x2": "", "color": "red"})
        self.assertEqual(errs, {})
        out = prediction.predict_row(self.res["pipeline"], schema, clean, "classification", self.prep["classes"])
        self.assertIn(out["prediction"], ("pos", "neg"))
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0)
        self.assertGreater(out["latency_ms"], 0)
        _, errs = prediction.validate_input(schema, {"x1": "abc", "color": "purple"})
        self.assertIn("x1", errs); self.assertIn("color", errs)
        _, errs = prediction.validate_input(schema, {"x1": "nan"})
        self.assertIn("x1", errs)


class Drift(unittest.TestCase):
    def test_psi_identical_is_near_zero_and_shift_is_large(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 2000); b = rng.normal(0, 1, 2000); c = rng.normal(2, 1, 2000)
        self.assertLess(drift.psi_numeric(a, b), 0.1)
        self.assertGreater(drift.psi_numeric(a, c), 0.25)
        _s, p_same = drift.ks_test(a, b); _s, p_shift = drift.ks_test(a, c)
        self.assertGreater(p_same, 0.01); self.assertLess(p_shift, 1e-6)

    def test_categorical_drift(self):
        ref = ["a"] * 300 + ["b"] * 300; same = ["a"] * 290 + ["b"] * 310; moved = ["a"] * 100 + ["b"] * 500
        self.assertLess(drift.psi_categorical(ref, same), 0.1)
        self.assertGreater(drift.psi_categorical(ref, moved), 0.25)
        self.assertLess(drift.chi_square_test(ref, moved)[1], 0.01)

    def test_constant_reference_and_none_values(self):
        self.assertIsNotNone(drift.psi_numeric([1] * 50, [1] * 50 + [5] * 5))
        self.assertIsNone(drift.psi_numeric([], [1, 2]))
        self.assertEqual(drift.psi_numeric([None, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, None]) >= 0, True)

    def test_detect_drift_end_to_end_and_insufficient_data(self):
        prep = training.prepare_data(make_cls(400), "label", "classification", 42)
        res = training.fit_and_evaluate(prep, "random_forest")
        ref_rows = res["reference"]["rows"]; ref_pred = res["reference"]["predictions"]
        shifted = make_cls(300, seed=9, shift=3.0)
        cur_rows = shifted[["x1", "x2", "color"]].replace({np.nan: None}).to_dict("records")
        cur_pred = res["pipeline"].predict(training._normalise(shifted[["x1", "x2", "color"]], ["color"], ["x1", "x2"])).tolist()
        rep = drift.detect_drift(ref_rows, cur_rows, prep["schema"], ref_pred, cur_pred, "classification")
        self.assertEqual(rep["status"], "drift"); self.assertIn("x1", rep["affected_features"])
        stable = make_cls(300, seed=11)
        rows2 = stable[["x1", "x2", "color"]].replace({np.nan: None}).to_dict("records")
        rep2 = drift.detect_drift(ref_rows, rows2, prep["schema"], ref_pred, ref_pred[:300] if len(ref_pred) >= 300 else ref_pred, "classification")
        self.assertNotIn("x1", rep2["affected_features"])
        few = drift.detect_drift(ref_rows, cur_rows[:5], prep["schema"], ref_pred, cur_pred[:5], "classification")
        self.assertEqual(few["status"], "insufficient_data"); self.assertEqual(few["features"], [])


def _log(i, out, actual=None, status="success", lat=1.0, ver=1):
    return {"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i % 3), "status": status,
            "latency_ms": lat if status == "success" else None, "output": out, "actual_label": actual, "model_version": ver}


class Monitoring(unittest.TestCase):
    def test_missing_ground_truth_is_not_invented(self):
        logs = [_log(i, "a") for i in range(10)]
        s = monitoring.summarize_logs(logs, "classification", {"f1_weighted": 0.9})
        self.assertFalse(s["performance"]["available"]); self.assertEqual(s["performance"]["n_labeled"], 0)
        self.assertNotIn("metrics", s["performance"])

    def test_volume_latency_failure(self):
        logs = [_log(i, "a", lat=float(i + 1)) for i in range(20)] + [_log(0, None, status="failed")] * 5
        s = monitoring.summarize_logs(logs, "classification")
        self.assertEqual(s["total"], 25); self.assertAlmostEqual(s["failure_rate"], 0.2)
        self.assertAlmostEqual(s["latency"]["median_ms"], 10.5); self.assertAlmostEqual(s["latency"]["p95_ms"], 19.05)

    def test_performance_and_degradation_flag(self):
        logs = [_log(i, "a", "a" if i % 2 else "b") for i in range(40)]
        s = monitoring.summarize_logs(logs, "classification", {"f1_weighted": 0.95})
        self.assertTrue(s["performance"]["available"]); self.assertAlmostEqual(s["performance"]["metrics"]["accuracy"], 0.5)
        self.assertTrue(s["performance"]["degradation"]["degraded"])
        few = monitoring.summarize_logs(logs[:5], "classification", {"f1_weighted": 0.95})
        self.assertIsNone(few["performance"]["degradation"])

    def test_regression_performance(self):
        logs = [_log(i, str(float(i)), str(float(i) + 1)) for i in range(40)]
        s = monitoring.summarize_logs(logs, "regression", {"rmse": 0.5})
        self.assertAlmostEqual(s["performance"]["metrics"]["mae"], 1.0)
        self.assertTrue(s["performance"]["degradation"]["degraded"])


if __name__ == "__main__":
    unittest.main()
