"""Django tests: models, workflow, views. Isolated: Django's test DB + temp artifact/dataset dirs."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from automl import models
from automl.services import datasets, workflow

from .test_ml_services import make_cls, make_reg


def csv_bytes(df):
    return df.to_csv(index=False).encode()


class BaseCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._ov = override_settings(ARTIFACT_DIR=Path(cls._tmp.name) / "art", DATASET_DIR=Path(cls._tmp.name) / "ds")
        cls._ov.enable()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._ov.disable()
        cls._tmp.cleanup()

    def setUp(self):
        U = get_user_model()
        self.staff = U.objects.create_user("staff", password="pw-for-tests-123", is_staff=True)
        self.user = U.objects.create_user("plain", password="pw-for-tests-123")

    def make_cls_models(self, algos=("logistic_regression", "random_forest")):
        ds = workflow.register_dataset("cls", csv_bytes(make_cls(300)), "cls.csv", actor="t")
        trained, failures = workflow.train_models(ds, "label", "classification", list(algos), "t")
        self.assertEqual(failures, {})
        return ds, trained


class WorkflowTests(BaseCase):
    def test_dataset_registration_and_rejection(self):
        ds = workflow.register_dataset("d", csv_bytes(make_cls()), "d.csv")
        self.assertEqual(ds.n_rows, 200)
        self.assertTrue((Path(self.__class__._tmp.name) / "ds" / ds.file_path).exists())
        with self.assertRaises(datasets.DatasetError):
            workflow.register_dataset("bad", b"not,a\nvalid", "bad.csv")
        self.assertEqual(models.Dataset.objects.count(), 1)

    def test_integration_dataset_train_predict_log(self):
        ds, trained = self.make_cls_models()
        self.assertEqual(len(trained), 2)
        tm = trained[0]
        self.assertTrue(tm.artifact_sha256 and tm.model_size_bytes > 0 and tm.inference_latency_ms > 0)
        entry, errors = workflow.run_prediction(tm, {"x1": "1.5", "x2": "5", "color": "red"}, "t")
        self.assertEqual(errors, {})
        self.assertEqual(entry.status, "success")
        self.assertIn(entry.output, ("pos", "neg"))
        self.assertGreater(entry.latency_ms, 0)
        self.assertEqual(models.PredictionLog.objects.count(), 1)

    def test_invalid_input_and_tampered_artifact_logged_as_failed(self):
        _ds, trained = self.make_cls_models(("logistic_regression",))
        tm = trained[0]
        entry, errors = workflow.run_prediction(tm, {"x1": "abc", "x2": "", "color": "red"})
        self.assertIn("x1", errors); self.assertEqual(entry.status, "failed")
        (Path(self.__class__._tmp.name) / "art" / tm.artifact_file).write_bytes(b"evil")
        entry, _ = workflow.run_prediction(tm, {"x1": "1", "x2": "1", "color": "red"})
        self.assertEqual(entry.status, "failed"); self.assertIn("integrity", entry.error)

    def test_regression_training_and_comparison_best(self):
        ds = workflow.register_dataset("reg", csv_bytes(make_reg()), "r.csv")
        trained, _ = workflow.train_models(ds, "y", "regression", ["linear_regression", "decision_tree", "ridge"])
        best = workflow.best_model(trained)
        self.assertEqual(best.primary_metric, "rmse")
        self.assertEqual(best.primary_value, min(m.primary_value for m in trained))

    def test_production_promote_and_rollback(self):
        _ds, trained = self.make_cls_models()
        a, b = trained
        self.assertIsNone(workflow.get_production())
        workflow.promote(a, "staff"); workflow.promote(b, "staff")
        self.assertEqual(workflow.get_production(), b)
        self.assertEqual(workflow.rollback("staff"), a)
        self.assertEqual(workflow.get_production(), a)
        self.assertEqual(models.ProductionModel.objects.filter(is_current=True).count(), 1)
        self.assertTrue(models.AuditLog.objects.filter(action="rollback").exists())

    def test_rollback_without_history_errors(self):
        _ds, trained = self.make_cls_models(("logistic_regression",))
        with self.assertRaises(ValueError): workflow.rollback()
        workflow.promote(trained[0])
        with self.assertRaises(ValueError): workflow.rollback()

    def test_labels_monitoring_and_missing_ground_truth(self):
        _ds, trained = self.make_cls_models(("logistic_regression",))
        tm = trained[0]
        entries = [workflow.run_prediction(tm, {"x1": str(v), "x2": "5", "color": "red"})[0] for v in np.linspace(-2, 2, 40)]
        data, n = workflow.compute_monitoring(tm)
        self.assertEqual(n, 40); self.assertFalse(data["performance"]["available"])
        with self.assertRaises(ValueError): workflow.add_actual_label(entries[0], "maybe")
        for e, v in zip(entries, np.linspace(-2, 2, 40)):
            workflow.add_actual_label(e, "pos" if v > 0 else "neg")
        data, _ = workflow.compute_monitoring(tm)
        self.assertTrue(data["performance"]["available"]); self.assertEqual(data["performance"]["n_labeled"], 40)
        snap = workflow.save_snapshot(tm); self.assertEqual(snap.window_size, 40)

    def test_drift_insufficient_then_detected(self):
        _ds, trained = self.make_cls_models(("random_forest",))
        tm = trained[0]
        self.assertEqual(workflow.run_drift(tm).status, "insufficient_data")
        for v in np.random.default_rng(0).normal(4, 0.5, 60):  # far outside the training x1 range
            workflow.run_prediction(tm, {"x1": str(v), "x2": "5", "color": "red"})
        rep = workflow.run_drift(tm)
        self.assertEqual(rep.status, "drift"); self.assertIn("x1", rep.results["affected_features"])


class ViewTests(BaseCase):
    def test_login_required(self):
        for name in ("overview", "datasets", "train", "compare", "predict", "history", "monitoring", "drift", "versions", "system"):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302, name); self.assertIn("/accounts/login/", r["Location"])

    def test_health_is_public(self):
        r = self.client.get(reverse("health"))
        self.assertEqual(r.status_code, 200); self.assertEqual(r.json()["status"], "ok")

    def test_all_pages_render_empty_states(self):
        self.client.force_login(self.staff)
        for name in ("overview", "datasets", "train", "compare", "predict", "history", "monitoring", "drift", "versions", "system"):
            r = self.client.get(reverse(name)); self.assertEqual(r.status_code, 200, name)

    def test_upload_view_rejects_bad_files(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        self.client.force_login(self.staff)
        r = self.client.post(reverse("datasets"), {"name": "x", "file": SimpleUploadedFile("x.txt", b"hello")})
        self.assertContains(r, "Only .csv files"); self.assertEqual(models.Dataset.objects.count(), 0)
        r = self.client.post(reverse("datasets"), {"name": "x", "file": SimpleUploadedFile("x.csv", b"a,b\n1,2\n")})
        self.assertContains(r, "Too few rows")
        r = self.client.post(reverse("datasets"), {"name": "ok", "file": SimpleUploadedFile("ok.csv", csv_bytes(make_cls()))})
        self.assertEqual(r.status_code, 302); self.assertEqual(models.Dataset.objects.count(), 1)

    def test_full_flow_through_views(self):
        self.client.force_login(self.staff)
        ds = workflow.register_dataset("flow", csv_bytes(make_cls(300)), "f.csv")
        r = self.client.post(reverse("dataset_detail", args=[ds.pk]), {"target_column": "label", "problem_type": "classification"})
        self.assertEqual(r.status_code, 302)
        r = self.client.post(reverse("train"), {"dataset": ds.pk, "target_column": "label", "problem_type": "classification",
                                                "algorithms": ["classification:logistic_regression", "classification:decision_tree"], "random_state": 42})
        self.assertEqual(r.status_code, 302, getattr(r, "context", None) and r.context["form"].errors)
        self.assertEqual(models.TrainedModel.objects.count(), 2)
        self.assertEqual(self.client.get(reverse("compare")).status_code, 200)
        tm = models.TrainedModel.objects.order_by("version").first()
        self.client.post(reverse("promote", args=[tm.pk]))
        r = self.client.post(reverse("predict"), {"model": tm.pk, "f0": "1.0", "f1": "5", "f2": "red"})
        self.assertContains(r, "Result"); self.assertEqual(models.PredictionLog.objects.filter(status="success").count(), 1)
        self.assertContains(self.client.get(reverse("history")), "success")
        self.assertEqual(self.client.get(reverse("monitoring") + f"?model={tm.pk}").status_code, 200)
        self.assertEqual(self.client.post(reverse("drift"), {"model": tm.pk}).status_code, 302)
        self.assertContains(self.client.get(reverse("drift") + f"?model={tm.pk}"), "insufficient_data")

    def test_train_form_rejects_mismatched_algorithms_and_target(self):
        self.client.force_login(self.staff)
        ds = workflow.register_dataset("flow", csv_bytes(make_cls()), "f.csv")
        r = self.client.post(reverse("train"), {"dataset": ds.pk, "target_column": "nope", "problem_type": "classification",
                                                "algorithms": ["regression:ridge"], "random_state": 1})
        self.assertEqual(r.status_code, 200); self.assertEqual(models.TrainedModel.objects.count(), 0)

    def test_non_staff_cannot_promote_or_rollback(self):
        _ds = workflow.register_dataset("d", csv_bytes(make_cls()), "d.csv")
        trained, _ = workflow.train_models(_ds, "label", "classification", ["logistic_regression"])
        self.client.force_login(self.user)
        self.assertEqual(self.client.post(reverse("promote", args=[trained[0].pk])).status_code, 403)
        self.assertEqual(self.client.post(reverse("rollback")).status_code, 403)
        self.assertIsNone(workflow.get_production())

    def test_csrf_enforced_on_post(self):
        from django.test import Client
        c = Client(enforce_csrf_checks=True); c.force_login(self.staff)
        self.assertEqual(c.post(reverse("load_samples")).status_code, 403)

    def test_sample_loading_idempotent(self):
        self.client.force_login(self.staff)
        self.client.post(reverse("load_samples")); n = models.Dataset.objects.count()
        self.assertGreaterEqual(n, 3); self.assertTrue(all(d.is_sample for d in models.Dataset.objects.all()))
        self.client.post(reverse("load_samples")); self.assertEqual(models.Dataset.objects.count(), n)

    def test_history_search_and_pagination(self):
        self.client.force_login(self.staff)
        ds = workflow.register_dataset("d", csv_bytes(make_cls(300)), "d.csv")
        trained, _ = workflow.train_models(ds, "label", "classification", ["logistic_regression"])
        for i in range(25):
            workflow.run_prediction(trained[0], {"x1": str(i / 10), "x2": "5", "color": "blue"})
        r = self.client.get(reverse("history")); self.assertEqual(len(r.context["page"].object_list), 20)
        r = self.client.get(reverse("history") + "?q=color=blue&page=2"); self.assertEqual(len(r.context["page"].object_list), 5)
        r = self.client.get(reverse("history") + "?q=zzzz-nothing"); self.assertContains(r, "match your filters")
