"""Metric calculation. Every metric is computed from predictions - nothing is hardcoded."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

from .utils import to_native

PRIMARY = {"classification": ("f1_weighted", True), "regression": ("rmse", False)}  # (name, higher_is_better)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_weighted": "Precision (weighted avg)",
    "recall_weighted": "Recall (weighted avg)",
    "f1_weighted": "F1-score (weighted avg)",
    "f1_macro": "F1-score (macro avg)",
    "roc_auc": "ROC-AUC (binary, or one-vs-rest macro)",
    "mae": "MAE (lower is better)",
    "mse": "MSE (lower is better)",
    "rmse": "RMSE (lower is better)",
    "r2": "R² (higher is better)",
}


def classification_metrics(y_true, y_pred, y_proba=None, classes=None):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = [str(c) for c in (classes if classes is not None else sorted(set(y_true) | set(y_pred)))]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": None,
    }
    if y_proba is not None and len(labels) >= 2:
        proba = np.asarray(y_proba)
        present = len(set(y_true))
        try:
            if len(labels) == 2 and present == 2:
                metrics["roc_auc"] = roc_auc_score((y_true == labels[1]).astype(int), proba[:, 1])
            elif len(labels) > 2 and present == len(labels):
                metrics["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            metrics["roc_auc"] = None  # not computable on this sample
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    details = {"confusion_matrix": {"labels": labels, "matrix": cm.tolist()}, "classification_report": report}
    return to_native(metrics), to_native(details)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None
    return to_native({"mae": mean_absolute_error(y_true, y_pred), "mse": mse,
                      "rmse": float(np.sqrt(mse)), "r2": r2})


def primary_metric(problem_type, metrics):
    name, higher = PRIMARY[problem_type]
    return name, metrics.get(name), higher
