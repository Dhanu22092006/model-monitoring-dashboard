"""ORM glue: connects the pure ML services to Django models. All state changes are audited."""
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from automl import models
from . import artifacts, datasets, drift, evaluation, monitoring, prediction, training
from .utils import to_native

log = logging.getLogger(__name__)


def audit(action, detail="", actor=""):
    models.AuditLog.objects.create(action=action, detail=str(detail)[:500], actor=actor)


# ---------------- datasets ----------------
def register_dataset(name, data: bytes, filename="", is_sample=False, actor=""):
    df = datasets.read_csv_bytes(data, settings.MAX_UPLOAD_BYTES)
    settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}.csv"
    (settings.DATASET_DIR / stored).write_bytes(data)
    ds = models.Dataset.objects.create(
        name=name[:120], original_filename=Path(filename).name[:255], file_path=stored, is_sample=is_sample,
        n_rows=df.shape[0], n_columns=df.shape[1], profile=datasets.profile_dataframe(df), uploaded_by=actor)
    audit("dataset_uploaded", f"{ds.name} ({ds.n_rows}x{ds.n_columns})", actor)
    return ds


def load_dataframe(ds) -> pd.DataFrame:
    base = Path(settings.DATASET_DIR).resolve()
    path = (base / ds.file_path).resolve()
    if base not in path.parents or not path.is_file():
        raise datasets.DatasetError("Dataset file is missing from storage.")
    return datasets.read_csv_bytes(path.read_bytes(), settings.MAX_UPLOAD_BYTES * 4)


def load_samples(actor=""):
    created = []
    for f in sorted(Path(settings.SAMPLE_DATA_DIR).glob("SAMPLE_*.csv")):
        name = f"[Sample] {f.stem.replace('SAMPLE_', '').replace('_', ' ')}"
        if models.Dataset.objects.filter(name=name, is_sample=True).exists():
            continue
        ds = register_dataset(name, f.read_bytes(), f.name, is_sample=True, actor=actor)
        df = load_dataframe(ds)
        target = df.columns[-1]
        ds.target_column, ds.problem_type = target, datasets.infer_problem_type(df[target])
        ds.save(update_fields=["target_column", "problem_type"])
        created.append(ds)
    return created


def set_target(ds, target, problem_type, actor=""):
    df = load_dataframe(ds)
    datasets.validate_target(df, target, problem_type)
    ds.target_column, ds.problem_type = target, problem_type
    ds.save(update_fields=["target_column", "problem_type"])
    audit("target_selected", f"{ds.name}: {target} ({problem_type})", actor)
    return datasets.target_distribution(df, target, problem_type)


# ---------------- training ----------------
def train_models(ds, target, problem_type, algorithm_keys, actor="", random_state=42):
    """Train every selected algorithm on one shared split. Returns (trained, failures)."""
    df = load_dataframe(ds)
    prep = training.prepare_data(df, target, problem_type, random_state)
    trained, failures = [], {}
    for key in algorithm_keys:
        try:
            res = training.fit_and_evaluate(prep, key)
            version = (models.TrainedModel.objects.aggregate(m=Max("version"))["m"] or 0) + 1
            fname, sha, size = artifacts.save_pipeline(
                res["pipeline"], settings.ARTIFACT_DIR, f"model_v{version}_{uuid.uuid4().hex[:8]}.joblib")
            pname, pval, _ = evaluation.primary_metric(problem_type, res["metrics"])
            tm = models.TrainedModel.objects.create(
                dataset=ds, version=version, algorithm=key, algorithm_label=res["algorithm_label"],
                problem_type=problem_type, target_column=target, artifact_file=fname, artifact_sha256=sha,
                model_size_bytes=size, random_state=random_state, feature_schema=prep["schema"],
                classes=prep["classes"], metrics=res["metrics"], validation_metrics=res["validation_metrics"],
                details=to_native({**res["details"], "dropped_columns": prep["dropped_columns"]}),
                primary_metric=pname, primary_value=pval, train_seconds=res["train_seconds"],
                inference_latency_ms=res["latency_ms"], split_info=prep["split_info"],
                reference=res["reference"], created_by=actor)
            trained.append(tm)
            audit("model_trained", f"v{version} {tm.algorithm_label} on {ds.name}", actor)
        except Exception as exc:  # one failing algorithm must not discard the others
            log.exception("training failed for %s", key)
            failures[key] = str(exc)
    return trained, failures


def best_model(qs):
    """Best *evaluated* model among comparable models (same dataset/target) - independent of production."""
    items = [m for m in qs if m.primary_value is not None]
    if not items:
        return None
    higher = evaluation.PRIMARY[items[0].problem_type][1]
    return (max if higher else min)(items, key=lambda m: m.primary_value)


# ---------------- production versions ----------------
def get_production():
    entry = models.ProductionModel.objects.filter(is_current=True).select_related("model", "model__dataset").first()
    return entry.model if entry else None


@transaction.atomic
def promote(model, actor="", note="", action="promote"):
    now = timezone.now()
    models.ProductionModel.objects.filter(is_current=True).update(is_current=False, deactivated_at=now)
    models.ProductionModel.objects.create(model=model, is_current=True, action=action, note=note[:255],
                                          activated_by=actor, activated_at=now)
    audit(action, f"v{model.version} {model.algorithm_label} {note}", actor)


@transaction.atomic
def rollback(actor=""):
    current = get_production()
    if current is None:
        raise ValueError("There is no production model to roll back from.")
    prev = (models.ProductionModel.objects.filter(is_current=False).exclude(model=current)
            .select_related("model").order_by("-activated_at", "-id").first())
    if prev is None:
        raise ValueError("No previous production model exists to roll back to.")
    promote(prev.model, actor, f"rollback from v{current.version}", action="rollback")
    return prev.model


# ---------------- prediction ----------------
def load_model_pipeline(tm):
    return artifacts.load_pipeline(settings.ARTIFACT_DIR, tm.artifact_file, tm.artifact_sha256)


def _summary(row):
    return ", ".join(f"{k}={v}" for k, v in row.items() if v is not None)[:2000]


def run_prediction(tm, raw, actor=""):
    """Validate -> load verified pipeline -> predict -> log (success or failure). Returns (log, errors)."""
    clean, errors = prediction.validate_input(tm.feature_schema, raw)
    if errors:
        entry = models.PredictionLog.objects.create(
            model=tm, model_version=tm.version, input_data=to_native(raw), input_summary=_summary(raw),
            status="failed", error="Validation failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items())[:450],
            created_by=actor)
        return entry, errors
    try:
        pipe = load_model_pipeline(tm)
        out = prediction.predict_row(pipe, tm.feature_schema, clean, tm.problem_type, tm.classes)
    except Exception as exc:
        log.exception("prediction failed")
        entry = models.PredictionLog.objects.create(
            model=tm, model_version=tm.version, input_data=to_native(clean), input_summary=_summary(clean),
            status="failed", error=str(exc)[:490], created_by=actor)
        return entry, {}
    entry = models.PredictionLog.objects.create(
        model=tm, model_version=tm.version, input_data=to_native(clean), input_summary=_summary(clean),
        output=str(out["prediction"]), probabilities=out["probabilities"], latency_ms=out["latency_ms"],
        status="success", created_by=actor)
    return entry, {}


def add_actual_label(entry, label, actor=""):
    label = (label or "").strip()
    if not label:
        raise ValueError("Enter the actual value.")
    tm = entry.model
    if tm.problem_type == "regression":
        try:
            float(label)
        except ValueError:
            raise ValueError("Actual value must be a number for a regression model.")
    elif tm.classes and label not in tm.classes:
        raise ValueError("Label must be one of: " + ", ".join(tm.classes))
    if entry.status != "success":
        raise ValueError("Labels can only be added to successful predictions.")
    entry.actual_label, entry.actual_at = label, timezone.now()
    entry.save(update_fields=["actual_label", "actual_at"])
    audit("label_added", f"prediction #{entry.pk} -> {label}", actor)


# ---------------- monitoring & drift ----------------
def _log_dicts(qs):
    return [{"created_at": l.created_at, "status": l.status, "latency_ms": l.latency_ms, "output": l.output,
             "actual_label": l.actual_label, "model_version": l.model_version} for l in qs]


def compute_monitoring(tm, limit=5000):
    logs = list(tm.predictions.order_by("-created_at")[:limit])
    return monitoring.summarize_logs(_log_dicts(logs), tm.problem_type, tm.metrics), len(logs)


def save_snapshot(tm, actor=""):
    data, n = compute_monitoring(tm)
    snap = models.MonitoringSnapshot.objects.create(model=tm, window_size=n, metrics=to_native(data))
    audit("snapshot_saved", f"v{tm.version}: {n} predictions", actor)
    return snap


def run_drift(tm, actor="", limit=1000):
    logs = list(tm.predictions.filter(status="success").order_by("-created_at")[:limit])
    cur_rows = [l.input_data for l in logs]
    cur_pred = [l.output if tm.problem_type == "classification" else float(l.output) for l in logs]
    ref = tm.reference or {}
    rep = drift.detect_drift(ref.get("rows", []), cur_rows, tm.feature_schema, ref.get("predictions", []),
                             cur_pred, tm.problem_type)
    obj = models.DriftReport.objects.create(
        model=tm, reference_size=rep["reference_size"], current_size=rep["current_size"], status=rep["status"],
        results=to_native(rep), thresholds=rep["thresholds"])
    audit("drift_checked", f"v{tm.version}: {rep['status']}", actor)
    return obj
