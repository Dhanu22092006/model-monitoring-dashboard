import logging
import platform
from urllib.parse import quote
import shutil
import sys

import django
import sklearn
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db import connection
from django.db.models import Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import forms, models
from .services import datasets as ds_service
from .services import evaluation, workflow

log = logging.getLogger(__name__)


def actor(request):
    return request.user.get_username()


def staff_required(view):
    def wrapper(request, *a, **kw):
        if not request.user.is_staff:
            raise PermissionDenied("Staff permission is required for this action.")
        return view(request, *a, **kw)
    wrapper.__name__ = view.__name__
    return wrapper


def bar(labels, values, label="Count"):
    return {"type": "bar", "labels": labels, "datasets": [{"label": label, "data": values}]}


# ---------------- health (unauthenticated, used by Docker and Jenkins) ----------------
def health(request):
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        return JsonResponse({"status": "ok", "database": "ok"})
    except Exception:
        log.exception("health check failed")
        return JsonResponse({"status": "error", "database": "unavailable"}, status=503)


# ---------------- 1. overview ----------------
@login_required
def overview(request):
    prod = workflow.get_production()
    logs = models.PredictionLog.objects
    ctx = {
        "n_datasets": models.Dataset.objects.count(), "n_models": models.TrainedModel.objects.count(),
        "n_predictions": logs.count(), "n_failed": logs.filter(status="failed").count(),
        "production": prod,
        "latest_drift": models.DriftReport.objects.select_related("model").first(),
        "recent_models": models.TrainedModel.objects.select_related("dataset")[:5],
        "recent_predictions": logs.select_related("model")[:5],
    }
    return render(request, "overview.html", ctx)


# ---------------- 2. datasets ----------------
@login_required
def datasets_view(request):
    form = forms.DatasetUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        try:
            ds = workflow.register_dataset(form.cleaned_data["name"], form.cleaned_data["file"].read(),
                                           form.cleaned_data["file"].name, actor=actor(request))
            messages.success(request, f"Dataset '{ds.name}' uploaded. Now choose its target column.")
            return redirect("dataset_detail", pk=ds.pk)
        except ds_service.DatasetError as exc:
            form.add_error("file", str(exc))
        except Exception:
            log.exception("upload failed")
            form.add_error("file", "Could not process this file.")
    return render(request, "datasets.html", {"form": form, "datasets": models.Dataset.objects.all(),
                                             "max_mb": settings.MAX_UPLOAD_BYTES // (1024 * 1024)})


@login_required
@require_POST
def load_samples_view(request):
    created = workflow.load_samples(actor(request))
    messages.success(request, f"Registered {len(created)} sample dataset(s)." if created
                     else "Sample datasets are already registered.")
    return redirect("datasets")


@login_required
def dataset_detail(request, pk):
    ds = get_object_or_404(models.Dataset, pk=pk)
    try:
        df = workflow.load_dataframe(ds)
    except ds_service.DatasetError as exc:
        messages.error(request, str(exc))
        return redirect("datasets")
    form = forms.TargetForm(request.POST or None, initial={
        "target_column": ds.target_column, "problem_type": ds.problem_type or "classification"})
    if request.method == "POST" and form.is_valid():
        try:
            workflow.set_target(ds, form.cleaned_data["target_column"], form.cleaned_data["problem_type"], actor(request))
            messages.success(request, "Target column saved.")
            return redirect("dataset_detail", pk=ds.pk)
        except ds_service.DatasetError as exc:
            form.add_error(None, str(exc))
    dist = None
    if ds.target_column and ds.target_column in df.columns and ds.problem_type:
        d = ds_service.target_distribution(df, ds.target_column, ds.problem_type)
        dist = bar(d["labels"], d["values"], "Rows")
    preview = df.head(15).astype(object).where(df.head(15).notna(), "").astype(str)
    return render(request, "dataset_detail.html", {
        "ds": ds, "profile": ds.profile, "form": form, "dist": dist,
        "columns": list(df.columns), "rows": preview.values.tolist(),
        "describe": ds.profile.get("describe", {}),
        "suggested": ds_service.infer_problem_type(df[df.columns[-1]]),
    })


# ---------------- 3. training ----------------
@login_required
def train(request):
    form = forms.TrainForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        d = form.cleaned_data
        keys = [a.split(":", 1)[1] for a in d["algorithms"]]
        try:
            trained, failures = workflow.train_models(d["dataset"], d["target_column"], d["problem_type"], keys,
                                                      actor(request), d["random_state"])
            for k, msg in failures.items():
                messages.warning(request, f"{k} failed: {msg}")
            if trained:
                messages.success(request, f"Trained {len(trained)} model(s) on one shared train/validation/test split.")
                return redirect(f"/compare/?dataset={d['dataset'].pk}&target={quote(d['target_column'])}")
        except ds_service.DatasetError as exc:
            form.add_error(None, str(exc))
        except Exception:
            log.exception("training failed")
            form.add_error(None, "Training failed unexpectedly. Check the application logs.")
    from .services import training
    all_ds = models.Dataset.objects.all()
    meta = {str(d.pk): {"columns": [f["name"] for f in d.profile.get("features", [])],
                        "target": d.target_column, "problem_type": d.problem_type} for d in all_ds}
    return render(request, "train.html", {
        "form": form, "meta": meta,
        "algos": [{"value": v, "label": l, "type": t} for v, l, t in training.algorithm_choices()]})


# ---------------- 4. comparison ----------------
@login_required
def compare(request):
    groups = (models.TrainedModel.objects.values("dataset_id", "dataset__name", "target_column", "problem_type")
              .annotate(n=Count("id")).order_by("-n", "dataset__name"))
    groups = list(groups)
    ds_id, target = request.GET.get("dataset"), request.GET.get("target")
    sel = next((g for g in groups if str(g["dataset_id"]) == str(ds_id) and g["target_column"] == target), None)
    if sel is None and groups:
        last = models.TrainedModel.objects.first()
        sel = next((g for g in groups if g["dataset_id"] == last.dataset_id and g["target_column"] == last.target_column), groups[0])
    ctx = {"groups": groups, "sel": sel}
    if sel:
        items = list(models.TrainedModel.objects.filter(dataset_id=sel["dataset_id"], target_column=sel["target_column"])
                     .select_related("dataset").order_by("version"))
        best = workflow.best_model(items)
        prod = workflow.get_production()
        names = [f"v{m.version} {m.algorithm_label}" for m in items]
        if sel["problem_type"] == "classification":
            keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
            charts = [{"id": "c1", "title": "Classification metrics (held-out test set)",
                       "cfg": {"type": "bar", "labels": names, "datasets": [
                           {"label": evaluation.METRIC_LABELS[k], "data": [m.metrics.get(k) for m in items]} for k in keys]}}]
            cols = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
        else:
            charts = [
                {"id": "c1", "title": "Error metrics - lower is better (test set)",
                 "cfg": {"type": "bar", "labels": names, "datasets": [
                     {"label": evaluation.METRIC_LABELS[k], "data": [m.metrics.get(k) for m in items]} for k in ("mae", "rmse")]}},
                {"id": "c2", "title": "R² - higher is better (test set)",
                 "cfg": {"type": "bar", "labels": names, "datasets": [{"label": "R²", "data": [m.metrics.get("r2") for m in items]}]}}]
            cols = ["mae", "mse", "rmse", "r2"]
        charts.append({"id": "c3", "title": "Training time (s) and inference latency (ms)",
                       "cfg": {"type": "bar", "labels": names, "datasets": [
                           {"label": "Training seconds", "data": [round(m.train_seconds, 4) for m in items]},
                           {"label": "Inference latency ms (1 row)", "data": [round(m.inference_latency_ms, 4) for m in items]}]}})
        ctx.update(items=items, best=best, production=prod, charts=charts,
                   cols=[(c, evaluation.METRIC_LABELS[c]) for c in cols],
                   primary=evaluation.PRIMARY[sel["problem_type"]])
    return render(request, "compare.html", ctx)


# ---------------- 5. prediction playground ----------------
@login_required
def predict(request):
    all_models = list(models.TrainedModel.objects.select_related("dataset").order_by("-version"))
    prod = workflow.get_production()
    chosen_id = request.POST.get("model") or request.GET.get("model")
    tm = next((m for m in all_models if str(m.pk) == str(chosen_id)), None) or prod or (all_models[0] if all_models else None)
    result, errors, values = None, {}, {}
    if tm and request.method == "POST":
        raw = {f["name"]: request.POST.get(f"f{i}", "") for i, f in enumerate(tm.feature_schema)}
        values = {f"f{i}": raw[f["name"]] for i, f in enumerate(tm.feature_schema)}
        entry, errors = workflow.run_prediction(tm, raw, actor(request))
        result = entry
        if errors:
            messages.error(request, "Please fix the highlighted inputs.")
    fields = []
    if tm:
        for i, f in enumerate(tm.feature_schema):
            fields.append({**f, "key": f"f{i}", "value": values.get(f"f{i}", ""), "error": errors.get(f["name"])})
    return render(request, "predict.html", {"models": all_models, "tm": tm, "production": prod,
                                            "fields": fields, "result": result})


# ---------------- 6. history ----------------
@login_required
def history(request):
    qs = models.PredictionLog.objects.select_related("model")
    q = (request.GET.get("q") or "").strip()
    status = request.GET.get("status") or ""
    if q:
        cond = Q(output__icontains=q) | Q(input_summary__icontains=q) | Q(error__icontains=q)
        if q.lstrip("vV").isdigit():
            cond |= Q(model_version=int(q.lstrip("vV")))
        qs = qs.filter(cond)
    if status in ("success", "failed"):
        qs = qs.filter(status=status)
    page = Paginator(qs, 20).get_page(request.GET.get("page"))
    return render(request, "history.html", {"page": page, "q": q, "status": status, "label_form": forms.LabelForm()})


@login_required
@require_POST
def add_label(request, pk):
    entry = get_object_or_404(models.PredictionLog, pk=pk)
    form = forms.LabelForm(request.POST)
    if form.is_valid():
        try:
            workflow.add_actual_label(entry, form.cleaned_data["actual_label"], actor(request))
            messages.success(request, f"Actual value saved for prediction #{entry.pk}.")
        except ValueError as exc:
            messages.error(request, str(exc))
    else:
        messages.error(request, "Enter the actual value.")
    nxt = request.POST.get("next", "")
    return redirect(nxt if nxt.startswith("/history/") else "history")


# ---------------- 7. monitoring ----------------
def _pick_model(request):
    mid = request.POST.get("model") or request.GET.get("model")
    all_models = list(models.TrainedModel.objects.order_by("-version"))
    tm = next((m for m in all_models if str(m.pk) == str(mid)), None) or workflow.get_production() or (all_models[0] if all_models else None)
    return tm, all_models


@login_required
def monitoring_view(request):
    tm, all_models = _pick_model(request)
    if tm and request.method == "POST":
        workflow.save_snapshot(tm, actor(request))
        messages.success(request, "Monitoring snapshot saved.")
        return redirect(f"/monitoring/?model={tm.pk}")
    ctx = {"tm": tm, "models": all_models}
    if tm:
        data, n = workflow.compute_monitoring(tm)
        ctx.update(data=data, window=n, snapshots=tm.snapshots.all()[:10],
                   production=workflow.get_production(),
                   volume=bar(data["daily_volume"]["labels"], data["daily_volume"]["values"], "Predictions / day"),
                   dist=bar(data["prediction_distribution"]["labels"], data["prediction_distribution"]["values"], "Predictions"),
                   lat=bar(["Average", "Median", "P95"], [data["latency"]["avg_ms"], data["latency"]["median_ms"], data["latency"]["p95_ms"]], "Latency (ms)"),
                   usage_all=list(models.PredictionLog.objects.values("model_version").annotate(n=Count("id")).order_by("model_version")))
        ctx["usage"] = bar([f"v{u['model_version']}" for u in ctx["usage_all"]], [u["n"] for u in ctx["usage_all"]], "Predictions")
    return render(request, "monitoring.html", ctx)


# ---------------- 8. drift ----------------
@login_required
def drift_view(request):
    tm, all_models = _pick_model(request)
    if tm and request.method == "POST":
        workflow.run_drift(tm, actor(request))
        return redirect(f"/drift/?model={tm.pk}")
    ctx = {"tm": tm, "models": all_models}
    if tm:
        reports = list(tm.drift_reports.all()[:10])
        ctx.update(report=reports[0] if reports else None, reports=reports,
                   n_current=tm.predictions.filter(status="success").count(),
                   n_ref=len((tm.reference or {}).get("rows", [])))
        if reports:
            feats = [f for f in reports[0].results.get("features", []) if f.get("psi") is not None]
            ctx["psi_chart"] = bar([f["name"] for f in feats], [round(f["psi"], 4) for f in feats], "PSI")
    return render(request, "drift.html", ctx)


# ---------------- 9. versions ----------------
@login_required
def versions(request):
    items = list(models.TrainedModel.objects.select_related("dataset"))
    best_ids = set()
    seen = {}
    for m in items:
        seen.setdefault((m.dataset_id, m.target_column), []).append(m)
    for grp in seen.values():
        b = workflow.best_model(grp)
        if b:
            best_ids.add(b.pk)
    return render(request, "versions.html", {
        "items": items, "best_ids": best_ids, "production": workflow.get_production(),
        "history": models.ProductionModel.objects.select_related("model")[:15]})


@login_required
@staff_required
@require_POST
def promote_view(request, pk):
    tm = get_object_or_404(models.TrainedModel, pk=pk)
    workflow.promote(tm, actor(request), request.POST.get("note", ""))
    messages.success(request, f"v{tm.version} is now the production model.")
    return redirect("versions")


@login_required
@staff_required
@require_POST
def rollback_view(request):
    try:
        tm = workflow.rollback(actor(request))
        messages.success(request, f"Rolled back: v{tm.version} is production again.")
    except ValueError as exc:
        messages.error(request, str(exc))
    return redirect("versions")


# ---------------- 10. system ----------------
@login_required
def system(request):
    usage = shutil.disk_usage(settings.ARTIFACT_DIR if settings.ARTIFACT_DIR.exists() else settings.BASE_DIR)
    info = [("Python", sys.version.split()[0]), ("Platform", platform.platform()), ("Django", django.get_version()),
            ("scikit-learn", sklearn.__version__), ("Debug mode", settings.DEBUG),
            ("Database file", settings.SQLITE_PATH), ("Artifact directory", str(settings.ARTIFACT_DIR)),
            ("Dataset directory", str(settings.DATASET_DIR)),
            ("Max upload", f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB"),
            ("Disk free", f"{usage.free / 1e9:.1f} GB of {usage.total / 1e9:.1f} GB"),
            ("Secure cookies", settings.SECURE_COOKIES), ("Health endpoint", "/health/")]
    return render(request, "system.html", {"info": info, "audit": models.AuditLog.objects.all()[:25],
                                           "counts": {"Datasets": models.Dataset.objects.count(),
                                                      "Models": models.TrainedModel.objects.count(),
                                                      "Predictions": models.PredictionLog.objects.count(),
                                                      "Drift reports": models.DriftReport.objects.count()}})
