from django.db import models
from django.utils import timezone


class Dataset(models.Model):
    name = models.CharField(max_length=120)
    original_filename = models.CharField(max_length=255, blank=True)
    file_path = models.CharField(max_length=255)  # file name inside DATASET_DIR (never user supplied)
    is_sample = models.BooleanField(default=False)
    target_column = models.CharField(max_length=200, blank=True)
    problem_type = models.CharField(max_length=20, blank=True)
    n_rows = models.PositiveIntegerField()
    n_columns = models.PositiveIntegerField()
    profile = models.JSONField(default=dict)
    uploaded_by = models.CharField(max_length=150, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"], name="ds_created_idx")]

    def __str__(self):
        return self.name


class TrainedModel(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="models")
    version = models.PositiveIntegerField(unique=True)
    algorithm = models.CharField(max_length=50)
    algorithm_label = models.CharField(max_length=100)
    problem_type = models.CharField(max_length=20)
    target_column = models.CharField(max_length=200)
    artifact_file = models.CharField(max_length=255)
    artifact_sha256 = models.CharField(max_length=64)
    model_size_bytes = models.PositiveIntegerField(default=0)
    random_state = models.IntegerField(default=42)
    feature_schema = models.JSONField(default=list)
    classes = models.JSONField(default=list)
    metrics = models.JSONField(default=dict)
    validation_metrics = models.JSONField(default=dict)
    details = models.JSONField(default=dict)
    primary_metric = models.CharField(max_length=30, blank=True)
    primary_value = models.FloatField(null=True, blank=True)
    train_seconds = models.FloatField(default=0)
    inference_latency_ms = models.FloatField(default=0)
    split_info = models.JSONField(default=dict)
    reference = models.JSONField(default=dict)
    created_by = models.CharField(max_length=150, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version"]
        indexes = [
            models.Index(fields=["dataset", "target_column"], name="tm_dataset_target_idx"),
            models.Index(fields=["-created_at"], name="tm_created_idx"),
        ]

    def __str__(self):
        return f"v{self.version} {self.algorithm_label}"


class ProductionModel(models.Model):
    """History of which model was serving. The row with is_current=True is production."""
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name="production_entries")
    is_current = models.BooleanField(default=False)
    action = models.CharField(max_length=20, default="promote")
    note = models.CharField(max_length=255, blank=True)
    activated_by = models.CharField(max_length=150, blank=True)
    activated_at = models.DateTimeField(default=timezone.now)
    deactivated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-activated_at", "-id"]
        indexes = [models.Index(fields=["is_current"], name="pm_current_idx")]


class PredictionLog(models.Model):
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name="predictions")
    model_version = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    input_data = models.JSONField(default=dict)
    input_summary = models.TextField(blank=True)
    output = models.CharField(max_length=255, blank=True)
    probabilities = models.JSONField(null=True, blank=True)
    latency_ms = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=10)
    error = models.CharField(max_length=500, blank=True)
    actual_label = models.CharField(max_length=255, null=True, blank=True)
    actual_at = models.DateTimeField(null=True, blank=True)
    created_by = models.CharField(max_length=150, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["model", "-created_at"], name="pl_model_created_idx"),
            models.Index(fields=["status"], name="pl_status_idx"),
        ]


class MonitoringSnapshot(models.Model):
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name="snapshots")
    created_at = models.DateTimeField(auto_now_add=True)
    window_size = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["model", "-created_at"], name="ms_model_created_idx")]


class DriftReport(models.Model):
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name="drift_reports")
    created_at = models.DateTimeField(auto_now_add=True)
    reference_size = models.PositiveIntegerField(default=0)
    current_size = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20)
    results = models.JSONField(default=dict)
    thresholds = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["model", "-created_at"], name="dr_model_created_idx")]


class AuditLog(models.Model):
    action = models.CharField(max_length=50)
    detail = models.CharField(max_length=500, blank=True)
    actor = models.CharField(max_length=150, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["-created_at"], name="al_created_idx")]
