# Hand-written to match automl/models.py (generated without a Django runtime).
# Run `python manage.py makemigrations --check --dry-run` to confirm it is in sync.
import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


def _id():
    return ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID"))


class Migration(migrations.Migration):
    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                _id(),
                ("name", models.CharField(max_length=120)),
                ("original_filename", models.CharField(blank=True, max_length=255)),
                ("file_path", models.CharField(max_length=255)),
                ("is_sample", models.BooleanField(default=False)),
                ("target_column", models.CharField(blank=True, max_length=200)),
                ("problem_type", models.CharField(blank=True, max_length=20)),
                ("n_rows", models.PositiveIntegerField()),
                ("n_columns", models.PositiveIntegerField()),
                ("profile", models.JSONField(default=dict)),
                ("uploaded_by", models.CharField(blank=True, max_length=150)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"],
                     "indexes": [models.Index(fields=["-created_at"], name="ds_created_idx")]},
        ),
        migrations.CreateModel(
            name="TrainedModel",
            fields=[
                _id(),
                ("version", models.PositiveIntegerField(unique=True)),
                ("algorithm", models.CharField(max_length=50)),
                ("algorithm_label", models.CharField(max_length=100)),
                ("problem_type", models.CharField(max_length=20)),
                ("target_column", models.CharField(max_length=200)),
                ("artifact_file", models.CharField(max_length=255)),
                ("artifact_sha256", models.CharField(max_length=64)),
                ("model_size_bytes", models.PositiveIntegerField(default=0)),
                ("random_state", models.IntegerField(default=42)),
                ("feature_schema", models.JSONField(default=list)),
                ("classes", models.JSONField(default=list)),
                ("metrics", models.JSONField(default=dict)),
                ("validation_metrics", models.JSONField(default=dict)),
                ("details", models.JSONField(default=dict)),
                ("primary_metric", models.CharField(blank=True, max_length=30)),
                ("primary_value", models.FloatField(blank=True, null=True)),
                ("train_seconds", models.FloatField(default=0)),
                ("inference_latency_ms", models.FloatField(default=0)),
                ("split_info", models.JSONField(default=dict)),
                ("reference", models.JSONField(default=dict)),
                ("created_by", models.CharField(blank=True, max_length=150)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("dataset", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                              related_name="models", to="automl.dataset")),
            ],
            options={"ordering": ["-version"],
                     "indexes": [models.Index(fields=["dataset", "target_column"], name="tm_dataset_target_idx"),
                                 models.Index(fields=["-created_at"], name="tm_created_idx")]},
        ),
        migrations.CreateModel(
            name="ProductionModel",
            fields=[
                _id(),
                ("is_current", models.BooleanField(default=False)),
                ("action", models.CharField(default="promote", max_length=20)),
                ("note", models.CharField(blank=True, max_length=255)),
                ("activated_by", models.CharField(blank=True, max_length=150)),
                ("activated_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("deactivated_at", models.DateTimeField(blank=True, null=True)),
                ("model", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                            related_name="production_entries", to="automl.trainedmodel")),
            ],
            options={"ordering": ["-activated_at", "-id"],
                     "indexes": [models.Index(fields=["is_current"], name="pm_current_idx")]},
        ),
        migrations.CreateModel(
            name="PredictionLog",
            fields=[
                _id(),
                ("model_version", models.PositiveIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("input_data", models.JSONField(default=dict)),
                ("input_summary", models.TextField(blank=True)),
                ("output", models.CharField(blank=True, max_length=255)),
                ("probabilities", models.JSONField(blank=True, null=True)),
                ("latency_ms", models.FloatField(blank=True, null=True)),
                ("status", models.CharField(max_length=10)),
                ("error", models.CharField(blank=True, max_length=500)),
                ("actual_label", models.CharField(blank=True, max_length=255, null=True)),
                ("actual_at", models.DateTimeField(blank=True, null=True)),
                ("created_by", models.CharField(blank=True, max_length=150)),
                ("model", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                            related_name="predictions", to="automl.trainedmodel")),
            ],
            options={"ordering": ["-created_at", "-id"],
                     "indexes": [models.Index(fields=["model", "-created_at"], name="pl_model_created_idx"),
                                 models.Index(fields=["status"], name="pl_status_idx")]},
        ),
        migrations.CreateModel(
            name="MonitoringSnapshot",
            fields=[
                _id(),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("window_size", models.PositiveIntegerField(default=0)),
                ("metrics", models.JSONField(default=dict)),
                ("model", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                            related_name="snapshots", to="automl.trainedmodel")),
            ],
            options={"ordering": ["-created_at", "-id"],
                     "indexes": [models.Index(fields=["model", "-created_at"], name="ms_model_created_idx")]},
        ),
        migrations.CreateModel(
            name="DriftReport",
            fields=[
                _id(),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("reference_size", models.PositiveIntegerField(default=0)),
                ("current_size", models.PositiveIntegerField(default=0)),
                ("status", models.CharField(max_length=20)),
                ("results", models.JSONField(default=dict)),
                ("thresholds", models.JSONField(default=dict)),
                ("model", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                            related_name="drift_reports", to="automl.trainedmodel")),
            ],
            options={"ordering": ["-created_at", "-id"],
                     "indexes": [models.Index(fields=["model", "-created_at"], name="dr_model_created_idx")]},
        ),
        migrations.CreateModel(
            name="AuditLog",
            fields=[
                _id(),
                ("action", models.CharField(max_length=50)),
                ("detail", models.CharField(blank=True, max_length=500)),
                ("actor", models.CharField(blank=True, max_length=150)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at", "-id"],
                     "indexes": [models.Index(fields=["-created_at"], name="al_created_idx")]},
        ),
    ]
