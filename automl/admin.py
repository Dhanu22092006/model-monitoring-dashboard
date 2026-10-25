from django.contrib import admin

from . import models

for m in (models.Dataset, models.TrainedModel, models.ProductionModel, models.PredictionLog,
          models.MonitoringSnapshot, models.DriftReport, models.AuditLog):
    admin.site.register(m)
