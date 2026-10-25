from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("", views.overview, name="overview"),
    path("datasets/", views.datasets_view, name="datasets"),
    path("datasets/samples/", views.load_samples_view, name="load_samples"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("train/", views.train, name="train"),
    path("compare/", views.compare, name="compare"),
    path("predict/", views.predict, name="predict"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/label/", views.add_label, name="add_label"),
    path("monitoring/", views.monitoring_view, name="monitoring"),
    path("drift/", views.drift_view, name="drift"),
    path("versions/", views.versions, name="versions"),
    path("versions/promote/<int:pk>/", views.promote_view, name="promote"),
    path("versions/rollback/", views.rollback_view, name="rollback"),
    path("system/", views.system, name="system"),
]
