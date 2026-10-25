FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SQLITE_PATH=/app/data/db.sqlite3 \
    ARTIFACT_DIR=/app/model_artifacts \
    DATASET_DIR=/app/datasets

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data /app/model_artifacts /app/datasets /app/staticfiles \
    && DJANGO_SECRET_KEY=build-time-only python manage.py collectstatic --noinput \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/', timeout=4)"

# migrate is idempotent and never drops data; the DB, datasets and artifacts live on mounted volumes.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py ensure_admin && exec gunicorn core_project.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120 --access-logfile -"]
