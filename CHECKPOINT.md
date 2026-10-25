# Checkpoint

## Milestones
1. Project + core dashboard (10 pages): written, not run.
2. Datasets / training / evaluation / comparison: services **run and tested**; views/templates not run.
3. Prediction history, versions, monitoring, drift: services **run and tested**; ORM glue/views not run.
4. Docker + persistence: written, not built.
5. Jenkins Freestyle guide: written, not executed.
6. Tests: 22 service tests PASS (executed). 18 Django tests written, NOT executed (no Django in sandbox).
7. Docs: README (architecture, viva, commands), docs/JENKINS_FREESTYLE.md.

## Known gaps / risks
- Hand-written migration unverified against Django's autodetector.
- Templates unrendered; only tag balance was checked.
- Dependency pins for Django/gunicorn/whitenoise/xmlrunner unverified.
- Training is synchronous (large data / SVC can be slow); no background worker.
- SQLite + 2 gunicorn workers is fine for demos, not for heavy concurrent writes.
