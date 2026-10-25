from django.core.management.base import BaseCommand

from automl.services import workflow


class Command(BaseCommand):
    help = "Register the bundled sample datasets (idempotent)."

    def handle(self, *args, **opts):
        created = workflow.load_samples("cli")
        self.stdout.write(f"Registered {len(created)} sample dataset(s).")
