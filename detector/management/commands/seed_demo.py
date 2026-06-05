"""
Seed a handful of demo homes and run solar detection on them.

Used by the Render build so the live demo's dashboard is populated on first
visit (instead of an empty state). Uses the built-in mock addresses (real
rooftops), so it needs no property API — just the Google Static Maps key to
fetch satellite imagery. Idempotent: skips if homes already exist.
"""
from django.core.management.base import BaseCommand
from homes.models import Home
from homes.services import _mock_home_data
from detector.tasks import process_home_sync


class Command(BaseCommand):
    help = "Seed demo homes and run solar detection (for the live demo)."

    def handle(self, *args, **opts):
        if Home.objects.exists():
            self.stdout.write("Homes already present — skipping demo seed.")
            return

        homes = _mock_home_data("", "")
        for home in homes:
            try:
                process_home_sync(home.id)
            except Exception as e:   # never fail the whole deploy on one bad scan
                self.stderr.write(f"  scan failed for {home.address}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Seeded + scanned {len(homes)} demo homes."))
