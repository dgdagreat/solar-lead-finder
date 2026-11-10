"""
Seed a handful of demo homes and run solar detection on them.

Seeds the live demo during the build, then scans its sample homes when the
service starts. Keeping network calls out of the build avoids baking transient
satellite API failures into the generated SQLite database.
"""
from django.core.management.base import BaseCommand
from homes.models import Home
from homes.services import _mock_home_data
from detector.tasks import process_home_sync

DEMO_ADDRESSES = (
    "2215 Rimcrest Dr",
    "3421 Ramona Ave",
    "1244 El Monte Ave",
    "4821 Hazel Ave",
    "7312 Quill Dr",
)


class Command(BaseCommand):
    help = "Seed demo homes or scan unresolved demo homes."

    def add_arguments(self, parser):
        parser.add_argument(
            "--scan",
            action="store_true",
            help="Scan unresolved demo homes (use at service runtime).",
        )
        parser.add_argument(
            "--seed-only",
            action="store_true",
            help="Seed demo rows without fetching satellite imagery.",
        )

    def handle(self, *args, **opts):
        if not Home.objects.exists():
            _mock_home_data("", "")
            self.stdout.write("Seeded demo homes.")

        if opts["seed_only"] or not opts["scan"]:
            return

        homes = Home.objects.filter(address__in=DEMO_ADDRESSES)
        homes = [home for home in homes if (
            home.solar_status in ("unknown", "error")
            or not (home.scan_detail or {}).get("roof_verification", {}).get("vegetation_checked")
        )]
        scanned = 0
        for home in homes:
            try:
                process_home_sync(home.id)
            except Exception as e:   # never fail the whole deploy on one bad scan
                self.stderr.write(f"  scan failed for {home.address}: {e}")
            scanned += 1

        self.stdout.write(self.style.SUCCESS(
            f"Scanned {scanned} unresolved demo homes."))
