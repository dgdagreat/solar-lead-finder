"""
Assemble the CNN training folder from confirmed reference images.

Every "Confirm Has Solar" / "Confirm No Solar" click saves a raw, in-domain
satellite image into media/solar_references or media/nosolar_references. This
command mirrors those into the ImageFolder layout the trainer expects:

    data/train/solar/*.jpg
    data/train/no_solar/*.jpg
"""
import shutil
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Build data/train/{solar,no_solar}/ from confirmed solar / no-solar references."

    def handle(self, *args, **opts):
        media = Path(settings.MEDIA_ROOT)
        sources = {
            "solar":    media / "solar_references",
            "no_solar": media / "nosolar_references",
        }
        out = Path(settings.BASE_DIR) / "data" / "train"

        counts = {}
        for label, src in sources.items():
            dst = out / label
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            if src.exists():
                for img in src.glob("*.jpg"):
                    shutil.copy(img, dst / img.name)
                    n += 1
            counts[label] = n

        total = sum(counts.values())
        self.stdout.write(self.style.SUCCESS(
            f"Prepared {out} → {counts} ({total} images total)."))
        if total < 50:
            self.stdout.write(self.style.WARNING(
                "Fewer than 50 images — confirm more homes (or ingest an external "
                "dataset like BDAPPV) before expecting a robust model."))
