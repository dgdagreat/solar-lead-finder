"""
Assemble the CNN training folder from confirmed reference images.

Every "Confirm Has Solar" / "Confirm No Solar" click saves a raw, in-domain
satellite image into media/solar_references or media/nosolar_references. This
command mirrors those into the ImageFolder layout the trainer expects:

    data/train/solar/*.jpg
    data/train/no_solar/*.jpg
"""
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Retired: confirmed Google Maps scans are not training data."

    def handle(self, *args, **opts):
        raise CommandError(
            "Training from confirmed app scans is disabled because those scans use "
            "Google Maps imagery. Use an independently sourced, licensed dataset "
            "with geographic train/validation/test splits instead."
        )
