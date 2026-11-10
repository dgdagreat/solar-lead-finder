from django.db import migrations, models


DEMO_FIXTURES = (
    ("2215 Rimcrest Dr", "91207"),
    ("3421 Ramona Ave", "95826"),
    ("1244 El Monte Ave", "94040"),
    ("4821 Hazel Ave", "95628"),
    ("7312 Quill Dr", "90242"),
)


def mark_legacy_demo_homes(apps, schema_editor):
    Home = apps.get_model("homes", "Home")
    for address, zip_code in DEMO_FIXTURES:
        Home.objects.filter(address=address, zip_code=zip_code).update(is_demo=True)


class Migration(migrations.Migration):

    dependencies = [
        ("homes", "0004_manual_review_status_label"),
    ]

    operations = [
        migrations.AddField(
            model_name="home",
            name="is_demo",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(mark_legacy_demo_homes, migrations.RunPython.noop),
    ]
