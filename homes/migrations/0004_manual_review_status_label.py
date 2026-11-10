from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("homes", "0003_alter_home_solar_status"),
    ]

    operations = [
        migrations.AlterField(
            model_name="home",
            name="solar_status",
            field=models.CharField(
                choices=[
                    ("unknown", "Unknown"),
                    ("unverified", "Needs manual review"),
                    ("has_solar", "Has Solar"),
                    ("no_solar", "No Solar"),
                    ("error", "Error"),
                ],
                default="unknown",
                max_length=20,
            ),
        ),
    ]
