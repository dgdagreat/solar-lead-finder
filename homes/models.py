from django.db import models


class Home(models.Model):
    SOLAR_STATUS = [
        ('unknown', 'Unknown'),
        ('has_solar', 'Has Solar'),
        ('no_solar', 'No Solar'),
        ('error', 'Error'),
    ]

    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=50)
    zip_code = models.CharField(max_length=20)
    sale_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    sold_date = models.DateField(null=True, blank=True)

    # Image from Street View or satellite
    image = models.ImageField(upload_to='home_images/', null=True, blank=True)
    image_url = models.URLField(null=True, blank=True)

    # Solar detection result
    solar_status = models.CharField(max_length=20, choices=SOLAR_STATUS, default='unknown')
    solar_confidence = models.FloatField(null=True, blank=True)  # 0–100 percentage
    stages_triggered = models.IntegerField(null=True, blank=True)  # how many CV stages fired
    scan_detail = models.JSONField(null=True, blank=True)  # full stage breakdown

    # Lead tracking
    is_lead = models.BooleanField(default=False)
    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-sold_date']

    def __str__(self):
        return f"{self.address}, {self.city}, {self.state}"

    @property
    def full_address(self):
        return f"{self.address}, {self.city}, {self.state} {self.zip_code}"
