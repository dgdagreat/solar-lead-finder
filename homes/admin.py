from django.contrib import admin
from .models import Home


@admin.register(Home)
class HomeAdmin(admin.ModelAdmin):
    list_display = ['address', 'city', 'state', 'sale_price', 'sold_date', 'solar_status', 'solar_confidence', 'is_lead']
    list_filter = ['solar_status', 'is_lead', 'state']
    search_fields = ['address', 'city', 'zip_code']
    readonly_fields = ['created_at', 'updated_at']
    list_per_page = 25
