from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Home
from .services import fetch_recently_sold_homes
from detector.tasks import process_home_sync
from detector.reference_comparator import add_reference


SUGGESTED_CITIES = [
    {"city": "San Francisco", "state": "CA"},
    {"city": "Los Angeles",   "state": "CA"},
    {"city": "San Diego",     "state": "CA"},
    {"city": "Sacramento",    "state": "CA"},
    {"city": "San Jose",      "state": "CA"},
    {"city": "Phoenix",       "state": "AZ"},
    {"city": "Austin",        "state": "TX"},
    {"city": "Denver",        "state": "CO"},
]


def dashboard(request):
    """
    Main leads dashboard. By default it shows ONLY the homes from the user's
    most recent search (tracked in the session); `?scope=all` shows everything.
    """
    show_all     = request.GET.get('scope') == 'all'
    solar_status = request.GET.get('status', '')

    current_ids  = request.session.get('current_home_ids')
    search_label = request.session.get('current_search_label', '')

    homes  = Home.objects.all()
    scoped = bool(current_ids) and not show_all
    if scoped:
        homes = homes.filter(id__in=current_ids)

    # Stats reflect the scoped set (before the status filter is applied)
    base      = homes
    leads     = base.filter(is_lead=True).count()
    no_solar  = base.filter(solar_status='no_solar').count()
    has_solar = base.filter(solar_status='has_solar').count()
    pending   = base.filter(solar_status='unknown').count()
    total     = base.count()

    if solar_status:
        homes = homes.filter(solar_status=solar_status)

    context = {
        'homes': homes,
        'leads': leads,
        'no_solar': no_solar,
        'has_solar': has_solar,
        'pending': pending,
        'total': total,
        'active_filter': solar_status,
        'scoped': scoped,
        'show_all': show_all,
        'search_label': search_label,
        'all_count': Home.objects.count(),
        'suggested_cities': SUGGESTED_CITIES,
    }
    return render(request, 'homes/dashboard.html', context)


def search_homes(request):
    """Search for recently sold homes in a city/state (optionally within a radius)."""
    if request.method == 'POST':
        city     = request.POST.get('city', '').strip()
        state    = request.POST.get('state', '').strip()
        zip_code = request.POST.get('zip_code', '').strip()
        radius   = request.POST.get('radius', '').strip()

        if not city or not state:
            messages.error(request, 'Please enter both city and state.')
            return redirect('dashboard')

        try:
            radius_miles = float(radius) if radius else None
        except ValueError:
            radius_miles = None

        homes = fetch_recently_sold_homes(city, state, zip_code or None,
                                          radius_miles=radius_miles)

        if homes:
            # Always reset and reprocess so stale results are cleared
            for home in homes:
                home.solar_status = 'unknown'
                home.solar_confidence = None
                home.is_lead = False
                home.save()
                process_home_sync(home.id)

            # Scope the dashboard to just this search
            request.session['current_home_ids'] = [h.id for h in homes]
            label = f"{city.title()}, {state.upper()}"
            if radius_miles:
                label += f" · within {radius_miles:g} mi"
            if zip_code:
                label += f" · {zip_code}"
            request.session['current_search_label'] = label

            messages.success(request, f'Found {len(homes)} homes and ran solar detection!')
        else:
            messages.warning(request, 'No homes found. Try a different city or a larger radius.')

        return redirect('dashboard')

    return redirect('dashboard')


def home_detail(request, home_id):
    """Detail view for a single home."""
    home = get_object_or_404(Home, id=home_id)
    return render(request, 'homes/home_detail.html', {'home': home})


def reprocess_home(request, home_id):
    """Re-run solar detection on a specific home."""
    home = get_object_or_404(Home, id=home_id)
    home.solar_status = 'unknown'
    home.save()
    process_home_sync(home.id)
    messages.info(request, f'Re-processed {home.address}.')
    return redirect('dashboard')


def process_all(request):
    """Run solar detection on all pending homes."""
    pending = Home.objects.filter(solar_status='unknown')
    for home in pending:
        process_home_sync(home.id)
    messages.info(request, f'Processed {pending.count()} homes.')
    return redirect('dashboard')


def confirm_solar(request, home_id):
    """
    User manually confirms this home HAS solar panels.
    Adds it to the reference library to improve future detections.
    """
    home = get_object_or_404(Home, id=home_id)
    home.solar_status  = 'has_solar'
    home.is_lead       = False
    home.save()

    added = add_reference(home.full_address)
    if added:
        messages.success(
            request,
            f'✅ {home.address} confirmed as solar — added to reference library! '
            f'Future scans will compare against this home.'
        )
    else:
        messages.info(request, f'{home.address} marked as solar.')
    return redirect('home_detail', home_id=home_id)


def confirm_no_solar(request, home_id):
    """User manually confirms this home has NO solar panels — it's a lead."""
    home = get_object_or_404(Home, id=home_id)
    home.solar_status = 'no_solar'
    home.is_lead      = True
    home.save()

    # Grow the NEGATIVE reference set so future scans discriminate better
    add_reference(home.full_address, label="no_solar")

    messages.success(request, f'🔥 {home.address} confirmed as NO solar — marked as hot lead!')
    return redirect('home_detail', home_id=home_id)
