from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.conf import settings
import logging
from .models import Home
from .services import fetch_recently_sold_homes, PropertySearchUnavailable
from detector.tasks import process_home_sync
from detector.reference_comparator import add_reference

logger = logging.getLogger(__name__)


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
    if not settings.DEBUG:
        # Keep legacy/local demo rows out of production search results and stats.
        homes = homes.filter(is_demo=False)
    scoped = bool(current_ids) and not show_all
    if scoped:
        homes = homes.filter(id__in=current_ids)

    # Stats reflect the scoped set (before the status filter is applied)
    base      = homes
    leads     = base.filter(is_lead=True).count()
    no_solar  = base.filter(solar_status='no_solar').count()
    has_solar = base.filter(solar_status='has_solar').count()
    pending   = base.filter(solar_status='unknown').count()
    unverified = base.filter(solar_status='unverified').count()
    total     = base.count()

    if solar_status:
        homes = homes.filter(solar_status=solar_status)

    if settings.RAPIDAPI_KEY or settings.ATTOM_API_KEY:
        search_mode = 'live'
    elif settings.DEBUG:
        search_mode = 'demo'
    else:
        search_mode = 'unavailable'

    context = {
        'homes': homes,
        'leads': leads,
        'no_solar': no_solar,
        'has_solar': has_solar,
        'pending': pending,
        'unverified': unverified,
        'total': total,
        'active_filter': solar_status,
        'scoped': scoped,
        'show_all': show_all,
        'search_label': search_label,
        'all_count': (Home.objects.count() if settings.DEBUG
                      else Home.objects.filter(is_demo=False).count()),
        'suggested_cities': SUGGESTED_CITIES,
        'search_mode': search_mode,
    }
    return render(request, 'homes/dashboard.html', context)


def search_homes(request):
    """Search for recently sold homes in a city/state (optionally within a radius)."""
    if request.method == 'POST':
        city     = request.POST.get('city', '').strip()
        state    = request.POST.get('state', '').strip()
        zip_code = request.POST.get('zip_code', '').strip()
        radius   = request.POST.get('radius', '').strip()
        if radius == 'custom':                       # dropdown → free-form box
            radius = request.POST.get('radius_custom', '').strip()

        if not city or not state:
            messages.error(request, 'Please enter both city and state.')
            return redirect('dashboard')

        try:
            radius_miles = float(radius) if radius else None
        except ValueError:
            messages.error(request, 'Enter a valid radius in miles.', extra_tags='danger')
            return redirect('dashboard')

        if radius_miles is not None and not 1 <= radius_miles <= 500:
            messages.error(request, 'Radius must be between 1 and 500 miles.', extra_tags='danger')
            return redirect('dashboard')

        try:
            homes = fetch_recently_sold_homes(
                city, state, zip_code or None, radius_miles=radius_miles
            )
        except PropertySearchUnavailable as exc:
            messages.error(request, str(exc), extra_tags='danger')
            return redirect('dashboard')
        except Exception as exc:
            # Provider and response parsing failures must return a useful page,
            # not Django's generic 500. Log the exception type without request
            # headers or provider credentials.
            logger.error(
                "Unexpected property search failure (%s)", type(exc).__name__
            )
            messages.error(
                request,
                'The property search could not be completed. Please try again.',
                extra_tags='danger',
            )
            return redirect('dashboard')

        if homes:
            # Always reset and reprocess so stale results are cleared
            for home in homes:
                home.solar_status = 'unknown'
                home.solar_confidence = None
                home.is_lead = False
                home.save()
                try:
                    process_home_sync(home.id)
                except Exception as exc:
                    # A single satellite/model failure should not discard the
                    # whole search result set or leave the POST as a 500.
                    logger.error(
                        "Unexpected solar scan failure for home %s (%s)",
                        home.id,
                        type(exc).__name__,
                    )
                    home.solar_status = 'error'
                    home.solar_confidence = None
                    home.is_lead = False
                    home.scan_detail = {
                        "method": "error",
                        "error": "Satellite scan failed unexpectedly. Retry this home.",
                    }
                    home.save()

            # Scope the dashboard to just this search
            request.session['current_home_ids'] = [h.id for h in homes]
            if not settings.RAPIDAPI_KEY and not settings.ATTOM_API_KEY:
                label = "Demo homes (not filtered by your search)"
            else:
                label = f"{city.title()}, {state.upper()}"
                if radius_miles:
                    label += f" · within {radius_miles:g} mi"
                if zip_code:
                    label += f" · {zip_code}"
            request.session['current_search_label'] = label

            if not settings.RAPIDAPI_KEY and not settings.ATTOM_API_KEY:
                messages.info(
                    request,
                    f'Loaded {len(homes)} demo homes. They are fixed sample addresses, '
                    'not results for the searched city; radius was not applied.'
                )
            else:
                failed_scans = sum(home.solar_status == 'error' for home in homes)
                if failed_scans:
                    messages.warning(
                        request,
                        f'Found {len(homes)} homes. {failed_scans} satellite scan(s) '
                        'could not finish; retry those homes from the dashboard.',
                    )
                else:
                    messages.success(request, f'Found {len(homes)} homes and ran solar detection!')
        else:
            messages.warning(
                request,
                'No recently sold homes matched this search. The previous results were kept.'
            )

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
    """Run solar detection for pending homes and retry previous scan errors."""
    retryable = Home.objects.filter(solar_status__in=('unknown', 'error'))
    count = retryable.count()
    for home in retryable:
        process_home_sync(home.id)
    messages.info(request, f'Processed or retried {count} homes.')
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
