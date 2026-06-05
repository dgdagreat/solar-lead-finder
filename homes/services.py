"""
Services for fetching recently sold home data.

Priority order:
  1. RapidAPI — Realty in US  (realty-in-us.p.rapidapi.com)
  2. ATTOM Data API            (api.gateway.attomdata.com)
  3. Mock data                 (development fallback)
"""
import math
import requests
from datetime import datetime
from django.conf import settings
from .models import Home
import logging

logger = logging.getLogger(__name__)

RAPIDAPI_HOST = "realty-in-us.p.rapidapi.com"


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def fetch_recently_sold_homes(city: str, state: str, zip_code: str = None,
                              max_results: int = 20, radius_miles: float = None):
    """
    Fetch recently sold homes. Tries RapidAPI → ATTOM → mock data.

    radius_miles: if set, results are narrowed to homes within this many miles
    of the searched city/zip centre (RapidAPI path only; geocoded via Google).
    """
    if settings.RAPIDAPI_KEY:
        logger.info("Using RapidAPI (Realty in US)")
        homes = _fetch_from_rapidapi(city, state, zip_code, max_results, radius_miles)
        if homes:
            return homes
        logger.warning("RapidAPI returned no results — falling back")

    if settings.ATTOM_API_KEY:
        logger.info("Using ATTOM API")
        return _fetch_from_attom(city, state, zip_code, max_results)

    logger.warning("No API keys set — using mock data")
    return _mock_home_data(city, state)


# ──────────────────────────────────────────────────────────────────────────
# RapidAPI — Realty in US
# ──────────────────────────────────────────────────────────────────────────

def _fetch_from_rapidapi(city: str, state: str, zip_code: str, max_results: int,
                         radius_miles: float = None):
    """
    Fetch recently sold homes from the Realty-in-US RapidAPI.
    Docs: https://rapidapi.com/apidojo/api/realty-in-us
    """
    # When a radius is requested, geocode the search centre and widen the
    # candidate pool so there's enough to filter down by distance.
    center = None
    if radius_miles:
        center = _geocode(zip_code or f"{city}, {state}")
        if center:
            max_results = max(max_results, 50)
        else:
            logger.warning("Radius requested but geocoding unavailable — results not distance-filtered")

    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": settings.RAPIDAPI_KEY,
    }

    # Build location filter — prefer zip_code, fall back to city+state
    body = {
        "limit": max_results,
        "offset": 0,
        "status": ["sold"],
        "sort": {"direction": "desc", "field": "sold_date"},
    }

    if zip_code:
        body["postal_code"] = zip_code
    else:
        body["city"] = city
        body["state_code"] = state

    try:
        resp = requests.post(
            f"https://{RAPIDAPI_HOST}/properties/v3/list",
            headers=headers,
            json=body,
            timeout=15,
        )

        if resp.status_code == 403:
            logger.error("RapidAPI 403 — not subscribed to Realty-in-US API.")
            return []

        resp.raise_for_status()
        data = resp.json()

        # Navigate the response tree  (v3 API uses "results" not "properties")
        properties = (
            data.get("data", {})
                .get("home_search", {})
                .get("results") or []
        )

        if not properties:
            logger.warning(f"RapidAPI: no properties returned for {city}, {state}")
            return []

        homes = []
        for prop in properties:
            if center and radius_miles and _beyond_radius(prop, center, radius_miles):
                continue
            home = _parse_rapidapi_property(prop, city, state)
            if home:
                homes.append(home)

        suffix = f" within {radius_miles:g} mi" if (center and radius_miles) else ""
        logger.info(f"RapidAPI: fetched {len(homes)} homes in {city}, {state}{suffix}")
        return homes

    except requests.RequestException as e:
        logger.error(f"RapidAPI error: {e}")
        return []


def _parse_rapidapi_property(prop: dict, default_city: str, default_state: str) -> Home | None:
    """Parse a single property from the RapidAPI v3 response into a Home object."""
    try:
        location = prop.get("location", {})
        address  = location.get("address", {})

        street   = address.get("line", "").strip()
        city     = address.get("city", default_city).strip()
        state    = address.get("state_code", default_state).strip()
        zip_code = address.get("postal_code", "").strip()

        if not street:
            return None

        # Only single-family homes — skip condos, land, multi-family
        prop_type = prop.get("description", {}).get("type", "")
        if prop_type and prop_type not in ("single_family", "mobile", "farm"):
            logger.debug(f"Skipping {street} — type: {prop_type}")
            return None

        # Price: prefer actual sold price over list price
        price = prop.get("last_sold_price") or prop.get("list_price")

        # Sold date: format "2026-05-29"
        sold_date = None
        raw_date  = prop.get("last_sold_date")
        if raw_date:
            try:
                sold_date = datetime.fromisoformat(raw_date[:10]).date()
            except (ValueError, TypeError):
                pass

        home, created = Home.objects.get_or_create(
            address=street,
            city=city,
            state=state,
            defaults={
                "zip_code":   zip_code,
                "sale_price": price,
                "sold_date":  sold_date,
            }
        )

        if created:
            logger.info(f"New home added: {street}, {city}, {state}")

        return home

    except Exception as e:
        logger.error(f"Error parsing property: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# Geocoding + radius helpers
# ──────────────────────────────────────────────────────────────────────────

def _geocode(query: str):
    """Geocode a city/zip string to (lat, lng) via the Google Geocoding API."""
    key = settings.GOOGLE_STREET_VIEW_API_KEY
    if not key:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": key},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results:
            loc = results[0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
    except Exception as e:
        logger.warning(f"Geocoding failed for '{query}': {e}")
    return None


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lng points."""
    R = 3958.8  # Earth radius, miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlmb   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _beyond_radius(prop: dict, center: tuple, radius_miles: float) -> bool:
    """
    True if a RapidAPI property is farther than radius_miles from `center`.
    Fails open (returns False) when the property has no coordinates.
    """
    coord = ((prop.get("location") or {}).get("address") or {}).get("coordinate") or {}
    lat, lon = coord.get("lat"), coord.get("lon")
    if lat is None or lon is None:
        return False
    return _haversine_miles(center[0], center[1], lat, lon) > radius_miles


# ──────────────────────────────────────────────────────────────────────────
# ATTOM Data API
# ──────────────────────────────────────────────────────────────────────────

def _fetch_from_attom(city: str, state: str, zip_code: str, max_results: int):
    url = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/sale/snapshot"
    headers = {"apikey": settings.ATTOM_API_KEY, "Accept": "application/json"}
    params  = {
        "cityname": city, "State": state,
        "pagesize": max_results, "orderby": "SaleSearchDate DESC",
    }
    if zip_code:
        params["postalcode"] = zip_code

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        homes = []
        for prop in data.get("property", []):
            addr = prop.get("address", {})
            sale = prop.get("sale", {}).get("salesSearchResult", {})
            home, _ = Home.objects.get_or_create(
                address=addr.get("line1", ""),
                city=addr.get("locality", city),
                state=addr.get("countrySubd", state),
                zip_code=addr.get("postal1", zip_code or ""),
                defaults={
                    "sale_price": sale.get("saleAmt"),
                    "sold_date":  sale.get("saleSearchDate"),
                }
            )
            homes.append(home)
        return homes

    except requests.RequestException as e:
        logger.error(f"ATTOM API error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────
# Mock data (development fallback)
# ──────────────────────────────────────────────────────────────────────────

def _mock_home_data(city: str, state: str):
    """
    Real-looking addresses that actually resolve to residential rooftops
    in Google Maps satellite view (verified manually).
    """
    mock_homes = [
        {"address": "2215 Rimcrest Dr",      "city": "Glendale",       "state": "CA", "zip_code": "91207", "sale_price": 1100000},
        {"address": "3421 Ramona Ave",        "city": "Sacramento",     "state": "CA", "zip_code": "95826", "sale_price": 485000},
        {"address": "1244 El Monte Ave",      "city": "Mountain View",  "state": "CA", "zip_code": "94040", "sale_price": 1750000},
        {"address": "4821 Hazel Ave",         "city": "Fair Oaks",      "state": "CA", "zip_code": "95628", "sale_price": 620000},
        {"address": "7312 Quill Dr",          "city": "Downey",         "state": "CA", "zip_code": "90242", "sale_price": 710000},
    ]
    homes = []
    for data in mock_homes:
        # Override city/state with what user searched so results appear relevant
        home, _ = Home.objects.get_or_create(
            address=data["address"],
            city=city or data["city"],
            state=state or data["state"],
            defaults={"zip_code": data["zip_code"], "sale_price": data["sale_price"]}
        )
        homes.append(home)
    return homes
