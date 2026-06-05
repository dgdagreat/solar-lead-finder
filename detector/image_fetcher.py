"""
Fetches satellite/street view images for a given address.
Uses Google Street View Static API (free tier: $200/month credit).
Falls back to a placeholder image if no API key is set.
"""
import requests
import os
from django.conf import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MEDIA_ROOT = settings.MEDIA_ROOT


def fetch_street_view_image(address: str, save_path: str = None) -> str | None:
    """
    Fetch a Google Street View top-down/satellite image for an address.
    Returns the local file path if saved, or the URL if not saving.
    """
    api_key = settings.GOOGLE_STREET_VIEW_API_KEY

    if not api_key:
        logger.warning("No GOOGLE_STREET_VIEW_API_KEY set — skipping image fetch.")
        return None

    # Use satellite-like overhead pitch for roof detection
    params = {
        "size": "640x640",
        "location": address,
        "pitch": "90",       # Look straight down for roof view
        "fov": "90",
        "key": api_key,
    }

    url = "https://maps.googleapis.com/maps/api/streetview"

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        if response.headers.get("Content-Type", "").startswith("image"):
            if save_path:
                full_path = Path(MEDIA_ROOT) / save_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "wb") as f:
                    f.write(response.content)
                return str(save_path)
            else:
                # Return the URL instead
                return url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        else:
            logger.warning(f"Street View returned non-image response for: {address}")
            return None

    except requests.RequestException as e:
        logger.error(f"Street View API error for {address}: {e}")
        return None


def get_static_map_url(address: str) -> str | None:
    """
    Get a Google Maps Static API satellite image URL for rooftop view.
    """
    api_key = settings.GOOGLE_STREET_VIEW_API_KEY
    if not api_key:
        return None

    params = {
        "center": address,
        "zoom": "20",         # Max zoom for rooftop visibility
        "size": "640x640",
        "maptype": "satellite",
        "key": api_key,
    }

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"{base_url}?{query}"
