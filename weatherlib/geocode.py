"""Turn a city name into coordinates.

The committed ``city_cache.csv`` is consulted first so the ~30 cities looked up
before load without touching Nominatim. On Vercel the filesystem is read-only, so
a genuinely new city is resolved live but not written back -- it just gets
re-resolved next time.
"""
import os
import threading

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_CACHE_FILE = os.path.join(_ROOT, 'city_cache.csv')

_geolocator = Nominatim(user_agent="compare-cities-weather")
_lock = threading.Lock()


class GeocodeError(RuntimeError):
    """Raised when a city cannot be resolved."""


class GeocodeUnavailable(RuntimeError):
    """Raised when the geocoding service itself is unreachable."""


def normalize_city_key(city_name):
    """Normalize a city name for cache lookups so 'Merida, mexico' and
    'merida mexico' resolve to the same cached entry."""
    cleaned = "".join(c if c.isalnum() else ' ' for c in city_name.lower())
    return " ".join(cleaned.split())


def _lookup_cache(city_name):
    if not os.path.exists(CITY_CACHE_FILE):
        return None
    cache = pd.read_csv(CITY_CACHE_FILE)
    match = cache[cache['city'].map(normalize_city_key) == normalize_city_key(city_name)]
    if match.empty:
        return None
    return float(match.iloc[0]['lat']), float(match.iloc[0]['lon'])


def _try_write_cache(city_name, lat, lon):
    """Best effort: silently give up on a read-only filesystem (e.g. Vercel)."""
    try:
        new_row = pd.DataFrame([{'city': city_name, 'lat': lat, 'lon': lon}])
        header = not os.path.exists(CITY_CACHE_FILE)
        new_row.to_csv(CITY_CACHE_FILE, mode='a', header=header, index=False)
    except OSError:
        pass


def geocode(city_name):
    """Return ``(lat, lon, from_cache)`` for a city name.

    Raises GeocodeError if the name cannot be resolved, GeocodeUnavailable if the
    geocoding service is down.
    """
    city_name = (city_name or "").strip()
    if not city_name:
        raise GeocodeError("Please enter a city name.")

    cached = _lookup_cache(city_name)
    if cached is not None:
        return cached[0], cached[1], True

    try:
        with _lock:
            location = _geolocator.geocode(city_name, timeout=10)
    except (GeocoderUnavailable, GeocoderTimedOut) as exc:
        raise GeocodeUnavailable("Geocoding service unavailable, please try again later.") from exc

    if not location:
        raise GeocodeError(f"Could not locate '{city_name}'. Adding a country often helps.")

    _try_write_cache(city_name, location.latitude, location.longitude)
    return float(location.latitude), float(location.longitude), False
