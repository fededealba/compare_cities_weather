"""Turn a city name into coordinates.

Lookup order:

1. the committed ``city_cache.csv`` (instant, no network) -- authoritative for
   the cities already in ``weather_cache/``;
2. the Open-Meteo geocoding API.

Nominatim is deliberately not used here. Its usage policy disallows this kind of
serverless/bulk traffic and it rate-limits or outright blocks shared cloud IPs
(which is what Vercel functions run on), which showed up as 429s. Open-Meteo's
geocoder is built for app use, needs no key, and shares a vendor with the weather
data. ``app.py`` still uses geopy/Nominatim; that dependency now lives only in
``requirements-streamlit.txt``.
"""
import os

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_CACHE_FILE = os.path.join(_ROOT, 'city_cache.csv')

_GEOCODE_URL = 'https://geocoding-api.open-meteo.com/v1/search'
_TIMEOUT = (5, 15)


class GeocodeError(RuntimeError):
    """Raised when a city name cannot be resolved to a place."""


class GeocodeUnavailable(RuntimeError):
    """Raised when the geocoding service is unreachable or rate-limiting us."""


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


def _pick_result(results, hint):
    """Prefer a result whose country/region matches the text after the comma."""
    if hint:
        hint = hint.lower()
        for r in results:
            fields = (str(r.get('country', '')).lower(),
                      str(r.get('admin1', '')).lower(),
                      str(r.get('country_code', '')).lower())
            if any(hint == f or hint in f for f in fields):
                return r
    return results[0]


def _open_meteo_geocode(query):
    name = query.split(',')[0].strip()
    hint = query.split(',', 1)[1].strip() if ',' in query else ''
    try:
        resp = requests.get(_GEOCODE_URL, params={
            'name': name, 'count': 10, 'language': 'en', 'format': 'json',
        }, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise GeocodeUnavailable("Geocoding service unavailable, please try again later.") from exc

    if resp.status_code == 429:
        raise GeocodeUnavailable("Geocoding is rate-limited right now, please try again in a moment.")
    if resp.status_code != 200:
        raise GeocodeUnavailable(f"Geocoding service error ({resp.status_code}).")

    results = resp.json().get('results') or []
    if not results:
        raise GeocodeError(f"Could not locate '{query}'. Try just the city name, or add a country.")
    return _pick_result(results, hint)


def geocode(city_name):
    """Return ``(lat, lon, from_cache)`` for a city name.

    Raises GeocodeError if the name cannot be resolved, GeocodeUnavailable if the
    geocoding service is down or rate-limiting.
    """
    city_name = (city_name or "").strip()
    if not city_name:
        raise GeocodeError("Please enter a city name.")

    cached = _lookup_cache(city_name)
    if cached is not None:
        return cached[0], cached[1], True

    result = _open_meteo_geocode(city_name)
    lat, lon = float(result['latitude']), float(result['longitude'])
    _try_write_cache(city_name, lat, lon)
    return lat, lon, False
