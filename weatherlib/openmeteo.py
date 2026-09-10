"""Open-Meteo archive API: URL construction and fetching.

The URL builders are identical to the originals in ``app.py``. ``timezone=auto``
is what makes ``is_day`` (and the daily aggregation) follow the location's local
clock rather than UTC.
"""
import requests

API_BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'

# Open-Meteo's archive can be slow for multi-decade hourly pulls; give it room.
_TIMEOUT = (10, 110)


def build_api_url(lat, lon, start_str, end_str):
    """Build Open-Meteo API URL for the daily series."""
    return (
        f"{API_BASE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&daily=precipitation_sum,relative_humidity_2m_mean,sunshine_duration,temperature_2m_mean,temperature_2m_min,temperature_2m_max"
        f"&timezone=auto"
    )


def build_hourly_api_url(lat, lon, start_str, end_str):
    """Build Open-Meteo API URL for hourly temperature plus the day/night flag.

    `timezone=auto` makes is_day follow local sunrise/sunset for the location.
    """
    return (
        f"{API_BASE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&hourly=temperature_2m,is_day"
        f"&timezone=auto"
    )


class OpenMeteoError(RuntimeError):
    """Raised when the archive API returns an error or no usable data."""


def _get_json(url, what):
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise OpenMeteoError(f"Could not reach Open-Meteo ({what}): {exc}") from exc
    if resp.status_code != 200:
        raise OpenMeteoError(f"Open-Meteo API error ({what}): {resp.status_code}")
    return resp.json()


def fetch_daily(lat, lon, start_str, end_str):
    """Return the parsed daily-series JSON, or raise OpenMeteoError."""
    data = _get_json(build_api_url(lat, lon, start_str, end_str), "daily")
    if not data.get("daily"):
        raise OpenMeteoError("No daily data available for this location and period.")
    return data


def fetch_hourly(lat, lon, start_str, end_str):
    """Return the parsed hourly-series JSON, or raise OpenMeteoError."""
    data = _get_json(build_hourly_api_url(lat, lon, start_str, end_str), "hourly")
    if not data.get("hourly"):
        raise OpenMeteoError("No hourly data available for this location and period.")
    return data
