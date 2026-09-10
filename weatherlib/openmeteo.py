"""Open-Meteo archive API: URL construction and fetching.

The URL builders are identical to the originals in ``app.py``. ``timezone=auto``
is what makes ``is_day`` (and the daily aggregation) follow the location's local
clock rather than UTC.
"""
import time

import requests

API_BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'

# Open-Meteo's archive can be slow for multi-decade hourly pulls; give it room.
_TIMEOUT = (10, 110)

# Vercel functions share egress IPs with every other Vercel project, so
# Open-Meteo's per-IP limits get hit by neighbours. A 429 usually clears within
# seconds, so retry a couple of times before giving up.
_RETRY_WAITS = (1.5, 4.0)


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


class OpenMeteoRateLimited(OpenMeteoError):
    """The archive API is rate-limiting us right now; retrying later may work."""


def _get_json(url, what):
    last_error = None
    for attempt in range(len(_RETRY_WAITS) + 1):
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            last_error = OpenMeteoError(f"Could not reach Open-Meteo ({what}): {exc}")
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                last_error = OpenMeteoRateLimited(
                    f"Open-Meteo is rate-limiting requests right now ({what}). "
                    f"Please try again in a minute."
                )
            elif resp.status_code >= 500:
                last_error = OpenMeteoError(f"Open-Meteo server error ({what}): {resp.status_code}")
            else:
                # Other 4xx (bad coordinates, out-of-range dates) will not fix
                # themselves on retry -- surface the API's own reason and stop.
                try:
                    reason = resp.json().get('reason')
                except ValueError:
                    reason = None
                raise OpenMeteoError(reason or f"Open-Meteo API error ({what}): {resp.status_code}")

        if attempt < len(_RETRY_WAITS):
            time.sleep(_RETRY_WAITS[attempt])

    raise last_error


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
