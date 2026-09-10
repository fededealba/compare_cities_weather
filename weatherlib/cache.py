"""Read-only access to the committed ``weather_cache/`` summaries.

The original app both read and wrote this directory. On Vercel the deployment
filesystem is read-only, so this module only reads: a cache hit skips the
Open-Meteo call, a miss just means we fetch and reduce on the fly (without
persisting). Filenames match what ``app.py`` writes, so the committed CSVs are
picked up unchanged.
"""
import os
from datetime import datetime

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEATHER_CACHE_DIR = os.path.join(_ROOT, 'weather_cache')
VOLATILE_CACHE_DIR = os.path.join(WEATHER_CACHE_DIR, 'recent')
VOLATILE_CACHE_DAYS = 7


def as_date(value):
    """Accept either a date or a datetime and return a date."""
    return value.date() if isinstance(value, datetime) else value


def is_volatile_range(end):
    """True for ranges running up to roughly now, whose data is still settling.

    These are never served from the committed cache -- they would be stale within
    a day -- so the API always refetches them.
    """
    return (datetime.today().date() - as_date(end)).days <= VOLATILE_CACHE_DAYS


def get_cache_slug(city_name, lat, lon):
    """The coordinates are part of the key, not just the name: sanitizing drops
    punctuation and accents, so two different places can share a slug.
    """
    safe_city_name = "".join(c for c in city_name.lower() if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    return f'{safe_city_name}_{lat:.4f}_{lon:.4f}'


def get_cache_path(city_name, lat, lon, start, end, kind):
    """Build the cache file path for one city / date range / kind of data."""
    directory = VOLATILE_CACHE_DIR if is_volatile_range(end) else WEATHER_CACHE_DIR
    cache_key = f'{get_cache_slug(city_name, lat, lon)}_{as_date(start):%Y-%m-%d}_{as_date(end):%Y-%m-%d}'
    return os.path.join(directory, f'{cache_key}_{kind}.csv')


def read_kinds(city_name, lat, lon, start, end, kinds, parse_dates=None):
    """Return ``{kind: DataFrame}`` if every requested kind is on disk, else None.

    Volatile ranges always return None so the caller refetches fresh data.
    """
    if is_volatile_range(end):
        return None
    paths = {kind: get_cache_path(city_name, lat, lon, start, end, kind) for kind in kinds}
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    parse_dates = parse_dates or {}
    try:
        return {
            kind: pd.read_csv(path, parse_dates=parse_dates.get(kind))
            for kind, path in paths.items()
        }
    except (OSError, ValueError, pd.errors.ParserError):
        return None
