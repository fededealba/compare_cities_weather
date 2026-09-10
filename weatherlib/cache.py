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


def write_cache(df, path, decimals=2):
    """Write one summary DataFrame to ``path``. Used by ``scripts/warm_cache.py``.

    Cached numbers are display-grade, so full float precision is just bloat --
    round the float columns (rounding a datetime column would only warn).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rounded = df.copy()
    float_columns = rounded.select_dtypes(include='float').columns
    rounded[float_columns] = rounded[float_columns].round(decimals)
    rounded.to_csv(path, index=False)


def prune_superseded(city_name, lat, lon, start, keep_end):
    """Drop earlier 'up to today' files for the same city and start date.

    Without this the volatile directory gains a fresh set of files every day,
    each strictly superseded by the next.
    """
    if not os.path.isdir(VOLATILE_CACHE_DIR):
        return
    prefix = f'{get_cache_slug(city_name, lat, lon)}_{as_date(start):%Y-%m-%d}_'
    keep_prefix = f'{prefix}{as_date(keep_end):%Y-%m-%d}_'
    for name in os.listdir(VOLATILE_CACHE_DIR):
        if name.startswith(prefix) and not name.startswith(keep_prefix):
            try:
                os.remove(os.path.join(VOLATILE_CACHE_DIR, name))
            except OSError:
                pass


def read_kinds(city_name, lat, lon, start, end, kinds, parse_dates=None):
    """Return ``{kind: DataFrame}`` if every requested kind is on disk, else None.

    Volatile (near-today) ranges are read too when a file exists: the warm-cache
    job (``scripts/warm_cache.py``, run daily by CI) keeps them fresh, and serving
    a few hours stale beats a burst of Open-Meteo calls that get rate-limited.
    A miss just falls through to a live fetch as before.
    """
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
