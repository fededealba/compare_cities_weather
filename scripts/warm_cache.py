#!/usr/bin/env python3
"""Pre-generate ``weather_cache/`` entries so the deployed app makes no live
Open-Meteo calls for known cities.

Open-Meteo rate-limits bursts of multi-decade requests hard, and Vercel functions
share egress IPs with every other Vercel project, so live fetching from the app
is unreliable. Instead this script -- run locally, and daily by
``.github/workflows/warm-cache.yml`` from GitHub's runners -- fetches the data
ahead of time and commits it.

For each city it warms:
  * the historical baseline range (2010-01-01 .. end of last full year), skipped
    when every file is already on disk;
  * the current-year range (Jan 1 .. today), always refreshed -- yesterday's
    files are pruned.

Usage:
  python scripts/warm_cache.py                      # city_cache.csv + the defaults
  python scripts/warm_cache.py Tokyo "Cape Town, ZA"
  python scripts/warm_cache.py --current-year-only  # skip the historical pass
"""
import os
import sys
import time
from datetime import date

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weatherlib import aggregate, cache, geocode  # noqa: E402
from weatherlib.openmeteo import OpenMeteoError, fetch_daily, fetch_hourly  # noqa: E402

HISTORICAL_START = date(2010, 1, 1)
DEFAULT_CITIES = ['Paris', 'Madrid', 'Berlin']
CITIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cities.txt')
SLEEP_BETWEEN_FETCHES = 3.0
SLEEP_BETWEEN_CITIES = 4.0
# When a city fails even after openmeteo's own retries, Open-Meteo has throttled
# this IP for a while. Wait it out once before giving up on the city.
COOLDOWN_SECONDS = 90.0

NON_DAILY_KINDS = ['monthly', 'records', 'precip_climatology', 'daytime', 'daytime_climatology']


def _iso(d):
    return d.isoformat()


def _write_from_daily(city, lat, lon, start, end, paths, include_daily):
    df = aggregate.process_daily_data(fetch_daily(lat, lon, _iso(start), _iso(end)))
    cache.write_cache(aggregate.aggregate_to_calendar_months(df), paths['monthly'])
    cache.write_cache(aggregate.compute_temperature_records(df), paths['records'])
    cache.write_cache(aggregate.build_precipitation_climatology(df), paths['precip_climatology'])
    if include_daily:
        cache.write_cache(aggregate.build_accumulated_precipitation(df), paths['precip_daily'])


def _write_from_hourly(city, lat, lon, start, end, paths, include_daily):
    hourly = fetch_hourly(lat, lon, _iso(start), _iso(end))
    stats = aggregate.aggregate_daytime_to_calendar_months(hourly)
    daily = aggregate.aggregate_daytime_by_day(hourly)
    cache.write_cache(stats, paths['daytime'])
    cache.write_cache(aggregate.build_daytime_climatology(daily), paths['daytime_climatology'])
    if include_daily:
        cache.write_cache(daily, paths['daytime_daily'])


def warm_range(city, lat, lon, start, end, include_daily):
    kinds = list(NON_DAILY_KINDS)
    if include_daily:
        kinds += ['precip_daily', 'daytime_daily']
    paths = {k: cache.get_cache_path(city, lat, lon, start, end, k) for k in kinds}

    if not include_daily and all(os.path.exists(p) for p in paths.values()):
        print(f"    {start}..{end}: already cached")
        return

    _write_from_daily(city, lat, lon, start, end, paths, include_daily)
    time.sleep(SLEEP_BETWEEN_FETCHES)
    _write_from_hourly(city, lat, lon, start, end, paths, include_daily)
    print(f"    {start}..{end}: wrote {len(kinds)} files")


def _curated_names():
    if not os.path.exists(CITIES_FILE):
        return []
    names = []
    with open(CITIES_FILE) as handle:
        for line in handle:
            line = line.split('#', 1)[0].strip()
            if line:
                names.append(line)
    return names


def city_names():
    explicit = [a for a in sys.argv[1:] if not a.startswith('--')]
    if explicit:
        return explicit

    names = list(DEFAULT_CITIES)
    seen_slugs = set()
    # Cities already resolved (from prior use or a past warm run) come first, so a
    # curated name that matches one reuses its coordinates and cache files.
    if os.path.exists(geocode.CITY_CACHE_FILE):
        for _, row in pd.read_csv(geocode.CITY_CACHE_FILE).iterrows():
            slug = cache.get_cache_slug(row['city'], float(row['lat']), float(row['lon']))
            if slug in seen_slugs or row['city'] in names:
                continue
            seen_slugs.add(slug)
            names.append(row['city'])
    for name in _curated_names():
        if name not in names:
            names.append(name)
    return names


def main():
    current_year_only = '--current-year-only' in sys.argv
    today = date.today()
    hist_end = date(today.year - 1, 12, 31)
    cy_start = date(today.year, 1, 1)

    failures = []
    for name in city_names():
        print(name)
        try:
            lat, lon, _ = geocode.geocode(name)
        except (geocode.GeocodeError, geocode.GeocodeUnavailable) as exc:
            print(f"    geocode failed: {exc}")
            failures.append(name)
            continue

        def warm_all():
            if not current_year_only and hist_end > HISTORICAL_START:
                warm_range(name, lat, lon, HISTORICAL_START, hist_end, include_daily=False)
                time.sleep(SLEEP_BETWEEN_FETCHES)
            if today > cy_start:
                warm_range(name, lat, lon, cy_start, today, include_daily=True)
                cache.prune_superseded(name, lat, lon, cy_start, today)

        try:
            warm_all()
        except OpenMeteoError as exc:
            print(f"    rate-limited ({exc}); cooling down {COOLDOWN_SECONDS:.0f}s and retrying once")
            time.sleep(COOLDOWN_SECONDS)
            try:
                warm_all()
            except OpenMeteoError as exc2:
                print(f"    still failing: {exc2}")
                failures.append(name)

        time.sleep(SLEEP_BETWEEN_CITIES)

    if failures:
        print(f"\n{len(failures)} city/cities failed: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll cities warmed.")


if __name__ == '__main__':
    main()
