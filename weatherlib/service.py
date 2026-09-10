"""Orchestration: assemble everything the UI needs for one city into a JSON blob.

This mirrors the data-fetching flow that used to live in the body of ``app.py``:
monthly summaries + records, daytime stats merged onto the months, and the
"current year so far vs the baseline" series for the temperature and rainfall
comparison charts. A cache hit (committed ``weather_cache/`` file) skips the
Open-Meteo call; a miss fetches and reduces on the fly without persisting.
"""
from datetime import datetime, date

import pandas as pd

from . import aggregate, cache
from .openmeteo import fetch_daily, fetch_hourly, OpenMeteoError


def _iso(value):
    return cache.as_date(value).isoformat()


def _round(value, ndigits=2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)


def _monthly_records(df):
    """DataFrame -> list of dicts, with NaN as null and month strings preserved."""
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        item = {}
        for col, val in row.items():
            if isinstance(val, float) and pd.isna(val):
                item[col] = None
            elif isinstance(val, (pd.Timestamp, datetime, date)):
                item[col] = pd.Timestamp(val).strftime('%Y-%m-%d')
            elif hasattr(val, 'item'):
                item[col] = val.item()
            else:
                item[col] = val
        out.append(item)
    return out


def _daily_series(df, value_col, out_key):
    if df is None or df.empty:
        return None
    return [
        {'date': pd.Timestamp(d).strftime('%Y-%m-%d'), out_key: _round(v)}
        for d, v in zip(df['date'], df[value_col])
    ]


def _doy_series(df, value_col):
    if df is None or df.empty:
        return None
    return [
        {'m': int(m), 'd': int(dd), 'avg': _round(v)}
        for m, dd, v in zip(df['month'], df['day'], df[value_col])
    ]


class _CityFetcher:
    """Per-request cache so an identical date range is only fetched/read once.

    The baseline climatology range and the main plotted range are usually the
    same call; without this that would be two fetches of the same 15 years.
    """

    def __init__(self, city, lat, lon):
        self.city = city
        self.lat = lat
        self.lon = lon
        self._summaries = {}
        self._daytime = {}

    def summaries(self, start, end):
        key = (_iso(start), _iso(end))
        if key not in self._summaries:
            self._summaries[key] = self._load_summaries(start, end)
        return self._summaries[key]

    def daytime(self, start, end):
        key = (_iso(start), _iso(end))
        if key not in self._daytime:
            self._daytime[key] = self._load_daytime(start, end)
        return self._daytime[key]

    def _load_summaries(self, start, end):
        volatile = cache.is_volatile_range(end)
        # For volatile ranges the per-day accumulation is what the chart plots, so
        # it has to be part of the cached set or we fall through to a live fetch.
        kinds = ['monthly', 'records', 'precip_climatology']
        if volatile:
            kinds.append('precip_daily')
        cached = cache.read_kinds(
            self.city, self.lat, self.lon, start, end, kinds,
            parse_dates={'precip_daily': ['date']},
        )
        if cached is not None:
            return {
                'monthly': cached['monthly'],
                'records': cached['records'],
                'precip_climatology': cached['precip_climatology'],
                'precip_daily': cached.get('precip_daily'),
            }
        data = fetch_daily(self.lat, self.lon, _iso(start), _iso(end))
        df = aggregate.process_daily_data(data)
        return {
            'monthly': aggregate.aggregate_to_calendar_months(df),
            'records': aggregate.compute_temperature_records(df),
            'precip_climatology': aggregate.build_precipitation_climatology(df),
            'precip_daily': aggregate.build_accumulated_precipitation(df) if volatile else None,
        }

    def _load_daytime(self, start, end):
        volatile = cache.is_volatile_range(end)
        kinds = ['daytime', 'daytime_climatology']
        if volatile:
            kinds.append('daytime_daily')
        cached = cache.read_kinds(
            self.city, self.lat, self.lon, start, end, kinds,
            parse_dates={'daytime_daily': ['date']},
        )
        if cached is not None:
            return {
                'daytime': cached['daytime'],
                'climatology': cached['daytime_climatology'],
                'daily': cached.get('daytime_daily'),
            }
        data = fetch_hourly(self.lat, self.lon, _iso(start), _iso(end))
        daytime_stats = aggregate.aggregate_daytime_to_calendar_months(data)
        if daytime_stats.empty:
            raise OpenMeteoError(f"No daylight hours found for {self.city} in this period.")
        daytime_daily = aggregate.aggregate_daytime_by_day(data)
        climatology = aggregate.build_daytime_climatology(daytime_daily)
        return {
            'daytime': daytime_stats,
            'climatology': climatology,
            'daily': daytime_daily if volatile else None,
        }


def build_city_payload(city, lat, lon, start, end):
    """Return the full JSON-serialisable payload for a single city.

    Raises OpenMeteoError only when the core monthly/records data cannot be
    obtained; softer failures (daytime series, current-year comparison) are
    collected in ``warnings`` and the corresponding fields come back null.
    """
    fetcher = _CityFetcher(city, lat, lon)
    warnings = []

    core = fetcher.summaries(start, end)
    monthly = core['monthly']
    if monthly is None or monthly.empty:
        raise OpenMeteoError(f"No data available for {city} in this period.")

    try:
        daytime = fetcher.daytime(start, end)
        monthly = monthly.merge(
            daytime['daytime'].drop(columns=['time']), on='calendar_month', how='left',
        )
    except OpenMeteoError as exc:
        warnings.append(str(exc))

    today = datetime.today().date()
    current_year = today.year
    cy_start = date(current_year, 1, 1)
    cy_end = today
    baseline_end = min(cache.as_date(end), date(current_year - 1, 12, 31))
    baseline_label = f"{cache.as_date(start).year}-{baseline_end.year}"
    has_baseline_range = baseline_end > cache.as_date(start)

    current = {
        'year': current_year,
        'baselineLabel': baseline_label,
        'hasBaselineRange': has_baseline_range,
        'daytimeThisYear': None,
        'daytimeClimatology': None,
        'precipThisYear': None,
        'precipClimatology': None,
    }

    if cy_end > cy_start:
        try:
            this_year = fetcher.daytime(cy_start, cy_end)
            current['daytimeThisYear'] = _daily_series(this_year['daily'], 'daytime_temperature', 't')
        except OpenMeteoError as exc:
            warnings.append(str(exc))

        try:
            this_year_sum = fetcher.summaries(cy_start, cy_end)
            current['precipThisYear'] = _daily_series(
                this_year_sum['precip_daily'], 'accumulated_precipitation', 'acc',
            )
        except OpenMeteoError as exc:
            warnings.append(str(exc))

        if has_baseline_range:
            try:
                baseline = fetcher.daytime(start, baseline_end)
                current['daytimeClimatology'] = _doy_series(
                    baseline['climatology'], 'avg_daytime_temperature',
                )
            except OpenMeteoError as exc:
                warnings.append(str(exc))

            try:
                baseline_sum = fetcher.summaries(start, baseline_end)
                current['precipClimatology'] = _doy_series(
                    baseline_sum['precip_climatology'], 'avg_accumulated_precipitation',
                )
            except OpenMeteoError as exc:
                warnings.append(str(exc))

    records_df = core['records']
    records = _monthly_records(records_df)

    return {
        'city': city,
        'lat': lat,
        'lon': lon,
        'range': {'start': _iso(start), 'end': _iso(end)},
        'monthly': _monthly_records(monthly),
        'records': records[0] if records else None,
        'currentYear': current,
        'warnings': warnings,
    }
