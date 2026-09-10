"""Reduce the raw Open-Meteo daily/hourly series to the small summaries the UI plots.

These functions are copied unchanged from the original Streamlit ``app.py`` (only
the ``st.*`` progress chatter was dropped). The subtle bits -- leap-year handling
in the climatologies, ``is_day`` filtering for daytime temperature, per-month
percentiles -- must keep behaving exactly as before.
"""
import calendar

import pandas as pd

PERCENTILES = {'low': 0.1, 'high': 0.9}


def process_daily_data(daily_data):
    """Process daily weather data into DataFrame"""
    df = pd.DataFrame(daily_data["daily"])
    # Convert sunshine_duration from seconds to hours
    if "sunshine_duration" in df.columns:
        df["sunshine_hours"] = df["sunshine_duration"] / 3600
    # Convert 'time' to datetime
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_to_calendar_months(df):
    """Aggregate daily data directly to calendar months with correct statistics"""
    df['calendar_month'] = df['time'].dt.month
    results = []

    def get_monthly_total_stats(month_data, metric):
        """Helper for metrics that are summed monthly (e.g., precipitation)."""
        stats = {}
        if metric in df.columns and not month_data[metric].isna().all():
            month_data_copy = month_data.copy()
            month_data_copy['year_month'] = month_data_copy['time'].dt.to_period('M')
            monthly_totals = month_data_copy.groupby('year_month')[metric].sum()
            if not monthly_totals.empty:
                stats[metric] = float(monthly_totals.mean())
                if len(monthly_totals) > 1:
                    stats[f'{metric}_p10'] = float(monthly_totals.quantile(PERCENTILES['low']))
                    stats[f'{metric}_p90'] = float(monthly_totals.quantile(PERCENTILES['high']))
                else:
                    val = float(monthly_totals.iloc[0])
                    stats[f'{metric}_p10'] = val
                    stats[f'{metric}_p90'] = val
        return stats

    def get_daily_value_stats(month_data, metric):
        """Helper for metrics based on daily values (e.g., temperature)."""
        stats = {}
        if metric in df.columns and not month_data[metric].isna().all():
            stats[metric] = float(month_data[metric].mean())
            stats[f'{metric}_p10'] = float(month_data[metric].quantile(PERCENTILES['low']))
            stats[f'{metric}_p90'] = float(month_data[metric].quantile(PERCENTILES['high']))
        return stats

    for month in range(1, 13):
        month_data = df[df['calendar_month'] == month]
        if month_data.empty:
            continue

        month_stats = {'calendar_month': month, 'month': month, 'time': calendar.month_abbr[month]}

        # Metrics summed monthly
        month_stats.update(get_monthly_total_stats(month_data, 'precipitation_sum'))
        month_stats.update(get_monthly_total_stats(month_data, 'sunshine_hours'))

        # Metrics averaged from daily values
        month_stats.update(get_daily_value_stats(month_data, 'temperature_2m_mean'))
        month_stats.update(get_daily_value_stats(month_data, 'relative_humidity_2m_mean'))

        # Temperature extremes
        if "temperature_2m_min" in df.columns and not month_data['temperature_2m_min'].isna().all():
            month_stats['temperature_2m_min_absolute'] = float(month_data['temperature_2m_min'].min())
        if "temperature_2m_max" in df.columns and not month_data['temperature_2m_max'].isna().all():
            month_stats['temperature_2m_max_absolute'] = float(month_data['temperature_2m_max'].max())

        results.append(month_stats)

    return pd.DataFrame(results)


def extract_daylight_hours(hourly_data):
    """Return the hourly rows that fall between local sunrise and sunset.

    Open-Meteo's `is_day` flag is 1 between local sunrise and sunset, so filtering
    on it drops the night hours that pull the usual 24h mean down.
    """
    hourly = pd.DataFrame(hourly_data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    return hourly[(hourly["is_day"] == 1) & hourly["temperature_2m"].notna()]


def aggregate_daytime_by_day(hourly_data):
    """Daytime mean temperature for each individual date."""
    daylight = extract_daylight_hours(hourly_data)
    if daylight.empty:
        return pd.DataFrame(columns=['date', 'daytime_temperature', 'daylight_hours_sampled'])
    by_day = daylight.groupby(daylight["time"].dt.date)["temperature_2m"].agg(['mean', 'size'])
    return pd.DataFrame({
        'date': pd.to_datetime(by_day.index),
        'daytime_temperature': by_day['mean'].to_numpy(),
        'daylight_hours_sampled': by_day['size'].to_numpy(),
    }).reset_index(drop=True)


def aggregate_daytime_to_calendar_months(hourly_data):
    """Aggregate hourly temperatures to calendar months, keeping only daylight hours.

    Months with no daylight at all (polar night) are left out rather than
    reported as 0.
    """
    daylight = extract_daylight_hours(hourly_data)

    results = []
    for month in range(1, 13):
        temps = daylight.loc[daylight["time"].dt.month == month, "temperature_2m"]
        if temps.empty:
            continue
        results.append({
            'calendar_month': month,
            'time': calendar.month_abbr[month],
            'daytime_temperature': float(temps.mean()),
            'daytime_temperature_p10': float(temps.quantile(PERCENTILES['low'])),
            'daytime_temperature_p90': float(temps.quantile(PERCENTILES['high'])),
            'daylight_hours_sampled': int(len(temps)),
        })
    return pd.DataFrame(results)


def build_daytime_climatology(daily_daytime):
    """Average daytime temperature for each day of the year, across all years.

    Keyed on (month, day) rather than day-of-year so leap years line up: without
    this, every date after Feb 28 would be compared against the wrong day.
    """
    if daily_daytime.empty:
        return pd.DataFrame(columns=['month', 'day', 'avg_daytime_temperature', 'years_sampled'])

    # Both groupers come from 'date', so they need renaming before reset_index()
    months = daily_daytime['date'].dt.month.rename('month')
    days = daily_daytime['date'].dt.day.rename('day')
    climatology = daily_daytime.groupby([months, days])['daytime_temperature'].agg(['mean', 'size']).reset_index()
    climatology.columns = ['month', 'day', 'avg_daytime_temperature', 'years_sampled']
    return climatology


def build_precipitation_climatology(df):
    """Average accumulated precipitation for each day of the year.

    Built from the mean rainfall on each calendar day, accumulated in calendar
    order, so the curve can never step backwards. Accumulating each year first
    and averaging the totals looks equivalent but is not: Feb 29 would then be
    averaged over leap years only, leaving a visible dip on Mar 1 where the
    sample changes. Instead Feb 29 is weighted by how often it actually occurs,
    which also keeps the year-end value equal to the mean annual total.

    Years whose data does not start on Jan 1 are skipped: a partial year's
    running total is not a year-to-date figure.
    """
    empty = pd.DataFrame(columns=['month', 'day', 'avg_accumulated_precipitation', 'years_sampled'])
    if "precipitation_sum" not in df.columns:
        return empty
    data = df[['time', 'precipitation_sum']].dropna().sort_values('time')
    if data.empty:
        return empty

    years = data['time'].dt.year
    full_years = [year for year, group in data.groupby(years)
                  if group['time'].min().month == 1 and group['time'].min().day == 1]
    data = data[years.isin(full_years)]
    if data.empty:
        return empty

    months = data['time'].dt.month.rename('month')
    days = data['time'].dt.day.rename('day')
    daily = data.groupby([months, days])['precipitation_sum'].agg(['mean', 'size']).reset_index()
    daily.columns = ['month', 'day', 'mean_daily', 'years_sampled']
    daily = daily.sort_values(['month', 'day']).reset_index(drop=True)

    leap_share = sum(calendar.isleap(year) for year in full_years) / len(full_years)
    weight = pd.Series(1.0, index=daily.index)
    weight[(daily['month'] == 2) & (daily['day'] == 29)] = leap_share

    daily['avg_accumulated_precipitation'] = (daily['mean_daily'] * weight).cumsum()
    return daily[['month', 'day', 'avg_accumulated_precipitation', 'years_sampled']]


def build_accumulated_precipitation(df):
    """Running precipitation total for a single year, one row per date."""
    if "precipitation_sum" not in df.columns:
        return pd.DataFrame(columns=['date', 'precipitation_sum', 'accumulated_precipitation'])
    data = df[['time', 'precipitation_sum']].dropna().sort_values('time').copy()
    data['accumulated_precipitation'] = data['precipitation_sum'].cumsum()
    return data.rename(columns={'time': 'date'}).reset_index(drop=True)


def compute_temperature_records(df):
    """The single hottest and coldest days of the range, by daily mean temperature.

    This is the only thing the app needs from the full daily series, so it is what
    gets cached: keeping the series itself costs megabytes per city to preserve
    four numbers, and it is one API call to rebuild.
    """
    if "temperature_2m_mean" not in df.columns or df["temperature_2m_mean"].isna().all():
        return pd.DataFrame()
    temps = df["temperature_2m_mean"]
    return pd.DataFrame([{
        'record_low': float(temps.min()),
        'record_low_date': df.loc[temps.idxmin(), 'time'].strftime('%Y-%m-%d'),
        'record_high': float(temps.max()),
        'record_high_date': df.loc[temps.idxmax(), 'time'].strftime('%Y-%m-%d'),
    }])
