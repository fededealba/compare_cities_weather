import streamlit as st
from datetime import datetime, time, timedelta
import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import requests
import calendar
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import os

# Configuration constants
CITY_CACHE_FILE = 'city_cache.csv'
COLORS = {'blue': 'blue', 'red': 'red', 'green': 'green'}
FILL_COLORS = {
    'blue': 'rgba(0,0,255,0.1)',
    'red': 'rgba(255,0,0,0.1)', 
    'green': 'rgba(0,128,0,0.1)'
}
ANOMALY_COLORS = {'warm': 'rgba(214,39,40,0.45)', 'cool': 'rgba(31,119,180,0.45)'}
PERCENTILES = {'low': 0.1, 'high': 0.9}
API_BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'
DEFAULT_CITIES = {'city1': 'Paris', 'city2': 'Madrid', 'city3': 'Berlin'}

# Setup
st.set_page_config(page_title="Weather Comparison", layout="wide")
st.title("🌦️ Compare Historical Weather Between Cities")
st.text('Source: https://open-meteo.com')

# Geocoding functions
def normalize_city_key(city_name):
    """Normalize a city name for cache lookups so 'Merida, mexico' and
    'merida mexico' resolve to the same cached entry."""
    cleaned = "".join(c if c.isalnum() else ' ' for c in city_name.lower())
    return " ".join(cleaned.split())

def get_city_latlon(city_name, geolocator):
    # Check if cache file exists and city is cached
    if os.path.exists(CITY_CACHE_FILE):
        cache = pd.read_csv(CITY_CACHE_FILE)
        match = cache[cache['city'].map(normalize_city_key) == normalize_city_key(city_name)]
        if not match.empty:
            return match.iloc[0]['lat'], match.iloc[0]['lon']
    # If not cached, geocode
    location = geolocator.geocode(city_name)
    if location:
        # Save to cache
        new_row = pd.DataFrame([{'city': city_name, 'lat': location.latitude, 'lon': location.longitude}])
        if os.path.exists(CITY_CACHE_FILE):
            new_row.to_csv(CITY_CACHE_FILE, mode='a', header=False, index=False)
        else:
            new_row.to_csv(CITY_CACHE_FILE, index=False)
        return location.latitude, location.longitude
    return None, None

# Date and city selection in sidebar
with st.sidebar:
    st.header("Settings")
    # Kept outside the form so ticking it reveals the City 3 input immediately;
    # widgets inside a form only take effect once the form is submitted.
    add_third_city = st.checkbox("Add a third city?")
    with st.form("settings_form"):
        today = datetime.today().date()
        start = st.date_input(
            "Start date",
            datetime(2010, 1, 1),
            min_value=datetime(1970, 1, 1),
            max_value=today - timedelta(days=1),
            key="start_date"
        )
        end = st.date_input(
            "End date",
            datetime(2025, 12, 31),
            min_value=datetime(1970, 1, 1),
            max_value=today,
            key="end_date"
        )
        city1 = st.text_input("City 1", DEFAULT_CITIES['city1'], key="city1")
        city2 = st.text_input("City 2", DEFAULT_CITIES['city2'], key="city2")
        city3 = st.text_input("City 3", DEFAULT_CITIES['city3'], key="city3") if add_third_city else None
        # Show resolved addresses for each city
        geolocator = Nominatim(user_agent="streamlit-weather-app-sidebar")
        city_locations = []
        unresolved = []
        for label, city in [("City 1", city1), ("City 2", city2)] + ([("City 3", city3)] if city3 else []):
            if not city or not city.strip():
                st.caption(f"{label}: please enter a city name.")
                unresolved.append(label)
                continue
            try:
                lat, lon = get_city_latlon(city, geolocator)
                if lat is not None and lon is not None:
                    st.caption(f"{label} resolved as: {city} (lat: {lat:.4f}, lon: {lon:.4f})")
                    city_locations.append({"city": city, "lat": lat, "lon": lon})
                else:
                    st.caption(f"{label} not found.")
                    unresolved.append(f"{label} ({city})")
            except (GeocoderUnavailable, GeocoderTimedOut):
                st.caption(f"{label}: Geocoding service unavailable, please try again later.")
                unresolved.append(f"{label} ({city})")
            except Exception as e:
                st.caption(f"{label}: Geocoding error: {e}")
                unresolved.append(f"{label} ({city})")
        submitted = st.form_submit_button("Submit")

# Only run the rest of the app if the form is submitted (or on first load)
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if submitted:
    st.session_state.form_submitted = True

if st.session_state.form_submitted:
    if unresolved:
        st.error(
            "Could not locate: " + ", ".join(unresolved)
            + ". Check the spelling in the sidebar (adding a country often helps) and submit again."
        )
        st.stop()

    if start >= end:
        st.error("End date must be after start date.")
        st.stop()

    # Convert to datetime.datetime if needed
    if isinstance(start, datetime):
        start_dt = start
    else:
        start_dt = datetime.combine(start, time.min)

    if isinstance(end, datetime):
        end_dt = end
    else:
        end_dt = datetime.combine(end, time.min)

    # Plotting helper functions
    def plot_metric_with_percentiles(cities_data, metric, p10_col, p90_col, title, unit,
                                     reference_metric=None, reference_label=None):
        """Generic function to plot metrics with percentile ranges.

        An optional reference_metric is drawn as a thin dashed line per city, for
        charts where the point is the gap between two related series.
        """
        fig = go.Figure()
        for city, monthly_df, color in cities_data:
            x = monthly_df["time"]
            if metric not in monthly_df.columns:
                st.info(f"{title} data not available for {city}.")
                continue
            y = monthly_df[metric]
            y_p10 = monthly_df[p10_col] if p10_col in monthly_df.columns else None
            y_p90 = monthly_df[p90_col] if p90_col in monthly_df.columns else None
            
            # Fill between p10 and p90
            if y_p10 is not None and y_p90 is not None:
                fig.add_trace(go.Scatter(
                    x=x, y=y_p90, mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip', name=f"{city} 90th percentile"
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=y_p10, mode='lines', fill='tonexty',
                    fillcolor=FILL_COLORS[color], line=dict(width=0),
                    showlegend=True, name=f"{city} 10th-90th percentile"
                ))
            # Mean line
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                   name=f"{city} Mean", line=dict(color=color, width=2)))
            # Optional reference series for comparison
            if reference_metric and reference_metric in monthly_df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=monthly_df[reference_metric], mode='lines',
                    name=f"{city} {reference_label}",
                    line=dict(color=color, width=1, dash='dot')
                ))
        fig.update_layout(title=f"{title} (10th-90th Percentile Range)", xaxis_title="Month",
                         yaxis_title=unit, height=400)
        st.plotly_chart(fig)
    
    # Weather data processing functions
    def build_api_url(lat, lon, start_str, end_str):
        """Build Open-Meteo API URL"""
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

    # File-based cache for weather data.
    #
    # Only reduced summaries are ever written - never the raw daily or hourly series,
    # which run to megabytes per city and are one API call away. Ranges that end at
    # (or near) today land in a separate directory: they are rewritten under a new
    # name every day, so they are kept out of version control and pruned on write.
    WEATHER_CACHE_DIR = 'weather_cache'
    VOLATILE_CACHE_DIR = os.path.join(WEATHER_CACHE_DIR, 'recent')
    VOLATILE_CACHE_DAYS = 7

    def as_date(value):
        """Accept either a date or a datetime and return a date."""
        return value.date() if isinstance(value, datetime) else value

    def is_volatile_range(end):
        """True for ranges running up to roughly now, whose data is still settling."""
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

    def prune_superseded_volatile_cache(city_name, lat, lon, start, keep_end):
        """Drop earlier 'up to today' files for the same city and start date.

        Without this the volatile directory gains a full set of files every day,
        each one strictly superseded by the next.
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

    def write_cache(df, path, decimals=2):
        """Cached numbers are display-grade, so full float precision is just bloat."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rounded = df.copy()
        # Only the float columns: rounding a datetime column is a no-op that warns.
        float_columns = rounded.select_dtypes(include='float').columns
        rounded[float_columns] = rounded[float_columns].round(decimals)
        rounded.to_csv(path, index=False)

    def get_open_meteo_data_by_latlon(city_name, lat, lon, start, end):
        """
        Fetches weather data for a given city, using a file-based cache to avoid
        repeated API calls.

        The daily series is reduced to summaries before caching. Returns
        (summaries_dict, error_message), where the dict is keyed by cache kind;
        'precip_daily' is only present for volatile ranges, where the running
        total for a single year is what the chart plots.
        """
        kinds = ['monthly', 'records', 'precip_climatology']
        keeps_daily = is_volatile_range(end)
        if keeps_daily:
            kinds.append('precip_daily')
        paths = {kind: get_cache_path(city_name, lat, lon, start, end, kind) for kind in kinds}

        # Check if cached data exists
        if all(os.path.exists(path) for path in paths.values()):
            try:
                st.caption(f"Reading from cache for {city_name}...")
                summaries = {kind: pd.read_csv(path) for kind, path in paths.items()}
                if keeps_daily:
                    summaries['precip_daily']['date'] = pd.to_datetime(summaries['precip_daily']['date'])
                return summaries, None
            except Exception as e:
                st.warning(f"Could not read cache for {city_name}. Refetching. Error: {e}")

        # If not cached, fetch from API
        st.caption(f"Fetching new data from API for {city_name}...")
        daily_url = build_api_url(lat, lon, as_date(start).strftime('%Y-%m-%d'),
                                  as_date(end).strftime('%Y-%m-%d'))
        try:
            daily_resp = requests.get(daily_url)
            if daily_resp.status_code != 200:
                return None, f"Open-Meteo API error (daily): {daily_resp.status_code}"
            daily_data = daily_resp.json()
            if not daily_data.get("daily"):
                return None, f"No data available for {city_name} in this period."

            # Process the data
            df = process_daily_data(daily_data)
            summaries = {
                'monthly': aggregate_to_calendar_months(df),
                'records': compute_temperature_records(df),
                'precip_climatology': build_precipitation_climatology(df),
            }
            if keeps_daily:
                summaries['precip_daily'] = build_accumulated_precipitation(df)

            # Save to cache
            try:
                for kind, path in paths.items():
                    write_cache(summaries[kind], path)
                if keeps_daily:
                    prune_superseded_volatile_cache(city_name, lat, lon, start, end)
                st.caption(f"Saved to cache for {city_name}.")
            except Exception as e:
                st.warning(f"Could not save cache for {city_name}. Error: {e}")

            return summaries, None
        except Exception as e:
            return None, f"Error processing weather data for {city_name}: {str(e)}"

    def get_daytime_data_by_latlon(city_name, lat, lon, start, end):
        """
        Fetches hourly temperatures and returns daylight-only summaries.

        Returns (monthly_stats_df, climatology_df, daily_df, error_message). The
        per-day series is only cached for volatile ranges, where it is what the
        comparison chart plots; for long historical ranges the day-of-year
        climatology is the only thing consumed, and it is ~15x smaller.
        """
        daytime_cache_path = get_cache_path(city_name, lat, lon, start, end, 'daytime')
        climatology_cache_path = get_cache_path(city_name, lat, lon, start, end, 'daytime_climatology')
        daily_cache_path = get_cache_path(city_name, lat, lon, start, end, 'daytime_daily')
        keeps_daily = is_volatile_range(end)

        cached_paths = [daytime_cache_path, climatology_cache_path]
        if keeps_daily:
            cached_paths.append(daily_cache_path)
        if all(os.path.exists(path) for path in cached_paths):
            try:
                st.caption(f"Reading daytime cache for {city_name}...")
                daily = pd.read_csv(daily_cache_path, parse_dates=['date']) if keeps_daily else None
                return (pd.read_csv(daytime_cache_path),
                        pd.read_csv(climatology_cache_path),
                        daily,
                        None)
            except Exception as e:
                st.warning(f"Could not read daytime cache for {city_name}. Refetching. Error: {e}")

        st.caption(f"Fetching hourly data from API for {city_name}...")
        hourly_url = build_hourly_api_url(lat, lon, as_date(start).strftime('%Y-%m-%d'),
                                          as_date(end).strftime('%Y-%m-%d'))
        try:
            resp = requests.get(hourly_url)
            if resp.status_code != 200:
                return None, None, None, f"Open-Meteo API error (hourly) for {city_name}: {resp.status_code}"
            hourly_data = resp.json()
            if not hourly_data.get("hourly"):
                return None, None, None, f"No hourly data available for {city_name} in this period."

            daytime_stats = aggregate_daytime_to_calendar_months(hourly_data)
            daytime_daily = aggregate_daytime_by_day(hourly_data)
            climatology = build_daytime_climatology(daytime_daily)
            if daytime_stats.empty:
                return None, None, None, f"No daylight hours found for {city_name} in this period."

            try:
                write_cache(daytime_stats, daytime_cache_path)
                write_cache(climatology, climatology_cache_path)
                if keeps_daily:
                    write_cache(daytime_daily, daily_cache_path)
                    prune_superseded_volatile_cache(city_name, lat, lon, start, end)
                st.caption(f"Saved daytime cache for {city_name}.")
            except Exception as e:
                st.warning(f"Could not save daytime cache for {city_name}. Error: {e}")

            return daytime_stats, climatology, (daytime_daily if keeps_daily else None), None
        except Exception as e:
            return None, None, None, f"Error processing daytime data for {city_name}: {str(e)}"

    # Prepare city data with lat/lon
    city_latlons = {c['city']: (c['lat'], c['lon']) for c in city_locations}
    with st.spinner("Fetching weather data from Open-Meteo..."):
        summaries1, err1 = get_open_meteo_data_by_latlon(city1, *city_latlons[city1], start_dt, end_dt)
        summaries2, err2 = get_open_meteo_data_by_latlon(city2, *city_latlons[city2], start_dt, end_dt)
        summaries3, err3 = (get_open_meteo_data_by_latlon(city3, *city_latlons[city3], start_dt, end_dt) if city3 else (None, None))

    data1, data2, data3 = (s['monthly'] if s else None for s in (summaries1, summaries2, summaries3))
    records1, records2, records3 = (s['records'] if s else None for s in (summaries1, summaries2, summaries3))

    if err1:
        st.error(err1)
    if err2:
        st.error(err2)
    if city3 and err3:
        st.error(err3)

    if err1 or err2 or (city3 and err3):
        st.stop()

    if data1 is not None and data2 is not None:
        # For plotting, build a list of (city, data, color)
        cities_data = [(city1, data1, COLORS['blue']), (city2, data2, COLORS['red'])]
        if city3 and data3 is not None:
            cities_data.append((city3, data3, COLORS['green']))

        # Daylight-only temperatures come from a separate hourly request. A failure
        # here only costs those charts, so it is reported but not fatal.
        daytime_errors = []
        with st.spinner("Fetching hourly data for daylight temperatures..."):
            merged_cities_data = []
            for city, monthly_df, color in cities_data:
                daytime_stats, _, _, daytime_err = get_daytime_data_by_latlon(
                    city, *city_latlons[city], start_dt, end_dt
                )
                if daytime_err:
                    daytime_errors.append(daytime_err)
                else:
                    monthly_df = monthly_df.merge(
                        daytime_stats.drop(columns=['time']),
                        on='calendar_month', how='left'
                    )
                merged_cities_data.append((city, monthly_df, color))
            cities_data = merged_cities_data

        # This year so far, fetched separately since it runs past the selected range.
        # The baseline stops at the end of last year so the current year is never
        # compared against itself; for the usual ranges that clamp changes nothing.
        current_year = datetime.today().date().year
        current_year_start = datetime(current_year, 1, 1).date()
        current_year_end = datetime.today().date()
        baseline_end = min(as_date(end_dt), datetime(current_year - 1, 12, 31).date())
        current_year_errors = []
        current_year_by_city = {}
        climatology_by_city = {}
        precip_this_year_by_city = {}
        precip_climatology_by_city = {}
        if current_year_end > current_year_start:
            with st.spinner(f"Fetching {current_year} weather so far..."):
                for city, _, _ in cities_data:
                    _, _, this_year_daily, this_year_err = get_daytime_data_by_latlon(
                        city, *city_latlons[city], current_year_start, current_year_end
                    )
                    if this_year_err:
                        current_year_errors.append(this_year_err)
                    else:
                        current_year_by_city[city] = this_year_daily

                    this_year_summaries, precip_err = get_open_meteo_data_by_latlon(
                        city, *city_latlons[city], current_year_start, current_year_end
                    )
                    if precip_err:
                        current_year_errors.append(precip_err)
                    elif not this_year_summaries['precip_daily'].empty:
                        precip_this_year_by_city[city] = this_year_summaries['precip_daily']

                    if baseline_end > as_date(start_dt):
                        _, climatology, _, baseline_err = get_daytime_data_by_latlon(
                            city, *city_latlons[city], start_dt, baseline_end
                        )
                        if not baseline_err and climatology is not None and not climatology.empty:
                            climatology_by_city[city] = climatology

                        baseline_summaries, baseline_precip_err = get_open_meteo_data_by_latlon(
                            city, *city_latlons[city], start_dt, baseline_end
                        )
                        if not baseline_precip_err and not baseline_summaries['precip_climatology'].empty:
                            precip_climatology_by_city[city] = baseline_summaries['precip_climatology']

        # Show map with city dots at the top of the main page
        if city_locations:
            map_df = pd.DataFrame(city_locations)
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=1000, width=400, height=200)

        # Tabs for plots and prediction
        plots_tab, averages_tab, current_year_tab = st.tabs(
            ["📊 Plots", "🗓️ Monthly Averages", f"🌡️ Comparison to {current_year}"]
        )

        with plots_tab:
            st.subheader("🌡️ Monthly Temperature (°C)")
            st.caption("This chart shows the mean temperature for each month, with the shaded area representing the 10th to 90th percentile range.")
            plot_metric_with_percentiles(cities_data, "temperature_2m_mean",
                                       "temperature_2m_mean_p10", "temperature_2m_mean_p90",
                                       "Temperature", "°C")

            st.subheader("☀️ Monthly Daytime Temperature (°C)")
            st.caption(
                "This chart uses only the hours between local sunrise and sunset, so the cold "
                "night hours are excluded. The shaded area is the 10th to 90th percentile of "
                "those daylight hours, and the dotted line is the all-day (24h) mean for "
                "comparison — the gap between the two is how much the nights pull the average down."
            )
            if daytime_errors:
                for daytime_err in daytime_errors:
                    st.info(daytime_err)
            if any("daytime_temperature" in monthly_df.columns for _, monthly_df, _ in cities_data):
                plot_metric_with_percentiles(cities_data, "daytime_temperature",
                                           "daytime_temperature_p10", "daytime_temperature_p90",
                                           "Daytime Temperature", "°C",
                                           reference_metric="temperature_2m_mean",
                                           reference_label="All-day mean")
            else:
                st.info("Daytime temperature data not available for the selected cities.")

            st.subheader("💧 Average Humidity (%)")
            st.caption("This chart shows the mean relative humidity for each month, with the shaded area representing the 10th to 90th percentile range.")
            if any("relative_humidity_2m_mean" in monthly_df.columns for _, monthly_df, _ in cities_data):
                plot_metric_with_percentiles(cities_data, "relative_humidity_2m_mean", 
                                           "relative_humidity_2m_mean_p10", "relative_humidity_2m_mean_p90", 
                                           "Humidity", "%")
            else:
                st.info("Humidity data not available for one or more cities.")

            st.subheader("🌧️ Monthly Precipitation (mm)")
            st.caption("This chart shows the mean total precipitation for each month, with the shaded area representing the 10th to 90th percentile range of the monthly totals.")
            plot_metric_with_percentiles(cities_data, "precipitation_sum", 
                                       "precipitation_sum_p10", "precipitation_sum_p90", 
                                       "Precipitation", "mm")

            st.subheader("🌞 Monthly Sunshine Hours")
            st.caption("This chart shows the mean total sunshine hours for each month, with the shaded area representing the 10th to 90th percentile range of the monthly totals.")
            if any("sunshine_hours" in monthly_df.columns for _, monthly_df, _ in cities_data):
                plot_metric_with_percentiles(cities_data, "sunshine_hours", 
                                           "sunshine_hours_p10", "sunshine_hours_p90", 
                                           "Sunshine", "Hours")
            else:
                st.info("Sunshine duration data not available for one or more cities.")

            st.subheader("🌡️ Temperature Extremes (°C)")
            st.caption("This chart shows the absolute highest and lowest temperatures recorded for each calendar month in the selected period.")
            def plot_min_max_temperature():
                fig = go.Figure()
                for city, monthly_df, color in cities_data:
                    x = monthly_df["time"]
                    y_min = monthly_df["temperature_2m_min_absolute"] if "temperature_2m_min_absolute" in monthly_df.columns else None
                    y_max = monthly_df["temperature_2m_max_absolute"] if "temperature_2m_max_absolute" in monthly_df.columns else None
                    if y_min is None or y_max is None:
                        st.info(f"Temperature extremes data not available for {city}.")
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y_min, mode='lines+markers', name=f"{city} Coldest", line=dict(color=color, dash='dot')))
                    fig.add_trace(go.Scatter(x=x, y=y_max, mode='lines+markers', name=f"{city} Hottest", line=dict(color=color, dash='dash')))
                fig.update_layout(title="Absolute Temperature Extremes by Month", xaxis_title="Month", yaxis_title="°C", height=400)
                st.plotly_chart(fig)
            plot_min_max_temperature()

            st.subheader("🌡️ Record Temperature Extremes")
            st.caption("This table and chart show the single hottest and coldest days (based on daily average temperature) recorded across the entire selected date range.")
            abs_min_max = []
            for city, city_records in zip([city1, city2] + ([city3] if city3 else []),
                                          [records1, records2] + ([records3] if city3 else [])):
                if city_records is None or city_records.empty:
                    st.info(f"Record temperature data not available for {city}.")
                    continue
                record = city_records.iloc[0]
                abs_min_max.append({
                    "City": city,
                    "Record Low (°C)": record["record_low"],
                    "Coldest Day": record["record_low_date"],
                    "Record High (°C)": record["record_high"],
                    "Hottest Day": record["record_high_date"]
                })
            abs_min_max_df = pd.DataFrame(abs_min_max)
            if abs_min_max_df.empty:
                st.info("No record temperature data available for the selected cities.")
            else:
                with st.expander("View Record Data Table"):
                    st.table(abs_min_max_df)
                def plot_abs_min_max():
                    fig = go.Figure()
                    for _, row in abs_min_max_df.iterrows():
                        fig.add_trace(go.Bar(
                            x=[row["City"]],
                            y=[row["Record High (°C)"]],
                            name=f"{row['City']} Record High",
                            marker_color='red',
                            text=[row["Hottest Day"]]
                        ))
                        fig.add_trace(go.Bar(
                            x=[row["City"]],
                            y=[row["Record Low (°C)"]],
                            name=f"{row['City']} Record Low",
                            marker_color='blue',
                            text=[row["Coldest Day"]]
                        ))
                    fig.update_layout(barmode='group', title="All-Time Temperature Records", xaxis_title="City", yaxis_title="°C", height=400)
                    st.plotly_chart(fig)
                plot_abs_min_max()

        with averages_tab:
            tomorrow = datetime.today().date() + timedelta(days=1)
            prediction_date = st.date_input("Prediction date", tomorrow, key="prediction_date")
            st.header(f"Prediction for {prediction_date.strftime('%B %d')}")
            pred_month = prediction_date.month

            # Helper function to safely extract scalar values
            def get_scalar(row, col_name):
                if col_name not in row.columns or row[col_name].empty:
                    return None
                val = row[col_name].iloc[0]
                return val.item() if hasattr(val, 'item') else val

            prediction_rows = []
            metrics_to_predict = [
                {"label": "Temperature Mean", "unit": "°C", "p_label": "Temperature 10th-90th", "mean": "temperature_2m_mean", "p10": "temperature_2m_mean_p10", "p90": "temperature_2m_mean_p90", "format": ".1f"},
                {"label": "Humidity Mean", "unit": "%", "p_label": "Humidity 10th-90th", "mean": "relative_humidity_2m_mean", "p10": "relative_humidity_2m_mean_p10", "p90": "relative_humidity_2m_mean_p90", "format": ".1f"},
                {"label": "Precipitation (mm, avg monthly)", "unit": "", "mean": "precipitation_sum", "format": ".1f"},
                {"label": "Sunshine (hours, avg monthly)", "unit": "", "mean": "sunshine_hours", "format": ".1f"},
            ]

            for city, monthly_df, color in cities_data:
                pred_row_data = monthly_df[monthly_df['month'] == pred_month]
                row = {"City": city}
                if pred_row_data.empty:
                    for metric in metrics_to_predict:
                        key = f'{metric["label"]} ({metric["unit"]})' if metric["unit"] else metric["label"]
                        row[key] = "No data"
                        if 'p10' in metric:
                            p_key = f'{metric["p_label"]} ({metric["unit"]})' if metric["unit"] else metric["p_label"]
                            row[p_key] = "No data"
                else:
                    for metric in metrics_to_predict:
                        mean_val = get_scalar(pred_row_data, metric['mean'])
                        key = f'{metric["label"]} ({metric["unit"]})' if metric["unit"] else metric["label"]
                        row[key] = f"{mean_val:{metric['format']}}" if mean_val is not None else "No data"
                        
                        if 'p10' in metric:
                            p10_val = get_scalar(pred_row_data, metric['p10'])
                            p90_val = get_scalar(pred_row_data, metric['p90'])
                            p_key = f'{metric["p_label"]} ({metric["unit"]})' if metric["unit"] else metric["p_label"]
                            if p10_val is not None and p90_val is not None:
                                row[p_key] = f"{p10_val:.1f}–{p90_val:.1f}"
                            else:
                                row[p_key] = "No data"
                prediction_rows.append(row)

            prediction_df = pd.DataFrame(prediction_rows)
            st.table(prediction_df)

        with current_year_tab:
            baseline_label = f"{as_date(start_dt).year}-{baseline_end.year}"
            st.subheader(f"☀️ Daytime Temperature: {current_year} so far vs the {baseline_label} average")
            st.caption(
                f"Each day of {current_year} is compared against the average daytime temperature "
                f"for that same calendar day across {baseline_label}. Red means {current_year} ran "
                "warmer than usual that day, blue means cooler. Daylight hours only — night is excluded."
            )

            for current_year_err in current_year_errors:
                st.info(current_year_err)

            comparable_cities = [
                (city, color) for city, _, color in cities_data
                if city in climatology_by_city and city in current_year_by_city
            ]

            # Whether a baseline is possible at all depends only on the dates, not on
            # whether a fetch succeeded, so the two cases get different explanations.
            has_baseline_range = baseline_end > as_date(start_dt)

            if not comparable_cities:
                if not has_baseline_range:
                    st.info(
                        f"The selected date range leaves no earlier years to compare {current_year} "
                        f"against. Pick a start date before {current_year} to see this comparison."
                    )
                else:
                    st.info(f"No {current_year} daytime data available yet for the selected cities.")
            else:
                smoothing = st.slider(
                    "Smoothing (days)", min_value=1, max_value=31, value=1, key="anomaly_smoothing",
                    help="1 shows the raw daily values. Raise it to average over a rolling "
                         "window, which makes longer warm and cool spells easier to pick out."
                )

                def build_comparison(city):
                    """Line up this year's daily daytime temps against the day-of-year average."""
                    climatology = climatology_by_city[city]
                    if climatology.empty:
                        return None
                    this_year = current_year_by_city[city].copy()
                    this_year['month'] = this_year['date'].dt.month
                    this_year['day'] = this_year['date'].dt.day
                    merged = this_year.merge(climatology, on=['month', 'day'], how='left')
                    merged = merged.dropna(subset=['daytime_temperature', 'avg_daytime_temperature'])
                    return merged.sort_values('date') if not merged.empty else None

                comparisons = {city: build_comparison(city) for city, _ in comparable_cities}

                # Summary of how the year is running, from the unsmoothed daily values
                summary_rows = []
                for city, _ in comparable_cities:
                    merged = comparisons[city]
                    if merged is None:
                        continue
                    anomaly = merged['daytime_temperature'] - merged['avg_daytime_temperature']
                    warmest = merged.loc[anomaly.idxmax()]
                    coldest = merged.loc[anomaly.idxmin()]
                    summary_rows.append({
                        "City": city,
                        "Days compared": len(merged),
                        "Warmer than average": int((anomaly > 0).sum()),
                        "Cooler than average": int((anomaly < 0).sum()),
                        "Mean difference (°C)": f"{anomaly.mean():+.2f}",
                        "Biggest warm day": f"{warmest['date']:%b %d} ({anomaly.max():+.1f} °C)",
                        "Biggest cool day": f"{coldest['date']:%b %d} ({anomaly.min():+.1f} °C)",
                    })
                if summary_rows:
                    st.table(pd.DataFrame(summary_rows))

                def plot_current_year_vs_average(city, merged):
                    """Plot this year against the daily average, shaded red above / blue below."""
                    if smoothing > 1:
                        smoothed = merged[['daytime_temperature', 'avg_daytime_temperature']].rolling(
                            smoothing, center=True, min_periods=1
                        ).mean()
                    else:
                        smoothed = merged[['daytime_temperature', 'avg_daytime_temperature']]
                    x = merged['date']
                    actual = smoothed['daytime_temperature']
                    average = smoothed['avg_daytime_temperature']
                    # .where(cond, other) keeps the value where cond holds, so these are
                    # the elementwise max and min of the two series.
                    above = actual.where(actual > average, average)
                    below = actual.where(actual < average, average)

                    fig = go.Figure()
                    # Warm shading: fill from the average up to this year where it is hotter
                    fig.add_trace(go.Scatter(x=x, y=average, mode='lines', line=dict(width=0),
                                             showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=x, y=above, mode='lines', fill='tonexty',
                                             fillcolor=ANOMALY_COLORS['warm'], line=dict(width=0),
                                             name="Warmer than average", hoverinfo='skip'))
                    # Cool shading: fill from the average down to this year where it is colder
                    fig.add_trace(go.Scatter(x=x, y=average, mode='lines', line=dict(width=0),
                                             showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=x, y=below, mode='lines', fill='tonexty',
                                             fillcolor=ANOMALY_COLORS['cool'], line=dict(width=0),
                                             name="Cooler than average", hoverinfo='skip'))
                    # The two lines themselves, drawn on top of the shading
                    fig.add_trace(go.Scatter(x=x, y=average, mode='lines',
                                             name=f"{baseline_label} average",
                                             line=dict(color='gray', width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=x, y=actual, mode='lines', name=str(current_year),
                                             line=dict(color='#222222', width=1.5)))
                    fig.update_layout(title=f"{city} — daytime temperature vs average",
                                      xaxis_title="Date", yaxis_title="°C", height=400,
                                      hovermode='x unified')
                    st.plotly_chart(fig)

                for city, _ in comparable_cities:
                    merged = comparisons[city]
                    if merged is None:
                        st.info(f"Not enough historical data to compare {city}.")
                        continue
                    plot_current_year_vs_average(city, merged)

            st.subheader(f"🌧️ Accumulated Precipitation: {current_year} so far vs the {baseline_label} average")
            st.caption(
                f"Rainfall added up from 1 January. The dashed line is how much a typical "
                f"{baseline_label} year had by the same date, so the gap between the two lines "
                f"is the running surplus or shortfall."
            )

            precip_cities = [
                (city, color) for city, _, color in cities_data
                if city in precip_climatology_by_city and city in precip_this_year_by_city
            ]

            if not precip_cities:
                if not has_baseline_range:
                    st.info(f"No earlier years in the selected range to compare {current_year} rainfall against.")
                else:
                    st.info(f"No {current_year} precipitation data available yet for the selected cities.")
            else:
                def build_precip_comparison(city):
                    """Line up this year's running rainfall total against the day-of-year average."""
                    this_year = precip_this_year_by_city[city].copy()
                    this_year['month'] = this_year['date'].dt.month
                    this_year['day'] = this_year['date'].dt.day
                    merged = this_year.merge(precip_climatology_by_city[city], on=['month', 'day'], how='left')
                    merged = merged.dropna(subset=['accumulated_precipitation', 'avg_accumulated_precipitation'])
                    return merged.sort_values('date') if not merged.empty else None

                precip_comparisons = {city: build_precip_comparison(city) for city, _ in precip_cities}

                precip_summary = []
                for city, _ in precip_cities:
                    merged = precip_comparisons[city]
                    if merged is None:
                        continue
                    latest = merged.iloc[-1]
                    so_far = latest['accumulated_precipitation']
                    normal = latest['avg_accumulated_precipitation']
                    precip_summary.append({
                        "City": city,
                        f"{current_year} so far (mm)": f"{so_far:.0f}",
                        "Typical by this date (mm)": f"{normal:.0f}",
                        "Difference (mm)": f"{so_far - normal:+.0f}",
                        "Share of typical": f"{so_far / normal:.0%}" if normal else "n/a",
                    })
                if precip_summary:
                    st.table(pd.DataFrame(precip_summary))

                precip_fig = go.Figure()
                for city, color in precip_cities:
                    merged = precip_comparisons[city]
                    if merged is None:
                        st.info(f"Not enough historical data to compare precipitation for {city}.")
                        continue
                    precip_fig.add_trace(go.Scatter(
                        x=merged['date'], y=merged['avg_accumulated_precipitation'], mode='lines',
                        name=f"{city} typical", line=dict(color=color, width=1.5, dash='dash')
                    ))
                    precip_fig.add_trace(go.Scatter(
                        x=merged['date'], y=merged['accumulated_precipitation'], mode='lines',
                        name=f"{city} {current_year}", line=dict(color=color, width=2.5)
                    ))
                precip_fig.update_layout(
                    title=f"Accumulated precipitation since 1 January",
                    xaxis_title="Date", yaxis_title="mm", height=450, hovermode='x unified'
                )
                st.plotly_chart(precip_fig)
else:
    st.info("Welcome! Please select your cities and date range in the sidebar and click 'Submit' to see the weather comparison.")
