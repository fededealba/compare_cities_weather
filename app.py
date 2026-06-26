import streamlit as st
from datetime import datetime, time, timedelta
import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
PERCENTILES = {'low': 0.1, 'high': 0.9}
API_BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'
DEFAULT_CITIES = {'city1': 'Paris', 'city2': 'Madrid', 'city3': 'Berlin'}

# Setup
st.set_page_config(page_title="Weather Comparison", layout="wide")
st.title("🌦️ Compare Historical Weather Between Two Cities")
st.text('Source: https://open-meteo.com')


def create_retry_session():
    """HTTP session with automatic retries and backoff for rate-limited APIs."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "compare-cities-weather/1.0 (streamlit)"})
    return session


# Geocoding functions
def get_city_latlon(city_name, geolocator):
    if os.path.exists(CITY_CACHE_FILE):
        cache = pd.read_csv(CITY_CACHE_FILE)
        match = cache[cache['city'].str.lower() == city_name.lower()]
        if not match.empty:
            return match.iloc[0]['lat'], match.iloc[0]['lon']
    location = geolocator.geocode(city_name)
    if location:
        new_row = pd.DataFrame([{'city': city_name, 'lat': location.latitude, 'lon': location.longitude}])
        if os.path.exists(CITY_CACHE_FILE):
            new_row.to_csv(CITY_CACHE_FILE, mode='a', header=False, index=False)
        else:
            new_row.to_csv(CITY_CACHE_FILE, index=False)
        return location.latitude, location.longitude
    return None, None


# Weather data processing functions
def build_api_url(lat, lon, start_str, end_str):
    return (
        f"{API_BASE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&daily=precipitation_sum,relative_humidity_2m_mean,sunshine_duration,temperature_2m_mean,temperature_2m_min,temperature_2m_max"
        f"&timezone=auto"
    )


def process_daily_data(daily_data):
    df = pd.DataFrame(daily_data["daily"])
    if "sunshine_duration" in df.columns:
        df["sunshine_hours"] = df["sunshine_duration"] / 3600
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_to_calendar_months(df):
    df['calendar_month'] = df['time'].dt.month
    results = []

    def get_monthly_total_stats(month_data, metric):
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

        month_stats.update(get_monthly_total_stats(month_data, 'precipitation_sum'))
        month_stats.update(get_monthly_total_stats(month_data, 'sunshine_hours'))
        month_stats.update(get_daily_value_stats(month_data, 'temperature_2m_mean'))
        month_stats.update(get_daily_value_stats(month_data, 'relative_humidity_2m_mean'))

        if "temperature_2m_min" in df.columns and not month_data['temperature_2m_min'].isna().all():
            month_stats['temperature_2m_min_absolute'] = float(month_data['temperature_2m_min'].min())
        if "temperature_2m_max" in df.columns and not month_data['temperature_2m_max'].isna().all():
            month_stats['temperature_2m_max_absolute'] = float(month_data['temperature_2m_max'].max())

        results.append(month_stats)

    return pd.DataFrame(results)


def add_compatibility_columns(calendar_stats):
    if "temperature_2m_min_absolute" in calendar_stats.columns:
        calendar_stats["temperature_2m_mean_min"] = calendar_stats["temperature_2m_min_absolute"]
    if "temperature_2m_max_absolute" in calendar_stats.columns:
        calendar_stats["temperature_2m_mean_max"] = calendar_stats["temperature_2m_max_absolute"]
    return calendar_stats


def create_absolute_temperature_summary(df):
    if "temperature_2m_mean" not in df.columns:
        return {}

    df['calendar_month'] = df['time'].dt.month
    extremes = {}

    for month in range(1, 13):
        month_data = df[df['calendar_month'] == month]
        if not month_data.empty:
            extremes[month] = {
                'coldest_temp': month_data['temperature_2m_mean'].min(),
                'coldest_date': month_data.loc[month_data['temperature_2m_mean'].idxmin(), 'time'],
                'hottest_temp': month_data['temperature_2m_mean'].max(),
                'hottest_date': month_data.loc[month_data['temperature_2m_mean'].idxmax(), 'time']
            }

    return extremes


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_weather_data(city_name, lat, lon, start_str, end_str):
    """Fetch and process weather data from Open-Meteo, cached for 24 hours per city+date range."""
    daily_url = build_api_url(lat, lon, start_str, end_str)
    session = create_retry_session()
    try:
        resp = session.get(daily_url, timeout=30)
        resp.raise_for_status()
        daily_data = resp.json()
        if not daily_data.get("daily"):
            return None, f"No data available for {city_name} in this period.", None
        df = process_daily_data(daily_data)
        calendar_stats = aggregate_to_calendar_months(df)
        calendar_stats = add_compatibility_columns(calendar_stats)
        return calendar_stats, None, df
    except requests.exceptions.HTTPError as e:
        return None, f"Open-Meteo API error for {city_name}: {e.response.status_code}", None
    except Exception as e:
        return None, f"Error fetching data for {city_name}: {str(e)}", None


def get_open_meteo_data_by_latlon(city_name, lat, lon, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    return _fetch_weather_data(city_name, lat, lon, start_str, end_str)


# Plotting helper functions
def plot_metric_with_percentiles(cities_data, metric, p10_col, p90_col, title, unit):
    fig = go.Figure()
    for city, monthly_df, color in cities_data:
        x = monthly_df["time"]
        if metric not in monthly_df.columns:
            st.info(f"{title} data not available for {city}.")
            continue
        y = monthly_df[metric]
        y_p10 = monthly_df[p10_col] if p10_col in monthly_df.columns else None
        y_p90 = monthly_df[p90_col] if p90_col in monthly_df.columns else None

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
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                 name=f"{city} Mean", line=dict(color=color, width=2)))
    fig.update_layout(title=f"{title} (10th-90th Percentile Range)", xaxis_title="Month",
                      yaxis_title=unit, height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_simple_comparison(cities_data, metric, title, unit, show_total=False):
    fig = go.Figure()
    totals = []
    for city, monthly_df, color in cities_data:
        if metric not in monthly_df.columns:
            st.info(f"{title} data not available for {city}.")
            continue
        x = monthly_df["time"]
        y = monthly_df[metric]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                 name=f"{city}", line=dict(color=color, width=2)))
        if show_total:
            total = y.sum()
            if hasattr(total, 'iloc'):
                total = total.iloc[0] if len(total) > 0 else 0
            totals.append((city, total))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=unit, height=400)
    st.plotly_chart(fig, use_container_width=True)
    if show_total and totals:
        st.markdown("**Typical annual total:**")
        for city, total in totals:
            st.write(f"{city}: {total:.1f} {unit}")


# Date and city selection in sidebar
with st.sidebar:
    with st.form("settings_form"):
        st.header("Settings")
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
            datetime(2024, 12, 31),
            min_value=datetime(1970, 1, 1),
            max_value=today,
            key="end_date"
        )
        city1 = st.text_input("City 1", DEFAULT_CITIES['city1'], key="city1")
        city2 = st.text_input("City 2", DEFAULT_CITIES['city2'], key="city2")
        add_third_city = st.checkbox("Add a third city?")
        city3 = None
        if add_third_city:
            city3 = st.text_input("City 3", DEFAULT_CITIES['city3'], key="city3")
        geolocator = Nominatim(user_agent="streamlit-weather-app-sidebar")
        city_locations = []
        for label, city in [("City 1", city1), ("City 2", city2)] + ([("City 3", city3)] if city3 else []):
            if city:
                try:
                    lat, lon = get_city_latlon(city, geolocator)
                    if lat is not None and lon is not None:
                        st.caption(f"{label} resolved as: {city} (lat: {lat:.4f}, lon: {lon:.4f})")
                        city_locations.append({"city": city, "lat": lat, "lon": lon})
                    else:
                        st.caption(f"{label} not found.")
                except (GeocoderUnavailable, GeocoderTimedOut):
                    st.caption(f"{label}: Geocoding service unavailable, please try again later.")
                except Exception as e:
                    st.caption(f"{label}: Geocoding error: {e}")
        submitted = st.form_submit_button("Submit")

# Only run the rest of the app if the form is submitted (or on first load)
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if submitted:
    st.session_state.form_submitted = True

if st.session_state.form_submitted:
    if start >= end:
        st.error("End date must be after start date.")
        st.stop()

    if isinstance(start, datetime):
        start_dt = start
    else:
        start_dt = datetime.combine(start, time.min)

    if isinstance(end, datetime):
        end_dt = end
    else:
        end_dt = datetime.combine(end, time.min)

    city_latlons = {c['city']: (c['lat'], c['lon']) for c in city_locations}
    with st.spinner("Fetching weather data from Open-Meteo..."):
        data1, err1, df1 = get_open_meteo_data_by_latlon(city1, *city_latlons[city1], start_dt, end_dt)
        data2, err2, df2 = get_open_meteo_data_by_latlon(city2, *city_latlons[city2], start_dt, end_dt)
        data3, err3, df3 = (get_open_meteo_data_by_latlon(city3, *city_latlons[city3], start_dt, end_dt) if city3 else (None, None, None))

    if err1:
        st.error(err1)
    if err2:
        st.error(err2)
    if city3 and err3:
        st.error(err3)

    if err1 or err2 or (city3 and err3):
        st.stop()

    if data1 is not None and data2 is not None:
        cities_data = [(city1, data1, COLORS['blue']), (city2, data2, COLORS['red'])]
        if city3 and data3 is not None:
            cities_data.append((city3, data3, COLORS['green']))

        if city_locations:
            map_df = pd.DataFrame(city_locations)
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=1000, use_container_width=False, width=400, height=200)

        plots_tab, averages_tab = st.tabs(["📊 Plots", "🗓️ Monthly Averages"])

        with plots_tab:
            st.subheader("🌡️ Monthly Temperature (°C)")
            st.caption("This chart shows the mean temperature for each month, with the shaded area representing the 10th to 90th percentile range.")
            plot_metric_with_percentiles(cities_data, "temperature_2m_mean",
                                         "temperature_2m_mean_p10", "temperature_2m_mean_p90",
                                         "Temperature", "°C")

            st.subheader("💧 Average Humidity (%)")
            st.caption("This chart shows the mean relative humidity for each month, with the shaded area representing the 10th to 90th percentile range.")
            if any("relative_humidity_2m_mean" in df.columns for _, df, _ in cities_data):
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
            if all("sunshine_hours" in df.columns for _, df, _ in cities_data):
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
                    y_min = monthly_df["temperature_2m_mean_min"] if "temperature_2m_mean_min" in monthly_df.columns else None
                    y_max = monthly_df["temperature_2m_mean_max"] if "temperature_2m_mean_max" in monthly_df.columns else None
                    if y_min is None or y_max is None:
                        st.info(f"Temperature extremes data not available for {city}.")
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y_min, mode='lines+markers', name=f"{city} Coldest", line=dict(color=color, dash='dot')))
                    fig.add_trace(go.Scatter(x=x, y=y_max, mode='lines+markers', name=f"{city} Hottest", line=dict(color=color, dash='dash')))
                fig.update_layout(title="Absolute Temperature Extremes by Month", xaxis_title="Month", yaxis_title="°C", height=400)
                st.plotly_chart(fig, use_container_width=True)
            plot_min_max_temperature()

            st.subheader("🌡️ Record Temperature Extremes")
            st.caption("This table and chart show the single hottest and coldest days (based on daily average temperature) recorded across the entire selected date range.")
            abs_min_max = []
            for city, df in zip([city1, city2] + ([city3] if city3 else []), [df1, df2] + ([df3] if city3 else [])):
                if df is not None:
                    min_temp = df["temperature_2m_mean"].min()
                    max_temp = df["temperature_2m_mean"].max()
                    min_date = df.loc[df["temperature_2m_mean"].idxmin(), "time"]
                    max_date = df.loc[df["temperature_2m_mean"].idxmax(), "time"]
                    abs_min_max.append({
                        "City": city,
                        "Record Low (°C)": min_temp,
                        "Date": min_date.strftime('%Y-%m-%d'),
                        "Record High (°C)": max_temp,
                        "Date ": max_date.strftime('%Y-%m-%d')
                    })
            abs_min_max_df = pd.DataFrame(abs_min_max)
            with st.expander("View Record Data Table"):
                st.table(abs_min_max_df)
            def plot_abs_min_max():
                fig = go.Figure()
                for i, row in abs_min_max_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["City"]],
                        y=[row["Record High (°C)"]],
                        name=f"{row['City']} Record High",
                        marker_color='red',
                        text=[f"{row['Date ']}"]
                    ))
                    fig.add_trace(go.Bar(
                        x=[row["City"]],
                        y=[row["Record Low (°C)"]],
                        name=f"{row['City']} Record Low",
                        marker_color='blue',
                        text=[f"{row['Date']}"]
                    ))
                fig.update_layout(barmode='group', title="All-Time Temperature Records", xaxis_title="City", yaxis_title="°C", height=400)
                st.plotly_chart(fig, use_container_width=True)
            plot_abs_min_max()

        with averages_tab:
            tomorrow = datetime.today().date() + timedelta(days=1)
            prediction_date = st.date_input("Prediction date", tomorrow, key="prediction_date")
            st.header(f"Prediction for {prediction_date.strftime('%B %d')}")
            pred_month = prediction_date.month

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
else:
    st.info("Welcome! Please select your cities and date range in the sidebar and click 'Submit' to see the weather comparison.")
