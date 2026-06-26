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
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "compare-cities-weather/1.0 (streamlit)"})
    return session


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


def build_api_url(lat, lon, start_str, end_str):
    return (
        f"{API_BASE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,sunshine_duration"
        f"&daily=sunrise,sunset"
        f"&timezone=auto"
    )


def process_hourly_data(api_data):
    df = pd.DataFrame(api_data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    df_daily = pd.DataFrame({
        "date": pd.to_datetime(api_data["daily"]["time"]).dt.date,
        "sunrise": pd.to_datetime(api_data["daily"]["sunrise"]),
        "sunset": pd.to_datetime(api_data["daily"]["sunset"]),
    })
    df = df.merge(df_daily[["date", "sunrise", "sunset"]], on="date", how="left")
    df["is_daytime"] = (df["time"] >= df["sunrise"]) & (df["time"] < df["sunset"])

    if "sunshine_duration" in df.columns:
        df["sunshine_hours"] = df["sunshine_duration"] / 3600

    return df


def _day_night_stats(month_data, col, prefix):
    """Per-month day and night stats from hourly values, based on daily means."""
    stats = {}
    for is_day, label in [(True, 'day'), (False, 'night')]:
        subset = month_data[month_data['is_daytime'] == is_day]
        if subset.empty or col not in subset.columns or subset[col].isna().all():
            continue
        daily_means = subset.groupby('date')[col].mean()
        key = f'{prefix}_{label}'
        stats[key] = float(daily_means.mean())
        if len(daily_means) > 1:
            stats[f'{key}_p10'] = float(daily_means.quantile(PERCENTILES['low']))
            stats[f'{key}_p90'] = float(daily_means.quantile(PERCENTILES['high']))
        else:
            val = float(daily_means.iloc[0])
            stats[f'{key}_p10'] = val
            stats[f'{key}_p90'] = val
    return stats


def _monthly_sum_stats(month_data, col, key):
    """Per-month stats for variables that are summed (precipitation, sunshine)."""
    stats = {}
    if col not in month_data.columns or month_data[col].isna().all():
        return stats
    monthly_totals = month_data.groupby('year_month')[col].sum()
    if monthly_totals.empty:
        return stats
    stats[key] = float(monthly_totals.mean())
    if len(monthly_totals) > 1:
        stats[f'{key}_p10'] = float(monthly_totals.quantile(PERCENTILES['low']))
        stats[f'{key}_p90'] = float(monthly_totals.quantile(PERCENTILES['high']))
    else:
        val = float(monthly_totals.iloc[0])
        stats[f'{key}_p10'] = val
        stats[f'{key}_p90'] = val
    return stats


def aggregate_to_calendar_months(df):
    df = df.copy()
    df['calendar_month'] = df['time'].dt.month
    df['year_month'] = df['time'].dt.to_period('M')
    results = []

    for month in range(1, 13):
        month_data = df[df['calendar_month'] == month]
        if month_data.empty:
            continue

        month_stats = {'calendar_month': month, 'month': month, 'time': calendar.month_abbr[month]}

        month_stats.update(_day_night_stats(month_data, 'temperature_2m', 'temperature'))
        month_stats.update(_day_night_stats(month_data, 'relative_humidity_2m', 'humidity'))
        month_stats.update(_monthly_sum_stats(month_data, 'precipitation', 'precipitation_sum'))
        month_stats.update(_monthly_sum_stats(month_data, 'sunshine_hours', 'sunshine_hours'))

        if 'temperature_2m' in df.columns and not month_data['temperature_2m'].isna().all():
            month_stats['temperature_min_absolute'] = float(month_data['temperature_2m'].min())
            month_stats['temperature_max_absolute'] = float(month_data['temperature_2m'].max())

        results.append(month_stats)

    return pd.DataFrame(results)


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_weather_data(city_name, lat, lon, start_str, end_str):
    daily_url = build_api_url(lat, lon, start_str, end_str)
    session = create_retry_session()
    try:
        resp = session.get(daily_url, timeout=30)
        resp.raise_for_status()
        api_data = resp.json()
        if not api_data.get("hourly"):
            return None, f"No data available for {city_name} in this period.", None
        df = process_hourly_data(api_data)
        calendar_stats = aggregate_to_calendar_months(df)
        return calendar_stats, None, df
    except requests.exceptions.HTTPError as e:
        return None, f"Open-Meteo API error for {city_name}: {e.response.status_code}", None
    except Exception as e:
        return None, f"Error fetching data for {city_name}: {str(e)}", None


def get_open_meteo_data_by_latlon(city_name, lat, lon, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    return _fetch_weather_data(city_name, lat, lon, start_str, end_str)


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

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if submitted:
    st.session_state.form_submitted = True

if st.session_state.form_submitted:
    if start >= end:
        st.error("End date must be after start date.")
        st.stop()

    start_dt = datetime.combine(start, time.min) if not isinstance(start, datetime) else start
    end_dt = datetime.combine(end, time.min) if not isinstance(end, datetime) else end

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
                st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
                       size=1000, use_container_width=False, width=400, height=200)

        plots_tab, averages_tab = st.tabs(["📊 Plots", "🗓️ Monthly Averages"])

        with plots_tab:
            st.subheader("☀️ Daytime Temperature (°C)")
            st.caption("Mean temperature during daylight hours (sunrise to sunset), with 10th–90th percentile range. Each data point is the mean of daily daytime averages across all years for that month.")
            plot_metric_with_percentiles(cities_data, "temperature_day",
                                         "temperature_day_p10", "temperature_day_p90",
                                         "Daytime Temperature", "°C")

            st.subheader("🌙 Nighttime Temperature (°C)")
            st.caption("Mean temperature during nighttime hours (sunset to sunrise), with 10th–90th percentile range.")
            plot_metric_with_percentiles(cities_data, "temperature_night",
                                         "temperature_night_p10", "temperature_night_p90",
                                         "Nighttime Temperature", "°C")

            st.subheader("☀️ Daytime Humidity (%)")
            st.caption("Mean relative humidity during daylight hours, with 10th–90th percentile range.")
            plot_metric_with_percentiles(cities_data, "humidity_day",
                                         "humidity_day_p10", "humidity_day_p90",
                                         "Daytime Humidity", "%")

            st.subheader("🌙 Nighttime Humidity (%)")
            st.caption("Mean relative humidity during nighttime hours, with 10th–90th percentile range.")
            plot_metric_with_percentiles(cities_data, "humidity_night",
                                         "humidity_night_p10", "humidity_night_p90",
                                         "Nighttime Humidity", "%")

            st.subheader("🌧️ Monthly Precipitation (mm)")
            st.caption("Mean total precipitation for each month, with 10th–90th percentile range of monthly totals.")
            plot_metric_with_percentiles(cities_data, "precipitation_sum",
                                         "precipitation_sum_p10", "precipitation_sum_p90",
                                         "Precipitation", "mm")

            st.subheader("🌞 Monthly Sunshine Hours")
            st.caption("Mean total sunshine hours for each month, with 10th–90th percentile range of monthly totals.")
            plot_metric_with_percentiles(cities_data, "sunshine_hours",
                                         "sunshine_hours_p10", "sunshine_hours_p90",
                                         "Sunshine", "Hours")

            st.subheader("🌡️ Temperature Extremes by Month (°C)")
            st.caption("Absolute highest and lowest hourly temperatures recorded for each calendar month in the selected period.")
            def plot_min_max_temperature():
                fig = go.Figure()
                for city, monthly_df, color in cities_data:
                    x = monthly_df["time"]
                    y_min = monthly_df.get("temperature_min_absolute")
                    y_max = monthly_df.get("temperature_max_absolute")
                    if y_min is None or y_max is None:
                        st.info(f"Temperature extremes data not available for {city}.")
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y_min, mode='lines+markers',
                                             name=f"{city} Coldest", line=dict(color=color, dash='dot')))
                    fig.add_trace(go.Scatter(x=x, y=y_max, mode='lines+markers',
                                             name=f"{city} Hottest", line=dict(color=color, dash='dash')))
                fig.update_layout(title="Absolute Temperature Extremes by Month",
                                  xaxis_title="Month", yaxis_title="°C", height=400)
                st.plotly_chart(fig, use_container_width=True)
            plot_min_max_temperature()

            st.subheader("🌡️ All-Time Temperature Records")
            st.caption("Single hottest and coldest hourly readings recorded across the entire selected date range.")
            abs_min_max = []
            for city, df in zip([city1, city2] + ([city3] if city3 else []),
                                 [df1, df2] + ([df3] if city3 else [])):
                if df is not None:
                    min_temp = df["temperature_2m"].min()
                    max_temp = df["temperature_2m"].max()
                    min_date = df.loc[df["temperature_2m"].idxmin(), "time"]
                    max_date = df.loc[df["temperature_2m"].idxmax(), "time"]
                    abs_min_max.append({
                        "City": city,
                        "Record Low (°C)": min_temp,
                        "Date": min_date.strftime('%Y-%m-%d %H:%M'),
                        "Record High (°C)": max_temp,
                        "Date ": max_date.strftime('%Y-%m-%d %H:%M'),
                    })
            abs_min_max_df = pd.DataFrame(abs_min_max)
            with st.expander("View Record Data Table"):
                st.table(abs_min_max_df)
            def plot_abs_min_max():
                fig = go.Figure()
                for i, row in abs_min_max_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["City"]], y=[row["Record High (°C)"]],
                        name=f"{row['City']} Record High", marker_color='red',
                        text=[row["Date "]]
                    ))
                    fig.add_trace(go.Bar(
                        x=[row["City"]], y=[row["Record Low (°C)"]],
                        name=f"{row['City']} Record Low", marker_color='blue',
                        text=[row["Date"]]
                    ))
                fig.update_layout(barmode='group', title="All-Time Temperature Records",
                                  xaxis_title="City", yaxis_title="°C", height=400)
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

            metrics_to_predict = [
                {"label": "Daytime Temp Mean", "unit": "°C", "p_label": "Daytime Temp 10th-90th",
                 "mean": "temperature_day", "p10": "temperature_day_p10", "p90": "temperature_day_p90", "format": ".1f"},
                {"label": "Nighttime Temp Mean", "unit": "°C", "p_label": "Nighttime Temp 10th-90th",
                 "mean": "temperature_night", "p10": "temperature_night_p10", "p90": "temperature_night_p90", "format": ".1f"},
                {"label": "Daytime Humidity Mean", "unit": "%", "p_label": "Daytime Humidity 10th-90th",
                 "mean": "humidity_day", "p10": "humidity_day_p10", "p90": "humidity_day_p90", "format": ".1f"},
                {"label": "Nighttime Humidity Mean", "unit": "%", "p_label": "Nighttime Humidity 10th-90th",
                 "mean": "humidity_night", "p10": "humidity_night_p10", "p90": "humidity_night_p90", "format": ".1f"},
                {"label": "Precipitation (mm, avg monthly)", "unit": "", "mean": "precipitation_sum", "format": ".1f"},
                {"label": "Sunshine (hours, avg monthly)", "unit": "", "mean": "sunshine_hours", "format": ".1f"},
            ]

            prediction_rows = []
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
