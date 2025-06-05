import streamlit as st
#from meteostat import Stations, Monthly
from datetime import datetime, time
import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import requests
import calendar

# Setup
st.set_page_config(page_title="Weather Comparison", layout="wide")
st.title("🌦️ Compare Historical Weather Between Two Cities")

# Date and city selection in sidebar
with st.sidebar:
    st.header("Settings")
    start = st.date_input("Start date", datetime(2010, 1, 1), key="start_date")
    end = st.date_input("End date", datetime(2024, 12, 31), key="end_date")
    city1 = st.text_input("City 1", "Paris", key="city1")
    city2 = st.text_input("City 2", "Madrid", key="city2")
    add_third_city = st.checkbox("Add a third city?")
    city3 = None
    if add_third_city:
        city3 = st.text_input("City 3", "Berlin", key="city3")

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

# Geocoding function
@st.cache_data
def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="streamlit-weather-app")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

# Fetch weather data from Open-Meteo
@st.cache_data
def get_open_meteo_data(city_name, start, end):
    coords = get_coordinates(city_name)
    if not coords:
        return None, f"Could not find location for {city_name}"
    lat, lon = coords
    # Open-Meteo API expects YYYY-MM-DD
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    daily_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&daily=precipitation_sum,relative_humidity_2m_mean,sunshine_duration,temperature_2m_mean"
        f"&timezone=auto"
    )
    daily_resp = requests.get(daily_url)
    if daily_resp.status_code != 200:
        return None, f"Open-Meteo API error (daily): {daily_resp.status_code}"
    daily_data = daily_resp.json()
    if not daily_data.get("daily"):
        return None, f"No data available for {city_name} in this period."
    df = pd.DataFrame(daily_data["daily"])
    # Convert sunshine_duration from seconds to hours
    if "sunshine_duration" in df.columns:
        df["sunshine_hours"] = df["sunshine_duration"] / 3600
    # Convert 'time' to datetime
    df["time"] = pd.to_datetime(df["time"])
    # Aggregate to monthly with only sum for precipitation and sunshine, mean for humidity, and for temperature: mean, 10th, 90th percentiles
    agg_dict = {}
    if "precipitation_sum" in df.columns:
        agg_dict["precipitation_sum"] = ["sum"]
    if "sunshine_hours" in df.columns:
        agg_dict["sunshine_hours"] = ["sum"]
    if "relative_humidity_2m_mean" in df.columns:
        agg_dict["relative_humidity_2m_mean"] = ["mean", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]
    if "temperature_2m_mean" in df.columns:
        agg_dict["temperature_2m_mean"] = ["mean", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]
    monthly_df = df.groupby(df["time"].dt.to_period("M")).agg(agg_dict)
    # Flatten MultiIndex columns
    def colname(col):
        if isinstance(col[1], str):
            return f"{col[0]}_{col[1]}"
        # For quantiles, use p10/p90
        if callable(col[1]):
            # Use function name or quantile value
            return f"{col[0]}_p10" if '0.1' in str(col[1]) else f"{col[0]}_p90"
        return f"{col[0]}_{col[1]}"
    monthly_df.columns = [colname(col) for col in monthly_df.columns]
    monthly_df = monthly_df.reset_index()
    monthly_df["time"] = monthly_df["time"].dt.to_timestamp()
    # For convenience, also add the main column for plotting
    if "precipitation_sum_sum" in monthly_df.columns:
        monthly_df["precipitation_sum"] = monthly_df["precipitation_sum_sum"]
    if "sunshine_hours_sum" in monthly_df.columns:
        monthly_df["sunshine_hours"] = monthly_df["sunshine_hours_sum"]
    if "relative_humidity_2m_mean_mean" in monthly_df.columns:
        monthly_df["relative_humidity_2m_mean"] = monthly_df["relative_humidity_2m_mean_mean"]
    if "relative_humidity_2m_mean_<lambda_0>" in monthly_df.columns:
        monthly_df["relative_humidity_2m_mean_p10"] = monthly_df["relative_humidity_2m_mean_<lambda_0>"]
    if "relative_humidity_2m_mean_<lambda_1>" in monthly_df.columns:
        monthly_df["relative_humidity_2m_mean_p90"] = monthly_df["relative_humidity_2m_mean_<lambda_1>"]
    # Robust assignment for temperature mean
    temp_mean_col = [col for col in monthly_df.columns if col.startswith("temperature_2m_mean") and ("mean" in col and "<lambda" not in col)]
    if temp_mean_col:
        monthly_df["temperature_2m_mean"] = monthly_df[temp_mean_col[0]]
    if "temperature_2m_mean_<lambda_0>" in monthly_df.columns:
        monthly_df["temperature_2m_mean_p10"] = monthly_df["temperature_2m_mean_<lambda_0>"]
    if "temperature_2m_mean_<lambda_1>" in monthly_df.columns:
        monthly_df["temperature_2m_mean_p90"] = monthly_df["temperature_2m_mean_<lambda_1>"]
    # Further aggregate by calendar month (across all years)
    monthly_df['month'] = monthly_df['time'].dt.month
    agg_stats = {}
    if "precipitation_sum" in monthly_df.columns:
        agg_stats["precipitation_sum"] = 'mean'
    if "sunshine_hours" in monthly_df.columns:
        agg_stats["sunshine_hours"] = 'mean'
    if "relative_humidity_2m_mean" in monthly_df.columns:
        agg_stats["relative_humidity_2m_mean"] = 'mean'
    if "relative_humidity_2m_mean_p10" in monthly_df.columns:
        agg_stats["relative_humidity_2m_mean_p10"] = 'mean'
    if "relative_humidity_2m_mean_p90" in monthly_df.columns:
        agg_stats["relative_humidity_2m_mean_p90"] = 'mean'
    if "temperature_2m_mean" in monthly_df.columns:
        agg_stats["temperature_2m_mean"] = 'mean'
    if "temperature_2m_mean_p10" in monthly_df.columns:
        agg_stats["temperature_2m_mean_p10"] = 'mean'
    if "temperature_2m_mean_p90" in monthly_df.columns:
        agg_stats["temperature_2m_mean_p90"] = 'mean'
    monthly_by_month = monthly_df.groupby('month').agg(agg_stats).reset_index()
    monthly_by_month['time'] = monthly_by_month['month'].apply(lambda m: calendar.month_abbr[m])
    return monthly_by_month, None

# Load data
with st.spinner("Fetching weather data from Open-Meteo..."):
    data1, err1 = get_open_meteo_data(city1, start_dt, end_dt)
    data2, err2 = get_open_meteo_data(city2, start_dt, end_dt)
    data3, err3 = (get_open_meteo_data(city3, start_dt, end_dt) if city3 else (None, None))

#st.dataframe(data1)

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
    cities_data = [(city1, data1, 'blue'), (city2, data2, 'red')]
    if city3 and data3 is not None:
        cities_data.append((city3, data3, 'green'))

    st.subheader("🌡️ Monthly Temperature (°C)")
    def plot_temperature():
        fig = go.Figure()
        for city, monthly_df, color in cities_data:
            x = monthly_df["time"]
            if "temperature_2m_mean" not in monthly_df.columns:
                st.info(f"Temperature data not available for {city}.")
                continue
            y = monthly_df["temperature_2m_mean"]
            y_p10 = monthly_df["temperature_2m_mean_p10"] if "temperature_2m_mean_p10" in monthly_df.columns else None
            y_p90 = monthly_df["temperature_2m_mean_p90"] if "temperature_2m_mean_p90" in monthly_df.columns else None
            # Fill between p10 and p90
            if y_p10 is not None and y_p90 is not None:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_p90,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f"{city} 90th percentile"
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_p10,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.1)' if color=='blue' else ('rgba(255,0,0,0.1)' if color=='red' else 'rgba(0,128,0,0.1)'),
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{city} 10th-90th percentile"
                ))
            # Mean line
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f"{city} Mean", line=dict(color=color, width=2)))
        fig.update_layout(title="Temperature (80% Range)", xaxis_title="Month", yaxis_title="°C", height=400)
        st.plotly_chart(fig, use_container_width=True)
    plot_temperature()

    st.subheader("�� Average Humidity (%)")
    def plot_humidity():
        fig = go.Figure()
        for city, monthly_df, color in cities_data:
            x = monthly_df["time"]
            y = monthly_df["relative_humidity_2m_mean"] if "relative_humidity_2m_mean" in monthly_df.columns else None
            y_p10 = monthly_df["relative_humidity_2m_mean_p10"] if "relative_humidity_2m_mean_p10" in monthly_df.columns else None
            y_p90 = monthly_df["relative_humidity_2m_mean_p90"] if "relative_humidity_2m_mean_p90" in monthly_df.columns else None
            if y is None:
                st.info(f"Humidity data not available for {city}.")
                continue
            # Fill between p10 and p90
            if y_p10 is not None and y_p90 is not None:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_p90,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f"{city} 90th percentile"
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y_p10,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.1)' if color=='blue' else ('rgba(255,0,0,0.1)' if color=='red' else 'rgba(0,128,0,0.1)'),
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{city} 10th-90th percentile"
                ))
            # Mean line
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f"{city} Mean", line=dict(color=color, width=2)))
        fig.update_layout(title="Humidity (80% Range)", xaxis_title="Month", yaxis_title="%", height=400)
        st.plotly_chart(fig, use_container_width=True)
    if any("relative_humidity_2m_mean" in df.columns for _, df, _ in cities_data):
        plot_humidity()
    else:
        st.info("Humidity data not available for one or more cities.")

    st.subheader("🌧️ Monthly Precipitation (mm)")
    def plot_comparison(metric, title, unit, round_decimals=None):
        fig = go.Figure()
        for city, monthly_df, color in cities_data:
            x = monthly_df["time"]
            y = monthly_df[metric]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f"{city}", line=dict(color=color, width=2)))
        fig.update_layout(title=title, xaxis_title="Month", yaxis_title=unit, height=400)
        st.plotly_chart(fig, use_container_width=True)
    plot_comparison("precipitation_sum", "Precipitation", "mm", round_decimals=1)

    st.subheader("🌞 Sunshine Hours")
    if all("sunshine_hours" in df.columns for _, df, _ in cities_data):
        plot_comparison("sunshine_hours", "Monthly Sunshine Duration", "Hours", round_decimals=1)
    else:
        st.info("Sunshine duration data not available for one or more cities.")