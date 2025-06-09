import streamlit as st
from datetime import datetime, time, timedelta
import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import requests
import calendar
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import os

# Setup
st.set_page_config(page_title="Weather Comparison", layout="wide")
st.title("🌦️ Compare Historical Weather Between Two Cities")
st.text('Source: https://open-meteo.com')

# Persistent geocoding cache
CITY_CACHE_FILE = 'city_cache.csv'
def get_city_latlon(city_name, geolocator):
    # Check if cache file exists and city is cached
    if os.path.exists(CITY_CACHE_FILE):
        cache = pd.read_csv(CITY_CACHE_FILE)
        match = cache[cache['city'].str.lower() == city_name.lower()]
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
        city1 = st.text_input("City 1", "Paris", key="city1")
        city2 = st.text_input("City 2", "Madrid", key="city2")
        add_third_city = st.checkbox("Add a third city?")
        city3 = None
        if add_third_city:
            city3 = st.text_input("City 3", "Berlin", key="city3")
        # Show resolved addresses for each city
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

    # Convert to datetime.datetime if needed
    if isinstance(start, datetime):
        start_dt = start
    else:
        start_dt = datetime.combine(start, time.min)

    if isinstance(end, datetime):
        end_dt = end
    else:
        end_dt = datetime.combine(end, time.min)

    # Fetch weather data from Open-Meteo using cached lat/lon
    @st.cache_data
    def get_open_meteo_data_by_latlon(city_name, lat, lon, start, end):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        daily_url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_str}&end_date={end_str}"
            f"&daily=precipitation_sum,relative_humidity_2m_mean,sunshine_duration,temperature_2m_mean,temperature_2m_min,temperature_2m_max"
            f"&timezone=auto"
        )
        daily_resp = requests.get(daily_url)
        if daily_resp.status_code != 200:
            return None, f"Open-Meteo API error (daily): {daily_resp.status_code}", None
        daily_data = daily_resp.json()
        if not daily_data.get("daily"):
            return None, f"No data available for {city_name} in this period.", None
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
        if "temperature_2m_min" in df.columns:
            agg_dict["temperature_2m_min"] = ["mean"]
        if "temperature_2m_max" in df.columns:
            agg_dict["temperature_2m_max"] = ["mean"]
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
        if "temperature_2m_min" in monthly_df.columns:
            agg_stats["temperature_2m_min"] = 'mean'
        if "temperature_2m_max" in monthly_df.columns:
            agg_stats["temperature_2m_max"] = 'mean'
        monthly_by_month = monthly_df.groupby('month').agg(agg_stats).reset_index()
        monthly_by_month['time'] = monthly_by_month['month'].apply(lambda m: calendar.month_abbr[m])
        # Compute min/max daily mean temperature for each month (across all years)
        if "temperature_2m_mean" in df.columns:
            df['month'] = df['time'].dt.month
            min_by_month = df.groupby('month')['temperature_2m_mean'].min()
            max_by_month = df.groupby('month')['temperature_2m_mean'].max()
            monthly_by_month['temperature_2m_mean_min'] = monthly_by_month['month'].map(min_by_month)
            monthly_by_month['temperature_2m_mean_max'] = monthly_by_month['month'].map(max_by_month)
        if "temperature_2m_min_mean" in monthly_df.columns:
            monthly_df["temperature_2m_min"] = monthly_df["temperature_2m_min_mean"]
        if "temperature_2m_max_mean" in monthly_df.columns:
            monthly_df["temperature_2m_max"] = monthly_df["temperature_2m_max_mean"]
        return monthly_by_month, None, df

    # Prepare city data with lat/lon
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
        # For plotting, build a list of (city, data, color)
        cities_data = [(city1, data1, 'blue'), (city2, data2, 'red')]
        if city3 and data3 is not None:
            cities_data.append((city3, data3, 'green'))

        # Show map with city dots at the top of the main page
        if city_locations:
            map_df = pd.DataFrame(city_locations)
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size = 1000,use_container_width=False, width=400, height=200)

        # Tabs for plots and prediction
        plots_tab, prediction_tab = st.tabs(["📊 Plots", "🔮 Prediction"])

        with plots_tab:
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

            st.subheader("💧 Average Humidity (%)")
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

            st.subheader("🌧️ Monthly Total Precipitation (mm)")
            def plot_comparison(metric, title, unit, round_decimals=None, show_total=False):
                fig = go.Figure()
                totals = []
                for city, monthly_df, color in cities_data:
                    x = monthly_df["time"]
                    y = monthly_df[metric]
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f"{city}", line=dict(color=color, width=2)))
                    if show_total:
                        total = y.sum()
                        totals.append((city, total))
                fig.update_layout(title=title, xaxis_title="Month", yaxis_title=unit, height=400)
                st.plotly_chart(fig, use_container_width=True)
                if show_total and totals:
                    st.markdown("**Total for average year:**")
                    for city, total in totals:
                        st.write(f"{city}: {total:.1f} {unit}")

            plot_comparison("precipitation_sum", "Precipitation", "mm", round_decimals=1, show_total=True)

            st.subheader("🌞 Total Sunshine Hours")
            if all("sunshine_hours" in df.columns for _, df, _ in cities_data):
                plot_comparison("sunshine_hours", "Monthly Sunshine Duration", "Hours", round_decimals=1, show_total=True)
            else:
                st.info("Sunshine duration data not available for one or more cities.")

            st.subheader("🌡️ Monthly Min/Max Temperature (°C)")
            def plot_min_max_temperature():
                fig = go.Figure()
                for city, monthly_df, color in cities_data:
                    x = monthly_df["time"]
                    y_min = monthly_df["temperature_2m_mean_min"] if "temperature_2m_mean_min" in monthly_df.columns else None
                    y_max = monthly_df["temperature_2m_mean_max"] if "temperature_2m_mean_max" in monthly_df.columns else None
                    if y_min is None or y_max is None:
                        st.info(f"Min/Max temperature data not available for {city}.")
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y_min, mode='lines+markers', name=f"{city} Min", line=dict(color=color, dash='dot')))
                    fig.add_trace(go.Scatter(x=x, y=y_max, mode='lines+markers', name=f"{city} Max", line=dict(color=color, dash='dash')))
                fig.update_layout(title="Min/Max Temperature (from daily means)", xaxis_title="Month", yaxis_title="°C", height=400)
                st.plotly_chart(fig, use_container_width=True)
            plot_min_max_temperature()

            st.subheader("🌡️ Absolute Hottest and Coldest Days (Daily Mean Temperature)")
            abs_min_max = []
            for city, df in zip([city1, city2] + ([city3] if city3 else []), [df1, df2] + ([df3] if city3 else [])):
                if df is not None:
                    min_temp = df["temperature_2m_mean"].min()
                    max_temp = df["temperature_2m_mean"].max()
                    min_date = df.loc[df["temperature_2m_mean"].idxmin(), "time"]
                    max_date = df.loc[df["temperature_2m_mean"].idxmax(), "time"]
                    abs_min_max.append({
                        "City": city,
                        "Coldest (°C)": min_temp,
                        "Coldest Date": min_date.strftime('%Y-%m-%d'),
                        "Hottest (°C)": max_temp,
                        "Hottest Date": max_date.strftime('%Y-%m-%d')
                    })
            abs_min_max_df = pd.DataFrame(abs_min_max)
            st.table(abs_min_max_df)
            def plot_abs_min_max():
                fig = go.Figure()
                for i, row in abs_min_max_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["City"]],
                        y=[row["Hottest (°C)"]],
                        name=f"{row['City']} Hottest",
                        marker_color='red',
                        text=[f"{row['Hottest Date']}"]
                    ))
                    fig.add_trace(go.Bar(
                        x=[row["City"]],
                        y=[row["Coldest (°C)"]],
                        name=f"{row['City']} Coldest",
                        marker_color='blue',
                        text=[f"{row['Coldest Date']}"]
                    ))
                fig.update_layout(barmode='group', title="Absolute Hottest and Coldest Daily Mean Temperatures", xaxis_title="City", yaxis_title="°C", height=400)
                st.plotly_chart(fig, use_container_width=True)
            plot_abs_min_max()

        with prediction_tab:
            tomorrow = datetime.today().date() + timedelta(days=1)
            prediction_date = st.date_input("Prediction date", tomorrow, key="prediction_date")
            st.header(f"Prediction for {prediction_date.strftime('%B %d')}")
            pred_month = prediction_date.month
            prediction_rows = []
            for city, monthly_df, color in cities_data:
                pred_row = monthly_df[monthly_df['month'] == pred_month]
                if pred_row.empty:
                    prediction_rows.append({
                        "City": city,
                        "Temperature Mean (°C)": "No data",
                        "Temperature 10th-90th (°C)": "No data",
                        "Humidity Mean (%)": "No data",
                        "Humidity 10th-90th (%)": "No data",
                        "Precipitation (mm, daily avg)": "No data",
                        "Sunshine (hours, daily avg)": "No data"
                    })
                    continue
                # Temperature
                temp = pred_row["temperature_2m_mean"].values[0] if "temperature_2m_mean" in pred_row else None
                temp_p10 = pred_row["temperature_2m_mean_p10"].values[0] if "temperature_2m_mean_p10" in pred_row else None
                temp_p90 = pred_row["temperature_2m_mean_p90"].values[0] if "temperature_2m_mean_p90" in pred_row else None
                # Humidity
                hum = pred_row["relative_humidity_2m_mean"].values[0] if "relative_humidity_2m_mean" in pred_row else None
                hum_p10 = pred_row["relative_humidity_2m_mean_p10"].values[0] if "relative_humidity_2m_mean_p10" in pred_row else None
                hum_p90 = pred_row["relative_humidity_2m_mean_p90"].values[0] if "relative_humidity_2m_mean_p90" in pred_row else None
                # Precipitation (mean daily)
                prcp = pred_row["precipitation_sum"].values[0] if "precipitation_sum" in pred_row else None
                # Sunshine (mean daily)
                sun = pred_row["sunshine_hours"].values[0] if "sunshine_hours" in pred_row else None
                # Days in month
                days_in_month = calendar.monthrange(prediction_date.year, pred_month)[1]
                prcp_daily = prcp / days_in_month if prcp is not None else None
                sun_daily = sun / days_in_month if sun is not None else None
                prediction_rows.append({
                    "City": city,
                    "Temperature Mean (°C)": f"{temp:.1f}" if temp is not None else "No data",
                    "Temperature 10th-90th (°C)": f"{temp_p10:.1f}–{temp_p90:.1f}" if temp_p10 is not None and temp_p90 is not None else "No data",
                    "Humidity Mean (%)": f"{hum:.1f}" if hum is not None else "No data",
                    "Humidity 10th-90th (%)": f"{hum_p10:.1f}–{hum_p90:.1f}" if hum_p10 is not None and hum_p90 is not None else "No data",
                    "Precipitation (mm, daily avg)": f"{prcp_daily:.2f}" if prcp_daily is not None else "No data",
                    "Sunshine (hours, daily avg)": f"{sun_daily:.2f}" if sun_daily is not None else "No data"
                })
            prediction_df = pd.DataFrame(prediction_rows)
            st.table(prediction_df)
