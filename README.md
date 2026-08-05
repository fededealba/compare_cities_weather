# 🌤️ City Weather Comparison App

This Streamlit app allows you to compare historical weather data — including temperature, humidity, sunshine hours, and rainfall — between two (or optionally three) cities. It's ideal for travelers, researchers, or anyone curious about climate patterns across different regions.

## 🚀 Features

- 📍 Compare 2 or 3 cities side-by-side  
- 📅 Select a date range for historical analysis  
- 📊 View temperature, humidity, rainfall, and sunshine hours  
- ☀️ Compare daytime-only temperatures, using just the hours between local sunrise and sunset  
- 🌡️ See how the current year is running day by day against the historical average, shaded red where it is warmer and blue where cooler  
- 🌧️ Track this year's accumulated rainfall against a typical year's, as a running surplus or shortfall  
- 🌐 Supports cities worldwide (based on OpenStreetMap Nominatim)  

## 🖥️ Live Demo

https://fededealba-compare-cities-weather-app-zglwfv.streamlit.app

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/python/)
- [Open-Meteo](https://open-meteo.com)

## 🛠️ Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/city-weather-comparison.git
   cd city-weather-comparison
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📁 File Structure

```
├── app.py              # Main Streamlit app
├── city_cache.csv      # Cached city → lat/lon geocoding results
├── weather_cache/      # Cached Open-Meteo summaries, keyed by city, coordinates and date range
│   └── recent/         # Ranges ending at today (gitignored, pruned automatically)
├── screenshots/        # Images used in this README
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Caching

The caches are committed to the repo, so previously-looked-up cities load without
hitting Nominatim or Open-Meteo. Deleting a cache file just means it gets re-fetched.

Only **reduced summaries** are stored, never the raw daily or hourly series the API
returns — those run to megabytes per city and are one request away:

| File | Contents |
|---|---|
| `*_monthly.csv` | 12 rows: per-calendar-month means and percentiles |
| `*_records.csv` | 1 row: the hottest and coldest day of the range, with dates |
| `*_daytime.csv` | 12 rows: per-calendar-month daylight-only temperature stats |
| `*_daytime_climatology.csv` | 366 rows: average daytime temperature per day of the year |
| `*_precip_climatology.csv` | 366 rows: average accumulated rainfall by each day of the year |
| `*_daytime_daily.csv`, `*_precip_daily.csv` | one row per date — only kept for `recent/` ranges, where they are what the charts plot |

Ranges whose end date is within a week of today go to `weather_cache/recent/`. They
would otherwise be rewritten under a new filename every day, so they are gitignored,
and superseded copies are deleted whenever a newer one is written.

## ⚠️ Notes

- Cities are matched based on a proximity search of weather stations. Results may vary depending on data availability.
- Some metrics (e.g. sunshine hours) might be missing for certain stations.

## 📸 Screenshots

![Weather Comparison Screenshot](screenshots/screenshot.png)