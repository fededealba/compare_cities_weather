# 🌤️ City Weather Comparison App

This Streamlit app allows you to compare historical weather data — including temperature, humidity, sunshine hours, and rainfall — between two (or optionally three) cities. It's ideal for travelers, researchers, or anyone curious about climate patterns across different regions.

## 🚀 Features

- 📍 Compare 2 or 3 cities side-by-side  
- 📅 Select a date range for historical analysis  
- 📊 View temperature, humidity, rainfall, and sunshine hours  
- 🌐 Supports cities worldwide (based on NOAA GSOD dataset)  

## 🖥️ Live Demo

To be hosted soon (Streamlit Cloud / EC2 / etc.)

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/python/)
- [NOAA GSOD](https://data.noaa.gov/) or similar weather dataset

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

4. **Environment variables** (if applicable):
   Create a `.env` file to define any required API keys or paths.

## 📁 File Structure

```
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## ⚠️ Notes

- Cities are matched based on a proximity search of weather stations. Results may vary depending on data availability.
- Some metrics (e.g. sunshine hours) might be missing for certain stations.

## 📸 Screenshots

_Add screenshots here if available_

## 📄 License

MIT License — feel free to use and modify!