# 🌤️ City Weather Comparison App

Compare historical weather — temperature, humidity, sunshine hours and rainfall —
between two (or optionally three) cities. Useful for travellers, researchers, or
anyone curious about how climates line up across regions.

The app runs on **Vercel**: a static frontend (vanilla JS + Plotly.js) talks to
Python serverless functions that reduce Open-Meteo data to small summaries. The
original **Streamlit** version is still in the repo as [`app.py`](app.py).

## 🚀 Features

- 📍 Compare 2 or 3 cities side-by-side
- 📅 Pick a date range for the historical baseline
- 📊 Monthly temperature, humidity, rainfall and sunshine, with 10th–90th percentile bands
- ☀️ Daytime-only temperatures, using just the hours between local sunrise and sunset
- 🌡️ How the current year is running day by day against the historical average, shaded red where warmer and blue where cooler
- 🌧️ This year's accumulated rainfall against a typical year's, as a running surplus or shortfall
- 🌐 Cities worldwide, geocoded via the Open-Meteo geocoding API

## 🧰 Tech Stack

- Frontend: static HTML/CSS + vanilla JS, charts by [Plotly.js](https://plotly.com/javascript/)
- API: [Vercel Python functions](https://vercel.com/docs/functions/runtimes/python) (`api/*.py`)
- Data crunching: [pandas](https://pandas.pydata.org/) in [`weatherlib/`](weatherlib/), shared with the Streamlit app
- Weather + geocoding: [Open-Meteo](https://open-meteo.com) (the Streamlit `app.py` still geocodes via Nominatim/geopy)

## 🖥️ Run it locally

```bash
pip install -r requirements.txt
python dev_server.py            # → http://localhost:3000
```

`dev_server.py` serves `public/` and routes `/api/*` through the same functions
Vercel runs. It is not deployed.

<details>
<summary>Running the original Streamlit app instead</summary>

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```
</details>

## ▲ Deploy to Vercel

No build step. With the [Vercel CLI](https://vercel.com/docs/cli):

```bash
vercel            # preview deploy
vercel --prod     # production
```

Or import the GitHub repo at [vercel.com/new](https://vercel.com/new) — the defaults
work (framework preset: **Other**). [`vercel.json`](vercel.json) sets the function
timeout to 60s and bundles the committed `.csv` caches into the function.

**What changes on Vercel:** the deployment filesystem is read-only, so a
newly-looked-up city is fetched live but not written back to the cache — it is
re-fetched next time. Cities covered by the warm-cache job (below) load with no
live API calls at all. Fully-successful responses carry an `s-maxage` header so
the CDN absorbs repeat traffic.

**About rate limits (429):** Open-Meteo throttles bursts of multi-decade
requests hard, and Vercel functions share egress IPs across all tenants, so live
fetching from the app is unreliable. The real fix is the warm-cache job — the
app should almost never fetch live. When it does: geocoding uses Open-Meteo, not
Nominatim (whose policy forbids cloud traffic); the archive fetch retries 429s
with backoff; the frontend staggers per-city requests; partial responses aren't
CDN-cached.

### Warm-cache job

[`scripts/warm_cache.py`](scripts/warm_cache.py) fetches Open-Meteo data ahead of
time and writes it into `weather_cache/`, so the deployed app serves known cities
entirely from committed files. It warms, per city:

- the historical baseline (`2010-01-01` → end of last full year) — skipped when already on disk;
- the current-year range (`Jan 1` → today) — refreshed each run, superseding yesterday's files.

[`.github/workflows/warm-cache.yml`](.github/workflows/warm-cache.yml) runs it
from GitHub's runners (not Vercel's shared IPs) and commits the result — each run
triggers one Vercel redeploy:

- **daily** (05:17 UTC) — current-year ranges only; keeps the comparison charts fresh;
- **weekly** (Mondays) — full run, picking up the historical baseline for any newly added city;
- **manual** (Actions → *Run workflow*) — full run; pass names to warm just those.

The city list is [`scripts/cities.txt`](scripts/cities.txt) (a curated set of
~70 world cities) plus everything already in `city_cache.csv` — add a line to
`cities.txt` and the next weekly (or manual) run resolves and caches it. There's
a ~5-hour window each day (after midnight UTC, before the job runs) where the
current-year range shifts by a day and the app fetches that one range live.

## 📁 Layout

```
├── public/             # Static frontend served at /
│   ├── index.html
│   ├── app.js          # Form handling + all the Plotly charts
│   └── styles.css
├── api/                # Vercel serverless functions
│   ├── weather.py      # GET /api/weather  — one city's full comparison payload
│   ├── geocode.py      # GET /api/geocode  — city name → lat/lon
│   └── _util.py        # shared request/response helpers (not a route)
├── weatherlib/         # Pure pandas logic, shared by the API and app.py
│   ├── aggregate.py    # daily/hourly series → the reduced summaries
│   ├── openmeteo.py    # Open-Meteo URL builders + fetch (with 429 retry)
│   ├── cache.py        # read/write access to weather_cache/
│   ├── geocode.py      # city_cache.csv, then Open-Meteo geocoding
│   └── service.py      # orchestration: assemble one city's payload
├── scripts/warm_cache.py         # pre-generate weather_cache/ entries
├── scripts/cities.txt            # curated city list for the warm job
├── .github/workflows/warm-cache.yml  # runs it daily / weekly
├── app.py              # Original Streamlit app (still works)
├── dev_server.py       # Local static + /api server
├── city_cache.csv      # Cached city → lat/lon
├── weather_cache/      # Cached Open-Meteo summaries (see below)
│   └── recent/         # Current-year ranges, refreshed by the warm-cache job
├── requirements.txt              # API / weatherlib deps (pandas, requests)
├── requirements-streamlit.txt    # + streamlit, plotly, geopy (for app.py)
└── vercel.json
```

### Caching

The caches are committed to the repo, so cities the warm-cache job covers load
without hitting Open-Meteo at all. Deleting a cache file just means it gets
re-fetched. Only **reduced summaries** are stored, never the raw daily or hourly
series the API returns — those run to megabytes per city and are one request away:

| File | Contents |
|---|---|
| `*_monthly.csv` | 12 rows: per-calendar-month means and percentiles |
| `*_records.csv` | 1 row: the hottest and coldest day of the range, with dates |
| `*_daytime.csv` | 12 rows: per-calendar-month daylight-only temperature stats |
| `*_daytime_climatology.csv` | 366 rows: average daytime temperature per day of the year |
| `*_precip_climatology.csv` | 366 rows: average accumulated rainfall by each day of the year |
| `*_daytime_daily.csv`, `*_precip_daily.csv` | one row per date — only for `recent/` (current-year) ranges, where they are what the comparison charts plot |

Ranges ending within a week of today go to `weather_cache/recent/` under a
filename that includes the end date, so they change every day. The warm-cache
job rewrites them and prunes the superseded copies; a local Streamlit run may
also leave files there (safe to `git checkout`).

## ⚠️ Notes

- Cities are matched by a proximity search of weather stations; results vary with data availability.
- Some metrics (e.g. sunshine hours) may be missing for certain stations.
- Long historical ranges for an uncached city trigger a multi-decade hourly fetch, which can take a few seconds.

## 📸 Screenshot

![Weather comparison screenshot](screenshots/screenshot.png)
