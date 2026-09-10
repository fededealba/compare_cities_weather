"""GET /api/weather?city=&lat=&lon=&start=YYYY-MM-DD&end=YYYY-MM-DD

Returns the full comparison payload for one city (monthly summaries, records and
the current-year-vs-baseline series). The frontend calls this once per city in
parallel. ``lat``/``lon`` are optional; when absent the city name is geocoded.
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.dirname(__file__), _os.path.dirname(_os.path.dirname(__file__))]

from datetime import date
from http.server import BaseHTTPRequestHandler

from _util import BadRequest, parse_date, query_params, resolve, send
from weatherlib import geocode, service

CACHE_SECONDS = 21600
MIN_DATE = date(1940, 1, 1)


def run(params):
    city = params.get('city', '').strip()
    if not city:
        raise BadRequest("Pass a '?city=' query parameter.")

    start = parse_date(params.get('start'), 'start')
    end = parse_date(params.get('end'), 'end')
    if start < MIN_DATE:
        raise BadRequest(f"'start' cannot be before {MIN_DATE.isoformat()}.")
    end = min(end, date.today())
    if start >= end:
        raise BadRequest("'end' must be after 'start'.")

    if 'lat' in params and 'lon' in params:
        try:
            lat, lon = float(params['lat']), float(params['lon'])
        except ValueError:
            raise BadRequest("'lat' and 'lon' must be numbers.")
    else:
        lat, lon, _ = geocode.geocode(city)

    return service.build_city_payload(city, lat, lon, start, end)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status, payload = resolve(run, query_params(self.path))
        # Only let the CDN hold a fully-successful payload. A partial one (some
        # upstream series rate-limited) should be re-tried on the next visit.
        complete = status == 200 and not payload.get('warnings')
        send(self, status, payload, CACHE_SECONDS if complete else 0)

    def log_message(self, *args):
        pass
