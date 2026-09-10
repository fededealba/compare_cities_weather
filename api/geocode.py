"""GET /api/geocode?city=Paris  ->  { city, lat, lon, cached }

Resolves a place name to coordinates, checking the committed ``city_cache.csv``
before falling back to Nominatim.
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.dirname(__file__), _os.path.dirname(_os.path.dirname(__file__))]

from http.server import BaseHTTPRequestHandler

from _util import BadRequest, query_params, resolve, send
from weatherlib import geocode

CACHE_SECONDS = 86400


def run(params):
    city = params.get('city', '').strip()
    if not city:
        raise BadRequest("Pass a '?city=' query parameter.")
    lat, lon, cached = geocode.geocode(city)
    return {'city': city, 'lat': lat, 'lon': lon, 'cached': cached}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status, payload = resolve(run, query_params(self.path))
        send(self, status, payload, CACHE_SECONDS)

    def log_message(self, *args):
        pass
