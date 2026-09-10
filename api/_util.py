"""Shared helpers for the Vercel Python endpoints.

Files whose name starts with ``_`` are not turned into routes by Vercel, so this
is a plain import target for ``api/weather.py``, ``api/geocode.py`` and the local
``dev_server.py``.
"""
import json
import os
import sys
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# The endpoints import the top-level ``weatherlib`` package; make sure the repo
# root is importable regardless of where the runtime starts us.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weatherlib import geocode  # noqa: E402
from weatherlib.openmeteo import OpenMeteoError, OpenMeteoRateLimited  # noqa: E402


class BadRequest(Exception):
    """A 400: the client sent something we can't use."""


def query_params(path):
    """Return the query string of a request path as a flat ``{key: value}`` dict."""
    raw = parse_qs(urlparse(path).query)
    return {key: values[0] for key, values in raw.items() if values}


def parse_date(value, field):
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (TypeError, ValueError):
        raise BadRequest(f"'{field}' must be a date in YYYY-MM-DD format.")


def resolve(fn, params):
    """Run an endpoint body, mapping known exceptions to ``(status, payload)``."""
    try:
        return 200, fn(params)
    except BadRequest as exc:
        return 400, {'error': str(exc)}
    except geocode.GeocodeError as exc:
        return 404, {'error': str(exc)}
    except geocode.GeocodeUnavailable as exc:
        return 503, {'error': str(exc)}
    except OpenMeteoRateLimited as exc:
        return 429, {'error': str(exc)}
    except OpenMeteoError as exc:
        return 502, {'error': str(exc)}
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the function
        return 500, {'error': f'Unexpected error: {exc}'}


def _cache_header(status, cache_seconds):
    if status == 200 and cache_seconds:
        return (
            f'public, max-age=0, s-maxage={cache_seconds}, '
            f'stale-while-revalidate=86400'
        )
    return 'no-store'


def send(handler, status, payload, cache_seconds=0):
    body = json.dumps(payload).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', _cache_header(status, cache_seconds))
    handler.end_headers()
    handler.wfile.write(body)
