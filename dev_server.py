#!/usr/bin/env python3
"""Local dev server -- `python dev_server.py`, then open http://localhost:3000

Serves ``public/`` statically and routes ``/api/*`` through the exact same
functions Vercel runs (``api/weather.py`` / ``api/geocode.py``). This file is
never deployed; it only exists so the site can be worked on without the Vercel
CLI. Install deps first: ``pip install -r requirements.txt``.
"""
import importlib.util
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, ROOT)


def _load(module_name, filename):
    path = os.path.join(ROOT, "api", filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the endpoints' own ``from _util import ...`` reuses
    # this instance instead of loading a second copy with its own exception types.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_util = _load("_util", "_util.py")
_weather = _load("_endpoint_weather", "weather.py")
_geocode = _load("_endpoint_geocode", "geocode.py")

ROUTES = {
    "/api/weather": _weather.run,
    "/api/geocode": _geocode.run,
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC, **kwargs)

    def do_GET(self):
        route = ROUTES.get(urlparse(self.path).path)
        if route is None:
            return super().do_GET()
        status, payload = _util.resolve(route, _util.query_params(self.path))
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"Serving {PUBLIC} and /api/* on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
