"""Whiff Scout tiny local web server (stdlib only, no new dependencies).

Run from the whiff-scout/ subfolder:
    ../.venv/bin/python -m src.server
Then open http://127.0.0.1:8000 -- type a pitcher name, get the HTML report.
Local demo use only: binds loopback (127.0.0.1), no auth, no public exposure.
"""

import html as htmlmod
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.report import build_report  # noqa: E402

INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>Whiff Scout</title></head>
<body><h1>Whiff Scout</h1>
<p>Type a pitcher name, get a data-backed scouting report.</p>
<form action="/report" method="get">
<input name="name" placeholder="e.g. Drew Rasmussen" size="30" autofocus>
<button type="submit">Scout</button>
</form>
<p>Try: Drew Rasmussen, Jacob Misiorowski</p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, INDEX.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/report":
            name = (parse_qs(parsed.query).get("name") or [""])[0].strip()
            if not name:
                self._send(400, b"Missing pitcher name.", "text/plain")
                return
            try:
                path, _result, _log = build_report(name)
                with open(path, "rb") as f:
                    body = f.read()
            except Exception as exc:  # unknown pitcher, missing cache, etc.
                msg = ("Could not build a report for "
                       f"{htmlmod.escape(name)}: {htmlmod.escape(str(exc))}").encode()
                self._send(404, msg, "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        self._send(404, b"Not found.", "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("WHIFF_SCOUT_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Whiff Scout running at http://127.0.0.1:{port} (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
