"""A tiny process-friendly HTTP server for local orchestration tests."""

from __future__ import annotations

import argparse
import json
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(self.server.health_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--health-status", type=int, default=200)
    args = parser.parse_args()
    stopped = Event()
    server = HTTPServer((args.host, args.port), Handler)
    server.health_status = args.health_status
    server.timeout = 0.1
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    while not stopped.is_set():
        server.handle_request()
    server.server_close()


if __name__ == "__main__":
    main()
