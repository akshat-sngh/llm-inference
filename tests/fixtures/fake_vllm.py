#!/usr/bin/env python3
"""Deterministic vLLM-shaped CLI for integration tests without vLLM or a GPU."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def argument_value(arguments: list[str], flag: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{flag}="):
            return argument.split("=", 1)[1]
    return None


def serve(arguments: list[str]) -> None:
    host = argument_value(arguments, "--host") or "127.0.0.1"
    port = int(argument_value(arguments, "--port") or "8000")
    stopped = Event()
    server = HTTPServer((host, port), HealthHandler)
    server.timeout = 0.05
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    while not stopped.is_set():
        server.handle_request()
    server.server_close()


def bench(arguments: list[str]) -> None:
    mode = os.environ.get("FAKE_VLLM_MODE", "success")
    if mode == "bench-timeout":
        time.sleep(5)
    result_dir = argument_value(arguments, "--result-dir")
    filename = argument_value(arguments, "--result-filename")
    if result_dir is None or filename is None:
        if os.environ.get("FAKE_VLLM_WARMUP_FAIL") == "1":
            raise SystemExit(7)
        print('{"warmup": "ok"}')
        return
    path = Path(result_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode != "missing-result":
        if mode == "invalid-result":
            path.write_text("not-json")
        else:
            path.write_text(
                json.dumps({"completed": 64, "request_throughput": 12.5}, sort_keys=True)
            )
    if mode in {"bench-fail", "partial-result"}:
        raise SystemExit(7)
    print('{"benchmark": "ok"}')


def main() -> None:
    arguments = sys.argv[1:]
    if arguments == ["--version"]:
        print("fake-vllm 0.0.1")
        return
    if arguments == ["collect-env"]:
        print("Fake vLLM environment")
        return
    if "--help" in arguments:
        print("fake vllm help")
        return
    if arguments and arguments[0] == "serve":
        serve(arguments[1:])
        return
    if arguments[:2] == ["bench", "serve"]:
        bench(arguments[2:])
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
