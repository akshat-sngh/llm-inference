#!/usr/bin/env python3
"""A deterministic stand-in for nvidia-smi used by GPU-independent tests."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _sample_number() -> int:
    counter_path = os.environ.get("FAKE_NVIDIA_COUNTER_FILE")
    if counter_path is None:
        return 0
    path = Path(counter_path)
    previous = int(path.read_text()) if path.exists() else 0
    path.write_text(str(previous + 1))
    return previous


def main() -> None:
    arguments = sys.argv[1:]
    mode = os.environ.get("FAKE_NVIDIA_MODE", "success")
    sleep_seconds = float(os.environ.get("FAKE_NVIDIA_SLEEP_SECONDS", "0"))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    if any(argument == "--id=9" for argument in arguments):
        print("configured GPU does not exist", file=sys.stderr)
        raise SystemExit(6)
    query = next((argument for argument in arguments if argument.startswith("--query-gpu=")), "")
    is_metadata = "driver_version" in query
    if is_metadata:
        if mode == "metadata-fail":
            print("metadata probe failed", file=sys.stderr)
            raise SystemExit(5)
        if mode == "unavailable-fields":
            print("0, N/A, Fake NVIDIA GPU, N/A, N/A, N/A, N/A")
            return
        print("0, GPU-fake-0, Fake NVIDIA GPU, 555.42, 24564, 450.0, P8")
        return

    sample_number = _sample_number()
    fail_after = os.environ.get("FAKE_NVIDIA_FAIL_AFTER")
    if mode == "sample-fail" or (fail_after is not None and sample_number >= int(fail_after)):
        print("sample failed", file=sys.stderr)
        raise SystemExit(7)
    if mode == "unavailable-fields":
        print(
            "2026/07/19 16:00:00, 0, GPU-fake-0, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A"
        )
        return
    print(
        "2026/07/19 16:00:00, 0, GPU-fake-0, 97, 42, 18200, 24564, "
        "331.4, 450.0, 69, 2715, 10501, P2"
    )


if __name__ == "__main__":
    main()
