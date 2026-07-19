"""A deterministic replacement for an external benchmark process."""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    args = parser.parse_args()
    time.sleep(args.sleep_seconds)
    print(json.dumps({"label": args.label, "requests": 4, "throughput": 12.5}, sort_keys=True))
    if args.fail:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
