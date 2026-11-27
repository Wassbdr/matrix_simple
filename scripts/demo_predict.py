#!/usr/bin/env python3
"""Send the demo incident to a running STA service and print the ranked services.

Standard library only, so it runs outside the project virtualenv:

    docker compose up --build -d
    python scripts/demo_predict.py [--url http://localhost:8080] [--incident examples/sockshop_demo/incident.json]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _get(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            if _get(f"{base}/ready").get("ready", True):
                return
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        if time.time() > deadline:
            sys.exit(
                f"service at {base} not ready after {timeout_s:.0f}s (is `docker compose up` running?)"
            )
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--incident", default=str(ROOT / "examples/sockshop_demo/incident.json"))
    ap.add_argument("--top-k", type=int, default=14)
    ap.add_argument("--repeats", type=int, default=5, help="requests sent (first one is cold)")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /ready")
    args = ap.parse_args()

    inc = json.loads(Path(args.incident).read_text())
    wait_ready(args.url, args.wait)
    body = json.dumps(
        {
            "window": inc["window"],
            "services": inc["services"],
            "features": inc["features"],
            "top_k": args.top_k,
        }
    ).encode()
    print(
        f"POST {args.url}/predict  ({len(inc['services'])} services x {len(inc['features'])} metrics "
        f"x {len(inc['window'][0][0])} steps)"
    )
    req = urllib.request.Request(
        f"{args.url}/predict", data=body, headers={"Content-Type": "application/json"}
    )
    lat = []
    for _ in range(max(1, args.repeats)):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code}: {e.read().decode()}")
        lat.append(resp["latency_ms"])

    truth = inc.get("true_root_cause")
    warm = sorted(lat[1:])[len(lat[1:]) // 2] if len(lat) > 1 else lat[0]
    print(
        f"model {resp['model_version']}  server latency: first call {lat[0]:.1f} ms, "
        f"then median {warm:.2f} ms over {max(len(lat) - 1, 1)} calls (CPU)\n"
    )
    print(f"{'rank':>4}  {'service':14s} {'score':>8}")
    for c in resp["candidates"]:
        mark = "  <- true root cause" if c["service"] == truth else ""
        print(f"{c['rank']:>4}  {c['service']:14s} {c['score']:>8.3f}{mark}")
    ranked = [c["service"] for c in resp["candidates"]]
    if truth:
        where = (
            f"rank {ranked.index(truth) + 1}" if truth in ranked else f"not in top {len(ranked)}"
        )
        print(f"\ntrue root cause {truth}: {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
