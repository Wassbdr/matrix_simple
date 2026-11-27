"""Display the cached RCAEval BARO ranking for the public showcase incident."""

import argparse
import json
from pathlib import Path

from rca_pipeline.data import DEMO_CASE

ROOT = Path(__file__).resolve().parents[1]


def baro_ranks() -> list[str]:
    run = json.loads((ROOT / "benchmarks" / "prior.json").read_text())
    row = next(row for row in run["cases"] if row["case"] == DEMO_CASE and row["method"] == "baro")
    if row["status"] != "ok":
        raise ValueError("No successful cached BARO result. Run make bench first.")
    return list(row["ranks"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ranks = baro_ranks()[:3]
    if args.json:
        print(json.dumps(ranks))
    else:
        print("RCAEval BARO (cached reference)")
        for index, service in enumerate(ranks, 1):
            print(f"{index:>4}  {service}")
        print("Injected service: checkoutservice")


if __name__ == "__main__":
    main()
