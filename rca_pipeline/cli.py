"""Rank microservice root causes or run the public RCAEval benchmark."""

import argparse
import json
import sys
from dataclasses import asdict


def main() -> None:
    parser = argparse.ArgumentParser(prog="rca", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rank_parser = commands.add_parser("rank", help="Rank services from an RCAEval case or metrics export.")
    rank_parser.add_argument("source", metavar="CASE_OR_METRICS")
    rank_parser.add_argument("--inject-time", type=int)
    rank_parser.add_argument("--top-k", type=int, default=3)
    rank_parser.add_argument("--json", action="store_true", help="Emit candidates as JSON.")
    commands.add_parser("bench", help="Run the public benchmark.")
    commands.add_parser("data", help="Download the pinned public dataset.")
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from rca_pipeline.bench import main as bench

        bench(sys.argv[2:], prog="rca bench")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "data":
        from rca_pipeline.data import main as data

        data(sys.argv[2:], prog="rca data")
        return
    args = parser.parse_args()
    from rca_pipeline.ranking import rank

    try:
        candidates = rank(args.source, inject_time=args.inject_time, top_k=args.top_k)
    except (OSError, TypeError, ValueError, ImportError) as exc:
        rank_parser.exit(2, f"rca rank: {exc}\n")
    if args.json:
        print(json.dumps([asdict(candidate) for candidate in candidates], allow_nan=False))
    else:
        print("Rank  Service  Score")
        for candidate in candidates:
            print(f"{candidate.rank:>4}  {candidate.service}  {candidate.score:.4f}")


if __name__ == "__main__":
    main()
