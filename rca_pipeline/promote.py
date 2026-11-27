"""Promotion gate: compare the challenger model against the current champion.

Reads the two registered versions' logged metrics from MLflow and prints a
verdict. Promotion stays MANUAL (human-in-the-loop) by default - a deliberate
gate for a single-environment cluster, not a limitation. Pass --apply to move
the ``champion`` alias to the challenger version when the gate passes.

    python -m rca_pipeline.promote --model sta-sockshop \
        --metric mrr_mean --min-delta 0.0
    python -m rca_pipeline.promote --model sta-sockshop --apply

Requires MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _metric_for_alias(client, model: str, alias: str, metric: str) -> Optional[float]:
    try:
        mv = client.get_model_version_by_alias(model, alias)
    except Exception:
        logger.warning("No version found for %s@%s", model, alias)
        return None
    run = client.get_run(mv.run_id)
    val = run.data.metrics.get(metric)
    if val is None:
        logger.warning("Run %s for %s@%s has no metric %r", mv.run_id, model, alias, metric)
    return val


def evaluate(
    model: str,
    metric: str = "mrr_mean",
    min_delta: float = 0.0,
    apply: bool = False,
) -> bool:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set.")
    import mlflow

    client = mlflow.MlflowClient()
    challenger = _metric_for_alias(client, model, "challenger", metric)
    champion = _metric_for_alias(client, model, "champion", metric)

    if challenger is None:
        raise SystemExit("No challenger to evaluate.")
    if champion is None:
        logger.info("No champion yet - challenger %s=%.4f will be promoted.", metric, challenger)
        passed = True
    else:
        delta = challenger - champion
        passed = delta >= min_delta
        logger.info(
            "%s: challenger=%.4f champion=%.4f delta=%+.4f (min_delta=%.4f) -> %s",
            metric, challenger, champion, delta, min_delta,
            "PROMOTE" if passed else "HOLD",
        )

    if passed and apply:
        mv = client.get_model_version_by_alias(model, "challenger")
        client.set_registered_model_alias(model, "champion", mv.version)
        logger.info("Champion alias moved to %s v%s", model, mv.version)
    elif passed:
        logger.info("Gate passed. Re-run with --apply to move the champion alias.")
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sta-sockshop")
    parser.add_argument("--metric", default="mrr_mean")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Challenger must beat champion by at least this much.")
    parser.add_argument("--apply", action="store_true",
                        help="Move the champion alias when the gate passes.")
    args = parser.parse_args(argv)
    passed = evaluate(args.model, args.metric, args.min_delta, args.apply)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
