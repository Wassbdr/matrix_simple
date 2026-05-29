"""
A/B harness for evaluating a detector as a pre-filter for an RCA model.

Wraps any model that produces logits per node and applies a pre-filter
(e.g., gray_failure_detector or a future cascade detector) that boosts
the logits of nodes flagged by the detector. Reports Avg@1 with and
without the boost on synthetic-controlled scenarios where the ground
truth root cause is known.

Two complementary uses:
    1. Validate that a detector improves a model on the scenario it
       targets (e.g., gray detector on a true sub-threshold signal).
    2. Demonstrate the design/data mismatch when the detector does NOT
       fire on the empirical failure mode of the model (e.g., gray
       detector on a cascade scenario).

Usage:
    python -m rca_pipeline.detector_ab_harness \\
        --scenario gray \\
        --boost 2.0 \\
        --n-nodes 14 \\
        --n-trials 200

Output:
    Avg@1 baseline | Avg@1 with detector boost | uplift | detector trigger rate
"""

from __future__ import annotations

import argparse

import numpy as np

from rca_pipeline.gray_failure_detector import detect_gray_failure

# ---------------------------------------------------------------------------
# Synthetic scenario generators
# ---------------------------------------------------------------------------

def make_window(
    scenario: str,
    n_nodes: int,
    n_features: int,
    window_len: int,
    rc_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a synthetic [N, F, W] window with a known root cause at rc_idx.

    Feature index convention (matches prometheus_connector):
        0: cpu_usage, 1: memory_bytes, 2: network_rx, 3: network_tx,
        4: latency_p99, 5: error_rate

    Scenarios:
        - hard:    rc_idx has a strong CPU spike at the end of the window.
        - gray:    rc_idx has a sub-threshold but persistent latency p99
                   stutter (triggers the P99/P50 detector).
        - cascade: rc_idx has near-nominal metrics; rc_idx's downstream
                   neighbor (rc_idx + 1) has a large latency spike.
                   The root cause is silent; the symptom screams.
    """
    base = rng.normal(loc=1.0, scale=0.1, size=(n_nodes, n_features, window_len))
    base = np.abs(base) + 0.5

    if scenario == "hard":
        # Strong CPU spike on RC for last half of window
        spike_start = window_len // 2
        base[rc_idx, 0, spike_start:] += rng.uniform(8.0, 12.0, size=window_len - spike_start)
        return base

    if scenario == "gray":
        # Truly sub-threshold for the local-norm model: tiny tail-only
        # latency stutters that fire the P99/P50 ratio detector
        # (p99/p50 > 3.0) but contribute little L2 deviation.
        # Also distract the local model by adding modest noise on a few
        # other nodes so RC does not dominate by raw magnitude.
        lat = base[rc_idx, 4, :].copy()
        # Two short tail spikes — enough for p99/p50 ratio, small in L2
        lat[-2] *= 4.0
        lat[-5] *= 3.5
        base[rc_idx, 4, :] = lat
        # Distractor noise on a few non-RC nodes
        distractor_nodes = rng.choice(
            [i for i in range(n_nodes) if i != rc_idx],
            size=min(3, n_nodes - 1), replace=False,
        )
        for d in distractor_nodes:
            base[d] += rng.normal(0, 0.4, size=(n_features, window_len))
            base[d] = np.abs(base[d]) + 0.1
        return base

    if scenario == "cascade":
        # RC stays nominal; downstream node has a huge latency spike
        downstream = (rc_idx + 1) % n_nodes
        base[downstream, 4, window_len // 3:] += rng.uniform(10.0, 15.0,
                                                              size=window_len - window_len // 3)
        # Also bump error_rate on downstream (typical cascade signature)
        base[downstream, 5, window_len // 3:] += rng.uniform(5.0, 8.0,
                                                              size=window_len - window_len // 3)
        return base

    raise ValueError(f"Unknown scenario: {scenario}")


# ---------------------------------------------------------------------------
# Mock model — produces per-node logits based on local feature magnitude
# ---------------------------------------------------------------------------

def mock_local_score_model(window: np.ndarray) -> np.ndarray:
    """
    Stand-in for an RCA model: scores each node by the L2 norm of its own
    feature deviations from the per-node temporal mean. This is a faithful
    abstraction of any model whose scoring is dominated by the local node
    features (true for STA's selector and most temporal encoders).

    Returns: logits [N].
    """
    n_nodes, n_features, window_len = window.shape
    per_node_mean = window.mean(axis=2, keepdims=True)  # [N, F, 1]
    deviations = window - per_node_mean
    return np.linalg.norm(deviations.reshape(n_nodes, -1), axis=1)


# ---------------------------------------------------------------------------
# A/B evaluation
# ---------------------------------------------------------------------------

def ab_eval(
    scenario: str,
    n_nodes: int,
    n_features: int,
    window_len: int,
    n_trials: int,
    boost: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    baseline_hits = 0
    boosted_hits = 0
    triggered = 0

    for _ in range(n_trials):
        rc_idx = int(rng.integers(0, n_nodes))
        win = make_window(scenario, n_nodes, n_features, window_len, rc_idx, rng)

        # A: model alone
        logits = mock_local_score_model(win)
        if int(np.argmax(logits)) == rc_idx:
            baseline_hits += 1

        # B: model + gray detector pre-filter boost
        gray = detect_gray_failure(
            win,
            latency_idx=4,
            memory_idx=1,
            cpu_idx=0,
        )
        if gray.is_gray:
            triggered += 1
            boosted = logits.copy()
            for n in gray.affected_nodes:
                if 0 <= n < n_nodes:
                    boosted[n] += boost
            if int(np.argmax(boosted)) == rc_idx:
                boosted_hits += 1
        else:
            if int(np.argmax(logits)) == rc_idx:
                boosted_hits += 1

    return {
        "scenario": scenario,
        "n_trials": n_trials,
        "baseline_avg1": baseline_hits / n_trials,
        "boosted_avg1": boosted_hits / n_trials,
        "uplift": (boosted_hits - baseline_hits) / n_trials,
        "detector_trigger_rate": triggered / n_trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+",
                        default=["hard", "gray", "cascade"],
                        help="Synthetic scenarios to evaluate.")
    parser.add_argument("--n-nodes", type=int, default=14)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--window-len", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--boost", type=float, default=2.0,
                        help="Score boost applied to nodes flagged by the detector.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to write results as JSON.")
    args = parser.parse_args()

    print(f"\nA/B harness — mock local-score model vs. model + gray_failure_detector boost")
    print(f"N={args.n_nodes}, F={args.n_features}, W={args.window_len}, "
          f"trials={args.n_trials}, boost={args.boost}\n")

    header = f"{'scenario':<10} {'baseline':>10} {'boosted':>10} {'uplift':>10} {'trigger':>10}"
    print(header)
    print("-" * len(header))
    all_results = []
    for scenario in args.scenarios:
        r = ab_eval(scenario, args.n_nodes, args.n_features, args.window_len,
                    args.n_trials, args.boost, args.seed)
        all_results.append(r)
        print(f"{r['scenario']:<10} {r['baseline_avg1']:>10.3f} "
              f"{r['boosted_avg1']:>10.3f} {r['uplift']:>+10.3f} "
              f"{r['detector_trigger_rate']:>10.3f}")
    print()

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump({
                "config": {
                    "n_nodes": args.n_nodes,
                    "n_features": args.n_features,
                    "window_len": args.window_len,
                    "n_trials": args.n_trials,
                    "boost": args.boost,
                    "seed": args.seed,
                },
                "results": all_results,
            }, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
