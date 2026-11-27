"""Scaling benchmark: STA (Select-Then-Analyze) vs full-graph models on seeded BA graphs.

For each N, a Barabasi-Albert microservice graph and its incident windows are
generated with a fixed seed (``simulated_data_scalable.generate_scalable_dataset``),
split 80/20 (stratified, seed 42), and every model is trained per training seed.

Two measurements per (model, N):
  * accuracy: top-1 / top-3 / MRR of the true root cause on held-out incidents,
    mean and std across training seeds;
  * latency: CPU, batch=1, ``torch.set_num_threads(1)``, median over repeated
    forward passes.

Latency is measured in a separate pass on freshly initialised weights: the
forward-pass cost of these models does not depend on weight values (STA always
keeps exactly k nodes), so the pass can extend past the trained sizes and can
be re-run alone on a quiet machine with ``--latency-only``.

Usage:
    python -m rca_pipeline.scaling_benchmark --output-json results/scaling_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from rca_pipeline.benchmark import (
    FullGraphGCN,
    compute_norm_stats,
    eval_gat,
    eval_gcn_baseline,
    eval_sta,
    get_anomalies,
    normalize_dataset,
    train_baseline_gcn,
    train_gat_compact,
    train_sta_compact,
)
from rca_pipeline.common import set_all_seeds, set_torch_deterministic, stratified_split_shuffle
from rca_pipeline.k_selection import get_optimal_k
from rca_pipeline.model_late_gat import SpatioTemporalRCA_GAT
from rca_pipeline.model_sta import SelectThenAnalyze
from rca_pipeline.simulated_data_scalable import generate_ba_topology, generate_scalable_dataset

MODELS = ("STA", "FullGraphGCN", "LateGATv2")


def _ba_m(n: int) -> int:
    return 2 if n <= 20 else 3


def _top3(y_true: List[int], logits: List[np.ndarray]) -> float:
    hits = [int(t in np.argsort(-lo)[:3]) for t, lo in zip(y_true, logits)]
    return float(np.mean(hits))


def measure_latency_ms(
    fn: Callable[[], object], repeats: int = 200, warmup: int = 20
) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts)
    return {"median_ms": float(np.median(arr)), "p90_ms": float(np.percentile(arr, 90))}


def _forward_fn(name: str, model: torch.nn.Module, x: torch.Tensor, ei: torch.Tensor):
    model.eval()
    return lambda: model(x, ei)


def build_untrained(name: str, n: int, num_features: int, hidden: int) -> torch.nn.Module:
    if name == "STA":
        return SelectThenAnalyze(num_features, hidden, n, get_optimal_k(n))
    if name == "FullGraphGCN":
        return FullGraphGCN(num_features, hidden, n)
    return SpatioTemporalRCA_GAT(num_features, hidden, n)


def run_size(n: int, args, models: List[str]) -> Dict:
    num_samples = max(
        args.min_samples, int(np.ceil(n * args.windows_per_class / args.anomaly_ratio))
    )
    t0 = time.time()
    dataset, node_names, edge_index = generate_scalable_dataset(
        num_nodes=n,
        ba_m=_ba_m(n),
        num_samples=num_samples,
        window_size=12,
        num_features=5,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.dataset_seed,
        generate_multimodal=False,
    )
    gen_s = time.time() - t0
    name_to_idx = {nm: i for i, nm in enumerate(node_names)}
    train, test = stratified_split_shuffle(
        dataset, train_ratio=0.8, name_to_idx=name_to_idx, seed=args.dataset_seed
    )
    mean, std = compute_norm_stats(train)
    normalize_dataset(train, mean, std)
    normalize_dataset(test, mean, std)
    train_anom = get_anomalies(train, name_to_idx)
    test_anom = get_anomalies(test, name_to_idx)
    device = torch.device("cpu")
    k = get_optimal_k(n)
    print(
        f"\n=== N={n} k={k} samples={num_samples} train_anom={len(train_anom)} "
        f"test_anom={len(test_anom)} (gen {gen_s:.1f}s)",
        flush=True,
    )

    per_model: Dict[str, Dict] = {}
    for name in models:
        runs, train_s = [], []
        for si, seed in enumerate(args.seeds):
            set_all_seeds(seed)
            t1 = time.time()
            if name == "STA":
                m = train_sta_compact(
                    train,
                    node_names,
                    edge_index,
                    device,
                    hidden_dim=args.hidden,
                    k=k,
                    warmup_epochs=args.sta_warmup_epochs,
                    supervised_epochs=args.epochs,
                    batch_size=args.batch_size,
                    k_anneal_epochs=args.sta_k_anneal_epochs,
                )
                train_s.append(time.time() - t1)
                r = eval_sta(m, test_anom, edge_index, name_to_idx, device)
            elif name == "FullGraphGCN":
                m = train_baseline_gcn(
                    train_anom,
                    edge_index,
                    name_to_idx,
                    n,
                    5,
                    hidden_dim=args.hidden,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                m.eval()
                train_s.append(time.time() - t1)
                r = eval_gcn_baseline(m, test_anom, edge_index, name_to_idx, n)
            else:
                m = train_gat_compact(
                    train,
                    node_names,
                    edge_index,
                    device,
                    hidden_dim=args.hidden,
                    contrastive_epochs=args.gat_contrastive_epochs,
                    supervised_epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                train_s.append(time.time() - t1)
                r = eval_gat(m, test_anom, edge_index, name_to_idx, device)
            runs.append(
                {
                    "seed": seed,
                    "top1": float(r["top1"]),
                    "mrr": float(r["mrr"]),
                    "top3": float(r.get("rcaeval_avg3", r.get("avg3", np.nan))),
                }
            )
            print(
                f"  {name:13s} seed={seed} top1={r['top1']:.3f} mrr={r['mrr']:.3f} "
                f"train={train_s[-1]:.0f}s",
                flush=True,
            )
        t1s = [r["top1"] for r in runs]
        mrrs = [r["mrr"] for r in runs]
        per_model[name] = {
            "top1_mean": float(np.mean(t1s)),
            "top1_std": float(np.std(t1s)),
            "mrr_mean": float(np.mean(mrrs)),
            "mrr_std": float(np.std(mrrs)),
            "train_seconds_mean": float(np.mean(train_s)),
            "runs": runs,
        }
    return {
        "N": n,
        "k": k,
        "num_samples": num_samples,
        "n_test_incidents": len(test_anom),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "random_top1": 1.0 / n,
        "models": per_model,
    }


def run_latency_only(n: int, models: List[str], repeats: int) -> Dict:
    _, ei, _, _ = generate_ba_topology(n, _ba_m(n), 42)
    torch.manual_seed(0)
    x = torch.randn(1, n, 5, 12)
    out = {}
    for name in models:
        torch.manual_seed(0)
        m = build_untrained(name, n, 5, 32)
        out[name] = {"latency": measure_latency_ms(_forward_fn(name, m, x, ei), repeats=repeats)}
    print(
        f"  latency N={n}: "
        + ", ".join(f"{k}={v['latency']['median_ms']:.2f}ms" for k, v in out.items()),
        flush=True,
    )
    return {"N": n, "k": get_optimal_k(n), "models": out}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="14,50,100,200,500")
    ap.add_argument("--latency-sizes", default="14,50,100,200,500,1000,2000")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--dataset-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sta-warmup-epochs", type=int, default=5)
    ap.add_argument("--sta-k-anneal-epochs", type=int, default=20)
    ap.add_argument("--gat-contrastive-epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--min-samples", type=int, default=3000)
    ap.add_argument("--windows-per-class", type=int, default=12)
    ap.add_argument("--anomaly-ratio", type=float, default=0.4)
    ap.add_argument("--latency-repeats", type=int, default=300)
    ap.add_argument("--train-threads", type=int, default=0, help="0 = torch default")
    ap.add_argument(
        "--latency-only",
        action="store_true",
        help="re-measure latency and update it in an existing --output-json",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep sizes already in --output-json and append the missing ones",
    )
    ap.add_argument(
        "--skip-latency", action="store_true", help="accuracy only (run --latency-only later)"
    )
    ap.add_argument("--output-json", default="results/scaling_benchmark.json")
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",") if s]
    models = [m for m in args.models.split(",") if m]
    logging.disable(logging.INFO)
    set_torch_deterministic(True)

    if args.latency_only:
        with open(args.output_json) as fh:
            out = json.load(fh)
    else:
        if args.train_threads:
            torch.set_num_threads(args.train_threads)
        out = {"benchmark": "scaling_sta_vs_full_graph", "accuracy": [], "invocations": []}
        if args.resume and os.path.exists(args.output_json):
            with open(args.output_json) as fh:
                out = json.load(fh)
            out.setdefault("invocations", [])
        out["config"] = {k: v for k, v in vars(args).items() if k not in ("latency_only", "resume")}
        out["invocations"].append(
            " ".join(["python -m rca_pipeline.scaling_benchmark", *(argv or sys.argv[1:])])
        )
        done = {r["N"] for r in out["accuracy"]}
        for n in [int(n) for n in args.sizes.split(",") if n]:
            if n in done:
                print(f"=== N={n}: kept from {args.output_json}", flush=True)
                continue
            out["accuracy"].append(run_size(n, args, models))
            out["accuracy"].sort(key=lambda r: r["N"])
            out["hardware"] = _hardware()
            _write(out, args.output_json)
    if args.skip_latency:
        print(f"\nWrote {args.output_json} (accuracy only)")
        return
    torch.set_num_threads(1)
    out["latency"] = [
        run_latency_only(int(n), models, args.latency_repeats)
        for n in args.latency_sizes.split(",")
        if n
    ]
    out["hardware"] = _hardware()
    _write(out, args.output_json)
    print(f"\nWrote {args.output_json}")


def _write(out: Dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, path)


def _hardware() -> Dict[str, object]:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "latency_threads": 1,
        "latency_batch": 1,
    }


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
