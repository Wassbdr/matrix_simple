"""Seeded Sock Shop incident demo: STA (from a bundle) vs BARO on the same candidates.

``build`` generates a synthetic Sock Shop dataset (14 services, public
weaveworks dependency graph from ``topologies.SOCK_SHOP``, seed 42), trains
STA on CPU, saves it as a serving bundle, evaluates STA and BARO on every
held-out incident, and exports one held-out cascade incident as JSON.

``run`` loads the bundle + incident, ranks the 14 services with STA and with
BARO, and reports top-1, the rank of the true root cause and inference time.

    python -m rca_pipeline.demo_incident build
    python -m rca_pipeline.demo_incident run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from rca_pipeline.baseline_baro import baro_scores_from_arrays
from rca_pipeline.common import set_all_seeds, set_torch_deterministic, stratified_split_shuffle
from rca_pipeline.k_selection import get_optimal_k
from rca_pipeline.metrics_rcaeval import RCAEvalMetrics
from rca_pipeline.model_bundle import LoadedBundle, align_window, load_bundle, predict, save_bundle
from rca_pipeline.simulated_data_scalable import generate_scalable_dataset
from rca_pipeline.topologies import SOCK_SHOP

# Synthetic per-service metrics, in the column order the bundle is trained on.
FEATURES = ["cpu", "memory", "latency_p99", "error_rate", "throughput"]
DEMO_DIR = Path("examples/sockshop_demo")
BUNDLE_DIR = DEMO_DIR / "bundle"
INCIDENT_JSON = DEMO_DIR / "incident.json"
EVAL_JSON = Path("results/sockshop_demo_eval.json")
SEED = 42
NORMAL_WINDOWS = 5  # healthy windows concatenated as BARO's normal period


def _clone(d):
    c = d.clone()
    c.x = d.x.clone()
    return c


def sta_full_ranking(bundle: LoadedBundle, window: np.ndarray) -> List[str]:
    return [s for s, _ in predict(bundle, window, top_k=len(bundle.node_names))]


def baro_ranking(node_names: List[str], normal: np.ndarray, window: np.ndarray) -> List[str]:
    scores = baro_scores_from_arrays(normal, window)
    # stable sort: ties keep service order, so the ranking is deterministic
    return [node_names[i] for i in np.argsort(-scores, kind="stable")]


def build(args) -> Dict:
    set_torch_deterministic(True)
    set_all_seeds(SEED)
    dataset, node_names, edge_index = generate_scalable_dataset(
        topology=SOCK_SHOP,
        num_samples=args.num_samples,
        window_size=12,
        num_features=len(FEATURES),
        seed=SEED,
        generate_multimodal=False,
    )
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    train, test = stratified_split_shuffle(
        dataset, train_ratio=0.8, name_to_idx=name_to_idx, seed=SEED
    )
    raw_train = [_clone(d) for d in train]
    raw_test = [_clone(d) for d in test]

    all_x = torch.cat([d.x for d in train], dim=0)
    mean = all_x.mean(dim=(0, 2))
    std = all_x.std(dim=(0, 2)).clamp(min=1e-8)
    for d in train:
        d.x = (d.x - mean.view(1, -1, 1)) / std.view(1, -1, 1)

    from rca_pipeline.benchmark import train_sta_compact  # heavy import, build only

    k = get_optimal_k(len(node_names))
    set_all_seeds(SEED)
    model = train_sta_compact(
        train,
        node_names,
        edge_index,
        torch.device("cpu"),
        hidden_dim=32,
        k=k,
        warmup_epochs=5,
        supervised_epochs=args.epochs,
        batch_size=256,
        k_anneal_epochs=20,
    )
    model.eval()

    healthy = [d for d in raw_train if int(d.y.item()) == 0][:NORMAL_WINDOWS]
    normal = torch.cat([d.x for d in healthy], dim=2).numpy()  # [N, F, NORMAL_WINDOWS*W]
    test_inc = [d for d in raw_test if int(d.y.item()) == 1 and d.root_cause[0] in name_to_idx]

    save_bundle(
        str(BUNDLE_DIR),
        model=model,
        node_names=node_names,
        edge_index=edge_index,
        scaler_mean=mean.numpy(),
        scaler_scale=std.numpy(),
        num_features=len(FEATURES),
        hidden_dim=32,
        k=k,
        feature_names=FEATURES,
        training_info={
            "data": "synthetic sock-shop (generate_scalable_dataset, topology=SOCK_SHOP)",
            "seed": SEED,
            "num_samples": args.num_samples,
            "epochs": args.epochs,
        },
    )
    bundle = load_bundle(str(BUNDLE_DIR))

    y_true, sta_logits, baro_logits, by_type = [], [], [], {}
    for d in test_inc:
        w = d.x.numpy()
        sta_rank = sta_full_ranking(bundle, w)
        baro_rank = baro_ranking(node_names, normal, w)
        t = name_to_idx[d.root_cause[0]]
        y_true.append(t)
        sta_logits.append(np.array([-sta_rank.index(n) for n in node_names], dtype=np.float32))
        baro_logits.append(np.array([-baro_rank.index(n) for n in node_names], dtype=np.float32))
        by_type.setdefault(d.failure_type, []).append(len(y_true) - 1)

    m = RCAEvalMetrics()

    def _metrics(idx):
        sta = m.compute([y_true[i] for i in idx], [sta_logits[i] for i in idx])
        baro = m.compute([y_true[i] for i in idx], [baro_logits[i] for i in idx])
        return {
            "n": len(idx),
            "STA": {k_: round(float(sta[k_]), 4) for k_ in ("avg1", "avg3", "mrr")},
            "BARO": {k_: round(float(baro[k_]), 4) for k_ in ("avg1", "avg3", "mrr")},
        }

    evaluation = {"all": _metrics(list(range(len(y_true))))}
    for ft, idx in sorted(by_type.items()):
        evaluation[ft] = _metrics(idx)

    demo = next(d for d in test_inc if d.failure_type == "cascade")
    incident = {
        "description": "Synthetic Sock Shop incident (held-out test window, first cascade failure).",
        "generator": f"python -m rca_pipeline.demo_incident build  (seed={SEED})",
        "licence": "synthetic, generated by this repository (no third-party data)",
        "services": node_names,
        "features": FEATURES,
        "edges": [list(e) for e in SOCK_SHOP.edges],
        "true_root_cause": demo.root_cause[0],
        "failure_type": demo.failure_type,
        "window": np.round(demo.x.numpy(), 5).tolist(),
        "normal_history": np.round(normal, 5).tolist(),
    }
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    INCIDENT_JSON.write_text(json.dumps(incident) + "\n")
    EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "sockshop_demo_heldout",
        "command": "python -m rca_pipeline.demo_incident build",
        "setup": {
            "topology": "sock_shop (14 services)",
            "seed": SEED,
            "num_samples": args.num_samples,
            "test_incidents": len(y_true),
            "baro_normal_period": f"{NORMAL_WINDOWS} healthy windows",
            "candidates": "all 14 services, same for STA and BARO",
        },
        "metrics": evaluation,
    }
    EVAL_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2))
    return out


def _timed(fn, repeats: int):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return res, float(np.median(ts))


def run_demo(
    bundle_dir: str = str(BUNDLE_DIR), incident_path: str = str(INCIDENT_JSON), repeats: int = 20
) -> Dict:
    torch.set_num_threads(1)
    bundle = load_bundle(bundle_dir)
    inc = json.loads(Path(incident_path).read_text())
    window = align_window(
        bundle,
        np.asarray(inc["window"], dtype=np.float32),
        services=inc["services"],
        features=inc["features"],
    )
    normal = align_window(
        bundle,
        np.asarray(inc["normal_history"], dtype=np.float32),
        services=inc["services"],
        features=inc["features"],
    )
    truth = inc["true_root_cause"]
    with torch.no_grad():
        sta, sta_ms = _timed(lambda: sta_full_ranking(bundle, window), repeats)
    baro, baro_ms = _timed(lambda: baro_ranking(bundle.node_names, normal, window), repeats)
    result = {
        "true_root_cause": truth,
        "failure_type": inc.get("failure_type"),
        "candidates": bundle.node_names,
    }
    for name, ranking, ms in (("STA", sta, sta_ms), ("BARO", baro, baro_ms)):
        result[name] = {
            "top1": ranking[0],
            "true_rank": ranking.index(truth) + 1,
            "top3": ranking[:3],
            "ranking": ranking,
            "inference_ms": round(ms, 3),
        }
    return result


def _print(result: Dict) -> None:
    print(
        f"Incident: true root cause = {result['true_root_cause']} ({result['failure_type']}), "
        f"{len(result['candidates'])} candidate services"
    )
    print(f"{'method':6s} {'top-1':14s} {'true rank':>9s} {'CPU ms':>8s}  top-3")
    for name in ("STA", "BARO"):
        r = result[name]
        print(
            f"{name:6s} {r['top1']:14s} {r['true_rank']:>9d} {r['inference_ms']:>8.2f}  {', '.join(r['top3'])}"
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--num-samples", type=int, default=4000)
    b.add_argument("--epochs", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("--bundle", default=str(BUNDLE_DIR))
    r.add_argument("--incident", default=str(INCIDENT_JSON))
    r.add_argument("--json", default=None, help="also write the result to this path")
    args = ap.parse_args(argv)
    logging.disable(logging.INFO)
    if args.cmd == "build":
        build(args)
    else:
        res = run_demo(args.bundle, args.incident)
        _print(res)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
