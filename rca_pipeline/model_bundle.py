"""
Model bundle: portable, self-describing artifact for serving an STA model.

A bundle directory contains:

    model.pt          # torch state_dict
    scaler.pt         # StandardScaler stats (mean, scale)
    metadata.json     # model config + training info + model card

This module provides ``save_bundle()`` and ``load_bundle()`` plus a thin
``ServingModel`` wrapper that hides the normalization + ranking details
from the FastAPI service.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rca_pipeline.model_sta import SelectThenAnalyze

logger = logging.getLogger(__name__)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return None


def save_bundle(
    bundle_dir: str,
    model: SelectThenAnalyze,
    node_names: List[str],
    edge_index: torch.Tensor,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    num_features: int,
    hidden_dim: int,
    k: int,
    training_info: Dict[str, Any],
    hpo_provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a model + everything needed to serve it later."""
    p = Path(bundle_dir)
    p.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict()}, p / "model.pt")
    torch.save(
        {
            "mean": torch.from_numpy(np.asarray(scaler_mean, dtype=np.float32)),
            "scale": torch.from_numpy(np.asarray(scaler_scale, dtype=np.float32)),
        },
        p / "scaler.pt",
    )

    metadata: Dict[str, Any] = {
        "model_class": "SelectThenAnalyze",
        "config": {
            "num_features": int(num_features),
            "hidden_dim": int(hidden_dim),
            "num_nodes": len(node_names),
            "k": int(k),
        },
        "node_names": list(node_names),
        "edge_index": edge_index.tolist(),
        "training": training_info,
        "model_card": {
            "created_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
    }
    if hpo_provenance is not None:
        metadata["hpo_provenance"] = hpo_provenance

    (p / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved bundle to %s (%d files)", bundle_dir, len(list(p.iterdir())))


@dataclass
class LoadedBundle:
    model: SelectThenAnalyze
    metadata: Dict[str, Any]
    node_names: List[str]
    edge_index: torch.Tensor
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def load_bundle(bundle_dir: str) -> LoadedBundle:
    p = Path(bundle_dir)
    metadata = json.loads((p / "metadata.json").read_text())
    cfg = metadata["config"]
    model = SelectThenAnalyze(
        num_features=cfg["num_features"],
        hidden_dim=cfg["hidden_dim"],
        num_nodes=cfg["num_nodes"],
        k=cfg["k"],
    )
    state = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    scaler = torch.load(p / "scaler.pt", map_location="cpu", weights_only=False)
    edge_index = torch.tensor(metadata["edge_index"], dtype=torch.long)

    return LoadedBundle(
        model=model,
        metadata=metadata,
        node_names=metadata["node_names"],
        edge_index=edge_index,
        scaler_mean=scaler["mean"].numpy(),
        scaler_scale=scaler["scale"].numpy(),
    )


def predict(
    bundle: LoadedBundle,
    window: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Run inference on a single window.

    ``window`` is ``[N, F, W]`` with N matching ``len(bundle.node_names)`` and
    raw (unnormalized) feature values. The bundle's scaler is applied here
    so callers do not need to know normalization details.
    """
    if window.shape[0] != len(bundle.node_names):
        raise ValueError(
            f"window has N={window.shape[0]} but model expects "
            f"N={len(bundle.node_names)}"
        )
    if window.shape[1] != bundle.scaler_mean.shape[0]:
        raise ValueError(
            f"window has F={window.shape[1]} features but scaler expects "
            f"F={bundle.scaler_mean.shape[0]}"
        )
    mean = bundle.scaler_mean.reshape(1, -1, 1)
    scale = bundle.scaler_scale.reshape(1, -1, 1)
    normed = (window - mean) / np.where(scale == 0, 1.0, scale)
    x = torch.from_numpy(normed.astype(np.float32)).unsqueeze(0)  # [1, N, F, W]

    with torch.no_grad():
        logits_full, _, top_k_indices, sel_logits = bundle.model(x, bundle.edge_index)
        # Dense ranking (mirrors benchmark_real._merge_sta_logits)
        merged = sel_logits.clone()
        topk_vals = torch.gather(logits_full, 1, top_k_indices)
        merged.scatter_(1, top_k_indices, topk_vals)
        # Soft rerank
        alpha = 0.15
        scores = (1 - alpha) * sel_logits + alpha * merged
        scores_np = scores.squeeze(0).cpu().numpy()

    order = np.argsort(-scores_np)
    out: List[Tuple[str, float]] = []
    for idx in order[:top_k]:
        out.append((bundle.node_names[int(idx)], float(scores_np[int(idx)])))
    return out


# -----------------------------------------------------------------------------
# CLI smoke test: load a bundle and print info
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir")
    args = parser.parse_args()

    b = load_bundle(args.bundle_dir)
    cfg = b.metadata["config"]
    print(f"Model: {b.metadata['model_class']}")
    print(f"  N={cfg['num_nodes']} F={cfg['num_features']} H={cfg['hidden_dim']} k={cfg['k']}")
    print(f"  trained: {b.metadata.get('training', {}).get('avg1_mean', 'n/a'):.4f} Avg@1")
    print(f"  node_names: {b.node_names[:8]}{'...' if len(b.node_names)>8 else ''}")
    n_params = sum(p.numel() for p in b.model.parameters())
    print(f"  params: {n_params}")
