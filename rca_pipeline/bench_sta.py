"""Train and score STA on telemetry-derived candidate sets."""

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rca_pipeline.bench_data import Case, candidates, service_name
from rca_pipeline.common import set_all_seeds
from rca_pipeline.model_sta import SelectThenAnalyze
from rca_pipeline.sta_fusion_utils import gated_selector_scorer_fusion

METRICS = ("cpu", "mem", "diskio", "socket", "workload", "latency", "error")
METRIC_ALIASES = {"latency-90": "latency", "load": "workload"}
STEPS = 12
VARIANTS = {
    "baseline": "Post-injection telemetry with train-only scaling",
    "relative": "Pre-injection median and IQR scaling per metric and case",
    "onset": "Median and IQR scaling plus onset and persistence channels",
    "tails": "Median and IQR scaling plus signed within-bin tail deviations",
}


@dataclass
class Incident:
    case: Case
    names: tuple[str, ...]
    x: torch.Tensor
    preparation_ms: float


def _downsample(values: np.ndarray) -> np.ndarray:
    if len(values) < STEPS:
        return np.interp(np.linspace(0, len(values) - 1, STEPS), np.arange(len(values)), values)
    return np.array([piece.mean() for piece in np.array_split(values, STEPS)])


def _tails(values: np.ndarray) -> np.ndarray:
    if len(values) < STEPS:
        return _downsample(values)
    bounds = np.array([np.quantile(piece, (0.1, 0.9)) for piece in np.array_split(values, STEPS)])
    return np.where(np.abs(bounds[:, 0]) > np.abs(bounds[:, 1]), bounds[:, 0], bounds[:, 1])


def prepare(case: Case, frame: pd.DataFrame, cut: int, variant: str) -> Incident:
    start = time.perf_counter()
    names = candidates(frame)
    by_name = {name: i for i, name in enumerate(names)}
    channels = len(METRICS) * {"onset": 3, "tails": 2}.get(variant, 1)
    x = np.zeros((len(names), channels, STEPS), dtype=np.float32)
    strength = np.full((len(names), len(METRICS)), -np.inf)
    mask = frame.time.to_numpy() < cut
    for col in frame.columns:
        if "_" not in col:
            continue
        metric = col.split("_", 1)[1]
        metric = METRIC_ALIASES.get(metric, metric)
        if metric not in METRICS:
            continue
        node, feature = by_name[service_name(col)], METRICS.index(metric)
        values = frame[col].to_numpy(dtype=np.float64)
        before, after = values[mask], values[~mask]
        if variant == "baseline":
            signed = np.sign(after) * np.log1p(np.abs(after))
            priority = float(np.var(signed))
            signal = _downsample(signed)
        else:
            lo, center, hi = np.quantile(before, (0.25, 0.5, 0.75))
            scale = hi - lo
            if scale <= 1e-12:
                scale = max(float(np.std(before)), abs(center) * 0.01, 1e-9)
            z = (after - center) / scale
            priority = float(np.max(np.abs(z)))
            signed = np.sign(z) * np.log1p(np.abs(z))
            signal = _downsample(signed)
        if priority <= strength[node, feature]:
            continue
        strength[node, feature] = priority
        x[node, feature] = signal
        if variant == "onset":
            x[node, len(METRICS) + feature] = np.diff(signal, prepend=0.0)
            x[node, 2 * len(METRICS) + feature] = _downsample((np.abs(z) >= 3).astype(float))
        elif variant == "tails":
            x[node, len(METRICS) + feature] = _tails(signed)
    return Incident(case, names, torch.from_numpy(x), 1e3 * (time.perf_counter() - start))


def fit(train: list[Incident], seed: int, epochs: int, variant: str) -> tuple:
    set_all_seeds(seed)
    measurements = torch.cat([sample.x for sample in train])
    mean = measurements.mean(dim=(0, 2)).view(1, -1, 1)
    scale = measurements.std(dim=(0, 2)).clamp_min(0.1).view(1, -1, 1)
    if variant != "baseline":
        mean = torch.zeros_like(mean)
        scale = torch.ones_like(scale) * 4
    grouped = defaultdict(list)
    for sample in train:
        if sample.case.service in sample.names:
            grouped[len(sample.names)].append(sample)
    blocks = []
    for size, group in sorted(grouped.items()):
        if not group:
            continue
        x = (torch.stack([sample.x for sample in group]) - mean) / scale
        y = torch.tensor([sample.names.index(sample.case.service) for sample in group])
        blocks.append((size, x, y))
    if not blocks:
        raise ValueError("No training case has its injected service in the available metrics.")
    model = SelectThenAnalyze(num_features=mean.numel(), hidden_dim=16, num_nodes=blocks[0][0], k=8)
    edges = torch.empty((2, 0), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(epochs):
        for block in torch.randperm(len(blocks)).tolist():
            size, x, y = blocks[block]
            model.num_nodes = size
            model.set_k(min(model.k, size))
            for ids in torch.randperm(len(y)).split(32):
                scores, _, top, selector = model(x[ids], edges)
                chosen = top.eq(y[ids, None]).any(dim=1)
                loss = F.cross_entropy(selector, y[ids])
                if chosen.any():
                    loss = loss + 0.5 * F.cross_entropy(scores[chosen], y[ids][chosen])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    model.eval()
    return model, edges, mean, scale


@torch.inference_mode()
def rank(fitted: tuple, sample: Incident) -> dict:
    model, edges, mean, scale = fitted
    if not sample.names:
        return {"ranks": [], "status": "failed", "error": "no candidates", "latency_ms": 0.0}
    start = time.perf_counter()
    model.num_nodes = len(sample.names)
    model.set_k(min(model.k, model.num_nodes))
    x = ((sample.x - mean) / scale).unsqueeze(0)
    scores, _, top, selector = model(x, edges)
    fused = gated_selector_scorer_fusion(selector[0], scores[0], top[0], gamma=0.35)
    ranks = [sample.names[i] for i in fused.argsort(descending=True, stable=True).tolist()]
    picked = [sample.names[i] for i in top[0].tolist()]
    return {
        "ranks": ranks,
        "status": "ok",
        "latency_ms": round(sample.preparation_ms + 1e3 * (time.perf_counter() - start), 4),
        "selector_hit": sample.case.service in picked,
    }
