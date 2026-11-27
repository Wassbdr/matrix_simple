"""Rank services in a metrics export using the packaged RE1 model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import torch

from rca_pipeline.bench_data import Case, candidates, observation_window, read_metrics
from rca_pipeline.bench_sta import METRICS, FittedSTA, incident_logits, prepare
from rca_pipeline.model_sta import SelectThenAnalyze

DEFAULT_BUNDLE = Path(__file__).with_name("re1.pt")


@dataclass(frozen=True)
class Candidate:
    service: str
    score: float
    rank: int


class Ranker:
    def __init__(self, bundle: Path = DEFAULT_BUNDLE) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", module=r"torch\._utils")
            state = torch.load(bundle, map_location="cpu", weights_only=True)
        model = SelectThenAnalyze(
            num_features=len(METRICS), hidden_dim=state["hidden_dim"], k=state["k"],
        )
        model.load_state_dict(state["state"])
        model.eval()
        self._fitted = FittedSTA(model, torch.empty((2, 0), dtype=torch.long), state["mean"], state["scale"])
        self._lock = Lock()
        self.version: str = state["version"]
        self.rows: int = state["rows"]

    def rank(self, frame: pd.DataFrame, *, inject_time: int, top_k: int = 3) -> list[Candidate]:
        if top_k < 1:
            raise ValueError("Set top_k to a positive integer.")
        if "time" not in frame or not frame.columns.is_unique:
            raise ValueError("Supply a 'time' column and unique metric names.")
        if not all(isinstance(name, str) for name in frame.columns):
            raise ValueError("Use string column names of the form service_metric.")

        columns = ["time", *(name for name in frame.columns if "_" in name)]
        try:
            numeric = frame[columns].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("Supply numeric timestamps and metric readings.") from exc
        if not np.isfinite(numeric["time"].to_numpy()).all():
            raise ValueError("Supply finite timestamps in the same units as inject_time.")
        window = observation_window(numeric, inject_time, self.rows)
        if not candidates(window):
            raise ValueError("Add at least one metric column named service_metric, such as checkout_cpu.")
        sample = prepare(Case(Path("."), "", "", "", ""), window, inject_time, "prior")
        with self._lock:
            logits, _ = incident_logits(self._fitted, sample)
        indices = logits.argsort(descending=True, stable=True)[:top_k].tolist()
        return [Candidate(sample.names[i], float(logits[i]), place) for place, i in enumerate(indices, 1)]


@lru_cache(maxsize=1)
def _default_ranker() -> Ranker:
    return Ranker()


def rank(
    source: str | Path | pd.DataFrame, *,
    inject_time: int | None = None, top_k: int = 3,
) -> list[Candidate]:
    """Rank services from a case directory, metrics file, or DataFrame.

    Use the same numeric time units for the readings and the injection time.
    Case directories and files with an adjacent inject_time.txt can omit it.
    Scores order candidates within this incident; they are not probabilities.
    """
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        path = Path(source)
        if path.is_dir():
            path = next((path / name for name in ("metrics.parquet", "metrics.json", "data.csv", "metrics.csv")
                         if (path / name).is_file()), path / "metrics.json")
        if not path.is_file():
            raise ValueError(f"No metrics at {path}. Pass a case directory, metrics.json, or CSV.")
        if inject_time is None:
            sidecar = path.with_name("inject_time.txt")
            if sidecar.is_file():
                try:
                    inject_time = int(sidecar.read_text().split()[0])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Write an integer timestamp to {sidecar} or set inject_time.") from exc
        frame = read_metrics(path)
    if inject_time is None:
        raise ValueError("Set inject_time to the incident start time, in the same units as the time column.")
    return _default_ranker().rank(frame, inject_time=inject_time, top_k=top_k)
