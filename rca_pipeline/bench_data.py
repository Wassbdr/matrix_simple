"""Shared case discovery and observation windows for the public benchmark."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CASE_ID = re.compile(r"(re[123])(ob|ss|tt)_([a-z0-9-]+)_([a-z0-9]+)_([0-9]+)")


def service_name(column: str) -> str:
    return column.split("_", 1)[0].replace("-db", "")


@dataclass(frozen=True)
class Case:
    path: Path
    suite: str
    system: str
    service: str
    fault: str

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def group(self) -> tuple[str, str]:
        return self.service, self.fault


def cases_in(root: Path) -> list[Case]:
    cases = []
    for path in sorted(root.iterdir()):
        match = CASE_ID.fullmatch(path.name)
        if path.is_dir() and match:
            suite, system, service, fault, _ = match.groups()
            cases.append(Case(path, suite, system, service_name(service), fault))
    if not cases:
        raise ValueError(f"No RCAEval cases in {root}; run make data or pass --data-root.")
    return cases


def read_window(case: Case, rows: int) -> tuple[pd.DataFrame, int]:
    if rows <= 0:
        raise ValueError("Observation window must be positive.")
    cut = int((case.path / "inject_time.txt").read_text().split()[0])
    parquet = case.path / "metrics.parquet"
    if parquet.is_file():
        frame = pd.read_parquet(parquet)
    else:
        raw = json.loads((case.path / "metrics.json").read_text())
        frame = pd.DataFrame({name: dict(points) for name, points in raw.items() if points})
        frame.index.name = "time"
        frame = frame.reset_index()
    frame = frame.loc[:, ~frame.columns.str.endswith("_latency-50")]
    frame = frame.drop(columns=["time.1"], errors="ignore").sort_values("time")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    normal = frame.loc[frame.time < cut].tail(rows)
    incident = frame.loc[frame.time >= cut].head(rows)
    if normal.empty or incident.empty:
        raise ValueError(f"{case.id} needs samples on both sides of injection ({cut}).")
    return pd.concat([normal, incident], ignore_index=True), cut


def candidates(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted({service_name(name) for name in frame.columns if "_" in name}))
