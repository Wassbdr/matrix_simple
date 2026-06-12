"""YAML + pydantic configuration layer for the CLI entry points.

Every training/benchmark entry point accepts ``--config path.yaml``. The YAML
is validated against the matching pydantic model below, then injected as
argparse *defaults*, so precedence is:

    explicit CLI flag  >  YAML value  >  argparse default

Keys use the argparse dest names (underscores). Unknown keys are rejected
(``extra="forbid"``) so a typo fails loudly instead of being ignored.

The real-data contrastive-warmup guardrail (README §5.4) is enforced here at
YAML load time, mirroring the runtime guard in ``benchmark_real.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

WARMUP_GUARD_MESSAGE = (
    "sta_contrastive_warmup > 0 is unsafe on real datasets (collapses Avg@1, "
    "see README §5.4). Set allow_real_warmup: true to acknowledge the risk."
)


class _YamlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkSimConfig(_YamlConfig):
    """Mirrors ``python -m rca_pipeline.benchmark`` flags."""

    sizes: Optional[str] = "20,50,100"
    ba_m_map: Optional[str] = "20:2,50:3,100:3"
    k_map: Optional[str] = "20:5,50:8,100:10"
    num_samples: Optional[int] = 3000
    window_size: Optional[int] = 12
    num_features: Optional[int] = 5
    anomaly_ratio: Optional[float] = 0.4
    dataset_seed: Optional[int] = 42
    seeds: Optional[str] = "42,123,7,1337,2025"
    hidden_dim: Optional[int] = 32
    batch_size: Optional[int] = 256
    supervised_epochs: Optional[int] = 150
    late_contrastive_epochs: Optional[int] = 80
    sta_warmup_epochs: Optional[int] = 20
    sta_k_anneal_epochs: Optional[int] = 60
    baseline_epochs: Optional[int] = 150
    no_normalize: Optional[bool] = False
    graph_type: Optional[Literal["structural", "causal", "hybrid"]] = "structural"
    cascade_loss_weight: Optional[float] = 0.0
    enable_drift: Optional[bool] = False
    drift_delta: Optional[float] = 0.002
    drift_cooldown_steps: Optional[int] = 30
    skip_late_gat: Optional[bool] = False
    output_json: Optional[str] = None
    deterministic: Optional[bool] = False
    no_gib: Optional[bool] = False
    no_mask_weighted_nce: Optional[bool] = False
    selector_type: Optional[Literal["ste", "hierarchical"]] = "ste"
    temporal_encoder: Optional[Literal["conv", "s4"]] = "conv"


class BenchmarkRealConfig(_YamlConfig):
    """Mirrors ``python -m rca_pipeline.benchmark_real`` flags."""

    dataset: Optional[str] = "trainticket"
    seeds: Optional[str] = "42,123,7"
    epochs: Optional[int] = 100
    hidden_dim: Optional[int] = 32
    sta_contrastive_warmup: Optional[int] = 0
    allow_real_warmup: Optional[bool] = False
    rcaeval_root: Optional[str] = "rcaeval"
    max_cases: Optional[int] = None
    use_mm: Optional[bool] = False
    mm_strict: Optional[bool] = False
    mm_logs_only: Optional[bool] = False
    mm_traces_only: Optional[bool] = False
    mm_trace_prior: Optional[bool] = False
    enable_drift: Optional[bool] = False
    drift_delta: Optional[float] = 0.002
    drift_cooldown_steps: Optional[int] = 30
    output_json: Optional[str] = None
    normalize: Optional[bool] = False
    deterministic: Optional[bool] = False
    observit_csv: Optional[str] = None
    observit_window: Optional[int] = 12
    observit_topology: Optional[str] = "custom_app"
    include_late_gib: Optional[bool] = False
    loki_url: Optional[str] = None
    loki_namespace: Optional[str] = "sock-shop"

    @model_validator(mode="after")
    def _real_warmup_guardrail(self) -> BenchmarkRealConfig:
        if (self.sta_contrastive_warmup or 0) > 0 and not self.allow_real_warmup:
            raise ValueError(WARMUP_GUARD_MESSAGE)
        return self

    @model_validator(mode="after")
    def _mm_flags_consistent(self) -> BenchmarkRealConfig:
        if self.mm_logs_only and self.mm_traces_only:
            raise ValueError("mm_logs_only and mm_traces_only are mutually exclusive.")
        if self.mm_trace_prior and (self.mm_logs_only or self.mm_traces_only):
            raise ValueError("mm_trace_prior is incompatible with mm_logs_only/mm_traces_only.")
        return self


class TrainConfig(_YamlConfig):
    """Mirrors ``python -m rca_pipeline.train_and_persist`` flags."""

    dataset_dir: Optional[str] = "datasets/sockshop_runs"
    topology: Optional[str] = "sock_shop"
    window_size: Optional[int] = 12
    hpo_json: Optional[str] = None
    seeds: Optional[str] = "42,123,7,1337,2025"
    output_bundle: Optional[str] = None
    no_warmup: Optional[bool] = False


class HPOConfig(_YamlConfig):
    """Mirrors ``python -m rca_pipeline.hpo_sta`` flags."""

    dataset_dir: Optional[str] = "datasets/sockshop_runs"
    topology: Optional[str] = "sock_shop"
    window_size: Optional[int] = 12
    seed: Optional[int] = 42
    n_trials: Optional[int] = 50
    study_name: Optional[str] = "sta_sockshop_n14"
    storage: Optional[str] = "sqlite:///results_transmission/hpo_sta_sockshop.db"
    output_json: Optional[str] = None
    timeout_seconds: Optional[int] = None


def load_yaml_overrides(path: str, model_cls: type) -> Dict[str, Any]:
    """Load + validate a YAML file; return only the keys it explicitly sets."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"--config {path}: expected a YAML mapping, got {type(raw).__name__}")
    try:
        cfg = model_cls(**raw)
    except Exception as exc:
        raise SystemExit(f"--config {path}: invalid configuration\n{exc}") from exc
    return cfg.model_dump(exclude_unset=True)


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    model_cls: type,
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Add ``--config`` and parse with YAML-as-defaults precedence."""
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file; explicit CLI flags override its values.",
    )
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        overrides = load_yaml_overrides(pre.config, model_cls)
        known_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(overrides) - known_dests)
        if unknown:
            raise SystemExit(
                f"--config {pre.config}: keys not matching any CLI flag: {unknown}"
            )
        parser.set_defaults(**overrides)
    return parser.parse_args(argv)
