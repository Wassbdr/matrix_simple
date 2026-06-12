"""Tests for the YAML + pydantic config layer (rca_pipeline/config.py)."""

import argparse

import pytest

from rca_pipeline.config import (
    BenchmarkRealConfig,
    BenchmarkSimConfig,
    TrainConfig,
    parse_args_with_config,
)


def _sim_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="20,50,100")
    parser.add_argument("--seeds", type=str, default="42,123,7,1337,2025")
    parser.add_argument("--supervised-epochs", type=int, default=150)
    return parser


def test_yaml_overrides_argparse_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("sizes: '20'\nsupervised_epochs: 5\n")
    args = parse_args_with_config(_sim_parser(), BenchmarkSimConfig,
                                  argv=["--config", str(cfg)])
    assert args.sizes == "20"
    assert args.supervised_epochs == 5
    # Untouched key keeps the argparse default.
    assert args.seeds == "42,123,7,1337,2025"


def test_explicit_cli_flag_beats_yaml(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("sizes: '20'\nsupervised_epochs: 5\n")
    args = parse_args_with_config(_sim_parser(), BenchmarkSimConfig,
                                  argv=["--config", str(cfg), "--supervised-epochs", "9"])
    assert args.supervised_epochs == 9
    assert args.sizes == "20"


def test_no_config_flag_keeps_plain_cli(tmp_path):
    args = parse_args_with_config(_sim_parser(), BenchmarkSimConfig,
                                  argv=["--sizes", "50"])
    assert args.sizes == "50"
    assert args.supervised_epochs == 150


def test_unknown_yaml_key_fails_loudly(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("sizs: '20'\n")  # typo
    with pytest.raises(SystemExit, match="sizs"):
        parse_args_with_config(_sim_parser(), BenchmarkSimConfig,
                               argv=["--config", str(cfg)])


def test_bad_yaml_type_fails_loudly(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("supervised_epochs: not-an-int\n")
    with pytest.raises(SystemExit, match="supervised_epochs"):
        parse_args_with_config(_sim_parser(), BenchmarkSimConfig,
                               argv=["--config", str(cfg)])


def test_real_warmup_guardrail_blocks_yaml():
    with pytest.raises(ValueError, match="README"):
        BenchmarkRealConfig(sta_contrastive_warmup=10)


def test_real_warmup_guardrail_explicit_override():
    cfg = BenchmarkRealConfig(sta_contrastive_warmup=10, allow_real_warmup=True)
    assert cfg.sta_contrastive_warmup == 10


def test_mm_flags_mutual_exclusion():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BenchmarkRealConfig(mm_logs_only=True, mm_traces_only=True)
    with pytest.raises(ValueError, match="incompatible"):
        BenchmarkRealConfig(mm_trace_prior=True, mm_logs_only=True)


def test_committed_configs_are_valid():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2] / "configs"
    mapping = {
        "benchmark_sim_smoke.yaml": BenchmarkSimConfig,
        "benchmark_real.yaml": BenchmarkRealConfig,
        "train_sockshop.yaml": TrainConfig,
    }
    for name, model_cls in mapping.items():
        data = yaml.safe_load((root / name).read_text())
        model_cls(**data)  # must not raise
