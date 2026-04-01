"""Sockshop is excluded from MM runs in validation_protocol (paper scope)."""

from rca_pipeline.validation_protocol import MM_RUN_NAMES, RUNS, _build_mm_skipped_stub


def test_mm_run_names_exclude_metrics_only():
    names = {c.name for c in RUNS}
    assert "metrics_only" in names
    assert "metrics_only" not in MM_RUN_NAMES
    assert MM_RUN_NAMES <= names


def test_three_mm_configs_exist():
    assert MM_RUN_NAMES == {"traces_only", "trace_prior", "full_mm_strict"}


def test_mm_skip_stub_exposes_dataset_status():
    stub = _build_mm_skipped_stub("sockshop")
    cfg = stub["sockshop"]["config"]
    assert cfg["mm_dataset_status"] == "skipped_with_reason"
    assert cfg["use_mm"] is True
