"""P0 audit guardrails: warmup guard, MM status, and per-root consistency."""

from types import SimpleNamespace

import numpy as np
import pytest

from rca_pipeline.benchmark_real import (
    _attach_per_root_metrics,
    _build_mm_global_summary,
    _compute_mm_dataset_status,
    _enforce_real_warmup_guard,
)
from rca_pipeline.metrics_rcaeval import RCAEvalMetrics, top1_per_service_detailed


def test_real_warmup_guard_blocks_non_zero_without_ack():
    with pytest.raises(SystemExit):
        _enforce_real_warmup_guard(sta_contrastive_warmup=10, allow_real_warmup=False)


def test_real_warmup_guard_allows_zero_or_acknowledged():
    _enforce_real_warmup_guard(sta_contrastive_warmup=0, allow_real_warmup=False)
    _enforce_real_warmup_guard(sta_contrastive_warmup=10, allow_real_warmup=True)


def test_mm_dataset_status_skipped_when_no_executed_seed():
    args = SimpleNamespace(use_mm=True, mm_strict=True)
    st = _compute_mm_dataset_status(args, requested_seeds=[42, 123], skipped_seeds=[42, 123], executed_seeds=0)
    assert st["status"] == "skipped_with_reason"
    assert st["requested_seeds"] == 2
    assert st["executed_seeds"] == 0


def test_mm_global_summary_blocks_if_any_dataset_not_comparable():
    all_results = {
        "trainticket": {"config": {"mm_dataset_status": "comparable"}},
        "sockshop": {"config": {"mm_dataset_status": "skipped_with_reason"}},
    }
    summary = _build_mm_global_summary(all_results)
    assert summary["allow_global_aggregation"] is False
    assert "sockshop" in summary["blocked_by_datasets"]


def test_per_root_weighted_top1_matches_global_avg1():
    y_true = [0, 0, 1, 2, 2, 2]
    y_logits = [
        np.array([5.0, 0.0, 0.0]),  # hit rc0
        np.array([0.0, 5.0, 0.0]),  # miss rc0
        np.array([0.0, 4.0, 0.0]),  # hit rc1
        np.array([0.0, 0.0, 5.0]),  # hit rc2
        np.array([0.0, 0.0, 4.0]),  # hit rc2
        np.array([3.0, 0.0, 0.0]),  # miss rc2
    ]
    detailed = top1_per_service_detailed(y_true, y_logits, node_names=["svc0", "svc1", "svc2"])
    global_avg1 = RCAEvalMetrics().compute(y_true, y_logits)["avg1"]
    assert abs(float(detailed["weighted_top1"]) - float(global_avg1)) <= 1e-12


def test_attach_per_root_metrics_exports_summary_payload():
    y_true = [0, 0, 1, 2]
    y_logits = [
        np.array([5.0, 0.0, 0.0]),
        np.array([0.0, 5.0, 0.0]),
        np.array([0.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
    ]
    ds_results = {}
    _attach_per_root_metrics(
        ds_results=ds_results,
        prefix="STA",
        model_key="STA",
        runs=[(y_true, y_logits)],
        node_names=["svc0", "svc1", "svc2"],
        rca_metrics=RCAEvalMetrics(),
        dataset_name="trainticket",
    )
    assert "STA_per_root_cause" in ds_results
    assert "STA_per_root_cause_summary" in ds_results
    summary = ds_results["STA_per_root_cause_summary"]
    assert summary["source_model_key"] == "STA"
    assert summary["num_seeds_aggregated"] == 1
    assert set(summary["mean_top1_by_service"].keys()) == {"svc0", "svc1", "svc2"}
