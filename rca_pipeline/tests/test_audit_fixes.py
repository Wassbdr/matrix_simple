"""Regression tests for audit-driven fixes (imports, bootstrap metadata)."""

import numpy as np

from rca_pipeline.metrics_rcaeval import bootstrap_ci


def test_bootstrap_ci_flags_low_seed_count():
    ci = bootstrap_ci([0.2, 0.25, 0.22], n_boot=100, seed=0)
    assert ci["n"] == 3
    assert ci["seed_level_bootstrap_reliable"] is False


def test_bootstrap_ci_reliable_with_enough_values():
    rng = np.random.default_rng(0)
    vals = [float(rng.random()) for _ in range(12)]
    ci = bootstrap_ci(vals, n_boot=200, seed=0)
    assert ci["n"] == 12
    assert ci["seed_level_bootstrap_reliable"] is True


def test_observit_loader_imports():
    from rca_pipeline.loaders import observit_loader

    assert callable(observit_loader.load_observit)
    assert callable(observit_loader.resolve_observit_csv)


def test_benchmark_ablation_imports():
    import rca_pipeline.benchmark_ablation as ba

    assert ba.stratified_split_shuffle is not None
    assert callable(ba.run_ablation)
