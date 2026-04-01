"""Tests for TestTimeAdaptation behavior."""

import torch

from rca_pipeline.drift_detector import TestTimeAdaptation, make_k8s_drift_detector
from rca_pipeline.model_sta import SelectThenAnalyze


def test_tta_should_adapt_no_attribute_error():
    model = SelectThenAnalyze(num_features=5, hidden_dim=16, num_nodes=10, k=3)
    detector = make_k8s_drift_detector()
    tta = TestTimeAdaptation(model, detector)
    assert isinstance(tta.should_adapt(), bool)


def test_tta_adapt_and_restore():
    model = SelectThenAnalyze(num_features=5, hidden_dim=16, num_nodes=10, k=3)
    detector = make_k8s_drift_detector()
    tta = TestTimeAdaptation(model, detector)

    windows = torch.randn(3, 10, 5, 12)
    edge_index = torch.randint(0, 10, (2, 20))

    tta.adapt(windows, edge_index)
    assert tta._is_adapted

    tta.restore()
    assert not tta._is_adapted
