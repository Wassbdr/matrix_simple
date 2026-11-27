import json

import torch

from rca_pipeline import scaling_benchmark
from rca_pipeline.model_sta import SelectThenAnalyze


def test_cpu_inference_preserves_selector_and_deep_scores():
    torch.manual_seed(9)
    model = SelectThenAnalyze(num_features=5, hidden_dim=16, num_nodes=20, k=6).eval()
    x = torch.randn(2, 20, 5, 12)
    edges = torch.tensor([[0, 1, 5, 6], [1, 0, 6, 5]])
    expected = model(x, edges)
    with torch.no_grad():
        actual = model(x, edges)
    for base, optimized in zip(expected, actual):
        torch.testing.assert_close(optimized, base, atol=1e-6, rtol=1e-5)


def test_latency_only_creates_report_and_caps_sta_candidates(tmp_path, monkeypatch):
    path = tmp_path / "latency.json"
    monkeypatch.setattr(scaling_benchmark.logging, "disable", lambda _: None)
    monkeypatch.setattr(scaling_benchmark, "set_torch_deterministic", lambda _: None)
    scaling_benchmark.main([
        "--latency-only", "--latency-sizes", "14,20", "--models", "STA,FullGraphGCN",
        "--latency-k", "4", "--latency-repeats", "1", "--output-json", str(path),
    ])
    report = json.loads(path.read_text())
    assert report["accuracy"] == []
    assert [row["k"] for row in report["latency"]] == [4, 4]
    assert report["latency_slope_ms_per_1000_nodes"].keys() == {"STA", "FullGraphGCN"}
