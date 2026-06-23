"""Tests for the MLflow tracking wrapper (rca_pipeline/tracking.py).

The critical contract is the no-op path: with no MLFLOW_TRACKING_URI every
call must be a silent no-op and run() must yield None, so research/CI never
depend on a tracking server.
"""

from rca_pipeline import tracking


def test_disabled_when_no_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.is_enabled() is False


def test_run_yields_none_when_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with tracking.run(experiment="x", run_name="y") as run_id:
        assert run_id is None
        # All logging calls must be safe no-ops inside a disabled run.
        tracking.log_params({"hidden_dim": 32, "nested": {"a": 1}})
        tracking.log_metrics({"avg1": 0.79, "none_val": None})
        tracking.log_artifacts("/nonexistent/dir")
    assert tracking.active_run_id() is None


def test_register_model_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.register_model("/some/bundle", name="sta-sockshop") is None


def test_uri_set_but_mlflow_missing_disables(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated missing mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tracking.is_enabled() is False
