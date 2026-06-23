"""Thin MLflow wrapper — a complete no-op when MLflow is unconfigured.

Design goal: research and CI must stay offline-capable. Nothing in the
training/benchmark code may hard-depend on an MLflow server. So:

  * if ``MLFLOW_TRACKING_URI`` is unset, or the ``mlflow`` package is not
    installed, every call here is a silent no-op and ``run()`` yields a
    null context;
  * only when a tracking URI is present do we import mlflow and log.

This keeps ``mlflow`` in the optional ``[tracking]`` extra, out of core
dependencies.

Typical use in an entry point::

    from rca_pipeline import tracking

    with tracking.run(experiment="sta-sockshop", run_name="train"):
        tracking.log_params({"hidden_dim": 32, "k": 5})
        tracking.log_metrics({"avg1_mean": 0.79})
        tracking.log_artifacts(bundle_dir)
        version = tracking.register_model(bundle_dir, name="sta-sockshop")
        run_id = tracking.active_run_id()
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    """True only if a tracking URI is set *and* mlflow imports."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return False
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "MLFLOW_TRACKING_URI is set but mlflow is not installed; "
            "install the [tracking] extra. Tracking is disabled."
        )
        return False


@contextmanager
def run(
    experiment: str = "default",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[Optional[str]]:
    """Start an MLflow run, or a null context when tracking is disabled.

    Yields the run_id (str) when active, else None.
    """
    if not is_enabled():
        yield None
        return
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as active:
        yield active.info.run_id


def active_run_id() -> Optional[str]:
    if not is_enabled():
        return None
    import mlflow
    active = mlflow.active_run()
    return active.info.run_id if active else None


def log_params(params: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    import mlflow
    # MLflow params must be scalars/strings; stringify nested structures.
    flat = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in params.items()}
    mlflow.log_params(flat)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if not is_enabled():
        return
    import mlflow
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    mlflow.log_metrics(clean, step=step)


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None:
    if not is_enabled():
        return
    import mlflow
    mlflow.log_artifacts(local_dir, artifact_path=artifact_path)


def register_model(
    bundle_dir: str,
    name: str,
    alias: str = "challenger",
    artifact_path: str = "bundle",
) -> Optional[str]:
    """Log the bundle dir as an artifact and register it as a model version.

    The bundle format (model.pt / scaler.pt / metadata.json) *is* the
    artifact — we deliberately do not wrap it in mlflow.pyfunc. Returns the
    new version string, or None when tracking is disabled.
    """
    if not is_enabled():
        return None
    import mlflow

    run_id = active_run_id()
    if run_id is None:
        logger.warning("register_model called with no active run; skipping.")
        return None
    mlflow.log_artifacts(bundle_dir, artifact_path=artifact_path)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=name)
    try:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name, alias, mv.version)
    except Exception as exc:  # alias API varies across mlflow versions
        logger.warning("Could not set alias %s on %s v%s: %s",
                       alias, name, mv.version, exc)
    logger.info("Registered model %s version %s (alias=%s)", name, mv.version, alias)
    return mv.version
