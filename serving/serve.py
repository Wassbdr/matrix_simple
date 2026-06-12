"""
FastAPI inference service for the STA RCA model.

Loads a serialized model bundle (see ``rca_pipeline/model_bundle.py``) at
startup and exposes a small REST API:

    POST /predict     ranked root-cause prediction on a single window
    GET  /health      liveness probe (always 200 once model is loaded)
    GET  /ready       readiness probe (smoke-tests a zero window)
    GET  /info        full bundle metadata.json
    GET  /metrics     Prometheus metrics (via prometheus-fastapi-instrumentator)

The model bundle directory is configured via the ``MODEL_BUNDLE``
environment variable (default: ``/app/checkpoints/sta_sockshop_v1``).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from rca_pipeline.model_bundle import LoadedBundle, load_bundle, predict
from serving.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

BUNDLE_PATH = os.environ.get("MODEL_BUNDLE", "/app/checkpoints/sta_sockshop_v1")
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "3"))

app = FastAPI(
    title="STA RCA Inference",
    description="Root cause analysis serving for microservice failures.",
    version="1.0.0",
)

# Custom counters / histograms (in addition to the auto-instrumentation)
PREDICT_COUNTER = Counter(
    "rca_predictions_total",
    "Total number of /predict calls.",
    ["status"],
)
PREDICT_LATENCY = Histogram(
    "rca_predict_latency_seconds",
    "End-to-end /predict latency including normalization.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
# Identifies the served model in dashboards and shadow comparisons.
MODEL_INFO = Gauge(
    "rca_model_info",
    "Static info about the loaded model bundle (value is always 1).",
    ["git_commit", "created_at", "num_nodes", "k"],
)

Instrumentator().instrument(app).expose(app)

_BUNDLE: Optional[LoadedBundle] = None


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.time()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    if request.url.path not in ("/metrics", "/health", "/ready"):
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round((time.time() - t0) * 1000.0, 2),
            },
        )
    return response


@app.on_event("startup")
def _load() -> None:
    global _BUNDLE
    if not os.path.isdir(BUNDLE_PATH):
        logger.error("MODEL_BUNDLE directory %s not found", BUNDLE_PATH)
        return
    t0 = time.time()
    _BUNDLE = load_bundle(BUNDLE_PATH)
    cfg = _BUNDLE.metadata["config"]
    card = _BUNDLE.metadata.get("model_card", {})
    MODEL_INFO.labels(
        git_commit=(card.get("git_commit") or "unknown")[:12],
        created_at=card.get("created_at_utc", "unknown"),
        num_nodes=str(cfg["num_nodes"]),
        k=str(cfg["k"]),
    ).set(1)
    n_params = sum(p.numel() for p in _BUNDLE.model.parameters())
    logger.info(
        "Loaded bundle from %s in %.2fs (N=%d F=%d H=%d k=%d, %d params)",
        BUNDLE_PATH, time.time() - t0, cfg["num_nodes"], cfg["num_features"],
        cfg["hidden_dim"], cfg["k"], n_params,
        extra={"bundle_commit": (card.get("git_commit") or "unknown")[:12]},
    )


# -----------------------------------------------------------------------------
# Pydantic request / response models
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-window prediction request.

    ``window`` shape: ``[N, F, W]`` raw (unnormalized) metric tensor.
    """
    window: List[List[List[float]]] = Field(
        ...,
        description="3D array shaped [num_nodes, num_features, window_size].",
    )
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50,
                       description="Number of top candidates to return.")


class PredictedCandidate(BaseModel):
    service: str
    score: float
    rank: int


class PredictResponse(BaseModel):
    # "model_version" intentionally shadows pydantic's protected "model_" namespace
    model_config = {"protected_namespaces": ()}

    model_version: str
    candidates: List[PredictedCandidate]
    latency_ms: float


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded")
    return {"status": "ok", "bundle": BUNDLE_PATH}


@app.get("/ready")
def ready():
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded")
    cfg = _BUNDLE.metadata["config"]
    zero = np.zeros((cfg["num_nodes"], cfg["num_features"], 12), dtype=np.float32)
    try:
        predict(_BUNDLE, zero, top_k=1)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Smoke predict failed: {exc}") from exc
    return {"status": "ready"}


@app.get("/info")
def info():
    if _BUNDLE is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded")
    return _BUNDLE.metadata


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if _BUNDLE is None:
        PREDICT_COUNTER.labels(status="unavailable").inc()
        raise HTTPException(status_code=503, detail="Bundle not loaded")

    cfg = _BUNDLE.metadata["config"]
    try:
        arr = np.asarray(req.window, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        PREDICT_COUNTER.labels(status="bad_request").inc()
        raise HTTPException(status_code=400, detail=f"window not array-like: {exc}") from exc
    if arr.ndim != 3:
        PREDICT_COUNTER.labels(status="bad_request").inc()
        raise HTTPException(
            status_code=400,
            detail=f"window must be 3D [N,F,W]; got shape={list(arr.shape)}",
        )
    if arr.shape[0] != cfg["num_nodes"]:
        PREDICT_COUNTER.labels(status="bad_request").inc()
        raise HTTPException(
            status_code=400,
            detail=f"N mismatch: model expects {cfg['num_nodes']}, got {arr.shape[0]}",
        )
    if arr.shape[1] != cfg["num_features"]:
        PREDICT_COUNTER.labels(status="bad_request").inc()
        raise HTTPException(
            status_code=400,
            detail=f"F mismatch: model expects {cfg['num_features']}, got {arr.shape[1]}",
        )

    t0 = time.time()
    try:
        ranked = predict(_BUNDLE, arr, top_k=req.top_k)
    except Exception as exc:
        PREDICT_COUNTER.labels(status="error").inc()
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"predict failed: {exc}") from exc
    elapsed = time.time() - t0
    PREDICT_LATENCY.observe(elapsed)
    PREDICT_COUNTER.labels(status="ok").inc()

    return PredictResponse(
        model_version=_BUNDLE.metadata["model_card"].get("created_at_utc", "unknown"),
        candidates=[
            PredictedCandidate(service=svc, score=score, rank=i + 1)
            for i, (svc, score) in enumerate(ranked)
        ],
        latency_ms=elapsed * 1000.0,
    )
