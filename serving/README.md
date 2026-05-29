# STA RCA Inference Service

End-to-end serving of the STA model trained on sock-shop cluster data.

## What this directory contains

| File | Purpose |
|---|---|
| `serve.py` | FastAPI inference service (`/predict`, `/health`, `/ready`, `/info`, `/metrics`) |
| `requirements.txt` | Runtime Python deps (CPU torch + fastapi stack) |
| `Dockerfile` | Multi-stage container build |
| `k8s-deployment.yaml` | Kubernetes Deployment + Service for `rca-sandbox` namespace |
| `tests/test_serve.py` | Smoke tests for the FastAPI service |

## Train, package, deploy

```bash
# 1) Optuna HPO on sock-shop windows (~30-60 min on CPU)
python -m rca_pipeline.hpo_sta \
    --dataset-dir datasets/sockshop_runs \
    --topology sock_shop \
    --n-trials 25 \
    --output-json results_transmission/hpo_sta_sockshop.json

# 2) Train final model with best params, persist as bundle (~10 min)
python -m rca_pipeline.train_and_persist \
    --dataset-dir datasets/sockshop_runs \
    --topology sock_shop \
    --hpo-json results_transmission/hpo_sta_sockshop.json \
    --seeds 42,123,7,1337,2025 \
    --output-bundle checkpoints/sta_sockshop_v1 \
    --no-warmup

# 3) Smoke-test the bundle locally
python -m rca_pipeline.model_bundle checkpoints/sta_sockshop_v1

# 4) Build container
docker build -t rca-sta-serve:v1 -f serving/Dockerfile .

# 5) Local smoke test
docker run --rm -p 8080:8080 rca-sta-serve:v1 &
curl http://localhost:8080/health
curl http://localhost:8080/info | jq .config

# 6) Push + deploy (registry placeholder)
docker tag rca-sta-serve:v1 <REGISTRY>/rca-sta-serve:v1
docker push <REGISTRY>/rca-sta-serve:v1
sed -i "s|image: rca-sta-serve:v1|image: <REGISTRY>/rca-sta-serve:v1|" serving/k8s-deployment.yaml
kubectl apply -f serving/k8s-deployment.yaml
kubectl rollout status deployment/rca-sta-serve -n rca-sandbox

# 7) Verify
kubectl port-forward -n rca-sandbox svc/rca-sta-serve 8080:8080 &
curl http://localhost:8080/info
```

## API

`POST /predict`:
```json
{
  "window": [[[...]]],   // shape [N, F, W]  (raw values, the service normalizes)
  "top_k": 3
}
```

Response:
```json
{
  "model_version": "2026-05-21T...Z",
  "candidates": [
    {"service": "catalogue-db", "score": 4.17, "rank": 1},
    {"service": "catalogue",    "score": 2.03, "rank": 2},
    {"service": "front-end",    "score": 1.85, "rank": 3}
  ],
  "latency_ms": 28.3
}
```

`GET /info` returns the full bundle metadata.json (training metrics,
HPO provenance, scaler stats, git commit, etc).

`GET /metrics` exposes Prometheus metrics including custom counters
`rca_predictions_total{status=ok|error|bad_request}` and the histogram
`rca_predict_latency_seconds`.
