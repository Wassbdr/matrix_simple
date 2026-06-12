# Operations Runbook

How to train, serve, observe, and roll back the STA model in practice.
(Phase-2 MLOps additions — MLflow registry, Helm, retraining CronJob — will
extend this document.)

## 1. Configuration layer

Every CLI entry point accepts `--config <yaml>` (validated by pydantic,
`rca_pipeline/config.py`). Precedence: **explicit CLI flag > YAML > default**.
Versioned configs live in `configs/`:

| File | Entry point |
|---|---|
| `configs/benchmark_sim_smoke.yaml` | `python -m rca_pipeline.benchmark` (CI smoke) |
| `configs/benchmark_real.yaml` | `python -m rca_pipeline.benchmark_real` |
| `configs/hpo_sockshop.yaml` | `python -m rca_pipeline.hpo_sta` |
| `configs/train_sockshop.yaml` | `python -m rca_pipeline.train_and_persist` |

Unknown keys and type errors fail loudly. The real-data guardrail
(`sta_contrastive_warmup > 0` on real datasets, README §5.4) is rejected at
YAML load time unless `allow_real_warmup: true` is set explicitly.

## 2. Model lifecycle (current state)

```
hpo_sta --config configs/hpo_sockshop.yaml          # Optuna study → best params JSON
train_and_persist --config configs/train_sockshop.yaml
                                                    # → bundle dir (model.pt, scaler.pt, metadata.json)
uvicorn serving.serve:app                           # serves the bundle
```

The bundle is **not** baked into the Docker image. The serve container reads
`MODEL_BUNDLE` (default `/app/checkpoints/sta_sockshop_v1`); in Kubernetes the
bundle is mounted from a volume (`serving/k8s-deployment.yaml`). Each bundle's
`metadata.json` carries full provenance: git commit, training info, HPO source,
torch/python versions.

## 3. Dependency pins

`requirements/constraints-cpu.txt` is the single source of truth for the
CPU ABI pins (`torch==2.0.1`, `torch_geometric==2.4.0`, `numpy<2`). CI, both
Dockerfiles and local installs consume it:

```
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,serve]" -c requirements/constraints-cpu.txt
```

Never install numpy>=2 in the same env as torch 2.0.1 (import-time ABI break).

## 4. Serving

```
MODEL_BUNDLE=checkpoints/sta_sockshop_v1 uvicorn serving.serve:app --port 8080
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | `{window: [N,F,W], top_k}` → ranked candidates |
| `GET /health` / `/ready` | liveness / readiness (smoke predict) |
| `GET /info` | full bundle metadata (provenance) |
| `GET /metrics` | Prometheus metrics |

Docker images (build from repo root):

```
docker build -f serving/Dockerfile -t rca-sta-serve .
docker build -f serving/Dockerfile.watcher -t rca-sta-watcher .
```

## 5. Logs & metrics

- `LOG_FORMAT=json` (recommended in-cluster) emits one JSON object per line
  (Loki-friendly); `LOG_FORMAT=text` for local runs. `LOG_LEVEL` honored.
- Each `/predict` log line carries `request_id` (also returned as the
  `x-request-id` response header) and `latency_ms`.
- Key Prometheus series:
  - `rca_predictions_total{status}` — ok / bad_request / error / unavailable
  - `rca_predict_latency_seconds` — histogram
  - `rca_model_info{git_commit,created_at,...}` — identifies the served bundle
  - `rca_watcher_*` — polling loop, alerts, ntfy deliveries

## 6. Watcher & alerting

`serving/rca_watcher.py` polls Prometheus, calls `/predict`, posts alerts to
Alertmanager and/or directly to ntfy. Alertmanager webhooks received on
`/ntfy-relay` feed the dashboard (`/dashboard`) and `/api/alerts`
(`source: webhook`, with AM polling as fallback).

## 7. Rollback (current state)

Bundles are directories: keep previous versions side by side and point
`MODEL_BUNDLE` (or the k8s volume) at the previous one, then restart the pod.
Phase 2 replaces this with MLflow registry aliases (`@champion`/`@challenger`)
and `helm rollback`.

## 8. Scaling notes (deliberate simplifications)

- MLflow (Phase 2) will run on SQLite + PVC: single writer, 1 replica,
  `Recreate` strategy. At scale: Postgres backend + object-store artifacts.
- CD will use a self-hosted runner on `workflow_dispatch`/`main` only
  (public repo: never run self-hosted on `pull_request`). At scale: Argo CD.
