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

## 7. MLOps lifecycle (Phase 2)

End-to-end: HPO → train+register → gate → deploy → monitor → drift → retrain.

### MLflow (tracking + registry)
`deploy/mlflow/mlflow.yaml` runs MLflow 2.16 in namespace `mlops`, SQLite +
artifacts on one PVC, `replicas: 1` + `Recreate` (SQLite single-writer),
basic-auth. UI: `kubectl -n mlops port-forward svc/mlflow 5000:5000`.

Training is tracking-aware but offline-capable: `rca_pipeline/tracking.py`
is a complete no-op unless `MLFLOW_TRACKING_URI` is set. With it set,
`train_and_persist` logs params/metrics, logs the bundle dir *as the
artifact* (no pyfunc wrapping), registers model `sta-sockshop` with alias
`challenger`, and writes the run_id into the bundle's `metadata.json`
(surfaced by serving `/info`).

### Images & CD
`.github/workflows/release.yml` builds+pushes `rca-sta-{serve,watcher,train}`
to GHCR (public, sha+semver) on main/tags. `docker/train.Dockerfile` is the
training/retraining/initContainer image. `.dockerignore` keeps data out of
the build context. `deploy.yml` runs `helm upgrade` on a **self-hosted**
runner (cluster has private IPs) — `main`/`workflow_dispatch` only, never
`pull_request`.

### Serving pulls the model by alias
The serve pod's initContainer runs `serving/fetch_bundle.py` to download
`models:/sta-sockshop@champion` into an emptyDir at `MODEL_BUNDLE`. Deleting
the pod re-pulls the current champion. Roll back by moving the alias and
restarting, or `helm rollback`.

### Retraining (continuous training)
Helm CronJob `rca-retrain` (suspended by default):
```
kubectl create job --from=cronjob/rca-retrain retrain-manual-$(date +%s) -n rca-sandbox
```
It runs `train_and_persist` against the dataset PVC, logging a new
`@challenger`. Promotion is a deliberate human-in-the-loop gate:
```
python -m rca_pipeline.promote --model sta-sockshop --metric mrr_mean   # verdict
python -m rca_pipeline.promote --model sta-sockshop --apply             # move @champion
```

### Shadow comparison (canary substitute)
`--set shadow.enabled=true` deploys `rca-sta-serve-shadow` tracking
`@challenger`. The watcher POSTs each window to both and exports
`rca_watcher_shadow_top1_agree_total` / `_score_delta`. Drive traffic with a
Chaos Mesh session, then read agreement/latency in Grafana before promoting.

### Drift detection
CronJob `rca-drift` (every 6h) runs `rca_pipeline/monitoring/drift_job.py`:
Evidently `DataDriftPreset` (PSI fallback without the extra) comparing the
bundle's `reference_windows.parquet` to the last N hours from Prometheus,
pushing `rca_drift_share` to Pushgateway. Alert `RCADataDriftSustained`
(`> 0.3 for 12h`) → ntfy: **alert, do not auto-retrain** — trigger manually.

### Dashboard
Import `deploy/grafana/rca-model-slo.json`: latency p95/p99, prediction/error
rate, model commit, shadow agreement, drift share, retrain status, alert
timeline.

## 8. Scaling notes (deliberate simplifications)

- MLflow on SQLite + PVC (1 replica, `Recreate`). At scale: Postgres backend
  + object-store artifacts (documented path, not implemented).
- CD via a self-hosted runner on `workflow_dispatch`/`main` only. At scale:
  Argo CD (pull-based GitOps, no runner exposure).
- Promotion is manual. At scale: automated gate + canary traffic-split.
