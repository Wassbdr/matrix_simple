# Rank an incident from your own metrics

Export timestamps and numeric readings to CSV. Use one row per timestamp, a `time` column, and one `service_metric` column per signal. Use a hyphen inside service names; the first underscore separates the service from its metric. Timestamps must use the same units as the incident start. Include a normal period before the start and readings at or after it. Do not export service identifiers that contain personal data.

```sh
rca rank examples/incident.csv --inject-time 1700000002 --top-k 2
rca rank examples/incident.csv --inject-time 1700000002 --json
```

To reuse a timestamp, put it in an adjacent `inject_time.txt`. A case directory may contain RCAEval `metrics.json`, `metrics.parquet` (install the `data` extra), `data.csv`, or `metrics.csv`.

```python
from rca_pipeline import rank

for candidate in rank("examples/incident.csv", inject_time=1700000002, top_k=2):
    print(candidate.rank, candidate.service, candidate.score)
```

Scores are relative to this incident, not probabilities. Check the leading service against deployment changes, errors, and upstream dependencies. If the root service has no metric in the export, the model cannot return it.

## HTTP

Start the default model with `docker compose up --build -d` or install the `serve` extra and run `uvicorn serving.serve:app` from the checkout. Send the same metrics in column-oriented JSON:

```sh
curl --fail-with-body http://localhost:8080/predict \
  -H 'Content-Type: application/json' --data-binary @examples/incident_request.json
```

Use `GET /ready` to check model readiness and `GET /info` for its version. A directory passed via `MODEL_BUNDLE` overrides the packaged model and keeps the older tensor request contract.
