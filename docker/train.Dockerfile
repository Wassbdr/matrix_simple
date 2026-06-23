# syntax=docker/dockerfile:1.6
# Training / retraining image: full rca_pipeline + CPU torch + mlflow.
# Used by the retraining CronJob (M2.5) and as the initContainer that
# downloads the model bundle from the registry (M2.3c). Build from repo root:
#   docker build -f docker/train.Dockerfile -t rca-sta-train .

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements/constraints-cpu.txt .
COPY pyproject.toml .
COPY rca_pipeline/ ./rca_pipeline/
RUN apt-get update && apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --user --no-cache-dir \
        torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir torch_geometric==2.4.0 && \
    pip install --user --no-cache-dir -e ".[tracking]" -c constraints-cpu.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/rca_pipeline /app/rca_pipeline
COPY configs/ /app/configs/
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json

# Default: print help. The CronJob / initContainer overrides the command.
CMD ["python", "-m", "rca_pipeline.train_and_persist", "--help"]
