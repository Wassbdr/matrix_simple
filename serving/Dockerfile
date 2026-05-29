# syntax=docker/dockerfile:1.6
# Multi-stage build for the STA inference service.
# Stage 1: install Python deps with pip user-site so we can copy them
# clean into a slim runtime image.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY serving/requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 2: lean runtime
FROM python:3.11-slim

# Install minimal runtime deps (curl needed only for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MODEL_BUNDLE=/app/checkpoints/sta_sockshop_v1

# Copy only what the service needs
COPY rca_pipeline/ /app/rca_pipeline/
COPY serving/serve.py /app/serving/serve.py
COPY checkpoints/sta_sockshop_v1/ /app/checkpoints/sta_sockshop_v1/

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "serving.serve:app", "--host", "0.0.0.0", "--port", "8080"]
