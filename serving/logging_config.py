"""Structured logging for the serving stack (stdlib only, no extra deps).

``LOG_FORMAT=json`` (default in containers) emits one JSON object per line -
directly ingestible by Loki/promtail. ``LOG_FORMAT=text`` keeps the classic
human-readable format for local runs.

Extra fields passed via ``logger.info(..., extra={...})`` are merged into the
JSON payload (e.g. request_id, latency_ms, bundle_commit).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

_STD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure the root logger once, honouring ``LOG_FORMAT`` and ``LOG_LEVEL``."""
    fmt = os.environ.get("LOG_FORMAT", "text").lower()
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
