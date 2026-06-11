"""Tests for the stdlib JSON log formatter (serving/logging_config.py)."""

import json
import logging

from serving.logging_config import JsonFormatter, configure_logging


def _format(record_args: dict) -> dict:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=record_args.pop("msg", "hello %s"), args=record_args.pop("args", ("world",)),
        exc_info=None,
    )
    for key, value in record_args.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_one_object_per_line():
    payload = _format({})
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["ts"].endswith("Z")


def test_json_formatter_merges_extra_fields():
    payload = _format({"request_id": "abc123", "latency_ms": 4.2})
    assert payload["request_id"] == "abc123"
    assert payload["latency_ms"] == 4.2


def test_configure_logging_json_toggle(monkeypatch, capsys):
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging()
    logging.getLogger("toggle-test").info("structured", extra={"k": "v"})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["msg"] == "structured"
    assert parsed["k"] == "v"

    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging()  # restore text handler for the rest of the session
