"""Unit tests for LokiClient (pure logic, no network)."""

from unittest.mock import patch

from rca_pipeline.loki_connector import LokiClient


class _FakeResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _mk_stream(svc: str, lines):
    return {
        "stream": {"name": svc},
        "values": [[str(1_700_000_000 * 1_000_000_000 + i), ln] for i, ln in enumerate(lines)],
    }


def test_query_range_parses_success():
    fake = {"status": "success", "data": {"result": [_mk_stream("carts", ["ERROR boom", "INFO ok"])]}}
    with patch("rca_pipeline.loki_connector.requests.get", return_value=_FakeResp(fake)):
        cli = LokiClient("http://loki:3100")
        out = cli.query_range('{namespace="sock-shop"}', 0, 1)
        assert len(out) == 1
        assert out[0]["stream"]["name"] == "carts"


def test_fetch_lines_by_service_falls_back_labels():
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["query"])
        # Succeed only when the container label is used.
        if 'container="carts"' in params["query"]:
            return _FakeResp({"status": "success", "data": {"result": [_mk_stream("carts", ["ERROR x"])]}})
        return _FakeResp({"status": "success", "data": {"result": []}})

    with patch("rca_pipeline.loki_connector.requests.get", side_effect=fake_get):
        cli = LokiClient("http://loki:3100")
        out = cli.fetch_lines_by_service(0, 1, ["carts"], namespace="sock-shop")
    assert out["carts"] == ["ERROR x"]
    # First attempt should use app_kubernetes_io_name, then container (fallback hit)
    assert any('app_kubernetes_io_name="carts"' in q for q in calls)
    assert any('container="carts"' in q for q in calls)


def test_empty_services_returns_empty_dict():
    cli = LokiClient("http://loki:3100")
    out = cli.fetch_lines_by_service(0, 1, [], namespace="sock-shop")
    assert out == {}
