"""
rca-watcher: polling loop + ntfy adapter + small dashboard.

Three roles:

1. polling loop -- queries Prometheus, calls /predict, fires alerts
   to Alertmanager (every POLL_INTERVAL_S seconds);
2. /ntfy-relay  -- receives Alertmanager webhook payloads and
   re-posts them as nicely-formatted ntfy.sh notifications with
   Title / Priority / Tags HTTP headers (so the phone push has an
   emoji and a one-line title instead of raw JSON);
3. /dashboard   -- single-page HTML overview combining active
   alerts, recent predictions, model bundle info, and live ntfy
   stream URL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven, sane defaults for in-cluster deploy)
# ---------------------------------------------------------------------------

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://prometheus-server.monitoring-metrics.svc.cluster.local",
)
INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://rca-sta-serve.rca-sandbox.svc.cluster.local:8080",
)
ALERTMANAGER_URL = os.environ.get(
    "ALERTMANAGER_URL",
    "http://monitoring-kube-prometheus-alertmanager.monitoring.svc.cluster.local:9093",
)
NTFY_BASE_URL = os.environ.get("NTFY_BASE_URL", "https://ntfy.sh")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "wassim-rca-ae47c51eb01cd276")
TARGET_NAMESPACE = os.environ.get("TARGET_NAMESPACE", "sock-shop")
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "60"))
WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "360"))  # 6 min
STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "30"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "5.0"))
ERROR_RATE_TRIGGER = float(os.environ.get("ERROR_RATE_TRIGGER", "0.01"))
TOP_K = int(os.environ.get("TOP_K", "3"))
RECENT_BUFFER = int(os.environ.get("RECENT_BUFFER", "50"))
# Direct ntfy push from the watcher itself (bypass Alertmanager). Useful when
# the cluster doesn't run prometheus-operator (no AlertmanagerConfig CRD,
# no AM Service). When enabled and an alert fires, the watcher posts to
# ntfy directly in addition to posting to AM.
NTFY_DIRECT = os.environ.get("NTFY_DIRECT", "true").lower() in ("1", "true", "yes")

NODE_NAMES: List[str] = [
    "front-end", "catalogue", "catalogue-db", "carts", "carts-db",
    "orders", "orders-db", "user", "user-db", "session-db",
    "payment", "shipping", "rabbitmq", "queue-master",
]
N = len(NODE_NAMES)
F = 6
W = WINDOW_SECONDS // STEP_SECONDS

# ---------------------------------------------------------------------------
# In-process state (lock-protected ring buffer of recent predictions)
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_RECENT: Deque[dict] = deque(maxlen=RECENT_BUFFER)

# ---------------------------------------------------------------------------
# Prometheus client metrics
# ---------------------------------------------------------------------------

WATCHER_POLLS = Counter("rca_watcher_polls_total", "Polls executed", ["status"])
WATCHER_ALERTS = Counter("rca_watcher_alerts_total", "Alerts fired",
                          ["root_cause", "severity"])
WATCHER_NTFY = Counter("rca_watcher_ntfy_total", "Ntfy relay deliveries",
                        ["status"])
WATCHER_LAST_TOP_SCORE = Gauge("rca_watcher_last_top_score",
                                "Score of the top-1 candidate at the last poll")
WATCHER_LAST_ERROR_RATE = Gauge("rca_watcher_last_observed_error_rate",
                                 "Cluster-wide error rate at the last poll")


# ---------------------------------------------------------------------------
# Prometheus query helpers
# ---------------------------------------------------------------------------

def _promql_query_range(query: str, start: float, end: float, step: int) -> dict:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=8,
    )
    r.raise_for_status()
    return r.json()


def _promql_query_instant(query: str) -> dict:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def _series_by_node(result: list, w: int) -> np.ndarray:
    arr = np.zeros((N, w), dtype=np.float32)
    for series in result:
        pod = series.get("metric", {}).get("pod", "")
        parts = pod.split("-")
        node = None
        for n_try in range(len(parts), 0, -1):
            cand = "-".join(parts[:n_try])
            if cand in NODE_NAMES:
                node = cand
                break
        if node is None:
            continue
        idx = NODE_NAMES.index(node)
        values = series.get("values", [])
        for j, (_ts, v) in enumerate(values[-w:]):
            try:
                arr[idx, j] = float(v)
            except (TypeError, ValueError):
                pass
    return arr


def collect_window() -> np.ndarray:
    now = time.time()
    start = now - WINDOW_SECONDS
    ns = TARGET_NAMESPACE
    queries = {
        "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}}[2m])) by (pod)',
        "ram_usage": f'sum(container_memory_working_set_bytes{{namespace="{ns}",container!=""}}) by (pod)',
        "network_tx": f'sum(rate(container_network_transmit_bytes_total{{namespace="{ns}"}}[2m])) by (pod)',
        "network_rx": f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}"}}[2m])) by (pod)',
        "error_rate": f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}",code=~"5.."}}[2m])) by (pod)',
        "latency_p99": (
            f'histogram_quantile(0.99,'
            f' sum(rate(http_request_duration_seconds_bucket{{namespace="{ns}"}}[2m])) by (pod, le))'
        ),
    }
    feature_order = ["cpu_usage", "ram_usage", "network_tx", "network_rx",
                     "error_rate", "latency_p99"]
    tensor = np.zeros((N, F, W), dtype=np.float32)
    for f_idx, feat in enumerate(feature_order):
        try:
            resp = _promql_query_range(queries[feat], start, now, STEP_SECONDS)
        except requests.RequestException as exc:
            logger.warning("Prometheus query failed for %s: %s", feat, exc)
            continue
        result = resp.get("data", {}).get("result", [])
        m = _series_by_node(result, W)
        tensor[:, f_idx, :] = m
    return tensor


def observed_error_rate() -> float:
    ns = TARGET_NAMESPACE
    q = (
        f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}",code=~"5.."}}[5m]))'
        f' / clamp_min(sum(rate(http_request_duration_seconds_count{{namespace="{ns}"}}[5m])), 1e-9)'
    )
    try:
        resp = _promql_query_instant(q)
        result = resp.get("data", {}).get("result", [])
        if not result:
            return 0.0
        return float(result[0]["value"][1])
    except (requests.RequestException, ValueError, KeyError):
        return 0.0


# ---------------------------------------------------------------------------
# Inference call
# ---------------------------------------------------------------------------

def call_predict(window: np.ndarray) -> Optional[List[Tuple[str, float, int]]]:
    payload = {"window": window.tolist(), "top_k": TOP_K}
    try:
        r = requests.post(f"{INFERENCE_URL}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        logger.warning("predict request failed: %s", exc)
        return None
    if r.status_code != 200:
        logger.warning("predict returned %d: %s", r.status_code, r.text[:200])
        return None
    body = r.json()
    cands = body.get("candidates", [])
    return [(c["service"], float(c["score"]), int(c["rank"])) for c in cands]


# ---------------------------------------------------------------------------
# Alerting (Alertmanager)
# ---------------------------------------------------------------------------

def post_alertmanager(root_cause: str, score: float, runners_up: List[Tuple[str, float]],
                       err_rate: float, severity: str = "warning") -> bool:
    payload = [
        {
            "labels": {
                "alertname": "RCAStaServeRootCauseDetected",
                "severity": severity,
                "component": "rca-watcher",
                "namespace": TARGET_NAMESPACE,
                "root_cause": root_cause,
                "model": "sta_sockshop_v1",
            },
            "annotations": {
                "summary": f"STA identifies root cause: {root_cause}",
                "description": (
                    f"Top-1 candidate = {root_cause} (score {score:.3f}). "
                    f"Cluster error rate = {err_rate:.1%}. "
                    f"Runners-up: {', '.join(f'{s}({sc:.2f})' for s, sc in runners_up)}."
                ),
            },
        }
    ]
    try:
        r = requests.post(f"{ALERTMANAGER_URL}/api/v2/alerts", json=payload, timeout=5)
        return r.status_code < 400
    except requests.RequestException as exc:
        logger.warning("Alertmanager POST failed: %s", exc)
        return False


def maybe_fire_alert(ranked: List[Tuple[str, float, int]], err_rate: float) -> Optional[dict]:
    if not ranked:
        return None
    top_svc, top_score, _ = ranked[0]
    WATCHER_LAST_TOP_SCORE.set(top_score)
    if top_score < CONFIDENCE_THRESHOLD:
        return None
    if err_rate < ERROR_RATE_TRIGGER:
        return None
    severity = "critical" if err_rate > 0.05 else "warning"
    runners = [(s, sc) for s, sc, _ in ranked[1:]]
    logger.warning("ALERT: root_cause=%s score=%.3f err_rate=%.1f%% severity=%s",
                   top_svc, top_score, err_rate * 100.0, severity)
    sent = post_alertmanager(top_svc, top_score, runners, err_rate, severity)
    WATCHER_ALERTS.labels(root_cause=top_svc, severity=severity).inc()
    logger.info("alertmanager delivery: %s", "ok" if sent else "failed")

    # Direct ntfy push (bypass AM). Useful when AM is not available on the
    # cluster (no prometheus-operator). Safe to keep on even when AM is
    # available: AM additionally renders a richer body via /ntfy-relay.
    ntfy_sent = False
    if NTFY_DIRECT:
        # No emoji in HTTP Title header (latin-1 only). Tags do it client-side.
        prefix = "CRITICAL" if severity == "critical" else "ALERT"
        title = f"[{prefix}] STA root cause: {top_svc}"
        body = (
            f"**Top-1 candidate**: `{top_svc}` (score `{top_score:.3f}`)\n\n"
            f"- cluster error rate: `{err_rate:.1%}`\n"
            f"- runners-up: {', '.join(f'{s} ({sc:.2f})' for s, sc in runners)}\n"
            f"- model: `sta_sockshop_v1`\n"
            f"- namespace: `{TARGET_NAMESPACE}`"
        )
        ntfy_sent = _emit_ntfy(title, body, severity)
        logger.info("ntfy direct delivery: %s", "ok" if ntfy_sent else "failed")

    return {
        "root_cause": top_svc,
        "score": top_score,
        "severity": severity,
        "delivered_am": sent,
        "delivered_ntfy": ntfy_sent,
    }


# ---------------------------------------------------------------------------
# FastAPI app + endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="RCA watcher", version="2.0.0")
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    return {
        "prometheus_url": PROMETHEUS_URL,
        "inference_url": INFERENCE_URL,
        "alertmanager_url": ALERTMANAGER_URL,
        "ntfy_base_url": NTFY_BASE_URL,
        "ntfy_topic": NTFY_TOPIC,
        "target_namespace": TARGET_NAMESPACE,
        "poll_interval_s": POLL_INTERVAL_S,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "n_samples_per_window": W,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "error_rate_trigger": ERROR_RATE_TRIGGER,
        "top_k": TOP_K,
        "nodes": NODE_NAMES,
    }


@app.get("/api/recent")
def api_recent():
    """Return the last N polls (timestamp, top-k candidates, err_rate, alert)."""
    with _STATE_LOCK:
        return {"items": list(_RECENT)}


@app.get("/api/alerts")
def api_alerts():
    """Proxy the operator Alertmanager active alerts (filtered to RCA only)."""
    try:
        r = requests.get(f"{ALERTMANAGER_URL}/api/v2/alerts", timeout=5)
        r.raise_for_status()
        alerts = r.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"AM unreachable: {exc}") from exc
    rca = [
        a for a in alerts
        if a.get("labels", {}).get("alertname", "").startswith("RCAStaServe")
    ]
    return {"count": len(rca), "alerts": rca}


@app.get("/api/info")
def api_info():
    """Proxy /info from the inference service."""
    try:
        r = requests.get(f"{INFERENCE_URL}/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Inference info unreachable: {exc}") from exc


# ---------------------------------------------------------------------------
# /ntfy-relay : translate Alertmanager webhook into nicely-formatted ntfy push
# ---------------------------------------------------------------------------

_SEVERITY_TAGS = {
    "critical": "rotating_light,fire",
    "warning":  "warning",
    "info":     "information_source",
}
_SEVERITY_PRIO = {
    "critical": "5",  # max
    "warning":  "3",  # default
    "info":     "2",  # low
}


def _emit_ntfy(title: str, body: str, severity: str, click_url: Optional[str] = None,
                resolved: bool = False) -> bool:
    headers = {
        "Title": title,
        "Tags": ("white_check_mark" if resolved else _SEVERITY_TAGS.get(severity, "bell")),
        "Priority": ("2" if resolved else _SEVERITY_PRIO.get(severity, "3")),
        "Markdown": "yes",
    }
    if click_url:
        headers["Click"] = click_url
    url = f"{NTFY_BASE_URL.rstrip('/')}/{NTFY_TOPIC}"
    try:
        r = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=5)
        ok = r.status_code < 400
        WATCHER_NTFY.labels(status="ok" if ok else "error").inc()
        return ok
    except requests.RequestException as exc:
        logger.warning("ntfy POST failed: %s", exc)
        WATCHER_NTFY.labels(status="exc").inc()
        return False


def _format_alert_body(alert: dict, common_labels: dict) -> str:
    labels = alert.get("labels", {})
    ann = alert.get("annotations", {})
    rc = labels.get("root_cause", "(n/a)")
    severity = labels.get("severity", "warning")
    summary = ann.get("summary", labels.get("alertname", "alert"))
    desc = ann.get("description", "")
    starts = alert.get("startsAt", "")
    fp = (alert.get("fingerprint", "") or "")[:8]
    parts = [
        f"**{summary}**",
        "",
        f"- root cause : `{rc}`",
        f"- severity   : `{severity}`",
        f"- namespace  : `{labels.get('namespace', '-') }`",
        f"- model      : `{labels.get('model', '-')}`",
        f"- startsAt   : {starts}",
        f"- fingerprint: {fp}",
    ]
    if desc:
        parts.extend(["", desc])
    return "\n".join(parts)


@app.post("/ntfy-relay")
async def ntfy_relay(req: Request):
    """Receive Alertmanager webhook payload; re-send as ntfy push per alert."""
    try:
        body = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    status = body.get("status", "firing")
    common = body.get("commonLabels", {})
    alerts = body.get("alerts", []) or []
    resolved = (status == "resolved")
    delivered = 0
    for a in alerts:
        labels = a.get("labels", {})
        severity = labels.get("severity", "warning")
        alertname = labels.get("alertname", "alert")
        rc = labels.get("root_cause", "")
        rc_part = f" | {rc}" if rc and rc != "-" else ""
        # HTTP headers are ISO-8859-1 only -- no emoji in Title. Emojis are
        # rendered client-side via the Tags header (handled by _emit_ntfy).
        prefix = "RESOLVED" if resolved else ("CRITICAL" if severity == "critical" else "ALERT")
        title = f"[{prefix}] {alertname}{rc_part}"
        body_md = _format_alert_body(a, common)
        click = a.get("generatorURL") or None
        if _emit_ntfy(title, body_md, severity, click_url=click, resolved=resolved):
            delivered += 1
    return {"received": len(alerts), "delivered": delivered, "status": status}


# ---------------------------------------------------------------------------
# /dashboard : single-page SPA
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>RCA Live Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { --bg:#0f1117; --fg:#e6e6e6; --muted:#8a8f98; --acc:#3aa0ff; --ok:#39d98a; --warn:#ffb547; --crit:#ff5252; --card:#161922; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif; }
  header { padding: 14px 20px; border-bottom:1px solid #222; display:flex; align-items:center; gap:14px; }
  header h1 { margin:0; font-size:18px; font-weight:600; }
  header .pulse { width:10px; height:10px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 0 var(--ok); animation:pulse 1.6s infinite; }
  header .right { margin-left:auto; font-size:12px; color:var(--muted); }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(57,217,138,.6)} 70%{box-shadow:0 0 0 9px rgba(57,217,138,0)} 100%{box-shadow:0 0 0 0 rgba(57,217,138,0)} }
  main { padding: 20px; display:grid; grid-template-columns: 1fr 1fr; gap: 18px; max-width: 1500px; margin: 0 auto; }
  .card { background: var(--card); border:1px solid #1f2330; border-radius: 10px; padding: 16px; }
  .card h2 { margin: 0 0 12px 0; font-size: 14px; font-weight:600; letter-spacing:.04em; color: var(--muted); text-transform: uppercase; }
  .card.full { grid-column: 1 / -1; }
  table { width:100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 6px; text-align: left; border-bottom: 1px solid #1f2330; }
  th { color: var(--muted); font-weight: 500; }
  .sev-warning { color: var(--warn); }
  .sev-critical { color: var(--crit); }
  .sev-resolved { color: var(--ok); }
  .pill { padding:2px 8px; border-radius:6px; font-size:11px; background:#252a36; }
  .pill.warning { background:#3a2a10; color:var(--warn); }
  .pill.critical { background:#3a1010; color:var(--crit); }
  .pill.ok { background:#102a1c; color:var(--ok); }
  .kv { display:grid; grid-template-columns: 140px 1fr; gap: 6px 12px; font-size: 13px; }
  .kv .k { color: var(--muted); }
  code { background:#252a36; padding:1px 6px; border-radius:4px; font-size:12px; }
  .small { font-size: 11px; color: var(--muted); }
  .row { display:flex; gap:8px; align-items:center; }
  .badge { padding:2px 8px; border-radius:10px; font-size:11px; background:#252a36; color: var(--fg); }
  .badge.up { background: rgba(57,217,138,.15); color: var(--ok); }
  .badge.down { background: rgba(255,82,82,.2); color: var(--crit); }
  .score { font-variant-numeric: tabular-nums; }
  .topcand { display:flex; gap:6px; flex-wrap: wrap; }
  .topcand .c { background:#1c2030; padding: 2px 8px; border-radius: 6px; font-size: 11px; }
  .topcand .c.top { background:#15384f; color:var(--acc); font-weight:600; }
  a, a:visited { color: var(--acc); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .timestamp { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 11px; }
</style>
</head>
<body>
<header>
  <div class="pulse"></div>
  <h1>RCA Live Dashboard</h1>
  <span class="badge" id="hdrModel">model: ?</span>
  <span class="badge" id="hdrTopic">ntfy: ?</span>
  <span class="right">poll: <span id="lastPollTs">?</span> &middot; auto-refresh 5s</span>
</header>

<main>
  <section class="card">
    <h2>Active alerts</h2>
    <div id="alerts">Loading...</div>
  </section>

  <section class="card">
    <h2>Bundle / Model card</h2>
    <div class="kv" id="bundle"></div>
  </section>

  <section class="card full">
    <h2>Recent polls (last <span id="recentN">0</span>)</h2>
    <table>
      <thead>
        <tr>
          <th>time</th>
          <th>top-1</th>
          <th>score</th>
          <th>err rate</th>
          <th>candidates</th>
          <th>alert</th>
        </tr>
      </thead>
      <tbody id="recent"></tbody>
    </table>
  </section>

  <section class="card full">
    <h2>Ntfy topic</h2>
    <div class="kv">
      <div class="k">URL</div>
      <div><a id="ntfyLink" target="_blank" rel="noreferrer"></a></div>
      <div class="k">App mobile</div>
      <div>Install "ntfy" sur Android/iOS, ajoute le topic ci-dessus, et tu recoivras les push instantanement.</div>
      <div class="k">CLI</div>
      <div><code id="ntfyCli"></code></div>
    </div>
  </section>
</main>

<script>
async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return await r.json();
}

function fmtTime(t) {
  if (!t) return "-";
  if (typeof t === "number") return new Date(t * 1000).toLocaleTimeString();
  return new Date(t).toLocaleString();
}

function fmtScore(s) {
  if (typeof s !== "number") return "-";
  return s.toFixed(3);
}

async function refreshAlerts() {
  try {
    const d = await fetchJSON("/api/alerts");
    if (d.count === 0) {
      document.getElementById("alerts").innerHTML =
        '<div class="small">Aucune alerte RCA active. <span class="badge ok">ok</span></div>';
      return;
    }
    const rows = d.alerts.map(a => {
      const L = a.labels || {};
      const A = a.annotations || {};
      const sev = (L.severity || "warning").toLowerCase();
      const rc = L.root_cause || "-";
      return `<tr>
        <td><span class="pill ${sev}">${sev}</span></td>
        <td><code>${rc}</code></td>
        <td>${L.alertname || "-"}</td>
        <td><span class="timestamp">${fmtTime(a.startsAt)}</span></td>
      </tr>`;
    }).join("");
    document.getElementById("alerts").innerHTML =
      `<table>
         <thead><tr><th>sev</th><th>root cause</th><th>alertname</th><th>startsAt</th></tr></thead>
         <tbody>${rows}</tbody>
       </table>`;
  } catch (e) {
    document.getElementById("alerts").innerHTML = '<div class="small">AM unreachable: ' + e + '</div>';
  }
}

async function refreshBundle() {
  try {
    const d = await fetchJSON("/api/info");
    const cfg = d.config || {};
    const train = d.training || {};
    const hpo = (d.hpo_provenance || {});
    const mc = d.model_card || {};
    document.getElementById("hdrModel").textContent = "model: " + (d.model_class || "?") +
      " (N=" + (cfg.num_nodes || "?") + ", k=" + (cfg.k || "?") + ")";
    document.getElementById("bundle").innerHTML = `
      <div class="k">Model class</div><div><code>${d.model_class || "?"}</code></div>
      <div class="k">Config</div><div>N=${cfg.num_nodes}, F=${cfg.num_features}, H=${cfg.hidden_dim}, k=${cfg.k}</div>
      <div class="k">Avg@1 (mean)</div><div class="score">${(train.avg1_mean || 0).toFixed(3)} &plusmn; ${(train.avg1_std || 0).toFixed(3)}</div>
      <div class="k">MRR (mean)</div><div class="score">${(train.mrr_mean || 0).toFixed(3)} &plusmn; ${(train.mrr_std || 0).toFixed(3)}</div>
      <div class="k">Best seed</div><div>${train.best_seed}</div>
      <div class="k">HPO best MRR</div><div class="score">${(hpo.hpo_best_value_mrr || 0).toFixed(4)}</div>
      <div class="k">Git commit</div><div><code>${(mc.git_commit || "?").substring(0, 12)}</code></div>
      <div class="k">Trained at</div><div class="timestamp">${mc.created_at_utc || "?"}</div>
    `;
  } catch (e) {
    document.getElementById("bundle").innerHTML = '<div class="small">/info unreachable: ' + e + '</div>';
  }
}

async function refreshRecent() {
  try {
    const d = await fetchJSON("/api/recent");
    const items = (d.items || []).slice().reverse();
    document.getElementById("recentN").textContent = items.length;
    if (items.length) {
      document.getElementById("lastPollTs").textContent = fmtTime(items[0].ts);
    }
    document.getElementById("recent").innerHTML = items.map(p => {
      const cands = (p.candidates || []).map((c, i) =>
        `<span class="c ${i===0 ? 'top' : ''}">${c[0]} <span class="score">${fmtScore(c[1])}</span></span>`
      ).join(" ");
      const alertBadge = p.alert
        ? `<span class="pill ${p.alert.severity}">${p.alert.severity}</span>`
        : '<span class="small">-</span>';
      const top = (p.candidates || [])[0];
      return `<tr>
        <td class="timestamp">${fmtTime(p.ts)}</td>
        <td><code>${top ? top[0] : "-"}</code></td>
        <td class="score">${top ? fmtScore(top[1]) : "-"}</td>
        <td class="score">${(p.err_rate * 100).toFixed(2)}%</td>
        <td><div class="topcand">${cands}</div></td>
        <td>${alertBadge}</td>
      </tr>`;
    }).join("");
  } catch (e) {
    document.getElementById("recent").innerHTML = '<tr><td colspan="6" class="small">' + e + '</td></tr>';
  }
}

async function refreshConfig() {
  try {
    const d = await fetchJSON("/config");
    const url = d.ntfy_base_url.replace(/\\/$/, "") + "/" + d.ntfy_topic;
    document.getElementById("hdrTopic").textContent = "ntfy: " + d.ntfy_topic;
    const a = document.getElementById("ntfyLink");
    a.href = url; a.textContent = url;
    document.getElementById("ntfyCli").textContent = "curl -sN " + url + "/sse";
  } catch (e) {}
}

async function refreshAll() {
  await Promise.all([refreshConfig(), refreshAlerts(), refreshBundle(), refreshRecent()]);
}

refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _start_loop():
    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()


def _push_recent(ts: float, ranked: Optional[List[Tuple[str, float, int]]],
                  err_rate: float, alert: Optional[dict]) -> None:
    item = {
        "ts": ts,
        "candidates": [[s, sc] for s, sc, _ in (ranked or [])],
        "err_rate": err_rate,
        "alert": alert,
    }
    with _STATE_LOCK:
        _RECENT.append(item)


def watch_loop():
    logger.info(
        "Watcher up: poll=%ds window=%ds step=%ds N=%d F=%d W=%d threshold=%.2f ntfy_topic=%s",
        POLL_INTERVAL_S, WINDOW_SECONDS, STEP_SECONDS, N, F, W,
        CONFIDENCE_THRESHOLD, NTFY_TOPIC,
    )
    while True:
        ts = time.time()
        try:
            window = collect_window()
            err = observed_error_rate()
            WATCHER_LAST_ERROR_RATE.set(err)
            ranked = call_predict(window)
            alert = None
            if ranked is None:
                WATCHER_POLLS.labels(status="predict_failed").inc()
            else:
                top = ranked[0]
                logger.info("poll ok: top=%s score=%.3f err_rate=%.1f%%",
                            top[0], top[1], err * 100.0)
                WATCHER_POLLS.labels(status="ok").inc()
                alert = maybe_fire_alert(ranked, err)
            _push_recent(ts, ranked, err, alert)
        except Exception:
            logger.exception("poll failed")
            WATCHER_POLLS.labels(status="error").inc()
        time.sleep(POLL_INTERVAL_S)
