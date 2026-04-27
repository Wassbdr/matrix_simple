"""
loki_connector.py -- Loki HTTP client for MM (metrics + logs) on observ-it.

Cluster observit-cluster1:
  - Loki: monitoring-logs/loki:3100 (ou NodePort 31100 via loki-gateway-nodeport)
  - port-forward typique:
      kubectl port-forward -n monitoring-logs svc/loki 3100:3100

API utilisée: GET /loki/api/v1/query_range (LogQL).

Flux MM partiel:
  1) Pour une fenêtre [t0, t1], on récupère les lignes de logs par service.
  2) `log_feature_extractor.extract_log_features_batch` produit N x 8 features.
  3) Fusion côté MM loaders/builders (métier trace/log existant).

Ce module est indépendant du dataset builder : utile aussi bien en post-run
(ré-hydrater les CSV avec des logs) qu'en live.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LokiClient:
    """Thin wrapper around Loki HTTP API (query_range)."""

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def query_range(
        self,
        logql: str,
        start: float,
        end: float,
        limit: int = 5000,
        step: Optional[str] = None,
        direction: str = "forward",
    ) -> List[Dict]:
        """
        Exécute une requête LogQL sur [start, end] (timestamps UNIX secondes).
        Retourne la liste des streams ({"stream": {...labels...}, "values": [[ts_ns, line], ...]}).
        """
        url = f"{self.endpoint}/loki/api/v1/query_range"
        params: Dict[str, object] = {
            "query": logql,
            "start": int(start * 1e9),  # Loki attend des ns
            "end": int(end * 1e9),
            "limit": limit,
            "direction": direction,
        }
        if step:
            params["step"] = step

        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") == "success":
                    return payload.get("data", {}).get("result", [])
                logger.warning("Loki returned non-success payload: %s", payload.get("status"))
                return []
            except Exception as e:
                last_err = e
                logger.warning("Loki attempt %s/4 failed: %s", attempt + 1, e)
                time.sleep(1.5)
        logger.error("Loki query failed after retries: %s", last_err)
        return []

    def fetch_lines_by_service(
        self,
        start: float,
        end: float,
        services: List[str],
        namespace: str,
        service_label: str = "app_kubernetes_io_name",
        fallback_label: str = "container",
        limit_per_service: int = 5000,
    ) -> Dict[str, List[str]]:
        """
        Pour chaque service, retourne la liste des lignes log brutes sur la fenêtre.

        Essaie d'abord `service_label` (ex: `app_kubernetes_io_name` posé par
        Promtail/Fluent-bit sur Rancher), puis bascule sur `fallback_label`
        (ex: `container`) qui matche par nom de conteneur.

        Sock-shop: les pods n'ont que le label `name` (voir kubectl get pods).
        Sur une stack Loki Rancher, ce label est souvent ré-émis en
        `name` / `container` ; le fallback `container` marche dans 95% des cas.
        """
        out: Dict[str, List[str]] = {s: [] for s in services}
        for svc in services:
            streams: List[Dict] = []
            for label in (service_label, fallback_label, "name", "app"):
                logql = f'{{namespace="{namespace}", {label}="{svc}"}}'
                streams = self.query_range(logql, start, end, limit=limit_per_service)
                if streams:
                    logger.debug("Loki matched %s via label=%s", svc, label)
                    break
            lines: List[str] = []
            for s in streams:
                for _ts_ns, line in s.get("values", []):
                    lines.append(line)
            out[svc] = lines
        return out


def enrich_window_with_logs(
    loki: LokiClient,
    start_ts: float,
    end_ts: float,
    node_names: List[str],
    namespace: str,
    window_seconds: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Helper haut niveau: fetch des lignes log par service pour une fenêtre donnée.
    Aligné sur l'ordre de `node_names` pour brancher directement
    `log_feature_extractor.extract_log_features_batch`.
    """
    _ = window_seconds  # kept for API symmetry
    return loki.fetch_lines_by_service(start_ts, end_ts, node_names, namespace)
