import subprocess
import time
import requests
import csv
import logging
import argparse
import os
import sys
import math
import json
from typing import List, Dict, Any, Optional
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrometheusClient:
    """Class to interact with Prometheus HTTP API."""
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def query_range(self, query: str, start: float, end: float, step: str) -> List[Dict[str, Any]]:
        """Executes a PromQL query over a time range."""
        url = f"{self.endpoint}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        last_err: Optional[Exception] = None
        for attempt in range(5):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    return data["data"]["result"]
                logger.error(f"Prometheus query failed: {data}")
                return []
            except Exception as e:
                last_err = e
                logger.warning("Prometheus query attempt %s/5 failed: %s", attempt + 1, e)
                time.sleep(1.5)
        logger.error(f"Error querying Prometheus after retries: {last_err}")
        return []

class ExperimentRunner:
    """Automatically runs Chaos Mesh experiments and extracts telemetry."""
    def __init__(self, prometheus_url: str, chaos_file: str, locust_file: str,
                 frontend_url: str, output_dir: str = ".",
                 num_users: int = 20, baseline_duration: int = 300,
                 anomaly_duration: int = 300,
                 root_cause: str = "unknown",
                 failure_type: str = "hard",
                 scenario: str = "default") -> None:
        self.prometheus = PrometheusClient(prometheus_url)
        self.chaos_file = chaos_file
        self.locust_file = locust_file
        self.frontend_url = frontend_url
        self.output_dir = output_dir
        self.num_users = num_users
        self.baseline_duration = baseline_duration
        self.anomaly_duration = anomaly_duration
        self.root_cause = root_cause
        self.failure_type = failure_type
        self.scenario = scenario
        self.run_id = str(uuid.uuid4())[:8]
        self.chaos_start_ts: float = 0.0
        self.chaos_end_ts: float = 0.0
        self._pf_proc: Optional[subprocess.Popen] = None
        # Keep only application pods to avoid control-plane/chaos noise.
        self._app_pod_regex = r"(frontend|auth|backend|cache|database).*"

    def _ensure_prometheus_port_forward(self) -> None:
        """Si l'URL pointe sur localhost:9090 et rien n'écoute, relance kubectl port-forward."""
        ep = self.prometheus.endpoint.lower()
        if "127.0.0.1:9090" not in ep and "localhost:9090" not in ep:
            return
        try:
            requests.get(f"{self.prometheus.endpoint}/-/healthy", timeout=3)
            return
        except Exception:
            pass
        if self._pf_proc is not None and self._pf_proc.poll() is None:
            try:
                self._pf_proc.terminate()
                self._pf_proc.wait(timeout=5)
            except Exception:
                pass
        logger.info("Démarrage kubectl port-forward vers rca-prometheus-server:9090...")
        self._pf_proc = subprocess.Popen(
            [
                "kubectl", "port-forward", "-n", "rca-sandbox",
                "svc/rca-prometheus-server", "9090:80",
                "--address=127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        try:
            requests.get(f"{self.prometheus.endpoint}/-/healthy", timeout=5)
        except Exception as e:
            logger.warning("Prometheus toujours injoignable après port-forward: %s", e)

    def _run_cmd(self, cmd: List[str]) -> None:
        """Helper to run a shell command."""
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def run_experiment(self) -> None:
        """Runs the whole loop: Locust -> Wait -> Chaos -> Wait -> Stop -> Extract."""
        logger.info(f"Starting experiment run_id={self.run_id}")
        
        # 1. Start Locust in background
        locust_cmd = [
            sys.executable, "-m", "locust", "-f", self.locust_file,
            "--headless",
            "-u", str(self.num_users),
            "-r", str(max(1, self.num_users // 4)),
            "-H", self.frontend_url
        ]
        logger.info("Starting Locust traffic generator (%d users)...", self.num_users)
        locust_process = subprocess.Popen(locust_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        end_time = 0.0
        start_time = time.time()
        
        try:
            # 2. Baseline Period
            logger.info("Waiting %d seconds for baseline period...", self.baseline_duration)
            time.sleep(self.baseline_duration)

            # 3. Apply Chaos
            self.chaos_start_ts = time.time()
            logger.info(f"Applying Chaos Mesh experiment from: {self.chaos_file}")
            # Defensive cleanup in case a previous run left the same resource behind.
            self._run_cmd(["kubectl", "delete", "-f", self.chaos_file, "--ignore-not-found=true"])
            self._run_cmd(["kubectl", "apply", "--validate=false", "-f", self.chaos_file])

            # 4. Anomaly Period
            logger.info("Waiting %d seconds for anomaly period...", self.anomaly_duration)
            time.sleep(self.anomaly_duration)

            # 5. Delete Chaos
            logger.info("Deleting Chaos Mesh experiment...")
            self._run_cmd(["kubectl", "delete", "-f", self.chaos_file, "--ignore-not-found=true"])
            self.chaos_end_ts = time.time()
            
            # Wait to ensure Prometheus has scraped the final data points
            logger.info("Waiting 8 seconds for final metrics to be scraped...")
            time.sleep(8)
            
            end_time = time.time()
            
        except KeyboardInterrupt:
            logger.warning("Experiment interrupted by user!")
        except Exception as e:
            logger.error(f"Experiment encountered an error: {e}")
        finally:
            logger.info("Stopping Locust traffic generator...")
            locust_process.terminate()
            locust_process.wait()

        # 6. Extract data and export to CSV
        if end_time > 0:
            self.extract_metrics(start_time, end_time, self.chaos_start_ts, self.chaos_end_ts)

    def extract_metrics(self, start_ts: float, end_ts: float, chaos_start_ts: float, chaos_end_ts: float) -> None:
        """Fetches metrics from Prometheus and saves them to a CSV file."""
        logger.info("Extracting metrics from Prometheus (10s step)...")
        self._ensure_prometheus_port_forward()
        namespace = "rca-sandbox"
        
        queries = {
            "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", container!="", pod=~"{self._app_pod_regex}"}}[1m])) by (pod)',
            "ram_usage": f'sum(container_memory_working_set_bytes{{namespace="{namespace}", container!="", pod=~"{self._app_pod_regex}"}}) by (pod)',
            "network_tx": f'sum(rate(container_network_transmit_bytes_total{{namespace="{namespace}", pod=~"{self._app_pod_regex}"}}[1m])) by (pod)',
            "network_rx": f'sum(rate(container_network_receive_bytes_total{{namespace="{namespace}", pod=~"{self._app_pod_regex}"}}[1m])) by (pod)',
            "error_rate": f'sum(rate(kube_pod_container_status_restarts_total{{namespace="{namespace}", pod=~"{self._app_pod_regex}"}}[1m])) by (pod)',
            # Gray-failure sensitive tail metric when service mesh histograms are available.
            "latency_p99": (
                f'histogram_quantile(0.99, '
                f'sum(rate(istio_request_duration_milliseconds_bucket{{namespace="{namespace}", '
                f'reporter="destination", destination_workload=~"{self._app_pod_regex}"}}[1m])) '
                f'by (destination_workload, le))'
            ),
        }
        
        dataset_records = []
        
        for metric_name, query in queries.items():
            results = self.prometheus.query_range(query, start_ts, end_ts, "2s")
            for result in results:
                metric_labels = result.get("metric", {})
                pod_name = (
                    metric_labels.get("pod")
                    or metric_labels.get("destination_workload")
                    or "unknown"
                )
                for timestamp, value in result["values"]:
                    dataset_records.append({
                        "timestamp": timestamp,
                        "pod": pod_name,
                        "metric": metric_name,
                        "value": value
                    })

        csv_file = os.path.join(self.output_dir, f"dataset_{self.run_id}.csv")
        self._save_csv(dataset_records, csv_file, chaos_start_ts, chaos_end_ts)

    def _save_csv(self, data: List[Dict[str, Any]], filename: str, chaos_start_ts: float = 0.0, chaos_end_ts: float = 0.0) -> None:
        """Pivots the metrics and exports to CSV with label and root_cause columns."""
        if not data:
            logger.warning("No data extracted to save.")
            return
            
        # Pivot dataset
        pivoted: Dict[str, Dict[str, Any]] = {}
        for row in data:
            key = f'{row["timestamp"]}_{row["pod"]}'
            if key not in pivoted:
                ts = float(row["timestamp"])
                # Label as anomalous only within the chaos injection window
                is_anomaly = 1 if (chaos_start_ts > 0 and chaos_start_ts <= ts <= chaos_end_ts) else 0
                pivoted[key] = {
                    "timestamp": row["timestamp"],
                    "pod": row["pod"],
                    "cpu_usage": 0.0,
                    "ram_usage": 0.0,
                    "network_tx": 0.0,
                    "network_rx": 0.0,
                    "error_rate": 0.0,
                    "latency_p99": 0.0,
                    "label": is_anomaly,
                    "root_cause": self.root_cause if is_anomaly else "",
                    "failure_type": self.failure_type if is_anomaly else "healthy",
                    "scenario": self.scenario,
                }
            
            # For latency, avoid NaN if requests are 0
            val = float(row["value"])
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            pivoted[key][row["metric"]] = val
            
        rows = list(pivoted.values())
        rows.sort(key=lambda x: (x["timestamp"], x["pod"]))
        
        fields = [
            "timestamp", "pod", "cpu_usage", "ram_usage", "network_tx", "network_rx",
            "error_rate", "latency_p99", "label", "root_cause", "failure_type", "scenario",
        ]
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            
        logger.info(f"Dataset successfully saved to {filename}")
        self._save_manifest(filename, rows)

    def _save_manifest(self, csv_file: str, rows: List[Dict[str, Any]]) -> None:
        """Store run metadata for traceability and campaign QA."""
        out = {
            "run_id": self.run_id,
            "csv_file": csv_file,
            "scenario": self.scenario,
            "chaos_file": self.chaos_file,
            "root_cause": self.root_cause,
            "failure_type": self.failure_type,
            "users": self.num_users,
            "baseline_duration_s": self.baseline_duration,
            "anomaly_duration_s": self.anomaly_duration,
            "prometheus_url": self.prometheus.endpoint,
            "rows": len(rows),
            "anomalous_rows": sum(1 for r in rows if int(r.get("label", 0)) == 1),
            "healthy_rows": sum(1 for r in rows if int(r.get("label", 0)) == 0),
        }
        manifest_path = csv_file.replace(".csv", ".meta.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("Run manifest saved to %s", manifest_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="RCA Dataset Builder -- Orchestrates Chaos Mesh experiments and extracts Prometheus metrics."
    )
    parser.add_argument(
        "--prometheus-url", type=str, default="http://localhost:9090",
        help="Prometheus HTTP API endpoint."
    )
    parser.add_argument(
        "--chaos-file", type=str, required=True,
        help="Path to the Chaos Mesh YAML file to apply."
    )
    parser.add_argument(
        "--locust-file", type=str, default="locustfile.py",
        help="Path to the Locust script."
    )
    parser.add_argument(
        "--frontend-url", type=str, default="http://localhost:8080",
        help="URL of the frontend service (via port-forward)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory where dataset CSV files will be saved."
    )
    parser.add_argument(
        "--users", type=int, default=20,
        help="Number of simulated Locust users."
    )
    parser.add_argument(
        "--baseline-duration", type=int, default=300,
        help="Duration of the healthy baseline period in seconds (default: 300)."
    )
    parser.add_argument(
        "--anomaly-duration", type=int, default=300,
        help="Duration of the chaos injection period in seconds (default: 300)."
    )
    parser.add_argument(
        "--root-cause", type=str, default="unknown",
        help="Name of the target pod/service for labeling (e.g. cache, database, auth)."
    )
    parser.add_argument(
        "--failure-type", type=str, default="hard",
        choices=["hard", "gray", "cascade", "metastable", "byzantine"],
        help="Failure family label for dataset audit/SOTA coverage."
    )
    parser.add_argument(
        "--scenario", type=str, default="default",
        help="Human-readable scenario label (e.g. stress-cpu-db)."
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        prometheus_url=args.prometheus_url,
        chaos_file=args.chaos_file,
        locust_file=args.locust_file,
        frontend_url=args.frontend_url,
        output_dir=args.output_dir,
        num_users=args.users,
        baseline_duration=args.baseline_duration,
        anomaly_duration=args.anomaly_duration,
        root_cause=args.root_cause,
        failure_type=args.failure_type,
        scenario=args.scenario,
    )
    runner.run_experiment()

