import subprocess
import time
import requests
import csv
import logging
import argparse
from typing import List, Dict, Any
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
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data["data"]["result"]
            else:
                logger.error(f"Prometheus query failed: {data}")
                return []
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return []

class ExperimentRunner:
    """Automatically runs Chaos Mesh experiments and extracts telemetry."""
    def __init__(self, prometheus_url: str, chaos_file: str, locust_file: str,
                 frontend_url: str, output_dir: str = ".",
                 num_users: int = 20, baseline_duration: int = 300,
                 anomaly_duration: int = 300,
                 root_cause: str = "unknown") -> None:
        self.prometheus = PrometheusClient(prometheus_url)
        self.chaos_file = chaos_file
        self.locust_file = locust_file
        self.frontend_url = frontend_url
        self.output_dir = output_dir
        self.num_users = num_users
        self.baseline_duration = baseline_duration
        self.anomaly_duration = anomaly_duration
        self.root_cause = root_cause
        self.run_id = str(uuid.uuid4())[:8]
        self.chaos_start_ts: float = 0.0

    def _run_cmd(self, cmd: List[str]) -> None:
        """Helper to run a shell command."""
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def run_experiment(self) -> None:
        """Runs the whole loop: Locust -> Wait -> Chaos -> Wait -> Stop -> Extract."""
        logger.info(f"Starting experiment run_id={self.run_id}")
        
        # 1. Start Locust in background
        locust_cmd = [
            "locust", "-f", self.locust_file,
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
            self._run_cmd(["kubectl", "apply", "-f", self.chaos_file])

            # 4. Anomaly Period
            logger.info("Waiting %d seconds for anomaly period...", self.anomaly_duration)
            time.sleep(self.anomaly_duration)

            # 5. Delete Chaos
            logger.info("Deleting Chaos Mesh experiment...")
            self._run_cmd(["kubectl", "delete", "-f", self.chaos_file, "--ignore-not-found=true"])
            
            # Wait to ensure Prometheus has scraped the final data points
            logger.info("Waiting 30 seconds for final metrics to be scraped...")
            time.sleep(30)
            
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
            self.extract_metrics(start_time, end_time, self.chaos_start_ts)

    def extract_metrics(self, start_ts: float, end_ts: float, chaos_start_ts: float) -> None:
        """Fetches metrics from Prometheus and saves them to a CSV file."""
        logger.info("Extracting metrics from Prometheus (10s step)...")
        namespace = "rca-sandbox"
        
        queries = {
            "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", container!=""}}[1m])) by (pod)',
            "ram_usage": f'sum(container_memory_working_set_bytes{{namespace="{namespace}", container!=""}}) by (pod)',
            "network_tx": f'sum(rate(container_network_transmit_bytes_total{{namespace="{namespace}"}}[1m])) by (pod)',
            "network_rx": f'sum(rate(container_network_receive_bytes_total{{namespace="{namespace}"}}[1m])) by (pod)',
            "error_rate": f'sum(rate(kube_pod_container_status_restarts_total{{namespace="{namespace}"}}[1m])) by (pod)'
        }
        
        dataset_records = []
        
        for metric_name, query in queries.items():
            results = self.prometheus.query_range(query, start_ts, end_ts, "10s")
            for result in results:
                pod_name = result["metric"].get("pod", "unknown")
                for timestamp, value in result["values"]:
                    dataset_records.append({
                        "timestamp": timestamp,
                        "pod": pod_name,
                        "metric": metric_name,
                        "value": value
                    })

        import os
        csv_file = os.path.join(self.output_dir, f"dataset_{self.run_id}.csv")
        self._save_csv(dataset_records, csv_file, chaos_start_ts)

    def _save_csv(self, data: List[Dict[str, Any]], filename: str, chaos_start_ts: float = 0.0) -> None:
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
                is_anomaly = 1 if (chaos_start_ts > 0 and ts >= chaos_start_ts) else 0
                pivoted[key] = {
                    "timestamp": row["timestamp"],
                    "pod": row["pod"],
                    "cpu_usage": 0.0,
                    "ram_usage": 0.0,
                    "network_tx": 0.0,
                    "network_rx": 0.0,
                    "error_rate": 0.0,
                    "label": is_anomaly,
                    "root_cause": self.root_cause if is_anomaly else "",
                }
            pivoted[key][row["metric"]] = float(row["value"])
            
        rows = list(pivoted.values())
        rows.sort(key=lambda x: (x["timestamp"], x["pod"]))
        
        fields = ["timestamp", "pod", "cpu_usage", "ram_usage", "network_tx", "network_rx", "error_rate", "label", "root_cause"]
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            
        logger.info(f"Dataset successfully saved to {filename}")

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
    )
    runner.run_experiment()

