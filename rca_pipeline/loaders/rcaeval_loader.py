"""
loaders/rcaeval_loader.py — Load RCAEval datasets (Pham et al., ASE 2024 / WWW 2025).

Converts RCAEval failure cases into the same format as generate_scalable_dataset():
    (List[Data], node_names, edge_index)

Supported datasets (9 in RCAEval):
    trainticket, sockshop, online-boutique, circa2021, gaia,
    aiops22, platform, metric-causality, cloud-ranger

Prerequisites:
    git clone https://github.com/NetManAIOps/RCAEval ./rcaeval/
    cd rcaeval && pip install -e . && cd ..
    python -m rcaeval.download --datasets all --output-dir ./rcaeval/data/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = [
    "trainticket", "sockshop", "online-boutique", "circa2021", "gaia",
    "aiops22", "platform", "metric-causality", "cloud-ranger",
]

# Map our benchmark dataset names to the dataset identifiers/layout used by
# the current RCAEval repository (RE1/RE2/RE3 suites).
#
# We keep the public names stable (used by benchmark_real.py) while allowing
# local data directories to follow the upstream RCAEval structure.
DATASET_LAYOUT_ALIASES: Dict[str, Dict[str, List[str]]] = {
    # Benchmark names -> RCAEval API dataset id + candidate directory paths.
    "trainticket": {
        "api_names": ["re2-tt", "train-ticket"],
        "candidate_subdirs": ["RE2/RE2-TT", "RE2-TT", "train-ticket", "RE2/RE2-TT/"],
    },
    "sockshop": {
        "api_names": ["re2-ss", "sock-shop-1", "sock-shop-2"],
        "candidate_subdirs": ["RE2/RE2-SS", "RE2-SS", "sock-shop-1", "sock-shop-2"],
    },
    "online-boutique": {
        "api_names": ["re2-ob", "online-boutique"],
        "candidate_subdirs": ["RE2/RE2-OB", "RE2-OB", "online-boutique"],
    },
}


def _first_existing_path(base: Path, subdirs: List[str]) -> Optional[Path]:
    for sub in subdirs:
        p = base / sub
        if p.exists() and any(p.iterdir()):
            return p
    return None


class RCAEvalLoader:
    """
    Load an RCAEval dataset and convert it to List[Data] compatible with
    our benchmark pipeline.

    Args:
        rcaeval_root: path to the cloned rcaeval/ directory.
        window_size:  temporal window W to extract around each failure.
        num_features: number of metric features per node.
    """

    def __init__(
        self,
        rcaeval_root: str = "rcaeval",
        window_size: int = 12,
        num_features: int = 5,
        use_api: bool = False,
        api_fallback_to_layout: bool = True,
        extract_multimodal: bool = False,
        max_log_lines_per_node: int = 2000,
        max_trace_spans_per_instance: int = 50000,
    ) -> None:
        self.root = Path(rcaeval_root)
        self.data_dir = self.root / "data"
        self.window_size = window_size
        self.num_features = num_features
        self.use_api = use_api
        self.api_fallback_to_layout = api_fallback_to_layout
        self.extract_multimodal = extract_multimodal
        self.max_log_lines_per_node = max_log_lines_per_node
        self.max_trace_spans_per_instance = max_trace_spans_per_instance

    def available_datasets(self) -> List[str]:
        """List datasets that have data downloaded."""
        avail = []
        for ds in SUPPORTED_DATASETS:
            ds_dir = self.data_dir / ds
            if ds_dir.exists() and any(ds_dir.iterdir()):
                avail.append(ds)
                continue

            # Try upstream-style layout aliases (RE1/RE2/RE3)
            alias = DATASET_LAYOUT_ALIASES.get(ds)
            if not alias:
                continue
            if _first_existing_path(self.data_dir, alias.get("candidate_subdirs", [])):
                avail.append(ds)
        return avail

    def load(
        self,
        dataset_name: str,
        max_cases: Optional[int] = None,
    ) -> Tuple[List[Data], List[str], torch.Tensor]:
        """
        Load a single RCAEval dataset.

        Returns:
            dataset:    List[Data] with .x [N, F, W], .edge_index, .y, .root_cause
            node_names: List[str]
            edge_index: [2, E]
        """
        assert dataset_name in SUPPORTED_DATASETS, \
            f"Unknown dataset '{dataset_name}'. Choose from {SUPPORTED_DATASETS}"

        ds_dir = self.data_dir / dataset_name
        if not (ds_dir.exists() and any(ds_dir.iterdir())):
            # Try upstream-style layout aliases (RE1/RE2/RE3)
            alias = DATASET_LAYOUT_ALIASES.get(dataset_name)
            if not alias:
                ds_dir_candidate = None
            else:
                ds_dir_candidate = _first_existing_path(self.data_dir, alias.get("candidate_subdirs", []))
            if ds_dir_candidate is not None:
                ds_dir = ds_dir_candidate
            else:
                raise FileNotFoundError(
                    f"Dataset directory not found for '{dataset_name}'.\n"
                    f"Tried: {self.data_dir / dataset_name} and known upstream aliases.\n"
                    f"Run a download using RCAEval utilities, e.g.: "
                    f"`python -m rcaeval.download --datasets all --output-dir {self.data_dir}`"
                )

        # Resolve the dataset identifier expected by RCAEval Python API.
        # Prefer upstream layout id if available.
        alias = DATASET_LAYOUT_ALIASES.get(dataset_name)
        api_dataset_name = dataset_name
        if alias and alias.get("api_names"):
            # We try all API names in order (the API will raise if unknown).
            api_dataset_names = alias["api_names"]
        else:
            api_dataset_names = [dataset_name]

        # If requested, try the RCAEval Python API first.
        # Otherwise, load via the dataset layout (structured numeric CSVs).
        if self.use_api:
            last_err: Optional[Exception] = None
            for api_name in api_dataset_names:
                try:
                    return self._load_via_api(api_name, max_cases)
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "rcaeval API load failed for api_name='%s' (%s).",
                        api_name,
                        e,
                    )

            # If API load fails for all api_names, optionally fallback.
            assert ds_dir is not None
            if self.api_fallback_to_layout:
                logger.warning("All API attempts failed (%s); using layout loader.", last_err)
                return self._load_from_csv(ds_dir, max_cases)
            raise RuntimeError(f"All RCAEval API attempts failed: {last_err}")

        # Layout loader (structured numeric time-series CSVs only)
        assert ds_dir is not None
        return self._load_from_csv(ds_dir, max_cases)

    def _load_via_api(
        self, dataset_name: str, max_cases: Optional[int]
    ) -> Tuple[List[Data], List[str], torch.Tensor]:
        """Load using the rcaeval Python package if installed."""
        # Upstream package is named `RCAEval` (capitalized). Some environments may
        # also expose a `rcaeval` namespace, so we try both.
        try:
            from RCAEval.classes.data import Dataset as RCADataset
        except Exception:
            from rcaeval.classes.data import Dataset as RCADataset  # type: ignore

        ds = RCADataset(dataset_name, data_dir=str(self.data_dir))
        cases = ds.get_cases()
        if max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]

        node_names = sorted(ds.get_services())
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        N = len(node_names)

        # Build edge_index from the dataset's graph
        try:
            graph = ds.get_graph()
            edges = list(graph.edges())
            if edges:
                src = [name_to_idx[s] for s, d in edges if s in name_to_idx and d in name_to_idx]
                dst = [name_to_idx[d] for s, d in edges if s in name_to_idx and d in name_to_idx]
                edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            else:
                edge_index = self._default_edge_index(N)
        except Exception:
            edge_index = self._default_edge_index(N)

        data_list: List[Data] = []
        for case in cases:
            metrics = case.get_metrics()  # DataFrame: columns=services, rows=timestamps
            rc_name = case.get_root_cause()

            x = self._df_to_tensor(metrics, node_names)
            d = Data(x=x, edge_index=edge_index)
            d.y = torch.tensor([1], dtype=torch.long)
            d.root_cause = [rc_name] if rc_name in name_to_idx else [""]
            d.failure_type = "real"
            data_list.append(d)

        logger.info("Loaded %d cases from '%s' (%d nodes)", len(data_list), dataset_name, N)
        return data_list, node_names, edge_index

    def _load_from_csv(
        self, ds_dir: Path, max_cases: Optional[int]
    ) -> Tuple[List[Data], List[str], torch.Tensor]:
        """
        Fallback CSV loader.

        The downloaded RCAEval data layout looks like:
          ds_dir/
            ts-<root_service>_<fault>/
              <instance_id>/
                metrics.csv        (numeric)
                simple_metrics.csv (numeric, fewer columns)
                ... other csv (logs/traces/pod mapping) - ignored here

        We infer:
          - node_names from the service names appearing in top-level `ts-*_*` dirs
          - root_cause from the top-level `ts-<service>_<fault>` directory name
          - x from metrics.csv (or simple_metrics.csv) time-series numeric columns
        """
        import json

        import pandas as pd

        # 1) Infer services from the top-level directory naming convention.
        #    The RCAEval layout differs by dataset:
        #      - sometimes: `ts-<service>_<fault>/...`
        #      - sometimes: `<service>_<fault>/...` (no `ts-` prefix)
        fault_names = ("cpu", "mem", "disk", "delay", "loss", "socket")
        service_dirs: List[Tuple[str, str, Path]] = []
        for p in ds_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            # Strip optional `ts-` prefix.
            rest = name[len("ts-"):] if name.startswith("ts-") else name
            if "_" not in rest:
                continue
            service_part, fault = rest.rsplit("_", 1)
            if fault not in fault_names:
                continue
            service_dirs.append((service_part, fault, p))

        if not service_dirs:
            raise FileNotFoundError(f"Could not detect RCAEval case layout under {ds_dir}")

        node_names = sorted({s for s, _, _ in service_dirs})
        N = len(node_names)
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        edge_index = self._default_edge_index(N)

        # Helper to load a single case instance.
        def _load_instance(case_dir: Path) -> pd.DataFrame:
            # Prefer `metrics.csv` for richer per-node feature columns.
            metrics_path = case_dir / "metrics.csv"
            simple_path = case_dir / "simple_metrics.csv"
            if metrics_path.exists():
                return pd.read_csv(metrics_path)
            if simple_path.exists():
                return pd.read_csv(simple_path)
            raise FileNotFoundError(f"Missing metrics.csv/simple_metrics.csv in {case_dir}")

        def _norm_service_name(s: str) -> str:
            # RCAEval often prefixes services with `ts-` in logs/traces.
            if s is None:
                return ""
            s = str(s)
            return s[3:] if s.startswith("ts-") else s

        def _extract_logs_features(inst_dir: Path, node_names: List[str]) -> Optional[torch.Tensor]:
            """
            Extract x_logs [N, 8] from logs.csv via log_feature_extractor.
            """
            try:
                from log_feature_extractor import extract_log_features_batch
            except ModuleNotFoundError:
                from rca_pipeline.log_feature_extractor import extract_log_features_batch

            logs_path = inst_dir / "logs.csv"
            if not logs_path.exists():
                return None

            cache_dir = inst_dir / ".mm_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_meta_path = cache_dir / "logs_meta.json"
            cache_feat_path = cache_dir / "x_logs.npy"
            try:
                st = logs_path.stat()
                meta = {
                    "path": str(logs_path),
                    "mtime_ns": int(st.st_mtime_ns),
                    "size": int(st.st_size),
                    "max_log_lines_per_node": int(self.max_log_lines_per_node),
                    "node_names": list(node_names),
                    "shape": [len(node_names), 8],
                }
                if cache_meta_path.exists() and cache_feat_path.exists():
                    cached = json.loads(cache_meta_path.read_text())
                    if cached == meta:
                        x = np.load(cache_feat_path)
                        if x.shape == (len(node_names), 8) and float(np.std(x)) > 1e-8:
                            return torch.from_numpy(x.astype(np.float32))
            except Exception:
                pass

            # Read full logs.csv robustly (mixed-type columns are common, e.g. `error`).
            df_logs = pd.read_csv(logs_path, low_memory=False, dtype=str)

            # Keep only columns we might use; missing columns are created as NaN.
            wanted_cols = ["timestamp", "container_name", "message", "level", "error", "req_path"]
            for c in wanted_cols:
                if c not in df_logs.columns:
                    df_logs[c] = np.nan
            df_logs = df_logs[wanted_cols]
            if "timestamp" in df_logs.columns:
                df_logs["timestamp"] = pd.to_numeric(df_logs["timestamp"], errors="coerce").fillna(0)
                df_logs = df_logs.sort_values("timestamp")
            else:
                df_logs = df_logs

            # Build logs_per_node: list length N
            logs_per_node: List[List[str]] = [[] for _ in range(len(node_names))]

            # Map container_name -> node index
            # We match both exact `ts-<node>` and suffix fallback.
            container_values = df_logs.get("container_name")
            if container_values is None:
                return None

            for node_i, node in enumerate(node_names):
                cvals = df_logs["container_name"].astype(str)

                # 1) Exact match on node name (common for RE2-SS layouts).
                mask_exact_node = cvals == str(node)
                if mask_exact_node.any():
                    sub = df_logs[mask_exact_node]
                else:
                    # 2) Exact match on `ts-<node>` prefix (some layouts).
                    mask_exact_ts = cvals == f"ts-{node}"
                    if mask_exact_ts.any():
                        sub = df_logs[mask_exact_ts]
                    else:
                        # 3) Suffix fallback (robust against naming mismatches).
                        mask_suffix = cvals.str.endswith(f"-{node}")
                        sub = df_logs[mask_suffix] if mask_suffix.any() else df_logs.iloc[0:0]

                if len(sub) == 0:
                    continue

                # Keep only last K lines to avoid huge memory/cost.
                sub = sub.tail(self.max_log_lines_per_node)

                lines: List[str] = []
                for row in sub.itertuples(index=False):
                    level = getattr(row, "level", "")
                    message = getattr(row, "message", "")
                    error = getattr(row, "error", "")
                    req_path = getattr(row, "req_path", "")
                    # Normalize NaNs
                    parts = []
                    for p in (level, message, error, req_path):
                        if p is None:
                            continue
                        ps = str(p)
                        if ps.lower() == "nan":
                            continue
                        parts.append(ps)
                    lines.append(" ".join(parts))

                logs_per_node[node_i] = lines

            x_logs_np = extract_log_features_batch(logs_per_node, window_seconds=60.0)
            if float(np.std(x_logs_np)) <= 1e-8:
                return None
            try:
                np.save(cache_feat_path, x_logs_np.astype(np.float32))
                cache_meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass
            return torch.from_numpy(x_logs_np).to(dtype=torch.float32)

        def _extract_traces_features(inst_dir: Path, node_names: List[str]) -> Optional[torch.Tensor]:
            """
            Extract x_traces [N, 6] from traces.csv via trace_feature_extractor.
            """
            try:
                from trace_feature_extractor import Span, extract_trace_features_batch
            except ModuleNotFoundError:
                from rca_pipeline.trace_feature_extractor import Span, extract_trace_features_batch

            traces_path = inst_dir / "traces.csv"
            if not traces_path.exists():
                return None

            cache_dir = inst_dir / ".mm_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_meta_path = cache_dir / "traces_meta.json"
            cache_feat_path = cache_dir / "x_traces.npy"
            try:
                st = traces_path.stat()
                meta = {
                    "path": str(traces_path),
                    "mtime_ns": int(st.st_mtime_ns),
                    "size": int(st.st_size),
                    "max_trace_spans_per_instance": int(self.max_trace_spans_per_instance),
                    "node_names": list(node_names),
                    "shape": [len(node_names), 6],
                    "duration_unit": "us_to_ms",
                }
                if cache_meta_path.exists() and cache_feat_path.exists():
                    cached = json.loads(cache_meta_path.read_text())
                    if cached == meta:
                        x = np.load(cache_feat_path)
                        if x.shape == (len(node_names), 6) and float(np.std(x)) > 1e-8:
                            return torch.from_numpy(x.astype(np.float32))
            except Exception:
                pass

            cols = [
                "traceID", "spanID", "parentSpanID", "serviceName",
                "operationName", "duration", "statusCode",
            ]
            # Only keep cols that exist
            df_cols_probe = pd.read_csv(traces_path, nrows=0)
            usecols = [c for c in cols if c in df_cols_probe.columns]
            if not usecols:
                return None

            df_tr = pd.read_csv(traces_path, usecols=usecols)

            # Optionally downsample by spans
            if self.max_trace_spans_per_instance is not None and len(df_tr) > self.max_trace_spans_per_instance:
                df_tr = df_tr.sample(n=self.max_trace_spans_per_instance, random_state=0)

            # Normalize and convert types
            df_tr["duration"] = pd.to_numeric(df_tr.get("duration", 0), errors="coerce").fillna(0.0)
            if "statusCode" in df_tr.columns:
                df_tr["statusCode"] = pd.to_numeric(df_tr["statusCode"], errors="coerce").fillna(0)
            else:
                df_tr["statusCode"] = 0
            df_tr["has_error"] = df_tr["statusCode"].astype(float) >= 400

            # Build spans
            spans: List[Span] = []
            if "traceID" not in df_tr.columns or "spanID" not in df_tr.columns:
                return None

            # Drop rows missing ids (and normalize to strings) once.
            df_tr = df_tr.dropna(subset=["traceID", "spanID"])
            if df_tr.empty:
                return None
            df_tr["traceID"] = df_tr["traceID"].astype(str)
            df_tr["spanID"] = df_tr["spanID"].astype(str)
            if "parentSpanID" in df_tr.columns:
                # parentSpanID can be missing
                df_tr["parentSpanID"] = df_tr["parentSpanID"].astype(str)

            # Group once to avoid O(T * n) filtering.
            for tid, g in df_tr.groupby("traceID", sort=False):
                if g.empty:
                    continue
                # Index spans by spanID within this trace
                rows = {row.spanID: row for row in g.itertuples(index=False)}

                # Memoized depth
                memo_depth: Dict[str, int] = {}

                def compute_depth(sid: str) -> int:
                    if sid in memo_depth:
                        return memo_depth[sid]
                    row = rows.get(sid)
                    if row is None:
                        memo_depth[sid] = 0
                        return 0
                    parent_sid = getattr(row, "parentSpanID", None)
                    if parent_sid is None:
                        memo_depth[sid] = 0
                        return 0
                    parent_sid_s = str(parent_sid)
                    if parent_sid_s.lower() == "nan" or parent_sid_s == "":
                        memo_depth[sid] = 0
                        return 0
                    if parent_sid_s not in rows:
                        memo_depth[sid] = 0
                        return 0
                    memo_depth[sid] = compute_depth(parent_sid_s) + 1
                    return memo_depth[sid]

                for sid, row in rows.items():
                    service_raw = getattr(row, "serviceName", "")
                    service = _norm_service_name(service_raw)
                    op = getattr(row, "operationName", "")
                    op = "" if op is None or str(op).lower() == "nan" else str(op)
                    # RCAEval traces.csv stores `duration` in microseconds.
                    # Our feature extractor expects milliseconds.
                    duration_ms = float(getattr(row, "duration", 0.0)) / 1000.0
                    has_error = bool(getattr(row, "has_error", False))

                    parent_sid = getattr(row, "parentSpanID", None)
                    parent_service = None
                    if parent_sid is not None:
                        psid = str(parent_sid)
                        if psid.lower() == "nan" or psid == "":
                            psid = ""
                        parent_row = rows.get(psid)
                        if parent_row is not None:
                            parent_raw = getattr(parent_row, "serviceName", "")
                            parent_service = _norm_service_name(parent_raw)

                    depth = compute_depth(sid)
                    spans.append(
                        Span(
                            service=service,
                            operation=op,
                            duration_ms=duration_ms,
                            has_error=has_error,
                            parent_service=parent_service,
                            depth=depth,
                        )
                    )

            x_traces_np = extract_trace_features_batch(spans, node_names)
            if float(np.std(x_traces_np)) <= 1e-8:
                return None
            try:
                np.save(cache_feat_path, x_traces_np.astype(np.float32))
                cache_meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass
            return torch.from_numpy(x_traces_np).to(dtype=torch.float32)

        data_list: List[Data] = []

        # 2) Build case list by iterating root_service dirs.
        #    Each instance folder corresponds to one failure case.
        for service_part, fault, root_dir in service_dirs:
            # instance directories are numeric (1..k)
            instance_dirs = sorted(
                [p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                key=lambda x: int(x.name),
            )
            for inst_dir in instance_dirs:
                df = _load_instance(inst_dir)

                # time-window extraction: we take last `window_size` rows.
                W = self.window_size
                if len(df) < W:
                    # Pad at the front by duplicating the first row.
                    k = W - len(df)
                    first = df.iloc[[0]].copy()
                    pad = pd.concat([first] * k, ignore_index=True)
                    df = pd.concat([pad, df], ignore_index=True)
                df_tail = df.iloc[-W:]

                # Build x tensor: [N, F, W]
                x = torch.zeros(N, self.num_features, W, dtype=torch.float32)
                for node_i, node_name in enumerate(node_names):
                    # RE2-TT uses `ts-<node>_<metric>`, RE2-OB/SS use `<node>_<metric>`.
                    cols = [c for c in df_tail.columns if c.startswith(f"ts-{node_name}_")]
                    if not cols:
                        cols = [c for c in df_tail.columns if c.startswith(f"{node_name}_")]
                    if not cols:
                        continue
                    for feat_i, col_name in enumerate(cols[: self.num_features]):
                        series = pd.to_numeric(df_tail[col_name], errors="coerce").fillna(0.0).values
                        x[node_i, feat_i, :] = torch.from_numpy(series.astype(np.float32))

                d = Data(x=x, edge_index=edge_index)
                d.y = torch.tensor([1], dtype=torch.long)
                d.root_cause = [service_part]
                d.failure_type = "real"
                d.case_path = str(inst_dir)  # raw case dir (metrics.csv + inject_time.txt)

                # Multimodal features
                if self.extract_multimodal:
                    x_logs = _extract_logs_features(inst_dir, node_names)
                    x_traces = _extract_traces_features(inst_dir, node_names)
                    if x_logs is not None:
                        d.x_logs = x_logs
                    if x_traces is not None:
                        d.x_traces = x_traces
                else:
                    # Attach tensors so models can run end-to-end even when MM is disabled.
                    d.x_logs = torch.zeros(N, 8, dtype=torch.float32)
                    d.x_traces = torch.zeros(N, 6, dtype=torch.float32)
                data_list.append(d)

                if max_cases is not None and max_cases > 0 and len(data_list) >= max_cases:
                    logger.info("Loaded max_cases=%d, stopping.", max_cases)
                    return data_list, node_names, edge_index

        logger.info("Loaded %d CSV cases from %s (%d nodes)", len(data_list), ds_dir, N)
        return data_list, node_names, edge_index

    def _df_to_tensor(self, metrics_df, node_names: List[str]) -> torch.Tensor:
        """Convert a metrics DataFrame to [N, F, W] tensor."""
        N = len(node_names)
        W = self.window_size
        F = self.num_features
        x = torch.zeros(N, F, W)

        for i, name in enumerate(node_names):
            cols = [c for c in metrics_df.columns if name in c]
            for j, c in enumerate(cols[:F]):
                vals = metrics_df[c].values[-W:]
                vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
                pad = W - len(vals)
                if pad > 0:
                    vals = np.concatenate([np.zeros(pad, dtype=np.float32), vals])
                x[i, j, :] = torch.from_numpy(vals)
        return x

    @staticmethod
    def _default_edge_index(n: int) -> torch.Tensor:
        """
        Fallback when ``get_graph()`` is empty or raises: **undirected chain** 0—1—…—(n-1).

        This is a deliberate simplification: rankings from GNN/GAT can differ from the
        true service dependency graph. JSON exports should record when this path was used
        (e.g. compare ``edge_index`` shape to the dataset’s published topology).
        """
        if n <= 1:
            return torch.zeros(2, 0, dtype=torch.long)
        src = list(range(n - 1))
        dst = list(range(1, n))
        return torch.tensor([src + dst, dst + src], dtype=torch.long)
