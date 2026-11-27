"""Shared fixtures for serving tests: a tiny STA bundle and the FastAPI app.

``serving.serve`` reads ``MODEL_BUNDLE`` at import time, so the bundle is
built and the env var set *before* the module is imported (session scope -
prometheus collectors cannot be registered twice in one process).
"""

import os

import numpy as np
import pytest
import torch

from rca_pipeline.model_bundle import save_bundle
from rca_pipeline.model_sta import SelectThenAnalyze

NODE_NAMES = ["svc-a", "svc-b", "svc-c", "svc-d"]
N, F, W, H, K = len(NODE_NAMES), 2, 8, 8, 2


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle")
    torch.manual_seed(0)
    model = SelectThenAnalyze(num_features=F, hidden_dim=H, num_nodes=N, k=K)
    save_bundle(
        str(path),
        model=model,
        node_names=NODE_NAMES,
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        scaler_mean=np.zeros(F, dtype=np.float32),
        scaler_scale=np.ones(F, dtype=np.float32),
        num_features=F,
        hidden_dim=H,
        k=K,
        training_info={"avg1_mean": 0.5, "seeds": [42]},
    )
    return path


@pytest.fixture(scope="session")
def serve_module(bundle_dir):
    os.environ["MODEL_BUNDLE"] = str(bundle_dir)
    from serving import serve
    return serve
