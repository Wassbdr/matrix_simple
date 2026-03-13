import sys
import logging
logging.basicConfig(level=logging.INFO)
print("Starting build...", flush=True)
from graph_builder import GraphBuilder
from common import stratified_split, NODE_NAMES

builder = GraphBuilder('../datasets/dataset_merged.csv')
snap = builder.build_snapshots()
print(f"Snapshots built: {len(snap)}", flush=True)
win = builder.create_sliding_windows(snap, window_size=6)
print(f"Windows created: {len(win)}", flush=True)

# Split before fitting the scaler to avoid data leakage
train_win, test_win = stratified_split(win, train_ratio=0.8)
print(f"Train: {len(train_win)}, Test: {len(test_win)}", flush=True)

# Fit on train only, then apply to both splits
scaler = GraphBuilder.fit_scaler(train_win)
train_win = GraphBuilder.apply_scaler(train_win, scaler)
test_win  = GraphBuilder.apply_scaler(test_win,  scaler)

import torch
torch.save(
    {"train": train_win, "test": test_win, "scaler": scaler, "node_names": NODE_NAMES},
    '../dataset_windows.pt'
)
print("Done. Saved train/test splits + scaler to dataset_windows.pt", flush=True)
