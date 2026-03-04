import sys
import logging
logging.basicConfig(level=logging.INFO)
print("Starting build...", flush=True)
from graph_builder import GraphBuilder
builder = GraphBuilder('../datasets/dataset_merged.csv')
snap = builder.build_snapshots()
print(f"Snapshots built: {len(snap)}", flush=True)
win = builder.create_sliding_windows(snap, window_size=6)
print(f"Windows created: {len(win)}", flush=True)
builder.save_windows(win, '../dataset_windows.pt')
print("Done.", flush=True)
