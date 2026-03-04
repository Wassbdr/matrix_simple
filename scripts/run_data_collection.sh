#!/bin/bash

# run_data_collection.sh
# Automates the execution of Chaos experiments to generate a large dataset.
# Usage: ./scripts/run_data_collection.sh

set -e

# Number of iterations per scenario
NUM_ITERATIONS=5
SCENARIOS=(
  "k8s/chaos/network-delay-frontend-auth.yaml,auth"
  "k8s/chaos/stress-cpu-db.yaml,database"
  "k8s/chaos/pod-kill-cache.yaml,cache"
)

echo "=========================================="
echo " Starting Automated RCA Data Collection "
echo "=========================================="

# Create target dir
mkdir -p datasets/batch

for i in $(seq 1 $NUM_ITERATIONS); do
  echo "--- ITERATION $i / $NUM_ITERATIONS ---"
  
  for SCENARIO in "${SCENARIOS[@]}"; do
    IFS=',' read -r CHAOS_FILE ROOT_CAUSE <<< "$SCENARIO"
    echo "Running Scenario: $CHAOS_FILE (Root Cause: $ROOT_CAUSE)"
    
    # We use a short baseline and anomaly duration to generate data faster
    python3 rca_pipeline/dataset_builder.py \
        --chaos-file "$CHAOS_FILE" \
        --locust-file rca_pipeline/locustfile.py \
        --frontend-url http://localhost:8080 \
        --prometheus-url http://localhost:9090 \
        --users 20 \
        --baseline-duration 60 \
        --anomaly-duration 90 \
        --root-cause "$ROOT_CAUSE" \
        --output-dir datasets/batch/
        
    echo "Sleeping 10s between experiments to let cluster stabilize..."
    sleep 10
  done
done

echo "=========================================="
echo " Data Collection Complete! "
echo " Merging batch datasets... "
echo "=========================================="

python3 <<EOF
import pandas as pd
import glob
import os

all_files = glob.glob("datasets/batch/*.csv")
if not all_files:
    print("No CSV files found in datasets/batch/")
    exit(0)

print(f"Found {len(all_files)} files. Merging...")
df_list = []
for f in all_files:
    try:
        df = pd.read_csv(f)
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.sort_values(by=['timestamp', 'pod'], inplace=True)
    merged_df.to_csv("datasets/dataset_merged_large.csv", index=False)
    print("Saved to datasets/dataset_merged_large.csv")
EOF

echo "--- Generating Windowed Graphs ---"
python3 -c "
from rca_pipeline.graph_builder import GraphBuilder
builder = GraphBuilder('datasets/dataset_merged_large.csv')
snap = builder.build_snapshots()
win = builder.create_sliding_windows(snap, window_size=6)
builder.save_windows(win, 'dataset_windows_large.pt')
"

echo "ALL DONE. The large dataset is available at dataset_windows_large.pt."
