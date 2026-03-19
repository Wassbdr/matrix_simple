#!/bin/bash
#
# run_hybrid_training.sh - Complete SOTA RCA training pipeline with multi-modal support
#
# Usage:
#   ./run_hybrid_training.sh                                              # Synthetic only
#   ./run_hybrid_training.sh --prometheus http://prometheus:9090          # Metrics only
#   ./run_hybrid_training.sh --prometheus http://prometheus:9090 \
#                            --jaeger http://jaeger:16686 \
#                            --loki http://loki:3100                      # Full multi-modal
#
set -e

# ============================================================================
# Configuration
# ============================================================================
NUM_NODES=${NUM_NODES:-100}
SYNTHETIC_SAMPLES=${SYNTHETIC_SAMPLES:-10000}
EPOCHS=${EPOCHS:-100}
SUPERVISED_EPOCHS=${SUPERVISED_EPOCHS:-150}
PROMETHEUS_URL=${PROMETHEUS_URL:-""}
JAEGER_URL=${JAEGER_URL:-""}
LOKI_URL=${LOKI_URL:-""}
CLUSTER_DURATION=${CLUSTER_DURATION:-"7d"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
SEED=${SEED:-42}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prometheus)
            PROMETHEUS_URL="$2"
            shift 2
            ;;
        --jaeger)
            JAEGER_URL="$2"
            shift 2
            ;;
        --loki)
            LOKI_URL="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --samples)
            SYNTHETIC_SAMPLES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --duration)
            CLUSTER_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prometheus URL    Prometheus URL for metrics"
            echo "  --jaeger URL        Jaeger URL for trace features"
            echo "  --loki URL          Loki URL for log features"
            echo "  --nodes N           Number of nodes (default: 100)"
            echo "  --samples N         Number of synthetic samples (default: 10000)"
            echo "  --epochs N          Contrastive epochs (default: 100)"
            echo "  --duration DURATION Cluster data duration (default: 7d)"
            echo "  --output-dir DIR    Output directory (default: outputs)"
            echo "  --seed N            Random seed (default: 42)"
            echo ""
            echo "Examples:"
            echo "  # Synthetic only"
            echo "  $0"
            echo ""
            echo "  # Metrics + Traces + Logs (full multi-modal)"
            echo "  $0 --prometheus http://prometheus:9090 --jaeger http://jaeger:16686 --loki http://loki:3100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

# Determine modality mode
MODALITY_MODE="synthetic"
MODALITIES="synthetic"
if [ -n "$PROMETHEUS_URL" ]; then
    MODALITY_MODE="metrics"
    MODALITIES="metrics"
    if [ -n "$JAEGER_URL" ]; then
        MODALITY_MODE="metrics+traces"
        MODALITIES="$MODALITIES+traces"
    fi
    if [ -n "$LOKI_URL" ]; then
        MODALITY_MODE="$MODALITY_MODE+logs"
        MODALITIES="$MODALITIES+logs"
    fi
fi

echo "=============================================="
echo "  RCA SOTA Training Pipeline (Multi-Modal)"
echo "=============================================="
echo "Nodes: $NUM_NODES"
echo "Synthetic samples: $SYNTHETIC_SAMPLES"
echo "Modalities: $MODALITIES"
echo "Prometheus: ${PROMETHEUS_URL:-'(none)'}"
echo "Jaeger: ${JAEGER_URL:-'(none)'}"
echo "Loki: ${LOKI_URL:-'(none)'}"
echo "Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# Step 1: Generate/Load Data
# ============================================================================
echo "[1/4] Generating dataset..."

DATASET_PATH="$OUTPUT_DIR/hybrid_dataset_N${NUM_NODES}.pt"

if [ -n "$PROMETHEUS_URL" ]; then
    echo "  → Hybrid mode: synthetic + cluster ($MODALITIES)"

    # Build command with optional modality URLs
    CMD="python unified_data.py \
        --num-nodes $NUM_NODES \
        --synthetic-samples $SYNTHETIC_SAMPLES \
        --prometheus-url \"$PROMETHEUS_URL\" \
        --cluster-duration \"$CLUSTER_DURATION\" \
        --output \"$DATASET_PATH\" \
        --seed $SEED"

    if [ -n "$JAEGER_URL" ]; then
        CMD="$CMD --jaeger-url \"$JAEGER_URL\""
    fi

    if [ -n "$LOKI_URL" ]; then
        CMD="$CMD --loki-url \"$LOKI_URL\""
    fi

    eval $CMD
else
    echo "  → Synthetic only mode"
    python simulated_data_scalable.py \
        --num-nodes $NUM_NODES \
        --num-samples $SYNTHETIC_SAMPLES \
        --v2 \
        --seasonal-noise \
        --correlated-features \
        --multi-root-prob 0.15 \
        --output "$DATASET_PATH" \
        --seed $SEED
fi

echo "  ✓ Dataset saved to $DATASET_PATH"

# ============================================================================
# Step 2: Train Model
# ============================================================================
echo ""
echo "[2/4] Training model..."

CHECKPOINT_PATH="$OUTPUT_DIR/sta_sota_N${NUM_NODES}.pt"

python train_sta.py \
    --dataset "$DATASET_PATH" \
    --num-nodes $NUM_NODES \
    --k 10 \
    --epochs $EPOCHS \
    --supervised-epochs $SUPERVISED_EPOCHS \
    --use-memory \
    --memory-slots 128 \
    --use-severity-weights \
    --curriculum \
    --lambda-causality 0.05 \
    --checkpoint-dir "$OUTPUT_DIR" \
    --out-checkpoint "sta_sota_N${NUM_NODES}.pt"

echo "  ✓ Model saved to $CHECKPOINT_PATH"

# ============================================================================
# Step 3: Evaluate
# ============================================================================
echo ""
echo "[3/4] Evaluating model..."

python evaluate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset "$DATASET_PATH" \
    --breakdown \
    2>&1 | tee "$OUTPUT_DIR/evaluation_results.txt"

echo "  ✓ Results saved to $OUTPUT_DIR/evaluation_results.txt"

# ============================================================================
# Step 4: Summary
# ============================================================================
echo ""
echo "[4/4] Summary"
echo "=============================================="
echo ""
echo "Modalities used: $MODALITIES"
echo ""
echo "Files created:"
echo "  - Dataset: $DATASET_PATH"
echo "  - Model:   $CHECKPOINT_PATH"
echo "  - Results: $OUTPUT_DIR/evaluation_results.txt"
echo ""
echo "To use the model for inference:"
echo "  python infer.py --checkpoint $CHECKPOINT_PATH --input <your_data>"
echo ""
echo "To visualize attention/memory:"
echo "  python -c \"from evaluate import load_model; m,_,_ = load_model('$CHECKPOINT_PATH'); print(m.get_memory_stats())\""
echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
