#!/bin/bash

# SLURM worker script that runs a single benchmark configuration
# Arguments: batch_size num_views dataset model_type checkpoint_path output_dir

set -euo pipefail

# Parse arguments
BATCH_SIZE=$1
NUM_VIEWS=$2
DATASET=$3
MODEL_TYPE=$4
CHECKPOINT_PATH=$5
OUTPUT_DIR=$6

echo "=========================================="
echo "SLURM Worker Starting"
echo "=========================================="
echo "Batch size: ${BATCH_SIZE}"
echo "Num views: ${NUM_VIEWS}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_TYPE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

# ===== Environment setup =====
eval "$(micromamba shell hook --shell bash)"
micromamba activate mapanything

# ===== Optimizations =====
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export SAVE_DEBUG_FILES=false

ulimit -n 32768 || true
ulimit -u 32768 || true

# ===== GPU Detection =====
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "⚠️  CUDA_VISIBLE_DEVICES not set, using nvidia-smi"
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi
echo "[INFO] Detected ${NUM_GPUS} visible GPUs"
echo "[INFO] Server: $(hostname)"
echo "[INFO] IP Address: $(hostname -I | awk '{print $1}')"

nvidia-smi

# ===== Run benchmark on uniscale.sh =====
echo ""
echo "Calling uniscale.sh with parameters..."
echo ""

# Export parameters for uniscale.sh to use
export BENCHMARK_BATCH_SIZE="${BATCH_SIZE}"
export BENCHMARK_NUM_VIEWS="${NUM_VIEWS}"
export BENCHMARK_DATASET="${DATASET}"
export BENCHMARK_MODEL_TYPE="${MODEL_TYPE}"
export BENCHMARK_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
export BENCHMARK_OUTPUT_DIR="${OUTPUT_DIR}"

# Call uniscale.sh
bash /mnt/nfs/slurm/home/gordon/map-anything/bash_scripts/benchmark/dense_n_view/uniscale.sh

echo ""
echo "=========================================="
echo "SLURM Worker Completed"
echo "=========================================="
