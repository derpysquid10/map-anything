#!/bin/bash

# Helper script to run benchmark with proper conda environment activation
# Usage: ./run_benchmark_with_conda.sh GPU_ID [benchmark_args...]

GPU_ID=$1
shift  # Remove GPU_ID from arguments, rest are benchmark args

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the mapanything environment
conda activate mapanything

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.current_device() if torch.cuda.is_available() else None}')"

# Run the benchmark with all remaining arguments
exec python3 "$@"