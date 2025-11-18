#!/bin/bash

# Exhaustive command runner for benchmarking with GPU scheduling
# This script runs commands over all combinations of specified variables
# and intelligently schedules them across available GPUs

set -u  # Exit on undefined variable
# Note: NOT using 'set -e' so script continues even if individual jobs fail

# ============================================================================
# CRITICAL ENVIRONMENT SETUP
# ============================================================================

# Initialize conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate mapanything
else
    echo "ERROR: Conda not found at /opt/conda"
    exit 1
fi

# Verify we're in the right environment
if [ "${CONDA_DEFAULT_ENV:-}" != "mapanything" ]; then
    echo "ERROR: Failed to activate mapanything environment"
    exit 1
fi

# Set library paths for CUDA
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:$PATH"

# Verify CUDA is available
echo "Verifying CUDA availability..."
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: CUDA is not available in Python"
    echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
    exit 1
fi
echo "✓ CUDA verification passed"
echo ""

# ============================================================================
# CONFIGURATION: Define your variable sets here
# ============================================================================

# Model checkpoint paths
MODEL_CHECKPOINTS=(
    # "/mnt/nfs/moma/checkpoints/exp36_whole_inject_before_norm_Prob_lowLR_again/checkpoint_0_30000.pt"
    "/mnt/nfs/SpatialAI/moma/logs/final_pose_cam_both0.5probs/checkpoint_0_38000.pt"
    # "/mnt/nfs/SpatialAI/moma/logs/final_swapped_pose_toCam/checkpoint_0_38000.pt"
)

# Variables for the first command (dense_n_view benchmark)
DATASETS_1=("benchmark_518_eth3d_snpp_tav2")
NUM_VIEWS=(2 4 8)
# NUM_VIEWS=(3)

BATCH_SIZES=(10)

# Variables for the second command (rmvd_mvs benchmark)
EVAL_DATASETS=("kitti" "scannet")
CONDITIONING_MODES=("image" "image+pose" "image+intrinsics" "image+intrinsics+pose")
ALIGNMENT_METHODS=("median" "none")
EVALUATION_VIEWS=("multi_view" "single_view")

# GPU scheduling configuration
GPU_MEMORY_THRESHOLD=20000  # MB - consider GPU free if less than this is used
CHECK_INTERVAL=10         # seconds - how often to check for free GPUs
MAX_RETRIES=3            # Maximum retries for failed jobs
FORCE_GPU_USE=false      # Set to true to use GPUs even if they appear busy (uses round-robin)

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
JOB_QUEUE_FILE="$LOG_DIR/job_queue_${TIMESTAMP}.txt"
JOB_STATUS_FILE="$LOG_DIR/job_status_${TIMESTAMP}.txt"

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

# ============================================================================
# GPU MANAGEMENT FUNCTIONS
# ============================================================================

# Get list of all available GPU IDs
get_all_gpus() {
    nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '
}

# Check if a GPU is free (no processes or minimal memory usage)
is_gpu_free() {
    local gpu_id=$1
    
    # Check memory usage
    local mem_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "999999")
    
    # Check for running processes (count lines, subtract header if present)
    local processes=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    
    if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD" ] && [ "$processes" -eq 0 ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is busy
    fi
}

# Find the first available free GPU
find_free_gpu() {
    local gpus=$(get_all_gpus)
    
    for gpu_id in $gpus; do
        if is_gpu_free "$gpu_id"; then
            echo "$gpu_id"
            return 0
        fi
    done
    
    return 1  # No free GPU found
}

# Wait for a GPU to become available
wait_for_gpu() {
    # If force mode is enabled, use round-robin GPU assignment
    if [ "$FORCE_GPU_USE" = true ]; then
        local all_gpus=($(get_all_gpus))
        local num_gpus=${#all_gpus[@]}
        local gpu_index=$((RANDOM % num_gpus))
        local selected_gpu=${all_gpus[$gpu_index]}
        log "Force mode: assigning GPU $selected_gpu" >&2
        echo "$selected_gpu"
        return 0
    fi
    
    log "Waiting for a free GPU..." >&2
    
    local wait_count=0
    while true; do
        local free_gpu=$(find_free_gpu)
        if [ -n "$free_gpu" ]; then
            echo "$free_gpu"
            return 0
        fi
        
        ((wait_count++))
        if [ $((wait_count % 6)) -eq 0 ]; then
            log "Still waiting for GPU... (${wait_count} checks, $((wait_count * CHECK_INTERVAL)) seconds)" >&2
            # Show GPU status
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -n 4 | tee -a "$LOG_FILE" >&2
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# ============================================================================
# JOB QUEUE MANAGEMENT
# ============================================================================

# Initialize job queue
init_job_queue() {
    : > "$JOB_QUEUE_FILE"
    : > "$JOB_STATUS_FILE"
    
    echo "job_id|type|checkpoint|params|status|gpu|attempts|start_time|end_time" >> "$JOB_STATUS_FILE"
}

# Add job to queue
add_job() {
    local job_type=$1
    shift
    local params="$*"
    
    local job_id=$(wc -l < "$JOB_QUEUE_FILE")
    ((job_id++))
    
    # Use tab as delimiter and space for params
    echo -e "$job_id\t$job_type\t$params" >> "$JOB_QUEUE_FILE"
    echo "$job_id"
}

# Update job status
update_job_status() {
    local job_id=$1
    local job_type=$2
    local checkpoint=$3
    local params=$4
    local status=$5
    local gpu=$6
    local attempts=$7
    local start_time=$8
    local end_time=$9
    
    echo "$job_id|$job_type|$checkpoint|$params|$status|$gpu|$attempts|$start_time|$end_time" >> "$JOB_STATUS_FILE"
}

# ============================================================================
# COMMAND 1: Dense N-View Benchmark
# ============================================================================

run_dense_n_view() {
    local gpu_id=$1
    local checkpoint=$2
    local dataset=$3
    local num_views=$4
    local batch_size=$5
    local job_id=$6
    local attempt=$7
    
    log "========================================="
    log "Job ID: $job_id (Attempt $attempt)"
    log "Running Dense N-View Benchmark on GPU $gpu_id"
    log "Checkpoint: $checkpoint"
    log "Dataset: $dataset"
    log "Num Views: $num_views"
    log "Batch Size: $batch_size"
    log "========================================="
    
    local start_time=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Extract directory name as checkpoint name and sanitize for Hydra
    local ckpt_name=$(basename "$(dirname "$checkpoint")" | tr '()' '__')
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export HYDRA_FULL_ERROR=1
    
    log "Environment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Conda env: ${CONDA_DEFAULT_ENV}"
    log "Python: $(which python3)"
    
    # Verify CUDA is available for this specific GPU
    local cuda_check=$(python3 -c "import torch; print(f'{torch.cuda.is_available()}:{torch.cuda.device_count()}')" 2>&1)
    log "CUDA check: available=$(echo $cuda_check | cut -d: -f1), devices=$(echo $cuda_check | cut -d: -f2)"
    
    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=default \
        dataset="$dataset" \
        dataset.num_workers=12 \
        dataset.num_views="$num_views" \
        batch_size="$batch_size" \
        model=pow3r_vggt \
        amp=0 \
        model.model_config.load_custom_ckpt=true \
        model.model_config.custom_ckpt_path="$checkpoint" \
        +model/task=non_metric_poses_non_metric_depth_sparse \
        hydra.run.dir='/home/binbin/gtan/uniscale_final_final/no_priors/mapanything/benchmarking/dense_'"${num_views}"'_view/'"${ckpt_name}"
    
    local exit_code=$?
    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    
    if [ $exit_code -eq 0 ]; then
        log "✓ Dense N-View benchmark completed successfully"
        update_job_status "$job_id" "dense_n_view" "$checkpoint" "$dataset|$num_views|$batch_size" "SUCCESS" "$gpu_id" "$attempt" "$start_time" "$end_time"
    else
        log "✗ Dense N-View benchmark failed with exit code $exit_code"
        update_job_status "$job_id" "dense_n_view" "$checkpoint" "$dataset|$num_views|$batch_size" "FAILED" "$gpu_id" "$attempt" "$start_time" "$end_time"
    fi
    log ""
    return $exit_code
}

# ============================================================================
# COMMAND 2: RMVD MVS Benchmark
# ============================================================================

run_rmvd_mvs() {
    local gpu_id=$1
    local checkpoint=$2
    local eval_dataset=$3
    local conditioning=$4
    local alignment=$5
    local views=$6
    local job_id=$7
    local attempt=$8
    
    log "========================================="
    log "Job ID: $job_id (Attempt $attempt)"
    log "Running RMVD MVS Benchmark on GPU $gpu_id"
    log "Checkpoint: $checkpoint"
    log "Dataset: $eval_dataset"
    log "Conditioning: $conditioning"
    log "Alignment: $alignment"
    log "Views: $views"
    log "========================================="
    
    local start_time=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Extract directory name as checkpoint name and sanitize for Hydra
    local ckpt_name=$(basename "$(dirname "$checkpoint")" | tr '()' '__')
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export HYDRA_FULL_ERROR=1
    
    log "Environment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Conda env: ${CONDA_DEFAULT_ENV}"
    log "Python: $(which python)"
    
    # Verify CUDA is available for this specific GPU
    local cuda_check=$(python -c "import torch; print(f'{torch.cuda.is_available()}:{torch.cuda.device_count()}')" 2>&1)
    log "CUDA check: available=$(echo $cuda_check | cut -d: -f1), devices=$(echo $cuda_check | cut -d: -f2)"
    
    local load_ckpt="true"
    local ckpt_path="$checkpoint"
    
    # python \
    #     benchmarking/rmvd_mvs_benchmark/benchmark.py \
    #     machine=default \
    #     eval_dataset="$eval_dataset" \
    #     evaluation_conditioning="$conditioning" \
    #     evaluation_alignment="$alignment" \
    #     evaluation_views="$views" \
    #     hydra.run.dir="/home/binbin/gtan/uniscale_pretest2/mapanything/benchmarking/rmvd_${conditioning}_${alignment}_${views}/${eval_dataset}/${ckpt_name}" \
    #     model=pow3r_vggt \
    #     evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
    #     model.model_config.load_custom_ckpt="$load_ckpt" \
    #     model.model_config.custom_ckpt_path="$ckpt_path"
    
    local exit_code=$?
    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    
    if [ $exit_code -eq 0 ]; then
        log "✓ RMVD MVS benchmark completed successfully"
        update_job_status "$job_id" "rmvd_mvs" "$checkpoint" "$eval_dataset|$conditioning|$alignment|$views" "SUCCESS" "$gpu_id" "$attempt" "$start_time" "$end_time"
    else
        log "✗ RMVD MVS benchmark failed with exit code $exit_code"
        update_job_status "$job_id" "rmvd_mvs" "$checkpoint" "$eval_dataset|$conditioning|$alignment|$views" "FAILED" "$gpu_id" "$attempt" "$start_time" "$end_time"
    fi
    log ""
    return $exit_code
}

# ============================================================================
# JOB EXECUTION WITH RETRY
# ============================================================================

execute_job_with_retry() {
    local job_id=$1
    local job_type=$2
    shift 2
    local params=("$@")
    
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        # Wait for and claim a free GPU
        local gpu_id=$(wait_for_gpu)
        log "Assigned GPU $gpu_id to job $job_id"
        
        # Execute the job based on type
        if [ "$job_type" = "dense_n_view" ]; then
            if run_dense_n_view "$gpu_id" "${params[@]}" "$job_id" "$attempt"; then
                return 0
            fi
        elif [ "$job_type" = "rmvd_mvs" ]; then
            if run_rmvd_mvs "$gpu_id" "${params[@]}" "$job_id" "$attempt"; then
                return 0
            fi
        fi
        
        # Job failed, increment attempt
        ((attempt++))
        
        if [ $attempt -le $MAX_RETRIES ]; then
            log "Retrying job $job_id (attempt $attempt/$MAX_RETRIES)..."
            sleep 5
        fi
    done
    
    log "Job $job_id failed after $MAX_RETRIES attempts"
    return 1
}

# ============================================================================
# BUILD JOB QUEUE
# ============================================================================

build_job_queue() {
    log "Building job queue..."
    
    local job_count=0
    
    # Iterate through all model checkpoints
    for checkpoint in "${MODEL_CHECKPOINTS[@]}"; do
        log "Adding jobs for checkpoint: $checkpoint"
        
        # Add Dense N-View jobs
        for dataset in "${DATASETS_1[@]}"; do
            for num_views in "${NUM_VIEWS[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    add_job "dense_n_view" "$checkpoint" "$dataset" "$num_views" "$batch_size" > /dev/null
                    ((job_count++))
                done
            done
        done
        
        # Add RMVD MVS jobs
        for eval_dataset in "${EVAL_DATASETS[@]}"; do
            for conditioning in "${CONDITIONING_MODES[@]}"; do
                for alignment in "${ALIGNMENT_METHODS[@]}"; do
                    for views in "${EVALUATION_VIEWS[@]}"; do
                        add_job "rmvd_mvs" "$checkpoint" "$eval_dataset" "$conditioning" "$alignment" "$views" > /dev/null
                        ((job_count++))
                    done
                done
            done
        done
    done
    
    log "Total jobs queued: $job_count"
    echo "$job_count"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log "========================================="
    log "Starting Exhaustive Benchmark Run with GPU Scheduling"
    log "Log file: $LOG_FILE"
    log "Job queue file: $JOB_QUEUE_FILE"
    log "Job status file: $JOB_STATUS_FILE"
    log "========================================="
    log ""
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR: nvidia-smi not found. GPU scheduling requires NVIDIA GPUs."
        exit 1
    fi
    
    # Display available GPUs
    log "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    log ""
    
    # Initialize job tracking
    init_job_queue
    
    # Build the job queue
    local total_jobs=$(build_job_queue)
    
    log "========================================="
    log "Starting job execution..."
    log "========================================="
    log ""
    
    local successful_jobs=0
    local failed_jobs=0
    local job_num=1
    
    # Process each job in the queue
    while IFS=$'\t' read -r job_id job_type params; do
        [ -z "$job_id" ] && continue  # Skip empty lines
        
        log "Processing job $job_num/$total_jobs (ID: $job_id)"
        
        # Parse parameters (space-separated)
        read -ra param_array <<< "$params"
        
        # Execute job with retry logic
        if execute_job_with_retry "$job_id" "$job_type" "${param_array[@]}"; then
            ((successful_jobs++))
        else
            ((failed_jobs++))
        fi
        
        ((job_num++))
        
    done < "$JOB_QUEUE_FILE"
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    log "========================================="
    log "EXECUTION SUMMARY"
    log "========================================="
    log "Total jobs: $total_jobs"
    log "Successful: $successful_jobs"
    log "Failed: $failed_jobs"
    log "Model checkpoints: ${#MODEL_CHECKPOINTS[@]}"
    log "========================================="
    log "Job status saved to: $JOB_STATUS_FILE"
    log ""
    
    # Display failed jobs if any
    if [ $failed_jobs -gt 0 ]; then
        log "Failed jobs:"
        grep "FAILED" "$JOB_STATUS_FILE" | while IFS='|' read -r jid jtype ckpt params status gpu att start end; do
            log "  Job $jid: $jtype with checkpoint $(basename $ckpt)"
        done
        log ""
    fi
    
    if [ $failed_jobs -eq 0 ]; then
        log "✓ All benchmarks completed successfully!"
        return 0
    else
        log "✗ Some benchmarks failed. Check the log for details."
        return 1
    fi
}

# Run main function
main