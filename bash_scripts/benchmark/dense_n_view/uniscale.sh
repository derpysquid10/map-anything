#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export SAVE_DEBUG_FILES=false

# Function to get available free GPUs
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 < 100 {print $1}' | \
    tr '\n' ' '
}

# Get parameters from environment variables (set by dense_n_worker.sh)
# If not set, use defaults for standalone execution
batch_size=${BENCHMARK_BATCH_SIZE:-1}
num_views=${BENCHMARK_NUM_VIEWS:-2}
dataset=${BENCHMARK_DATASET:-benchmark_518_eth3d_snpp_tav2}
model_type=${BENCHMARK_MODEL_TYPE:-pow3r_vggt}
checkpoint_path=${BENCHMARK_CHECKPOINT_PATH:-/mnt/glusterfs/SpatialAI/Experiments/gordon/weights/uniscale/longer_training_cont/checkpoint_2_50000.pt}
base_output_dir=${BENCHMARK_OUTPUT_DIR:-/mnt/glusterfs/SpatialAI/Experiments/gordon/experiments/dense_n}

echo "=========================================="
echo "Uniscale.sh Configuration"
echo "=========================================="
echo "Batch size: ${batch_size}"
echo "Num views: ${num_views}"
echo "Dataset: ${dataset}"
echo "Model type: ${model_type}"
echo "Checkpoint: ${checkpoint_path}"
echo "Base output dir: ${base_output_dir}"
echo "=========================================="
echo ""

prior_combinations=(
    "[]"
    "[intrinsics]"
    "[extrinsics]"
    "[intrinsics,extrinsics]"
)

# Get initial list of free GPUs
echo "Detecting available GPUs..."
free_gpus=($(get_free_gpus))
echo "Available GPUs: ${free_gpus[@]}"

if [ ${#free_gpus[@]} -eq 0 ]; then
    echo "No free GPUs available. Exiting."
    exit 1
fi

# Arrays to track running jobs
declare -a job_pids=()
declare -a job_gpus=()
declare -a running_jobs=()

# Function to run a single benchmark job
run_benchmark_job() {
    local gpu=$1
    local batch_size=$2
    local num_views=$3
    local dataset=$4
    local prior_combo=$5
    local prior_dir_name=$6
    
    echo "Starting job on GPU $gpu: $dataset with batch_size=$batch_size, num_views=$num_views, input_priors=$prior_combo"
    
    /mnt/nfs/slurm/home/gordon/map-anything/run_benchmark_with_conda.sh $gpu \
        benchmarking/dense_n_view/benchmark.py \
        machine=default \
        dataset=$dataset \
        dataset.num_workers=16 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=pow3r_vggt \
        model.model_config.load_custom_ckpt=true \
        model.model_config.custom_ckpt_path="/mnt/glusterfs/SpatialAI/Experiments/gordon/weights/uniscale/longer_training_cont/checkpoint_2_50000.pt" \
        hydra.run.dir='/mnt/glusterfs/SpatialAI/Experiments/gordon/experiments/dense_n/dense_'"${num_views}"'_view/longer_training_base_150k_'"${prior_dir_name}" \
        input_priors=$prior_combo \
        dataset.principal_point_centered=true
    
    echo "Finished job on GPU $gpu: $dataset with batch_size=$batch_size, num_views=$num_views, input_priors=$prior_combo"
}
# Function to wait for a free GPU and clean up finished jobs
wait_for_free_gpu() {
    while true; do
        # Check for finished jobs
        for i in "${!job_pids[@]}"; do
            if ! kill -0 "${job_pids[i]}" 2>/dev/null; then
                echo "Job ${running_jobs[i]} on GPU ${job_gpus[i]} has finished"
                unset job_pids[i]
                unset job_gpus[i]
                unset running_jobs[i]
            fi
        done
        
        # Rebuild arrays to remove gaps
        job_pids=($(printf '%s\n' "${job_pids[@]}" | grep -v '^$'))
        job_gpus=($(printf '%s\n' "${job_gpus[@]}" | grep -v '^$'))
        running_jobs=($(printf '%s\n' "${running_jobs[@]}" | grep -v '^$'))
        
        # Try to find a free GPU
        local available_gpu=$(get_next_gpu "${free_gpus[*]}" "${job_gpus[@]}")
        if [[ -n "$available_gpu" ]]; then
            echo $available_gpu
            return
        fi
        
        echo "All GPUs busy, waiting..."
        sleep 10
    done
}

# Generate all job combinations for this configuration
declare -a job_commands=()
declare -a job_descriptions=()
job_count=0

for prior_combo in "${prior_combinations[@]}"; do
    prior_dir_name=$(echo "$prior_combo" | sed 's/\[\]//g' | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/_/g')
    if [ -z "$prior_dir_name" ]; then
        prior_dir_name="no_priors"
    fi

    # Build the command template (GPU will be assigned dynamically)
    cmd="/mnt/nfs/slurm/home/gordon/map-anything/run_benchmark_with_conda.sh GPU_PLACEHOLDER"
    cmd="$cmd benchmarking/dense_n_view/benchmark.py"
    cmd="$cmd machine=default"
    cmd="$cmd dataset=$dataset"
    cmd="$cmd dataset.num_workers=16"
    cmd="$cmd dataset.num_views=$num_views"
    cmd="$cmd batch_size=$batch_size"
    cmd="$cmd model=$model_type"
    cmd="$cmd model.model_config.load_custom_ckpt=true"
    cmd="$cmd model.model_config.custom_ckpt_path=\"$checkpoint_path\""
    cmd="$cmd hydra.run.dir='${base_output_dir}_${prior_dir_name}'"
    cmd="$cmd input_priors=$prior_combo"
    cmd="$cmd dataset.principal_point_centered=true"

    # Store command and description
    job_commands[$job_count]="$cmd"
    job_descriptions[$job_count]="Dataset: $dataset | Batch: $batch_size | Views: $num_views | Priors: $prior_dir_name"

    job_count=$((job_count + 1))
done

echo "Total jobs to run: $job_count"
echo ""

# Create base output directory if it doesn't exist
mkdir -p "$base_output_dir"

# Function to run a job with assigned GPU
run_job() {
    local job_idx=$1
    local gpu_id=$2
    local cmd="${job_commands[$job_idx]}"
    local desc="${job_descriptions[$job_idx]}"
    local log_file="${base_output_dir}/job_${job_idx}.log"
    
    # Replace GPU placeholder with actual GPU ID
    cmd="${cmd/GPU_PLACEHOLDER/$gpu_id}"
    
    echo "=========================================="
    echo "Starting Job $((job_idx + 1))/$job_count"
    echo "$desc | GPU: $gpu_id"
    echo "Log: $log_file"
    echo "=========================================="
    
    # Run the command and redirect output to both console and log file
    echo "command being run: $cmd"
    eval "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Job $((job_idx + 1)) completed successfully on GPU $gpu_id"
    else
        echo "✗ Job $((job_idx + 1)) failed on GPU $gpu_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Export functions and variables for parallel execution
export -f run_job get_free_gpus
export -a job_commands
export -a job_descriptions
export job_count base_output_dir

echo "=========================================="
echo "Starting parallel execution"
echo "Dynamically detecting free GPUs for each batch"
echo "=========================================="
echo ""

current_job=0
batch_number=1

# Run jobs until all are completed
while [ $current_job -lt $job_count ]; do
    # Get currently available GPUs
    current_free_gpus=($(get_free_gpus))
    available_gpu_count=${#current_free_gpus[@]}
    
    if [ $available_gpu_count -eq 0 ]; then
        echo "No GPUs available, waiting 30 seconds..."
        sleep 30
        continue
    fi
    
    echo "Batch $batch_number: Found $available_gpu_count free GPUs: ${current_free_gpus[*]}"
    
    # Calculate how many jobs to run in this batch
    jobs_remaining=$((job_count - current_job))
    jobs_in_batch=$((available_gpu_count < jobs_remaining ? available_gpu_count : jobs_remaining))
    
    echo "Starting $jobs_in_batch jobs (Jobs $((current_job+1)) to $((current_job+jobs_in_batch)))"
    
    # Start jobs in parallel on available GPUs
    pids=()
    for ((k=0; k<jobs_in_batch; k++)); do
        job_idx=$((current_job + k))
        gpu_id=${current_free_gpus[k]}
        run_job $job_idx $gpu_id &
        pids+=($!)
    done
    
    # Wait for all jobs in this batch to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    current_job=$((current_job + jobs_in_batch))
    batch_number=$((batch_number + 1))
    
    echo ""
    echo "Batch $((batch_number - 1)) completed. $((job_count - current_job)) jobs remaining."
    echo ""
    
    # Small delay to allow GPU memory to clear
    if [ $current_job -lt $job_count ]; then
        echo "Waiting 10 seconds for GPU memory to clear..."
        sleep 10
    fi
done

echo "=========================================="
echo "All $job_count benchmark jobs completed"
echo "=========================================="
