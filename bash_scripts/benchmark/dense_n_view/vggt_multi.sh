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

# Configurable benchmark arguments
BENCHMARK_SCRIPT="/workspace/run_benchmark_with_conda.sh"
PYTHON_SCRIPT="benchmarking/dense_n_view/benchmark.py"
MACHINE="default"
NUM_WORKERS="16"
MODEL="vggt"
LOAD_CUSTOM_CKPT="false"
CUSTOM_CKPT_PATH="/mnt/nfs/SpatialAI/moma/logs/final_pose_cam_both0.5probs/checkpoint_0_38000.pt"
BASE_OUTPUT_DIR="/mnt/nfs/binbin/experiments_new_model/mapanything/benchmarking"
OUTPUT_DIR_SUFFIX="vggt"
PRINCIPAL_POINT_CENTERED="true"

# Function to get available free GPUs (100MB or less used)
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 <= 100 {print $1}' | \
    tr '\n' ' '
}

# Function to get next available GPU (max 1 process per GPU)
get_next_gpu() {
    local free_gpus=($1)
    local gpu_jobs=("${@:2}")
    
    for gpu in "${free_gpus[@]}"; do
        local gpu_busy=false
        for job_gpu in "${gpu_jobs[@]}"; do
            if [[ "$job_gpu" == "$gpu" ]]; then
                gpu_busy=true
                break
            fi
        done
        if [[ "$gpu_busy" == false ]]; then
            echo $gpu
            return
        fi
    done
    echo ""
}

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "10 2 benchmark_518_eth3d_snpp_tav2"
    "10 4 benchmark_518_eth3d_snpp_tav2"
    "10 8 benchmark_518_eth3d_snpp_tav2"
    "5 16 benchmark_518_eth3d_snpp_tav2"
    "1 50 benchmark_518_eth3d_snpp_tav2"
    "2 32 benchmark_518_eth3d_snpp_tav2"
    "4 24 benchmark_518_eth3d_snpp_tav2"
    
    
    # "1 100 benchmark_518_eth3d_snpp_tav2"
)

prior_combinations=(
    "[]"
    # "[intrinsics]"
    # "[extrinsics]"
    # "[intrinsics,extrinsics]"
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


# Generate all job combinations and create command queue
declare -a command_queue=()
declare -a queue_descriptions=()

echo "Creating command queue..."
for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"
    for prior_combo in "${prior_combinations[@]}"; do
        prior_dir_name=$(echo "$prior_combo" | sed 's/\[\]//g' | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/_/g')
        if [ -z "$prior_dir_name" ]; then
            prior_dir_name="no_priors"
        fi
        
        # Create the full command using configurable variables
        cmd="$BENCHMARK_SCRIPT GPU_PLACEHOLDER $PYTHON_SCRIPT machine=$MACHINE dataset=$dataset dataset.num_workers=$NUM_WORKERS dataset.num_views=$num_views batch_size=$batch_size model=$MODEL hydra.run.dir='$BASE_OUTPUT_DIR/dense_${num_views}_view/${OUTPUT_DIR_SUFFIX}_${prior_dir_name}' dataset.principal_point_centered=$PRINCIPAL_POINT_CENTERED"
        
        command_queue+=("$cmd")
        queue_descriptions+=("bs${batch_size}_nv${num_views}_${prior_dir_name}")
    done
done

echo "Created command queue with ${#command_queue[@]} commands"

# Function to process the command queue
process_command_queue() {
    local queue_index=0
    
    while [ $queue_index -lt ${#command_queue[@]} ]; do
        echo "Checking for available GPUs... (Queue position: $((queue_index + 1))/${#command_queue[@]})"
        
        # Check for finished jobs first
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
            # Get the next command from queue
            local cmd="${command_queue[$queue_index]}"
            local description="${queue_descriptions[$queue_index]}"
            
            # Replace GPU placeholder with actual GPU
            cmd="${cmd/GPU_PLACEHOLDER/$available_gpu}"
            
            echo "Starting job $((queue_index + 1))/${#command_queue[@]} on GPU $available_gpu: $description"
            
            # Execute command in background
            eval "$cmd" &
            local job_pid=$!
            
            # Track the job
            job_pids+=($job_pid)
            job_gpus+=($available_gpu)
            running_jobs+=("$description")
            
            echo "Started job (PID: $job_pid) on GPU $available_gpu"
            
            # Move to next command in queue
            ((queue_index++))
            
            # Small delay to prevent race conditions
            sleep 2
        else
            echo "No GPUs available. Waiting 10 seconds..."
            sleep 10
        fi
    done
}

# Process the command queue
echo "Starting command queue processing..."
process_command_queue

# Wait for all remaining jobs to complete
echo "Waiting for all jobs to complete..."
while [ ${#job_pids[@]} -gt 0 ]; do
    for i in "${!job_pids[@]}"; do
        if ! kill -0 "${job_pids[i]}" 2>/dev/null; then
            echo "Job ${running_jobs[i]} on GPU ${job_gpus[i]} has finished"
            unset job_pids[i]
            unset job_gpus[i] 
            unset running_jobs[i]
        fi
    done
    
    # Rebuild arrays
    job_pids=($(printf '%s\n' "${job_pids[@]}" | grep -v '^$'))
    job_gpus=($(printf '%s\n' "${job_gpus[@]}" | grep -v '^$'))
    running_jobs=($(printf '%s\n' "${running_jobs[@]}" | grep -v '^$'))
    
    if [ ${#job_pids[@]} -gt 0 ]; then
        echo "Still waiting for ${#job_pids[@]} jobs: ${running_jobs[*]}"
        sleep 10
    fi
done

echo "All benchmarking jobs completed!"
