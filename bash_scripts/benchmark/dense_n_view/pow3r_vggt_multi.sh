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

# Function to get next available GPU
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
    # "1 2 benchmark_518_eth3d_snpp_tav2"
    # "10 4 benchmark_518_eth3d_snpp_tav2"
    # "10 8 benchmark_518_eth3d_snpp_tav2"
    # "5 16 benchmark_518_eth3d_snpp_tav2"
    # "1 50 benchmark_518_eth3d_snpp_tav2"
    "2 32 benchmark_518_eth3d_snpp_tav2"
    "4 24 benchmark_518_eth3d_snpp_tav2"
    
    
    # "1 100 benchmark_518_eth3d_snpp_tav2"
)

prior_combinations=(
    # "[]"
    # "[intrinsics]"
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
    
    /workspace/run_benchmark_with_conda.sh $gpu \
        benchmarking/dense_n_view/benchmark.py \
        machine=default \
        dataset=$dataset \
        dataset.num_workers=16 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=pow3r_vggt \
        model.model_config.load_custom_ckpt=true \
        model.model_config.custom_ckpt_path="/mnt/nfs/binbin/weights/prob_multi_unnormalized_tartan_scannetpp/checkpoint_0.pt" \
        hydra.run.dir='/mnt/nfs/mapanything/benchmarking/dense_'"${num_views}"'_view/2week_fixed_extri_inv_'"${prior_dir_name}" \
        input_priors=$prior_combo
    
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

# Generate all job combinations
declare -a all_jobs=()
for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"
    for prior_combo in "${prior_combinations[@]}"; do
        prior_dir_name=$(echo "$prior_combo" | sed 's/\[\]//g' | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/_/g')
        if [ -z "$prior_dir_name" ]; then
            prior_dir_name="no_priors"
        fi
        all_jobs+=("$batch_size|$num_views|$dataset|$prior_combo|$prior_dir_name")
    done
done

echo "Total jobs to run: ${#all_jobs[@]}"

# Run all jobs in parallel across available GPUs
for job in "${all_jobs[@]}"; do
    IFS='|' read -r batch_size num_views dataset prior_combo prior_dir_name <<< "$job"
    
    # Wait for a free GPU
    gpu=$(wait_for_free_gpu)
    
    # Start the job in background
    run_benchmark_job "$gpu" "$batch_size" "$num_views" "$dataset" "$prior_combo" "$prior_dir_name" &
    job_pid=$!
    
    # Track the job
    job_pids+=($job_pid)
    job_gpus+=($gpu)
    running_jobs+=("bs${batch_size}_nv${num_views}_${prior_dir_name}")
    
    echo "Started job (PID: $job_pid) on GPU $gpu"
    
    # Small delay to prevent race conditions
    sleep 2
done

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
