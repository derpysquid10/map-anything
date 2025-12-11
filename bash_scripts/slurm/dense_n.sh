#!/bin/bash

# Job submission script that loops over configurations and submits individual SLURM jobs

set -euo pipefail

# ===== CONFIGURABLE PARAMETERS =====
# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "1 2 benchmark_518_eth3d_snpp_tav2"
    "10 4 benchmark_518_eth3d_snpp_tav2"
    "8 8 benchmark_518_eth3d_snpp_tav2"
    "5 16 benchmark_518_eth3d_snpp_tav2"
    "1 50 benchmark_518_eth3d_snpp_tav2"
    "2 32 benchmark_518_eth3d_snpp_tav2"
    "4 24 benchmark_518_eth3d_snpp_tav2"
)

# Base output directory (hydra.run.dir will be: ${HYDRA_RUN_DIR}/dense_${num_views}_view/${EXPERIMENT_NAME})
HYDRA_RUN_DIR="/mnt/glusterfs/SpatialAI/Experiments/gordon/experiments/dense_n"

# Experiment name (will be part of the output path)
EXPERIMENT_NAME="ray_fuse_90k"

# Model checkpoint path
MODEL_CHECKPOINT_PATH="/mnt/glusterfs/SpatialAI/Experiments/gordon/weights/uniscale/ray_head_fused/checkpoint_1_40000.pt"

# Model configuration
MODEL_TYPE="pow3r_vggt"

# SLURM configuration
PARTITION="short"
NUM_GPUS=4
NUM_CPUS=32
MEMORY="128G"
TIME_LIMIT="10:00:00"

# ===== JOB SUBMISSION LOOP =====
echo "=========================================="
echo "Submitting benchmark jobs to SLURM"
echo "=========================================="

job_count=0
for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"

    # Create job name
    job_name="dense_${num_views}v_bs${batch_size}"

    # Create output directory for this configuration
    output_dir="${HYDRA_RUN_DIR}/dense_${num_views}_view/${EXPERIMENT_NAME}"

    echo ""
    echo "Submitting job ${job_count}: ${job_name}"
    echo "  Batch size: ${batch_size}"
    echo "  Num views: ${num_views}"
    echo "  Dataset: ${dataset}"
    echo "  Output: ${output_dir}"

    # Submit SLURM job
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --gres=gpu:${NUM_GPUS} \
        --ntasks-per-node=1 \
        --cpus-per-task=${NUM_CPUS} \
        --time="${TIME_LIMIT}" \
        --mem="${MEMORY}" \
        --output="/mnt/nfs/slurm/home/gordon/slurm_logs/${job_name}-%j.out" \
        --error="/mnt/nfs/slurm/home/gordon/slurm_logs/${job_name}-%j.err" \
        /mnt/nfs/slurm/home/gordon/map-anything/bash_scripts/slurm/dense_n_worker.sh \
        "${batch_size}" \
        "${num_views}" \
        "${dataset}" \
        "${MODEL_TYPE}" \
        "${MODEL_CHECKPOINT_PATH}" \
        "${output_dir}"

    job_count=$((job_count + 1))
done

echo ""
echo "=========================================="
echo "Submitted ${job_count} jobs to SLURM"
echo "Check status with: squeue -u \$USER"
echo "=========================================="
