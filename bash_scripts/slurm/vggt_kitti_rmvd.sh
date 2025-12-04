#!/bin/bash
#SBATCH -J vggt-kitti-rmvd
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:1             # 通过 GRES 分配 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 72:00:00
#SBATCH --nodelist=sines-2-embody-ai-node-7
#SBATCH -o /mnt/nfs/slurm/home/gordon/slurm_logs/%x-%j.out
#SBATCH -e /mnt/nfs/slurm/home/gordon/slurm_logs/%x-%j.err
#SBATCH --mem=512G

set -euo pipefail


# ===== 环境 =====
eval "$(micromamba shell hook --shell bash)"
micromamba activate mapanything





# ===== 通用优化 =====
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

ulimit -n 32768 || true
ulimit -u 32768 || true


# ===== 根据 CUDA_VISIBLE_DEVICES 计算 GPU 数 =====
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "⚠️  CUDA_VISIBLE_DEVICES 未设置，默认使用 nvidia-smi 检测"
  NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  # 按逗号切分计算数量
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi
echo "[INFO] Detected ${NUM_GPUS} visible GPUs via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Server: $(hostname)"
echo "[INFO] IP Address: $(hostname -I | awk '{print $1}')"

nvidia-smi
python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=kitti \
	evaluation_conditioning=image \
	evaluation_alignment=median \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_median_multi_view/kitti/vggt" \
	model=vggt \
	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
