# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MapAnything is a transformer-based model for universal feed-forward metric 3D reconstruction. The repository supports over 12 different 3D reconstruction tasks including multi-image SfM, multi-view stereo, monocular metric depth estimation, registration, and depth completion.

Two model variants are available:
- `facebook/map-anything` (CC-BY-NC 4.0) - Research use, best performance
- `facebook/map-anything-apache` (Apache 2.0) - Commercial use

## Development Commands

### Installation

```bash
# Create environment
conda create -n mapanything python=3.12 -y
conda activate mapanything

# Install MapAnything
pip install -e .

# Install all optional dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

### Running Demos

```bash
# Start Rerun server (Terminal 1)
rerun --serve --port 2004 --web-viewer-port 2006

# Run image-only inference demo (Terminal 2)
python scripts/demo_images_only_inference.py \
    --image_folder /path/to/images \
    --viz \
    --save_glb \
    --output_path /path/to/output.glb \
    --memory_efficient_inference  # For large numbers of views

# Run on COLMAP outputs
python scripts/demo_inference_on_colmap_outputs.py \
    --colmap_path /path/to/colmap_output \
    --viz \
    --save_glb

# Run Gradio app locally
pip install -e ".[gradio]"
python scripts/gradio_app.py

# Use Apache 2.0 model
python scripts/demo_images_only_inference.py --apache <other_args>
```

### COLMAP Export

```bash
pip install -e ".[colmap]"

# Feed-forward prediction only
python scripts/demo_colmap.py \
    --scene_dir=/path/to/scene \
    --memory_efficient_inference

# With bundle adjustment
python scripts/demo_colmap.py \
    --scene_dir=/path/to/scene \
    --memory_efficient_inference \
    --use_ba

# Faster BA with reduced parameters
python scripts/demo_colmap.py \
    --scene_dir=/path/to/scene \
    --memory_efficient_inference \
    --use_ba \
    --max_query_pts=2048 \
    --query_frame_num=5
```

Scene directory must have `images/` subdirectory. Output goes to `sparse/` in COLMAP format.

### Training

#### Single Dataset Training (Quick Start)

```bash
# Train on BlendedMVS with 4 views
bash bash_scripts/train/examples/mapa_curri_4v_bmvs_48ipg_8g.sh 8

# Memory optimization flags
# Use in training config:
# model.info_sharing.module_args.gradient_checkpointing=true
# model.pred_head.gradient_checkpointing=true
```

#### Main Training Pipeline

```bash
# Stage 1: 4-view training
bash bash_scripts/train/main/mapa_curri_4v_13d_48ipg_64g.sh 8

# Stage 2: 24-view training
bash bash_scripts/train/main/mapa_curri_24v_13d_48ipg_64g.sh 8

# Apache 2.0 variants
bash bash_scripts/train/main/mapa_curri_4v_6d_48ipg_8g_apache.sh 8
bash bash_scripts/train/main/mapa_curri_24v_6d_48ipg_64g_apache.sh 8
```

#### Fine-tuning Other Models

```bash
# MoGe-2 fine-tuning
bash bash_scripts/train/finetuning/moge2_finetuning.sh 8

# VGGT fine-tuning
bash bash_scripts/train/finetuning/vggt_finetuning.sh 8

# π³ fine-tuning
bash bash_scripts/train/finetuning/pi3_finetuning.sh 8
```

#### Training with torchrun

```bash
# Basic training invocation
torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=<your_machine> \
    dataset=<dataset_config> \
    model=mapanything \
    loss=<loss_config> \
    train_params=<train_params_config> \
    hydra.run.dir='${root_experiments_dir}/path/to/experiment'
```

### Benchmarking

#### Convert HuggingFace Checkpoint

```bash
# Convert default model
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --output_path checkpoints/facebook_map-anything.pth

# Convert Apache model
python scripts/convert_hf_to_benchmark_checkpoint.py \
    --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

#### Run Benchmarks

```bash
# Dense N-view reconstruction
bash bash_scripts/benchmark/dense_n_view/mapa_24v_mvs_nm.sh

# Dense 2-view reconstruction
bash bash_scripts/benchmark/dense_2_view/mapa_4v.sh

# Calibration benchmark
bash bash_scripts/benchmark/calibration/mapanything.sh

# RobustMVD benchmark
bash bash_scripts/benchmark/rmvd_mvs_benchmark/multi_view/mapanything_scannet_image.sh
```

### Data Processing

#### Verify Dataloaders

```bash
# Test dataloader with visualization
python mapanything/datasets/wai/blendedmvs.py \
    --root_dir /path/to/blendedmvs \
    --dataset_metadata_dir /path/to/metadata \
    --num_of_views 4 \
    --viz

# Other datasets
python mapanything/datasets/wai/eth3d.py --viz
python mapanything/datasets/wai/scannetpp.py --viz
```

#### WAI Processing

Requires separate environment with Python 3.12:

```bash
conda create -n wai_processing python=3.12 -y
conda activate wai_processing
cd data_processing/wai_processing/
pip install -e .[all]

# Run conversion
python -m wai_processing.scripts.conversion.<dataset_name> \
    original_root=<original_path> \
    root=<wai_path>

# Run undistortion
python -m wai_processing.scripts.undistort \
    <config>.yaml \
    root=<dataset_path>

# Run covisibility
python -m wai_processing.scripts.covisibility \
    <config>.yaml \
    root=<dataset_path>
```

### SLURM Execution

```bash
# Submit SLURM job
sbatch bash_scripts/slurm/dense_n.sh

# Check SLURM logs
tail -f /mnt/nfs/slurm/home/gordon/slurm_logs/<job-name>-<job-id>.out
tail -f /mnt/nfs/slurm/home/gordon/slurm_logs/<job-name>-<job-id>.err

# Check queue
squeue -u $USER

# Check GPU allocation
squeue -p <partition> -t R -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

## Architecture

### Core Components

- **Models** (`mapanything/models/`):
  - `mapanything.py`: Core MapAnything model, ModularDUSt3R, and ablations
  - `external/`: Wrappers for external models (DUSt3R, MASt3R, MoGe, VGGT, etc.)
  - Model factory pattern for dynamic model loading

- **Datasets** (`mapanything/datasets/wai/`):
  - All datasets in WAI (WorldAI) format
  - 14 datasets supported: ASE, BlendedMVS, DL3DV, Dynamic Replica, ETH3D, MPSD, MegaDepth, MVS-Synth, Parallel Domain 4D, SAIL-VOS 3D, ScanNet++V2, Spring, TartanAirV2-WB, UnrealStereo4K
  - Each dataset has a standalone Python file with main call for testing

- **Training** (`mapanything/train/`):
  - Distributed training utilities
  - Training scripts in `scripts/train.py`

- **Utilities** (`mapanything/utils/`):
  - Image processing, geometry utilities
  - Visualization helpers

### Configuration System

Uses Hydra for hierarchical configuration:

- **Machine configs** (`configs/machine/`): Define paths for data, checkpoints, experiments
  - `default.yaml`: Currently configured for SLURM cluster at `/mnt/nfs/slurm/home/gordon/`
  - Update `root_data_dir`, `root_experiments_dir`, etc. for your environment

- **Model configs** (`configs/model/`): Model architectures, encoder settings, info sharing modules
  - `mapanything.yaml`: Main model
  - `encoder/`: DINOv2, CRoCo, RADIO options
  - `info_sharing/`: Attention mechanisms (AAT, GAT)

- **Dataset configs** (`configs/dataset/`): Per-dataset and combined training configs
  - Individual dataset configs in subdirectories
  - `megatrain_13d_*` configs combine all 13 training datasets
  - `megatrain_6d_*` configs for Apache 2.0 variant (6 datasets)

- **Loss configs** (`configs/loss/`): Loss function definitions and ablations

- **Training params** (`configs/train_params/`): Learning rates, epochs, batch sizes

### Key Patterns

#### Camera Convention
All camera poses use **OpenCV convention**: +X right, +Y down, +Z forward, cam2world format.

#### Input Flexibility
Model supports any combination of inputs per view:
- `img`: RGB image (required)
- `intrinsics` OR `ray_directions`: Camera calibration (mutually exclusive)
- `depth_z`: Z-depth maps (requires calibration)
- `camera_poses`: 4x4 matrices or (quaternions, translations)
- `is_metric_scale`: Boolean flag for metric scale inputs

Use `preprocess_inputs()` from `mapanything.utils.image` to prepare inputs.

#### Output Structure
Model predictions include:
- Geometry: `pts3d`, `pts3d_cam`, `depth_z`, `depth_along_ray`
- Camera: `ray_directions`, `intrinsics`, `camera_poses`, `cam_trans`, `cam_quats`
- Quality: `conf`, `mask`, `non_ambiguous_mask`
- Scaling: `metric_scaling_factor`

#### Memory Optimization
- Set `memory_efficient_inference=True` for large view counts (up to 2000 views on 140GB)
- Use gradient checkpointing in training
- Adjust `max_num_of_imgs_per_gpu` to control memory usage
- Scale learning rate proportionally: effective_batch_size = NUM_GPUS × max_num_of_imgs_per_gpu / num_of_views

#### Multi-Node Training
Training scripts support multi-node execution:
```bash
bash bash_scripts/train/main/mapa_curri_24v_13d_48ipg_64g.sh \
    NUM_GPUS NUM_NODES NODE_RANK JOB_ID HOST_NODE_ADDR MAX_RESTARTS
```

### Data Format

**WAI (WorldAI)** is the unified data format for all datasets. Each dataset has:
- Scene-based organization
- Standardized metadata structure
- Pre-computed covisibility graphs
- Consistent coordinate conventions

Pre-computed metadata available on HuggingFace: `facebook/map-anything` dataset.

### External Model Integration

The codebase supports benchmarking and fine-tuning external models:
- DUSt3R, MASt3R, MUSt3R (NAVER)
- MoGe (Microsoft)
- VGGT, VGGSfM (Meta)
- π³ (Pi3)
- Pow3R wrappers

Models are wrapped with consistent interfaces in `mapanything/models/external/`.

### Benchmarking Datasets

Test splits from:
- ETH3D: High-quality indoor/outdoor scenes
- ScanNet++V2: Indoor RGB-D scans
- TartanAirV2-WB: Wide-baseline outdoor drone flights

See `benchmarking/dense_n_view/README.md`, `benchmarking/calibration/README.md`, `benchmarking/rmvd_mvs_benchmark/README.md` for details.

## Environment Variables

```bash
# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NCCL settings for multi-GPU training
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# Thread settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Hydra error reporting
export HYDRA_FULL_ERROR=1
```

## Important Notes

- Images must be in directory named `images/` for COLMAP export
- Rerun server runs on ports 2004 (server) and 2006 (web viewer)
- Effective batch size formula: NUM_GPUS × max_num_of_imgs_per_gpu / num_of_views
- Use `--apache` flag to load Apache 2.0 model instead of default CC-BY-NC model
- WAI processing requires separate Python 3.12 environment
- Training checkpoints and HuggingFace checkpoints have different formats (use conversion script)
- First view (reference) must have `camera_poses` if any view has them
- Cannot provide both `intrinsics` and `ray_directions` simultaneously
- All ablation training scripts are in `bash_scripts/train/ablations/`
