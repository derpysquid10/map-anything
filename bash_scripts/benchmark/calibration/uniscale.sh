CUDA_VISIBLE_DEVICES=6 python3 \
    benchmarking/calibration/benchmark.py \
    machine=default \
    dataset=benchmark_sv_calib_518_many_ar_eth3d_snpp_tav2 \
    dataset.num_workers=12 \
    batch_size=20 \
    model=pow3r_vggt \
    hydra.run.dir='${root_experiments_dir}/mapanything/calibration_benchmarking/single_view/uniscale_other_prob' \
    model.model_config.load_custom_ckpt=true \
	model.model_config.custom_ckpt_path="/mnt/nfs/SpatialAI/moma/logs/final_pose_cam_both0.5probs/checkpoint_0_38000.pt"

CUDA_VISIBLE_DEVICES=5 python3 \
    benchmarking/calibration/benchmark.py \
    machine=default \
    dataset=benchmark_sv_calib_518_many_ar_eth3d_snpp_tav2 \
    dataset.num_workers=12 \
    batch_size=20 \
    model=pow3r_vggt \
    hydra.run.dir='${root_experiments_dir}/mapanything/calibration_benchmarking/single_view/uniscale_map_prob' \
    model.model_config.load_custom_ckpt=true \
	model.model_config.custom_ckpt_path="/mnt/nfs/SpatialAI/moma/logs/final_swapped_pose_toCam/checkpoint_0_38000.pt"

