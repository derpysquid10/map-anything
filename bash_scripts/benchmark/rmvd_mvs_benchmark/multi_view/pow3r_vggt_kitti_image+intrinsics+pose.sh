# Run for kitti
python \
    benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=kitti \
	evaluation_conditioning=image \
	evaluation_alignment=none \
	evaluation_views=single_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/final_sanity/kitti/pow3r_vggt" \
	model=pow3r_vggt \
	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
	model.model_config.load_custom_ckpt=true \
	model.model_config.custom_ckpt_path="/mnt/nfs/moma/checkpoints/exp36_whole_inject_before_norm_Prob_lowLR_again/checkpoint_0_30000.pt"

# # Run for scannet
# python \
#     benchmarking/rmvd_mvs_benchmark/benchmark.py \
# 	machine=default \
# 	eval_dataset=scannet \
# 	evaluation_conditioning=image \
# 	evaluation_alignment=median \
# 	evaluation_views=multi_view \
# 	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/final_sanity/scannet/pow3r_vggt" \
# 	model=pow3r_vggt \
# 	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
# 	model.model_config.load_custom_ckpt=true \
# 	model.model_config.custom_ckpt_path="/mnt/nfs/moma/checkpoints/exp36_whole_inject_before_norm_Prob_lowLR_again/checkpoint_0_30000.pt"