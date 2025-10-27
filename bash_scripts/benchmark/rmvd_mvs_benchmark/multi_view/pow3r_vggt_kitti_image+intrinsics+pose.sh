#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=kitti \
	evaluation_conditioning=image+intrinsics+pose \
	evaluation_alignment=median \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image+intrinsics+pose_median_multi_view/kitti/pow3r_vggt" \
	model=pow3r_vggt \
	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
	model.model_config.load_custom_ckpt=true \
	model.model_config.custom_ckpt_path="/work/weights/pow3r_vggt/argoverse_learning_rate_experiments/argoverse_2ep_1eM5/checkpoint_1.pt" \
