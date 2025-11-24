#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=eth3d \
	evaluation_conditioning=image+intrinsics+pose \
	evaluation_alignment=median \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image+intrinsics+pose_median_multi_view/kitti/mapanything" \
	model=mapanything \
	model.pretrained="/mnt/nfs/SpatialAI/weights/mapanything/converted_checkpoint.pth" \
	evaluation_resolution=\${dataset.resolution_options.518_1_52_ar} \
