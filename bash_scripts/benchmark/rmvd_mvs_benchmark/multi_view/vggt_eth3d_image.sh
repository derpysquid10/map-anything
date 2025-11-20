#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=eth3d \
	evaluation_conditioning=image \
	evaluation_alignment=median \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_median_multi_view/kitti/vggt" \
	model=vggt \
	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
