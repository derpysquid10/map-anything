#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=scannet \
	evaluation_conditioning=image \
	evaluation_alignment=median \
	evaluation_views=single_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_median_single_view/scannet/pow3r_vggt" \
	model=pow3r_vggt \
	evaluation_resolution=\${dataset.resolution_options.518_1_33_ar} \
