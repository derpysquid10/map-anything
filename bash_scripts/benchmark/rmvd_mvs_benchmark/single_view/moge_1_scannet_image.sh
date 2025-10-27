#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=scannet \
	evaluation_conditioning=image \
	evaluation_alignment=median \
	evaluation_views=single_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_median_single_view/scannet/moge_1" \
	model=moge_1 \
	evaluation_resolution="'(640, 480)'" \
