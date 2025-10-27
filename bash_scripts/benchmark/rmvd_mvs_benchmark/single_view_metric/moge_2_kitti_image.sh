#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=kitti \
	evaluation_conditioning=image \
	evaluation_alignment=none \
	evaluation_views=single_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_none_single_view/kitti/moge_2" \
	model=moge_2 \
	evaluation_resolution="'(1248,384)'" \
