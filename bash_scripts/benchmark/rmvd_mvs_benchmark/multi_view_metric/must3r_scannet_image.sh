#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=scannet \
	evaluation_conditioning=image \
	evaluation_alignment=none \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_none_multi_view/scannet/must3r" \
	model=must3r \
	evaluation_resolution=\${dataset.resolution_options.512_1_33_ar} \
