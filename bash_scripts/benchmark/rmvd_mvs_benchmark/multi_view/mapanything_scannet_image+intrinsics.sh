#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=scannet \
	evaluation_conditioning=image+intrinsics \
	evaluation_alignment=median \
	evaluation_views=multi_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image+intrinsics_median_multi_view/scannet/mapanything" \
	model=mapanything \
	model.pretrained=\${root_experiments_dir}/mapanything/training/mapa_curri_24v_13d_48ipg_64g/checkpoint-last.pth \
	evaluation_resolution=\${dataset.resolution_options.518_1_33_ar} \
