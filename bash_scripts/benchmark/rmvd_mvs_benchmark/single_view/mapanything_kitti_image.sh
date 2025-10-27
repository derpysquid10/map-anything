#!/bin/bash

python \
	benchmarking/rmvd_mvs_benchmark/benchmark.py \
	machine=default \
	eval_dataset=kitti \
	evaluation_conditioning=image \
	evaluation_alignment=median \
	evaluation_views=single_view \
	hydra.run.dir="\${root_experiments_dir}/mapanything/benchmarking/rmvd_image_median_single_view/kitti/mapanything" \
	model=mapanything \
	model.pretrained=\${root_experiments_dir}/mapanything/training/mapa_curri_24v_13d_48ipg_64g/checkpoint-last.pth \
	evaluation_resolution=\${dataset.resolution_options.518_3_20_ar} \
