#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Generate bash scripts for pow3r_vggt RMVD Benchmark
Only generates single-view experiments with median alignment
"""

import os

machine = "default"
self_folder = os.path.dirname(os.path.abspath(__file__))


def get_model_settings(model: str, dataset: str):
    """Get model-specific settings"""
    if model == "pow3r_vggt":
        settings = {
            "model": "pow3r_vggt",
            "evaluation_resolution": (
                "\\${dataset.resolution_options.518_1_33_ar}"
                if dataset != "kitti"
                else "\\${dataset.resolution_options.518_3_20_ar}"
            ),
        }

        # Optional: Uncomment these lines and set your checkpoint path
        # settings["model.model_config.load_custom_ckpt"] = "true"
        # settings["model.model_config.custom_ckpt_path"] = '"/path/to/your/checkpoint.pth"'

        return settings
    else:
        raise ValueError(f"Unknown model: {model}")


def generate_shell_for_single_experiment(
    model: str = "pow3r_vggt",
    dataset: str = "kitti",
    conditioning: str = "image",
    alignment: str = "median",
    view: str = "single_view",
):
    """Generate a single bash script for benchmarking"""

    benchmark_configs = {
        "machine": machine,
        "eval_dataset": dataset,
        "evaluation_conditioning": conditioning,
        "evaluation_alignment": alignment,
        "evaluation_views": view,
        "hydra.run.dir": '"\${root_experiments_dir}/mapanything/benchmarking/rmvd_'
        + f'{conditioning}_{alignment}_{view}/{dataset}/{model}"',
    }

    benchmark_configs.update(get_model_settings(model, dataset))

    file_content = "#!/bin/bash\n\n"
    file_content += "python \\\n"
    file_content += "\tbenchmarking/rmvd_mvs_benchmark/benchmark.py \\\n"

    for key, value in benchmark_configs.items():
        file_content += f"\t{key}={value} \\\n"

    # Compute the output path for the shell script
    shell_output_path = os.path.join(
        self_folder,
        (view + "_metric") if alignment == "none" else view,
        f"{model}_{dataset}_{conditioning}.sh",
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(shell_output_path), exist_ok=True)

    # Write the content to the file
    with open(shell_output_path, "w") as f:
        f.write(file_content)

    # Change the file permission to make it executable
    os.chmod(shell_output_path, 0o755)

    print(f"✓ Generated: {shell_output_path}")
    return shell_output_path


if __name__ == "__main__":
    print("="*80)
    print("Generating pow3r_vggt RMVD Benchmark Scripts")
    print("="*80)
    print()

    generated_scripts = []

    # Generate single-view with alignment experiments
    for dataset in ["kitti", "scannet"]:
        # No conditioning (image only)
        script = generate_shell_for_single_experiment(
            model="pow3r_vggt",
            dataset=dataset,
            conditioning="image",
            alignment="median",
            view="single_view",
        )
        generated_scripts.append(script)

        # With intrinsics conditioning
        script = generate_shell_for_single_experiment(
            model="pow3r_vggt",
            dataset=dataset,
            conditioning="image+intrinsics",
            alignment="median",
            view="single_view",
        )
        generated_scripts.append(script)

    print()
    print("="*80)
    print(f"✓ Successfully generated {len(generated_scripts)} benchmark scripts!")
    print("="*80)
    print()
    print("Generated scripts:")
    for script in generated_scripts:
        print(f"  - {os.path.relpath(script, self_folder)}")

    print()
    print("To run a benchmark, execute:")
    print(f"  cd /workspace")
    print(f"  bash {os.path.relpath(generated_scripts[0], '/workspace')}")
    print()
    print("To customize checkpoint path, edit this script and uncomment lines 30-31")
    print()
