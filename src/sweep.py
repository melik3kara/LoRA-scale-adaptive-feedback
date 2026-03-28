"""
Phase 1 — Static Sweep

Generates images across a grid of LoRA scale × ControlNet scale combinations.
All active LoRAs share the same scale value per run.

Usage:
    python src/sweep.py
    python src/sweep.py --output-dir data/results/sweep --seeds 42 123
"""

import argparse
import os
import torch
import pandas as pd
from PIL import Image

from pipeline import load_identities, build_pipeline, generate

LORA_SCALES = [0.4, 0.6, 0.8, 1.0]
CTRL_SCALES  = [0.3, 0.5, 0.7, 0.9]
SEEDS        = [42, 123, 777]


def run_static_sweep(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    output_dir: str = "data/results/sweep",
    lora_scales: list = LORA_SCALES,
    ctrl_scales: list = CTRL_SCALES,
    seeds: list = SEEDS,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    records = []
    total = len(lora_scales) * len(ctrl_scales) * len(seeds)
    done  = 0

    for lora_s in lora_scales:
        for ctrl_s in ctrl_scales:
            for seed in seeds:
                done += 1
                print(f"[sweep] ({done}/{total})  lora={lora_s}  ctrl={ctrl_s}  seed={seed}")

                lora_scales_dict = {k: lora_s for k in identities}
                img = generate(
                    pipe,
                    identities,
                    pose_image,
                    lora_scales=lora_scales_dict,
                    ctrl_scale=ctrl_s,
                    seed=seed,
                )

                fname = f"lora{lora_s}_ctrl{ctrl_s}_seed{seed}.png"
                img.save(os.path.join(output_dir, fname))

                records.append({
                    "lora_scale": lora_s,
                    "ctrl_scale": ctrl_s,
                    "seed":       seed,
                    "filename":   fname,
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sweep_runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"[sweep] Done. {len(records)} images saved to {output_dir}")
    print(f"[sweep] Run log: {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/results/sweep")
    parser.add_argument("--pose-image", default="data/pose_images/two_person_pose.png")
    parser.add_argument("--lora-scales", nargs="+", type=float, default=LORA_SCALES)
    parser.add_argument("--ctrl-scales", nargs="+", type=float, default=CTRL_SCALES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    identities = load_identities()
    pipe       = build_pipeline(identities)
    pose_image = Image.open(args.pose_image).convert("RGB")

    run_static_sweep(
        pipe,
        identities,
        pose_image,
        output_dir=args.output_dir,
        lora_scales=args.lora_scales,
        ctrl_scales=args.ctrl_scales,
        seeds=args.seeds,
    )
