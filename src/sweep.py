# src/sweep.py
"""
Phase 1 — Static Sweep with Integrated Scoring

Generates images across a grid of LoRA scale × ControlNet scale combinations,
scores each image immediately (ArcFace + pose), then discards the image tensor
to keep memory usage low.

Also records timing and peak memory for each run.

Usage:
    python src/sweep.py
    python src/sweep.py --output-dir data/results/sweep --seeds 42 123
    python src/sweep.py --save-images          # also save the generated images
    python src/sweep.py --default-only         # only run default weights baseline
"""

import argparse
import os
import time
import torch
import pandas as pd
from PIL import Image

from pipeline import load_identities, build_pipeline, generate
from scorer import FaceScorer
from pose_scorer import PoseScorer


LORA_SCALES = [0.4, 0.6, 0.8, 1.0]
CTRL_SCALES = [0.3, 0.5, 0.7, 0.9]
SEEDS       = [42, 123, 777]

# Default weights baseline — standard recommended values
DEFAULT_LORA_SCALE = 1.0
DEFAULT_CTRL_SCALE = 0.7


def get_identity_regions(identities: dict, width: int = 1024, height: int = 1024) -> dict:
    """Default left/right split for two identities."""
    names = list(identities.keys())
    mid = width // 2
    return {
        names[0]: (0,   0, mid,   height),
        names[1]: (mid, 0, width, height),
    }


def track_memory() -> dict:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def run_single(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    lora_scale: float,
    ctrl_scale: float,
    seed: int,
    save_path: str | None = None,
    use_regional_attention: bool = False,
) -> dict:
    """
    Generate one image, score it, optionally save it, and return the record.
    The image is discarded after scoring to free memory.
    """
    lora_scales_dict = {k: lora_scale for k in identities}

    # Reset peak memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Generate
    t_start = time.time()
    img = generate(
        pipe,
        identities,
        pose_image,
        lora_scales=lora_scales_dict,
        ctrl_scale=ctrl_scale,
        seed=seed,
        use_regional_attention=use_regional_attention,
        identity_regions=identity_regions,
    )
    t_generate = time.time() - t_start

    # Score — face (ArcFace)
    t_start = time.time()
    face_scores = face_scorer.score_image(img, identity_regions)
    t_face = time.time() - t_start

    # Score — pose
    t_start = time.time()
    pose_scores = pose_scorer.score_image(img, target_keypoints, identity_regions)
    t_pose = time.time() - t_start

    # Memory snapshot
    mem = track_memory()

    # Optionally save the image
    if save_path is not None:
        img.save(save_path)

    # Build record
    record = {
        "lora_scale": lora_scale,
        "ctrl_scale": ctrl_scale,
        "seed": seed,
        "time_generate_s": round(t_generate, 2),
        "time_face_score_s": round(t_face, 2),
        "time_pose_score_s": round(t_pose, 2),
        "gpu_peak_mb": mem["peak_mb"],
    }

    # Per-identity scores
    for identity_id in identities:
        fs = face_scores.get(identity_id, {})
        ps = pose_scores.get(identity_id, {})
        record[f"{identity_id}_arcface"] = round(fs.get("arcface", 0.0), 4)
        record[f"{identity_id}_face_detected"] = fs.get("detected", False)
        record[f"{identity_id}_pose"] = round(ps.get("pose", 0.0), 4)
        record[f"{identity_id}_pose_detected"] = ps.get("detected", False)
        record[f"{identity_id}_keypoints_found"] = ps.get("keypoints_found", 0)

    # Aggregate scores (average across identities)
    arcface_vals = [face_scores[k]["arcface"] for k in identities if face_scores[k]["detected"]]
    pose_vals = [pose_scores[k]["pose"] for k in identities if pose_scores[k]["detected"]]

    record["avg_arcface"] = round(sum(arcface_vals) / len(arcface_vals), 4) if arcface_vals else 0.0
    record["avg_pose"] = round(sum(pose_vals) / len(pose_vals), 4) if pose_vals else 0.0

    # Discard the image — this is the whole point of score-and-discard
    del img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


def run_default_baseline(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    output_dir: str,
    seeds: list = SEEDS,
    save_images: bool = False,
    use_regional_attention: bool = False,
) -> pd.DataFrame:
    """
    Baseline 1: Default weights — no tuning at all.
    Runs with standard LoRA and ControlNet scales across seeds.
    """
    print("\n" + "=" * 60)
    print("BASELINE 1: Default Weights")
    print(f"  LoRA scale = {DEFAULT_LORA_SCALE}, ControlNet scale = {DEFAULT_CTRL_SCALE}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for i, seed in enumerate(seeds):
        print(f"\n[default] ({i+1}/{len(seeds)})  seed={seed}")

        save_path = None
        if save_images:
            save_path = os.path.join(output_dir, f"default_seed{seed}.png")

        record = run_single(
            pipe, identities, pose_image,
            face_scorer, pose_scorer, target_keypoints, identity_regions,
            lora_scale=DEFAULT_LORA_SCALE,
            ctrl_scale=DEFAULT_CTRL_SCALE,
            seed=seed,
            save_path=save_path,
            use_regional_attention=use_regional_attention,
        )
        record["experiment"] = "default"
        records.append(record)

        print(f"  avg_arcface={record['avg_arcface']}  avg_pose={record['avg_pose']}  "
              f"gen_time={record['time_generate_s']}s  peak_mem={record['gpu_peak_mb']}MB")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "default_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[default] Results saved to {csv_path}")
    return df


def run_static_sweep(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    output_dir: str = "data/results/sweep",
    lora_scales: list = LORA_SCALES,
    ctrl_scales: list = CTRL_SCALES,
    seeds: list = SEEDS,
    save_images: bool = False,
    use_regional_attention: bool = False,
) -> pd.DataFrame:
    """
    Baseline 2: Static sweep — exhaustive grid search.
    """
    print("\n" + "=" * 60)
    print("BASELINE 2: Static Sweep")
    print(f"  LoRA scales: {lora_scales}")
    print(f"  ControlNet scales: {ctrl_scales}")
    print(f"  Seeds: {seeds}")
    print(f"  Total runs: {len(lora_scales) * len(ctrl_scales) * len(seeds)}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    records = []
    total = len(lora_scales) * len(ctrl_scales) * len(seeds)
    done = 0
    sweep_start = time.time()

    for lora_s in lora_scales:
        for ctrl_s in ctrl_scales:
            for seed in seeds:
                done += 1
                print(f"\n[sweep] ({done}/{total})  lora={lora_s}  ctrl={ctrl_s}  seed={seed}")

                save_path = None
                if save_images:
                    save_path = os.path.join(output_dir, f"lora{lora_s}_ctrl{ctrl_s}_seed{seed}.png")

                record = run_single(
                    pipe, identities, pose_image,
                    face_scorer, pose_scorer, target_keypoints, identity_regions,
                    lora_scale=lora_s,
                    ctrl_scale=ctrl_s,
                    seed=seed,
                    save_path=save_path,
                    use_regional_attention=use_regional_attention,
                )
                record["experiment"] = "sweep"
                records.append(record)

                print(f"  avg_arcface={record['avg_arcface']}  avg_pose={record['avg_pose']}  "
                      f"gen_time={record['time_generate_s']}s  peak_mem={record['gpu_peak_mb']}MB")

    sweep_time = time.time() - sweep_start

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE")
    print(f"  Total time: {sweep_time:.1f}s ({sweep_time/60:.1f} min)")
    print(f"  Images generated: {total}")
    print(f"  Results: {csv_path}")

    best = df.loc[df["avg_arcface"].idxmax()]
    print(f"\n  Best by ArcFace:")
    print(f"    lora={best['lora_scale']}  ctrl={best['ctrl_scale']}  seed={int(best['seed'])}")
    print(f"    avg_arcface={best['avg_arcface']}  avg_pose={best['avg_pose']}")

    best_pose = df.loc[df["avg_pose"].idxmax()]
    print(f"\n  Best by Pose:")
    print(f"    lora={best_pose['lora_scale']}  ctrl={best_pose['ctrl_scale']}  seed={int(best_pose['seed'])}")
    print(f"    avg_arcface={best_pose['avg_arcface']}  avg_pose={best_pose['avg_pose']}")

    # Combined score (simple average for now)
    df["combined"] = (df["avg_arcface"] + df["avg_pose"]) / 2
    best_combined = df.loc[df["combined"].idxmax()]
    print(f"\n  Best Combined (arcface + pose / 2):")
    print(f"    lora={best_combined['lora_scale']}  ctrl={best_combined['ctrl_scale']}  seed={int(best_combined['seed'])}")
    print(f"    avg_arcface={best_combined['avg_arcface']}  avg_pose={best_combined['avg_pose']}")
    print(f"{'=' * 60}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Static sweep with integrated scoring")
    parser.add_argument("--output-dir", default="data/results/sweep")
    parser.add_argument("--pose-image", default="data/pose_images/two_person_pose.png")
    parser.add_argument("--reference-dir", default="data/reference_faces")
    parser.add_argument("--lora-scales", nargs="+", type=float, default=LORA_SCALES)
    parser.add_argument("--ctrl-scales", nargs="+", type=float, default=CTRL_SCALES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--save-images", action="store_true", help="Save generated images to disk")
    parser.add_argument("--default-only", action="store_true", help="Only run default weights baseline")
    parser.add_argument("--regional-attention", action="store_true", help="Enable regional attention masking")
    args = parser.parse_args()

    # Load everything
    print("[sweep] Loading identities...")
    identities = load_identities()

    print("[sweep] Building pipeline...")
    pipe = build_pipeline(identities)

    print("[sweep] Loading pose image...")
    pose_image = Image.open(args.pose_image).convert("RGB")

    print("[sweep] Initializing face scorer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_scorer = FaceScorer(reference_dir=args.reference_dir, device=device)

    print("[sweep] Initializing pose scorer...")
    pose_scorer = PoseScorer()

    print("[sweep] Extracting target keypoints...")
    target_keypoints = pose_scorer.extract_and_cache_target(pose_image)

    identity_regions = get_identity_regions(identities)
    print(f"[sweep] Identity regions: {identity_regions}")

    # Run default baseline
    df_default = run_default_baseline(
        pipe, identities, pose_image,
        face_scorer, pose_scorer, target_keypoints, identity_regions,
        output_dir=args.output_dir,
        seeds=args.seeds,
        save_images=args.save_images,
        use_regional_attention=args.regional_attention,
    )

    if not args.default_only:
        # Run static sweep
        df_sweep = run_static_sweep(
            pipe, identities, pose_image,
            face_scorer, pose_scorer, target_keypoints, identity_regions,
            output_dir=args.output_dir,
            lora_scales=args.lora_scales,
            ctrl_scales=args.ctrl_scales,
            seeds=args.seeds,
            save_images=args.save_images,
            use_regional_attention=args.regional_attention,
        )

        # Combine both into one summary
        df_all = pd.concat([df_default, df_sweep], ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_baselines.csv")
        df_all.to_csv(combined_path, index=False)
        print(f"\n[sweep] Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
