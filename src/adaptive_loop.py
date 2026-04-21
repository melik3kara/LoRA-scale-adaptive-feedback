# src/adaptive_loop.py
"""
Phase 3 — Per-Identity Adaptive Feedback Loop

Each identity maintains its own LoRA scale (alpha), updated independently
each iteration based on ArcFace similarity and pose error.

Algorithm:
  - Initialize α_i = alpha_init for all identities
  - For each iteration:
      1. Generate image with current [α_A, α_B]
      2. Score each face against its reference (ArcFace)
      3. Score pose per identity
      4. For each identity i:
           if arcface_i < id_threshold   → bump α_i up
           if pose_i    < pose_threshold → bump α_i down (pose protection)
      5. Mark identity as converged if thresholds met
      6. Stop if all converged, else continue until max_iters
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import torch
from PIL import Image

from pipeline import generate
from scorer import FaceScorer
from pose_scorer import PoseScorer


@dataclass
class IdentityState:
    identity_id: str
    alpha: float
    converged: bool = False
    history: list = field(default_factory=list)


@dataclass
class AdaptiveResult:
    image: Image.Image
    identity_states: dict
    total_iterations: int
    status: str                  # "all_converged" | "partial" | "none_converged"
    final_alphas: dict
    final_arcface: dict
    final_pose: dict
    total_time_s: float


def multi_lora_adaptive_generate(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    ctrl_scale: float = 0.7,
    id_threshold: float = 0.40,
    pose_threshold: float = 0.50,
    alpha_init: float | dict = 0.5,
    alpha_min: float | dict = 0.2,
    alpha_max: float | dict = 1.0,
    delta_up: float = 0.15,
    delta_down: float = 0.10,
    max_iters: int = 5,
    seed: int = 42,
    use_regional_attention: bool = False,
    verbose: bool = True,
) -> AdaptiveResult:
    """
    Run the adaptive loop for a single prompt/seed combination.

    Args:
        pipe:                    Built SDXL+ControlNet+LoRA pipeline.
        identities:              Identity config from load_identities().
        pose_image:              Target OpenPose skeleton image.
        face_scorer:             Instantiated FaceScorer.
        pose_scorer:             Instantiated PoseScorer.
        target_keypoints:        Precomputed target keypoints.
        identity_regions:        {id: (x1, y1, x2, y2)} per identity.
        ctrl_scale:              ControlNet conditioning strength (fixed).
        id_threshold:            ArcFace similarity required to mark identity converged.
        pose_threshold:          Pose similarity below which we roll α back.
        alpha_init/min/max:      LoRA scale bounds.
        delta_up:                Step for identity correction (bump up).
        delta_down:              Step for pose correction (bump down).
        max_iters:               Safety cap on iterations.
        seed:                    Random seed (fixed across iterations for comparability).
        use_regional_attention:  Apply regional attention masking during generation.
        verbose:                 Print per-iteration progress.

    Returns:
        AdaptiveResult with the last generated image and full history.
    """
    t_start = time.time()

    # Per-identity or scalar config
    def _resolve(param, identity_id):
        return param[identity_id] if isinstance(param, dict) else param

    states = {
        k: IdentityState(identity_id=k, alpha=_resolve(alpha_init, k))
        for k in identities
    }

    last_image = None
    last_face_scores = {}
    last_pose_scores = {}

    for iteration in range(max_iters):
        # Build per-identity LoRA scales from current state
        lora_scales = {k: states[k].alpha for k in identities}

        if verbose:
            alpha_str = "  ".join(f"{k}={states[k].alpha:.2f}" for k in identities)
            print(f"\n[adaptive] iter {iteration+1}/{max_iters}  {alpha_str}")

        # Generate
        img = generate(
            pipe, identities, pose_image,
            lora_scales=lora_scales,
            ctrl_scale=ctrl_scale,
            seed=seed,
            use_regional_attention=use_regional_attention,
            identity_regions=identity_regions,
        )
        last_image = img

        # Score
        face_scores = face_scorer.score_image(img, identity_regions)
        pose_scores = pose_scorer.score_image(img, target_keypoints, identity_regions)
        last_face_scores = face_scores
        last_pose_scores = pose_scores

        # Update each identity's state
        for identity_id in identities:
            state = states[identity_id]
            arcface_sim = face_scores.get(identity_id, {}).get("arcface", 0.0)
            pose_sim    = pose_scores.get(identity_id, {}).get("pose",    0.0)

            state.history.append({
                "iteration": iteration,
                "alpha":     state.alpha,
                "arcface":   arcface_sim,
                "pose":      pose_sim,
            })

            # Convergence check: both identity and pose must be good
            id_ok   = arcface_sim >= id_threshold
            pose_ok = pose_sim    >= pose_threshold

            if id_ok and pose_ok:
                state.converged = True
            else:
                state.converged = False

            if verbose:
                status = "✓" if state.converged else " "
                print(f"[adaptive]   {status} {identity_id}: "
                      f"α={state.alpha:.2f}  arcface={arcface_sim:.3f}  pose={pose_sim:.3f}")

        # Early stop if all converged
        if all(s.converged for s in states.values()):
            if verbose:
                print(f"[adaptive] All identities converged at iteration {iteration+1}")
            break

        # Update alphas for unconverged identities
        # Priority: pose correction (down) overrides identity correction (up)
        for identity_id, state in states.items():
            if state.converged:
                continue

            arcface_sim = face_scores.get(identity_id, {}).get("arcface", 0.0)
            pose_sim    = pose_scores.get(identity_id, {}).get("pose",    0.0)

            a_min = _resolve(alpha_min, identity_id)
            a_max = _resolve(alpha_max, identity_id)

            if pose_sim < pose_threshold:
                # Pose drifted — pull alpha back
                state.alpha = max(state.alpha - delta_down, a_min)
            elif arcface_sim < id_threshold:
                # Identity weak — push alpha up
                state.alpha = min(state.alpha + delta_up, a_max)

    n_converged = sum(1 for s in states.values() if s.converged)
    if n_converged == len(states):
        status = "all_converged"
    elif n_converged > 0:
        status = "partial"
    else:
        status = "none_converged"

    total_time = time.time() - t_start

    return AdaptiveResult(
        image=last_image,
        identity_states=states,
        total_iterations=iteration + 1,
        status=status,
        final_alphas={k: states[k].alpha for k in states},
        final_arcface={k: last_face_scores.get(k, {}).get("arcface", 0.0) for k in states},
        final_pose={k: last_pose_scores.get(k, {}).get("pose", 0.0) for k in states},
        total_time_s=round(total_time, 2),
    )


def run_adaptive_experiment(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    seeds: list,
    output_dir: str = "data/results/adaptive",
    save_images: bool = True,
    use_regional_attention: bool = False,
    **adaptive_kwargs,
) -> list:
    """
    Run the adaptive loop across multiple seeds and collect summary records.

    Returns a list of record dicts (one per seed) — ready for CSV logging.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"ADAPTIVE run {i+1}/{len(seeds)}  seed={seed}")
        print(f"{'='*60}")

        result = multi_lora_adaptive_generate(
            pipe, identities, pose_image,
            face_scorer, pose_scorer, target_keypoints, identity_regions,
            seed=seed,
            use_regional_attention=use_regional_attention,
            **adaptive_kwargs,
        )

        if save_images and result.image is not None:
            save_path = os.path.join(output_dir, f"adaptive_seed{seed}.png")
            result.image.save(save_path)

        record = {
            "method":     "adaptive",
            "seed":       seed,
            "iterations": result.total_iterations,
            "status":     result.status,
            "total_time_s": result.total_time_s,
        }
        for k in identities:
            record[f"{k}_final_alpha"]   = round(result.final_alphas[k], 3)
            record[f"{k}_final_arcface"] = round(result.final_arcface[k], 4)
            record[f"{k}_final_pose"]    = round(result.final_pose[k],    4)
        records.append(record)

        # Free memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records
