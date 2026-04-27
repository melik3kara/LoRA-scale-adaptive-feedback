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
    image: Image.Image           # best image seen across iterations
    identity_states: dict
    total_iterations: int
    status: str                  # "all_converged" | "partial" | "none_converged"
    final_alphas: dict
    final_arcface: dict
    final_pose: dict
    total_time_s: float
    best_iteration: int = -1     # iteration index (0-based) of returned image
    last_image: Optional[Image.Image] = None


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
    delta_suppress: float = 0.07,
    dominance_threshold: float = 0.0,
    attribute_scorer=None,
    attribute_threshold: float = 0.0,
    delta_attr: float = 0.05,
    block_profile: dict | None = None,
    max_iters: int = 5,
    seed: int = 42,
    use_regional_attention: bool = False,
    use_spatial_lora_gate: bool = False,
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

    # Track best iteration — score = mean over identities of min(arcface, pose).
    # Using min(arcface, pose) per identity penalises configs where one metric
    # is great but the other collapsed (a common failure mode of this loop).
    best_image = None
    best_score = -float("inf")
    best_iter = -1
    best_face_scores = {}
    best_pose_scores = {}

    def _scaled_for(identity_id: str, alpha: float):
        """Apply block_profile (if any) as a per-block multiplier on alpha."""
        if block_profile is None or identity_id not in block_profile:
            return alpha
        return {
            block: alpha * ratio
            for block, ratio in block_profile[identity_id].items()
        }

    for iteration in range(max_iters):
        # Build per-identity LoRA scales from current state — apply block
        # profile so a single adaptive scalar drives layer-wise routing.
        lora_scales = {k: _scaled_for(k, states[k].alpha) for k in identities}

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
            use_spatial_lora_gate=use_spatial_lora_gate,
            identity_regions=identity_regions,
        )
        last_image = img

        # Score
        face_scores = face_scorer.score_image(img, identity_regions)
        pose_scores = pose_scorer.score_image(img, target_keypoints, identity_regions)
        attr_scores = (
            attribute_scorer.score_image(img, identity_regions)
            if attribute_scorer is not None else {}
        )
        last_face_scores = face_scores
        last_pose_scores = pose_scores

        # Rank this iteration against previous best
        per_id_scores = []
        for k in identities:
            arc = face_scores.get(k, {}).get("arcface", 0.0)
            pos = pose_scores.get(k, {}).get("pose",    0.0)
            # If pose was not measurable (0.0), fall back to arcface alone
            per_id_scores.append(arc if pos == 0.0 else min(arc, pos))
        iter_score = sum(per_id_scores) / max(len(per_id_scores), 1)
        if iter_score > best_score:
            best_score = iter_score
            best_image = img
            best_iter = iteration
            best_face_scores = face_scores
            best_pose_scores = pose_scores

        # Update each identity's state
        for identity_id in identities:
            state = states[identity_id]
            face_info  = face_scores.get(identity_id, {}) or {}
            arcface_sim = face_info.get("arcface",   0.0)
            dominance   = face_info.get("dominance", 0.0)
            wrong_winner = face_info.get("wrong_winner")
            pose_sim    = pose_scores.get(identity_id, {}).get("pose", 0.0)

            attr_margin = attr_scores.get(identity_id, {}).get("margin", None)

            state.history.append({
                "iteration":   iteration,
                "alpha":       state.alpha,
                "arcface":     arcface_sim,
                "dominance":   dominance,
                "wrong_winner": wrong_winner,
                "pose":        pose_sim,
                "attr_margin": attr_margin,
            })

            # Convergence check: identity good, pose good, no leakage.
            # pose_sim == 0.0 usually means the pose scorer failed to detect
            # or assign a person — treat it as "unknown" rather than "failed".
            id_ok        = arcface_sim >= id_threshold
            pose_ok      = pose_sim >= pose_threshold or pose_sim == 0.0
            leakage_ok   = dominance <= dominance_threshold

            if id_ok and pose_ok and leakage_ok:
                state.converged = True
            else:
                state.converged = False

            if verbose:
                status = "✓" if state.converged else " "
                leak = f"  dom={dominance:+.3f}" + (f"←{wrong_winner}" if wrong_winner and dominance > 0 else "")
                attr_str = f"  attr={attr_margin:+.3f}" if attr_margin is not None else ""
                print(f"[adaptive]   {status} {identity_id}: "
                      f"α={state.alpha:.2f}  arcface={arcface_sim:.3f}  pose={pose_sim:.3f}{leak}{attr_str}")

        # Early stop if all converged
        if all(s.converged for s in states.values()):
            if verbose:
                print(f"[adaptive] All identities converged at iteration {iteration+1}")
            break

        # Update alphas — dominance-aware:
        # 1) If a wrong identity is winning region i (dominance > 0), suppress
        #    that wrong identity's α AND boost the correct identity's α.
        # 2) Otherwise apply the original pose-down / arcface-up logic per id.
        # Suppression deltas are applied per pair so they accumulate across
        # all leaking regions.
        suppress_delta = {k: 0.0 for k in identities}
        boost_delta    = {k: 0.0 for k in identities}

        for identity_id in identities:
            face_info = face_scores.get(identity_id, {}) or {}
            dominance = face_info.get("dominance", 0.0)
            wrong = face_info.get("wrong_winner")
            if wrong and dominance > dominance_threshold and wrong in states:
                # Wrong identity is leaking into this region — suppress wrong, boost correct
                suppress_delta[wrong]       += delta_suppress
                boost_delta[identity_id]    += delta_up

        for identity_id, state in states.items():
            if state.converged:
                continue

            face_info  = face_scores.get(identity_id, {}) or {}
            arcface_sim = face_info.get("arcface", 0.0)
            pose_sim    = pose_scores.get(identity_id, {}).get("pose", 0.0)

            a_min = _resolve(alpha_min, identity_id)
            a_max = _resolve(alpha_max, identity_id)

            applied_pair_update = (
                suppress_delta[identity_id] > 0 or boost_delta[identity_id] > 0
            )

            if applied_pair_update:
                # Apply dominance-driven adjustments first
                state.alpha = max(
                    min(state.alpha + boost_delta[identity_id] - suppress_delta[identity_id], a_max),
                    a_min,
                )
                continue

            # Fall back to original logic when no leakage signal involves this id
            pose_reliable = pose_sim > 0.0
            attr_margin = attr_scores.get(identity_id, {}).get("margin", None)

            if pose_reliable and pose_sim < pose_threshold:
                state.alpha = max(state.alpha - delta_down, a_min)
            elif arcface_sim < id_threshold:
                state.alpha = min(state.alpha + delta_up, a_max)
            elif attr_margin is not None and attr_margin < attribute_threshold:
                # Identity face is OK but visual attributes (hair/attire) are wrong —
                # nudge alpha up to push the LoRA's stylistic features further.
                state.alpha = min(state.alpha + delta_attr, a_max)

    n_converged = sum(1 for s in states.values() if s.converged)
    if n_converged == len(states):
        status = "all_converged"
    elif n_converged > 0:
        status = "partial"
    else:
        status = "none_converged"

    total_time = time.time() - t_start

    # Return the BEST image seen, not the last — the loop can oscillate past
    # a good iteration. final_arcface/final_pose reflect the returned image.
    chosen_image = best_image if best_image is not None else last_image
    chosen_face  = best_face_scores if best_image is not None else last_face_scores
    chosen_pose  = best_pose_scores if best_image is not None else last_pose_scores

    if verbose and best_image is not None:
        print(f"[adaptive] Returning best iteration: {best_iter+1} "
              f"(composite score={best_score:.3f})")

    return AdaptiveResult(
        image=chosen_image,
        identity_states=states,
        total_iterations=iteration + 1,
        status=status,
        final_alphas={k: states[k].alpha for k in states},
        final_arcface={k: chosen_face.get(k, {}).get("arcface", 0.0) for k in states},
        final_pose={k: chosen_pose.get(k, {}).get("pose", 0.0) for k in states},
        total_time_s=round(total_time, 2),
        best_iteration=best_iter,
        last_image=last_image,
    )


def run_adaptive_best_of_n(
    pipe,
    identities: dict,
    pose_image: Image.Image,
    face_scorer: FaceScorer,
    pose_scorer: PoseScorer,
    target_keypoints: list,
    identity_regions: dict,
    seeds: list,
    use_regional_attention: bool = True,
    use_spatial_lora_gate: bool = False,
    verbose: bool = True,
    **adaptive_kwargs,
) -> AdaptiveResult:
    """
    Run the adaptive loop across multiple seeds and return the single best
    AdaptiveResult (by composite score: mean of per-identity min(arcface, pose)).

    Identity separation is very seed-sensitive with multi-LoRA pipelines — some
    seeds avoid the dominance collapse and others fall straight into it. Running
    a handful and keeping the best one is the cheapest way to get a usable
    image without tuning hyperparameters further.
    """
    def composite(result: AdaptiveResult) -> float:
        scores = []
        for k in identities:
            arc = result.final_arcface.get(k, 0.0)
            pos = result.final_pose.get(k, 0.0)
            scores.append(arc if pos == 0.0 else min(arc, pos))
        return sum(scores) / max(len(scores), 1)

    best: Optional[AdaptiveResult] = None
    best_seed: Optional[int] = None
    best_score = -float("inf")

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}\n[best-of-n] seed {seed} ({i+1}/{len(seeds)})\n{'='*60}")
        result = multi_lora_adaptive_generate(
            pipe, identities, pose_image,
            face_scorer, pose_scorer, target_keypoints, identity_regions,
            seed=seed,
            use_regional_attention=use_regional_attention,
            use_spatial_lora_gate=use_spatial_lora_gate,
            verbose=verbose,
            **adaptive_kwargs,
        )
        score = composite(result)
        if verbose:
            print(f"[best-of-n] seed {seed} composite score: {score:.3f}")
        if score > best_score:
            best_score = score
            best = result
            best_seed = seed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if verbose and best is not None:
        print(f"\n[best-of-n] Winner: seed={best_seed}  composite={best_score:.3f}")
        print(f"[best-of-n] Best alphas:  {best.final_alphas}")
        print(f"[best-of-n] Best arcface: {best.final_arcface}")
        print(f"[best-of-n] Best pose:    {best.final_pose}")

    return best


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
