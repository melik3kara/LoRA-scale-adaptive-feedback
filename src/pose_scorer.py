# src/pose_scorer.py
"""
Pose estimation and scoring for ControlNet feedback.

Extracts OpenPose keypoints from generated images and compares them
against the target pose skeleton to produce a pose similarity score.

Usage:
    from pose_scorer import PoseScorer

    pose_scorer = PoseScorer()

    # Load or define target keypoints
    target_kps = pose_scorer.extract_keypoints(pose_image)

    # Score a generated image
    scores = pose_scorer.score_image(generated_image, target_kps, identity_regions)
    # scores = {
    #     "hermione": {"pose": 0.87, "detected": True, "keypoints_found": 14},
    #     "daenerys": {"pose": 0.92, "detected": True, "keypoints_found": 16},
    # }
"""

import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector


# COCO 18-keypoint format (same as generate_pose.py)
KEYPOINT_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

NUM_KEYPOINTS = 18


class PoseScorer:

    def __init__(self, model_id: str = "lllyasviel/Annotators"):
        """
        Args:
            model_id:  HuggingFace model ID for the OpenPose detector.
        """
        print("[pose_scorer] Loading OpenPose detector...")
        self.detector = OpenposeDetector.from_pretrained(model_id)
        print("[pose_scorer] Ready.")

    def extract_keypoints(self, image: Image.Image) -> list[list]:
        """
        Extract per-person keypoints from an image.

        Args:
            image:  PIL image (can be a real photo, generated image, or skeleton).

        Returns:
            List of persons, where each person is a list of 18 keypoints.
            Each keypoint is (x_normalized, y_normalized, confidence) or None if not detected.
            Coordinates are normalized to [0, 1] relative to image dimensions.
        """
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]

        # Access the internal body estimation model
        candidate, subset = self.detector.body_estimation(img_array)

        persons = []
        for person_idx in range(len(subset)):
            keypoints = []
            for kp_idx in range(NUM_KEYPOINTS):
                idx = int(subset[person_idx][kp_idx])
                if idx == -1:
                    keypoints.append(None)
                else:
                    x, y = candidate[idx][0], candidate[idx][1]
                    score = candidate[idx][2] if len(candidate[idx]) > 2 else 1.0
                    # Normalize to [0, 1]
                    keypoints.append((x / w, y / h, float(score)))
            persons.append(keypoints)

        return persons

    def assign_persons_to_identities(
        self,
        persons: list[list],
        identity_regions: dict,
        image_width: int,
        image_height: int,
    ) -> dict:
        """
        Assign detected persons to identity regions based on average keypoint position.

        Args:
            persons:           Output of extract_keypoints().
            identity_regions:  {identity_id: (x1, y1, x2, y2)} in pixel coordinates.
            image_width:       Width of the image.
            image_height:      Height of the image.

        Returns:
            {identity_id: person_keypoints or None}
        """
        # Compute center of mass for each detected person (from valid keypoints)
        person_centers = []
        for person_kps in persons:
            valid = [(kp[0], kp[1]) for kp in person_kps if kp is not None]
            if valid:
                cx = np.mean([v[0] for v in valid])
                cy = np.mean([v[1] for v in valid])
                person_centers.append((cx, cy))
            else:
                person_centers.append(None)

        assignments = {}
        used = set()

        for identity_id, (rx1, ry1, rx2, ry2) in identity_regions.items():
            # Normalize region to [0, 1]
            nrx1, nrx2 = rx1 / image_width, rx2 / image_width
            nry1, nry2 = ry1 / image_height, ry2 / image_height
            region_cx = (nrx1 + nrx2) / 2
            region_cy = (nry1 + nry2) / 2

            best_idx = None
            best_dist = float("inf")

            for i, center in enumerate(person_centers):
                if i in used or center is None:
                    continue

                cx, cy = center
                inside = nrx1 <= cx <= nrx2 and nry1 <= cy <= nry2
                dist = ((cx - region_cx) ** 2 + (cy - region_cy) ** 2) ** 0.5

                if inside:
                    if best_idx is None or dist < best_dist:
                        best_idx = i
                        best_dist = dist
                elif best_idx is None:
                    if dist < best_dist:
                        best_idx = i
                        best_dist = dist

            if best_idx is not None:
                assignments[identity_id] = persons[best_idx]
                used.add(best_idx)
            else:
                assignments[identity_id] = None

        return assignments

    def keypoint_similarity(
        self,
        detected: list,
        target: list,
        per_keypoint: bool = False,
    ) -> float | dict:
        """
        Compute pose similarity between two sets of keypoints using OKS-inspired metric.

        For each keypoint present in both sets, computes:
            sim_i = exp(-d_i^2 / (2 * sigma^2))
        where d_i is the Euclidean distance in normalized coordinates.

        The overall score is the average across all mutually detected keypoints.

        Args:
            detected:      List of 18 keypoints from the generated image.
            target:         List of 18 keypoints from the target pose.
            per_keypoint:  If True, also return per-keypoint scores.

        Returns:
            Float similarity score in [0, 1], or dict with per-keypoint breakdown.
        """
        sigma = 0.1  # Controls sensitivity — lower = stricter matching

        scores = {}
        for i in range(NUM_KEYPOINTS):
            det_kp = detected[i]
            tgt_kp = target[i]

            if det_kp is None or tgt_kp is None:
                continue

            dx = det_kp[0] - tgt_kp[0]
            dy = det_kp[1] - tgt_kp[1]
            dist_sq = dx ** 2 + dy ** 2

            sim = np.exp(-dist_sq / (2 * sigma ** 2))
            scores[KEYPOINT_NAMES[i]] = float(sim)

        if not scores:
            overall = 0.0
        else:
            overall = float(np.mean(list(scores.values())))

        if per_keypoint:
            return {"overall": overall, "per_keypoint": scores}
        return overall

    def score_image(
        self,
        image: Image.Image,
        target_keypoints: list[list],
        identity_regions: dict,
    ) -> dict:
        """
        Score pose accuracy of a generated image against target keypoints.

        Args:
            image:              The generated PIL image.
            target_keypoints:   Per-person target keypoints (output of extract_keypoints on target).
            identity_regions:   {identity_id: (x1, y1, x2, y2)}.

        Returns:
            {
                identity_id: {
                    "pose": float (0 to 1),
                    "detected": bool,
                    "keypoints_found": int,
                },
                ...
            }
        """
        w, h = image.size

        # Extract keypoints from generated image
        gen_persons = self.extract_keypoints(image)

        # Assign generated persons to identity regions
        gen_assignments = self.assign_persons_to_identities(
            gen_persons, identity_regions, w, h
        )

        # Assign target persons to identity regions
        tgt_assignments = self.assign_persons_to_identities(
            target_keypoints, identity_regions, w, h
        )

        scores = {}
        for identity_id in identity_regions:
            gen_kps = gen_assignments.get(identity_id)
            tgt_kps = tgt_assignments.get(identity_id)

            if gen_kps is None:
                scores[identity_id] = {
                    "pose": 0.0,
                    "detected": False,
                    "keypoints_found": 0,
                }
                print(f"[pose_scorer] WARNING: No person detected for '{identity_id}'")
                continue

            if tgt_kps is None:
                scores[identity_id] = {
                    "pose": 0.0,
                    "detected": True,
                    "keypoints_found": sum(1 for kp in gen_kps if kp is not None),
                }
                print(f"[pose_scorer] WARNING: No target pose for '{identity_id}'")
                continue

            sim = self.keypoint_similarity(gen_kps, tgt_kps)
            n_found = sum(1 for kp in gen_kps if kp is not None)

            scores[identity_id] = {
                "pose": sim,
                "detected": True,
                "keypoints_found": n_found,
            }

        return scores

    def extract_and_cache_target(self, pose_image: Image.Image) -> list[list]:
        """
        Convenience method: extract keypoints from the target skeleton image
        and return them for reuse across multiple score_image calls.

        Call this once on your target pose, then pass the result to score_image.
        """
        keypoints = self.extract_keypoints(pose_image)
        print(f"[pose_scorer] Extracted {len(keypoints)} person(s) from target skeleton")
        for i, person in enumerate(keypoints):
            n_valid = sum(1 for kp in person if kp is not None)
            print(f"[pose_scorer]   Person {i}: {n_valid}/{NUM_KEYPOINTS} keypoints detected")
        return keypoints
