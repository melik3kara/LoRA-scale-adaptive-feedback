# src/scorer.py
"""
Face detection + ArcFace identity scoring + pose scoring.

Usage:
    from scorer import FaceScorer

    scorer = FaceScorer(reference_dir="data/reference_faces")

    # Score a generated multi-identity image
    scores = scorer.score_image(
        image,
        identity_regions={"hermione": (0, 0, 512, 1024), "daenerys": (512, 0, 1024, 1024)},
    )
    # scores = {
    #     "hermione": {"arcface": 0.82, "detected": True},
    #     "daenerys": {"arcface": 0.71, "detected": True},
    # }
"""

import os
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis


class FaceScorer:

    def __init__(
        self,
        reference_dir: str = "data/reference_faces",
        device: str = "cuda",
        det_size: int = 640,
    ):
        self.reference_dir = reference_dir
        self.device = device

        # Initialize InsightFace (bundles RetinaFace detector + ArcFace recognizer)
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(
            ctx_id=0 if device == "cuda" else -1,
            det_size=(det_size, det_size),
        )

        # Pre-compute reference embeddings
        self.reference_embeddings = {}
        self._load_references()

    def _load_references(self):
        """Load reference face images and extract their ArcFace embeddings."""
        if not os.path.exists(self.reference_dir):
            print(f"[scorer] WARNING: Reference directory {self.reference_dir} not found.")
            print(f"[scorer] Run extract_reference.py first.")
            return

        for fname in os.listdir(self.reference_dir):
            # Skip full portraits, only use cropped faces
            if fname.endswith("_full.png") or not fname.endswith(".png"):
                continue

            identity_id = fname.replace(".png", "")
            fpath = os.path.join(self.reference_dir, fname)

            img = np.array(Image.open(fpath).convert("RGB"))
            img_bgr = img[:, :, ::-1]

            faces = self.app.get(img_bgr)

            if len(faces) == 0:
                print(f"[scorer] WARNING: No face found in reference {fname}")
                print(f"[scorer] Re-run extract_reference.py or provide a better reference.")
                continue

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = face.normed_embedding

            self.reference_embeddings[identity_id] = embedding
            print(f"[scorer] Loaded reference embedding for '{identity_id}'")

        if not self.reference_embeddings:
            print("[scorer] WARNING: No reference embeddings loaded!")

    def detect_faces(self, image: Image.Image) -> list:
        """
        Detect all faces in an image.

        Returns list of dicts:
            [{"bbox": (x1, y1, x2, y2), "embedding": np.array, "center": (cx, cy)}, ...]
        """
        img_array = np.array(image.convert("RGB"))
        img_bgr = img_array[:, :, ::-1]

        faces = self.app.get(img_bgr)

        results = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            results.append({
                "bbox": (x1, y1, x2, y2),
                "embedding": face.normed_embedding,
                "center": (cx, cy),
            })

        return results

    def assign_faces_to_identities(
        self,
        faces: list,
        identity_regions: dict,
    ) -> dict:
        """
        Assign detected faces to identities based on spatial overlap with regions.

        For each identity, picks the face whose center falls inside (or closest to)
        that identity's bounding box region.

        Args:
            faces:             Output of detect_faces().
            identity_regions:  {identity_id: (x1, y1, x2, y2)} region per identity.

        Returns:
            {identity_id: face_dict or None}
        """
        assignments = {}
        used_faces = set()

        for identity_id, (rx1, ry1, rx2, ry2) in identity_regions.items():
            region_cx = (rx1 + rx2) / 2
            region_cy = (ry1 + ry2) / 2

            best_face = None
            best_dist = float("inf")

            for i, face in enumerate(faces):
                if i in used_faces:
                    continue

                fcx, fcy = face["center"]

                # Check if face center is inside the region
                inside = rx1 <= fcx <= rx2 and ry1 <= fcy <= ry2

                dist = ((fcx - region_cx) ** 2 + (fcy - region_cy) ** 2) ** 0.5

                if inside:
                    # Prefer faces inside the region; among those pick closest to center
                    if best_face is None or dist < best_dist:
                        best_face = i
                        best_dist = dist
                elif best_face is None:
                    # No face inside yet — pick closest overall as fallback
                    if dist < best_dist:
                        best_face = i
                        best_dist = dist

            if best_face is not None:
                assignments[identity_id] = faces[best_face]
                used_faces.add(best_face)
            else:
                assignments[identity_id] = None

        return assignments

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))

    def score_image(
        self,
        image: Image.Image,
        identity_regions: dict,
    ) -> dict:
        """
        Score a generated multi-identity image.

        Detects faces, assigns them to identity regions, and computes ArcFace
        similarity against reference embeddings.

        Args:
            image:             The generated PIL image.
            identity_regions:  {identity_id: (x1, y1, x2, y2)} per identity.

        Returns:
            {
                identity_id: {
                    "arcface": float (cosine similarity, -1 to 1),
                    "detected": bool,
                    "bbox": (x1, y1, x2, y2) or None,
                },
                ...
            }
        """
        faces = self.detect_faces(image)
        assignments = self.assign_faces_to_identities(faces, identity_regions)

        scores = {}
        for identity_id, face in assignments.items():
            if face is None:
                scores[identity_id] = {
                    "arcface": 0.0,
                    "detected": False,
                    "bbox": None,
                }
                print(f"[scorer] WARNING: No face detected for '{identity_id}'")
                continue

            ref_emb = self.reference_embeddings.get(identity_id)
            if ref_emb is None:
                scores[identity_id] = {
                    "arcface": 0.0,
                    "detected": True,
                    "bbox": face["bbox"],
                }
                print(f"[scorer] WARNING: No reference embedding for '{identity_id}'")
                continue

            sim = self.cosine_similarity(face["embedding"], ref_emb)

            scores[identity_id] = {
                "arcface": sim,
                "detected": True,
                "bbox": face["bbox"],
            }

        return scores

    def score_batch(
        self,
        images: list[Image.Image],
        identity_regions: dict,
    ) -> list[dict]:
        """Score multiple images. Convenience wrapper around score_image."""
        return [self.score_image(img, identity_regions) for img in images]
