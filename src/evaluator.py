import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from scipy.optimize import linear_sum_assignment

# MediaPipe is optional. Newer versions of mediapipe removed the legacy
# `mp.solutions` API in favour of `mp.tasks`. We try the legacy import
# first and silently fall back to None so ArcFace metrics still work.
try:
    import mediapipe as mp
    _MP_POSE = mp.solutions.pose if hasattr(mp, "solutions") else None
except Exception:
    mp = None
    _MP_POSE = None


class Evaluator:
    """
    Evaluates face identity similarity and pose keypoint error.

    pose_keypoint_error requires MediaPipe with the legacy `solutions` API
    (mediapipe <= 0.10.x). If MediaPipe is missing or too new, that method
    becomes a no-op returning 0.0; ArcFace methods continue to work.
    """
    def __init__(self, device="cuda"):
        self.device = device

        print("Initializing InsightFace for Evaluator...")
        self.face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        if _MP_POSE is not None:
            print("Initializing MediaPipe for Evaluator...")
            self.mp_pose = _MP_POSE
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            )
        else:
            print("[Evaluator] MediaPipe legacy API not available — pose_keypoint_error disabled.")
            print("[Evaluator] To enable: pip install 'mediapipe==0.10.14' and restart kernel.")
            self.mp_pose = None
            self.pose_detector = None

    def extract_face_embedding(self, img_pil: Image.Image) -> np.ndarray:
        """Extract normalized ArcFace embedding for the largest face in an image."""
        img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(img_bgr)
        if len(faces) == 0:
            return None
        # Return the embedding of the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def detect_and_assign_faces(self, output_img_pil: Image.Image, face1_embed: np.ndarray, face2_embed: np.ndarray):
        """
        Detects faces in the generated image and assigns them to the reference embeddings.
        Returns (sim1, sim2) cosine similarities.
        If less than 2 faces are detected, returns (0.0, 0.0) or actual similarity for the detected one.
        """
        img_bgr = cv2.cvtColor(np.array(output_img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(img_bgr)
        
        if len(faces) == 0:
            return 0.0, 0.0
            
        out_embeds = [f.normed_embedding for f in faces]
        ref_embeds = [face1_embed, face2_embed]
        
        # Compute cost matrix (negative cosine similarity)
        cost_matrix = np.zeros((len(faces), 2))
        for i, out_emb in enumerate(out_embeds):
            for j, ref_emb in enumerate(ref_embeds):
                cost_matrix[i, j] = -np.dot(out_emb, ref_emb)
                
        # Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        sim1 = 0.0
        sim2 = 0.0
        for r, c in zip(row_ind, col_ind):
            sim = -cost_matrix[r, c]
            if c == 0:
                sim1 = float(sim)
            elif c == 1:
                sim2 = float(sim)
                
        return sim1, sim2

    def arcface_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """Computes cosine similarity between two face embeddings."""
        if embed1 is None or embed2 is None:
            return 0.0
        return float(np.dot(embed1, embed2))
        
    def pose_keypoint_error(self, output_img_pil: Image.Image, pose_reference_pil: Image.Image) -> float:
        """
        Mean L2 distance between normalized pose keypoints of output and reference.
        Returns 0.0 if MediaPipe is unavailable (so calling code stays generic).
        """
        if self.pose_detector is None:
            return 0.0

        out_arr = np.array(output_img_pil.convert("RGB"))
        ref_arr = np.array(pose_reference_pil.convert("RGB"))

        out_results = self.pose_detector.process(out_arr)
        ref_results = self.pose_detector.process(ref_arr)

        if not out_results.pose_landmarks or not ref_results.pose_landmarks:
            return 1.0  # Max error if pose not found in either image

        out_kps = np.array([[lm.x, lm.y] for lm in out_results.pose_landmarks.landmark])
        ref_kps = np.array([[lm.x, lm.y] for lm in ref_results.pose_landmarks.landmark])

        error = np.linalg.norm(out_kps - ref_kps, axis=1).mean()
        return float(error)
