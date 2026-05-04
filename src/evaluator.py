import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from scipy.optimize import linear_sum_assignment
import mediapipe as mp

class Evaluator:
    """
    Evaluates face identity similarity and pose keypoint error.
    """
    def __init__(self, device="cuda"):
        self.device = device
        
        # InsightFace for face detection and arcface embeddings
        print("Initializing InsightFace for Evaluator...")
        self.face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            # By not specifying allowed_modules, it loads both detection and recognition
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # MediaPipe for pose evaluation
        print("Initializing MediaPipe for Evaluator...")
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=2, 
            min_detection_confidence=0.5
        )

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
        Calculates pose error. If reference is a real image, uses MediaPipe 33-keypoint L2 distance.
        If reference is an OpenPose skeleton image, extracts skeleton from output and uses pixel-wise MSE.
        """
        out_arr = np.array(output_img_pil.convert("RGB"))
        ref_arr = np.array(pose_reference_pil.convert("RGB"))
        
        # Check if reference is a skeleton image (mostly black)
        is_skeleton = np.mean(ref_arr < 15) > 0.60
        
        if is_skeleton:
            try:
                from controlnet_aux import OpenposeDetector
                if not hasattr(self, 'openpose'):
                    # Load OpenPose only once when needed
                    self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                
                # Extract skeleton from generated image
                out_skel = self.openpose(output_img_pil, hand_and_face=True)
                if isinstance(out_skel, np.ndarray):
                    out_skel = Image.fromarray(out_skel)
                    
                # Resize to match reference
                out_skel = out_skel.resize(pose_reference_pil.size)
                
                out_skel_arr = np.array(out_skel).astype(np.float32) / 255.0
                ref_skel_arr = ref_arr.astype(np.float32) / 255.0
                
                # Compute MSE over pixels. Since most pixels are black, MSE is naturally small.
                # Multiply by 10 to scale it to a similar range as the original MediaPipe error (0.0 to 1.0)
                mse = np.mean((out_skel_arr - ref_skel_arr) ** 2)
                return float(mse) * 10.0
                
            except ImportError:
                print("Warning: controlnet_aux not found. Cannot evaluate skeleton pose error.")
                return 1.0
                
        # --- Standard MediaPipe Evaluation for Real Photos ---
        out_results = self.pose_detector.process(out_arr)
        ref_results = self.pose_detector.process(ref_arr)
        
        if not out_results.pose_landmarks or not ref_results.pose_landmarks:
            return 1.0 # Max error if pose not found in either image
            
        out_kps = np.array([[lm.x, lm.y] for lm in out_results.pose_landmarks.landmark])
        ref_kps = np.array([[lm.x, lm.y] for lm in ref_results.pose_landmarks.landmark])
        
        # Calculate mean L2 distance across all 33 landmarks
        error = np.linalg.norm(out_kps - ref_kps, axis=1).mean()
        return float(error)
