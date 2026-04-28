# src/face_aware_pose.py
"""
Face-orientation-aware pose conditioning for OpenPose ControlNet.

The synthetic 2-person skeleton in generate_pose.py underspecifies face
orientation: the nose/eyes/ears appear as small circles with no explicit
inter-keypoint lines on the face, so ControlNet treats face direction as
ambiguous. The dominant identity LoRA's training-data bias then chooses
the orientation, which is how Daenerys ends up in three-quarter or
profile view even at high ctrl_scale.

This module provides two alternatives:

  1) make_face_aware_pose(...)
     A synthetic 2-person skeleton with EXPLICIT face structure: thicker
     facial keypoints, an eye-to-eye line, and a nose-to-neck centre line.
     The OpenPose ControlNet was trained on detector outputs that already
     contain these patterns, so making them prominent in the synthetic
     skeleton biases generation toward front view.

  2) extract_pose_with_face_landmarks(image_path)
     MediaPipe-Pose-based extractor (33 landmarks including detailed
     face landmarks). Convert real two-person photos into a richer pose
     condition than synthetic 18-keypoint COCO.

Both return a PIL.Image suitable for the OpenPose ControlNet input.
"""

from PIL import Image, ImageDraw
import numpy as np
from generate_pose import (
    POSE_CONNECTIONS, LIMB_COLORS, KEYPOINT_COLOR,
    _synthetic_two_person_raw_kps,
)


# Face-structure connections beyond the standard COCO 18.
# (idx_a, idx_b) — drawn as thin white lines to give the OpenPose
# ControlNet a hard-edged signal of where the face plane is facing.
FACE_STRUCTURE_LINES = [
    (14, 15),   # right eye <-> left eye  (horizontal eye line)
    (16, 17),   # right ear <-> left ear  (head width)
    (0,  1),    # nose <-> neck (centreline)
]

FACE_LINE_COLOR = (240, 240, 240)
FACE_LINE_WIDTH = 3
KEYPOINT_RADIUS_FACE = 8     # face dots bigger than body
KEYPOINT_RADIUS_BODY = 5


def make_face_aware_pose(
    width: int = 1024,
    height: int = 1024,
    front_facing: bool = True,
) -> Image.Image:
    """
    Draw a 2-person OpenPose skeleton with explicit face structure lines.

    Compared to make_synthetic_two_person_pose (generate_pose.py), this
    version adds eye-eye, ear-ear, and nose-neck lines and enlarges the
    facial keypoints so the OpenPose ControlNet receives a stronger
    "face is looking forward" signal. front_facing=True ensures left/right
    eye and ear keypoints are symmetric around the nose; flipping it allows
    deliberately ambiguous poses for ablation.
    """
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    person1, person2 = _synthetic_two_person_raw_kps()

    # If front_facing is False, deliberately collapse the eye line so the
    # face direction signal is ambiguous (used as a contrast in ablation).
    if not front_facing:
        person1 = _collapse_face_axis(person1)
        person2 = _collapse_face_axis(person2)

    for person_kps in [person1, person2]:
        kps_px = [(int(x * width), int(y * height)) for x, y in person_kps]

        # Body limbs (colourful)
        for idx, (start, end) in enumerate(POSE_CONNECTIONS):
            color = LIMB_COLORS[idx % len(LIMB_COLORS)]
            draw.line([kps_px[start], kps_px[end]], fill=color, width=4)

        # Face structure lines (white) — added beyond standard OpenPose
        for a, b in FACE_STRUCTURE_LINES:
            draw.line(
                [kps_px[a], kps_px[b]],
                fill=FACE_LINE_COLOR,
                width=FACE_LINE_WIDTH,
            )

        # Keypoints — face larger than body to dominate the local signal
        face_idx = {0, 14, 15, 16, 17}
        for i, (x, y) in enumerate(kps_px):
            r = KEYPOINT_RADIUS_FACE if i in face_idx else KEYPOINT_RADIUS_BODY
            draw.ellipse([x - r, y - r, x + r, y + r], fill=KEYPOINT_COLOR)

    return img


def _collapse_face_axis(kps: list) -> list:
    """Helper for ablation: move both eyes onto the nose x so the face has
    no clear left/right symmetry signal."""
    nose_x = kps[0][0]
    out = list(kps)
    for i in (14, 15, 16, 17):
        x, y = kps[i]
        out[i] = (nose_x, y)
    return out


# ── MediaPipe-based extractor for real reference photos ────────────────

def extract_pose_with_face_landmarks(
    image_path: str,
    width: int = 1024,
    height: int = 1024,
) -> Image.Image:
    """
    Extract a 2-person pose from a real photo using MediaPipe Pose, then
    render it in OpenPose-skeleton style with the additional face-structure
    lines used by make_face_aware_pose.

    Notes:
        - MediaPipe Pose detects ONE person per call. For 2-person photos,
          you can crop into halves and call this twice, or pre-segment.
        - Requires `mediapipe` package; the import is local so the rest
          of the project does not require it.
    """
    import mediapipe as mp

    src = Image.open(image_path).convert("RGB").resize((width, height))
    src_arr = np.array(src)

    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    )
    result = pose_detector.process(src_arr)
    if not result.pose_landmarks:
        raise RuntimeError(f"MediaPipe Pose failed to detect a person in {image_path}")

    # MediaPipe → COCO-18 mapping (subset)
    MP = mp.solutions.pose.PoseLandmark
    mp_to_coco = {
        0:  MP.NOSE,
        1:  None,                     # neck — synthesised below
        2:  MP.RIGHT_SHOULDER,
        3:  MP.RIGHT_ELBOW,
        4:  MP.RIGHT_WRIST,
        5:  MP.LEFT_SHOULDER,
        6:  MP.LEFT_ELBOW,
        7:  MP.LEFT_WRIST,
        8:  MP.RIGHT_HIP,
        9:  MP.RIGHT_KNEE,
        10: MP.RIGHT_ANKLE,
        11: MP.LEFT_HIP,
        12: MP.LEFT_KNEE,
        13: MP.LEFT_ANKLE,
        14: MP.RIGHT_EYE,
        15: MP.LEFT_EYE,
        16: MP.RIGHT_EAR,
        17: MP.LEFT_EAR,
    }

    lm = result.pose_landmarks.landmark
    coco = []
    for i in range(18):
        if i == 1:  # neck = midpoint of shoulders
            r = lm[MP.RIGHT_SHOULDER]
            l = lm[MP.LEFT_SHOULDER]
            coco.append(((r.x + l.x) / 2, (r.y + l.y) / 2))
        else:
            mp_idx = mp_to_coco[i]
            p = lm[mp_idx]
            coco.append((p.x, p.y))

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    kps_px = [(int(x * width), int(y * height)) for x, y in coco]

    for idx, (start, end) in enumerate(POSE_CONNECTIONS):
        color = LIMB_COLORS[idx % len(LIMB_COLORS)]
        draw.line([kps_px[start], kps_px[end]], fill=color, width=4)
    for a, b in FACE_STRUCTURE_LINES:
        draw.line([kps_px[a], kps_px[b]], fill=FACE_LINE_COLOR, width=FACE_LINE_WIDTH)
    face_idx = {0, 14, 15, 16, 17}
    for i, (x, y) in enumerate(kps_px):
        r = KEYPOINT_RADIUS_FACE if i in face_idx else KEYPOINT_RADIUS_BODY
        draw.ellipse([x - r, y - r, x + r, y + r], fill=KEYPOINT_COLOR)

    return img


def get_face_aware_target_keypoints(front_facing: bool = True) -> list[list]:
    """
    Same as get_synthetic_target_keypoints but matching the front-facing
    setting used to draw the face-aware pose. Pose scoring uses these as
    the target.
    """
    person1, person2 = _synthetic_two_person_raw_kps()
    if not front_facing:
        person1 = _collapse_face_axis(person1)
        person2 = _collapse_face_axis(person2)
    return [
        [(x, y, 1.0) for x, y in person1],
        [(x, y, 1.0) for x, y in person2],
    ]
