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

# Dense face structure — extra detail beyond the 18-keypoint COCO standard.
# Drawn as faint lines/arcs to give ControlNet stronger "front-facing face"
# evidence. Generated procedurally from existing nose/eye/ear keypoints.
DENSE_FACE_COLOR = (220, 200, 200)
DENSE_FACE_WIDTH = 2


def _draw_dense_face(draw: ImageDraw.ImageDraw, kps_px: list) -> None:
    """
    Draw extra face structure (jawline, brows, mouth, nose bridge) procedurally
    from the COCO-18 keypoints (nose=0, r_eye=14, l_eye=15, r_ear=16, l_ear=17,
    neck=1). All points are derived geometrically — no extra inputs needed.
    """
    nose   = kps_px[0]
    neck   = kps_px[1]
    r_eye  = kps_px[14]
    l_eye  = kps_px[15]
    r_ear  = kps_px[16]
    l_ear  = kps_px[17]

    # Eye level → above for brows, below for mouth
    eye_y    = (r_eye[1] + l_eye[1]) // 2
    face_w   = abs(l_ear[0] - r_ear[0])
    face_h   = max(int(face_w * 1.2), 1)
    face_cx  = (r_ear[0] + l_ear[0]) // 2
    chin_y   = eye_y + int(face_h * 0.6)
    mouth_y  = eye_y + int(face_h * 0.35)

    # ── Jawline arc (ear → chin → ear) approximated as 5-segment polyline
    jaw_pts = [
        r_ear,
        (r_ear[0] + int(face_w * 0.10), eye_y + int(face_h * 0.30)),
        (face_cx - int(face_w * 0.10), chin_y),
        (face_cx + int(face_w * 0.10), chin_y),
        (l_ear[0] - int(face_w * 0.10), eye_y + int(face_h * 0.30)),
        l_ear,
    ]
    for a, b in zip(jaw_pts[:-1], jaw_pts[1:]):
        draw.line([a, b], fill=DENSE_FACE_COLOR, width=DENSE_FACE_WIDTH)

    # ── Eyebrows (short lines just above each eye)
    brow_dx = max(int(face_w * 0.06), 6)
    brow_dy = max(int(face_h * 0.08), 4)
    for eye in (r_eye, l_eye):
        draw.line(
            [(eye[0] - brow_dx, eye[1] - brow_dy),
             (eye[0] + brow_dx, eye[1] - brow_dy)],
            fill=DENSE_FACE_COLOR, width=DENSE_FACE_WIDTH,
        )

    # ── Mouth (horizontal line below the nose)
    mouth_w = max(int(face_w * 0.22), 8)
    draw.line(
        [(face_cx - mouth_w, mouth_y), (face_cx + mouth_w, mouth_y)],
        fill=DENSE_FACE_COLOR, width=DENSE_FACE_WIDTH,
    )
    # Mouth corners as small dots so the line isn't ambiguous
    for dx in (-mouth_w, mouth_w):
        draw.ellipse(
            [face_cx + dx - 2, mouth_y - 2, face_cx + dx + 2, mouth_y + 2],
            fill=DENSE_FACE_COLOR,
        )

    # ── Nose bridge (eye-line midpoint → nose) and nose tip ticks
    eye_mid = ((r_eye[0] + l_eye[0]) // 2, (r_eye[1] + l_eye[1]) // 2)
    draw.line([eye_mid, nose], fill=DENSE_FACE_COLOR, width=DENSE_FACE_WIDTH)
    # Nostril hint — two small dots flanking nose tip
    nost_dx = max(int(face_w * 0.04), 4)
    for dx in (-nost_dx, nost_dx):
        draw.ellipse(
            [nose[0] + dx - 2, nose[1] - 2, nose[0] + dx + 2, nose[1] + 2],
            fill=DENSE_FACE_COLOR,
        )


def make_face_aware_pose(
    width: int = 1024,
    height: int = 1024,
    front_facing: bool = True,
    dense_face: bool = False,
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

        # Optional dense face structure (jawline, brows, mouth, nose bridge)
        if dense_face:
            _draw_dense_face(draw, kps_px)

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
