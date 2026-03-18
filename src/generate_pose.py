# src/generate_pose.py
"""
Generate multi-person OpenPose skeleton images for ControlNet conditioning.

Two modes:
  1. Extract from a real photo:
       python src/generate_pose.py --from-image path/to/two_people.jpg

  2. Draw a synthetic 2-person portrait skeleton:
       python src/generate_pose.py --synthetic
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw


# ── OpenPose body keypoint connections (COCO 18-keypoint format) ─────
# Each tuple: (start_keypoint_idx, end_keypoint_idx)
POSE_CONNECTIONS = [
    (0, 1),    # nose → neck
    (1, 2),    # neck → right shoulder
    (1, 5),    # neck → left shoulder
    (2, 3),    # right shoulder → right elbow
    (3, 4),    # right elbow → right wrist
    (5, 6),    # left shoulder → left elbow
    (6, 7),    # left elbow → left wrist
    (1, 8),    # neck → right hip
    (1, 11),   # neck → left hip
    (8, 9),    # right hip → right knee
    (9, 10),   # right knee → right ankle
    (11, 12),  # left hip → left knee
    (12, 13),  # left knee → left ankle
    (0, 14),   # nose → right eye
    (0, 15),   # nose → left eye
    (14, 16),  # right eye → right ear
    (15, 17),  # left eye → left ear
]

# Limb colors (BGR-style for visual distinction)
LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]

KEYPOINT_COLOR = (255, 0, 0)


def make_synthetic_two_person_pose(width=1024, height=1024) -> Image.Image:
    """
    Draw a synthetic OpenPose skeleton for two people standing side by side.
    This is a portrait-style composition suitable for the project.
    """
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Define keypoints for two people (normalised 0-1, then scaled)
    # Format: [nose, neck, r_shoulder, r_elbow, r_wrist,
    #          l_shoulder, l_elbow, l_wrist,
    #          r_hip, r_knee, r_ankle,
    #          l_hip, l_knee, l_ankle,
    #          r_eye, l_eye, r_ear, l_ear]

    # Person 1 — left side of frame
    person1 = [
        (0.30, 0.18),   # 0  nose
        (0.30, 0.25),   # 1  neck
        (0.24, 0.27),   # 2  right shoulder
        (0.20, 0.37),   # 3  right elbow
        (0.18, 0.45),   # 4  right wrist
        (0.36, 0.27),   # 5  left shoulder
        (0.40, 0.37),   # 6  left elbow
        (0.42, 0.45),   # 7  left wrist
        (0.26, 0.45),   # 8  right hip
        (0.25, 0.60),   # 9  right knee
        (0.25, 0.75),   # 10 right ankle
        (0.34, 0.45),   # 11 left hip
        (0.35, 0.60),   # 12 left knee
        (0.35, 0.75),   # 13 left ankle
        (0.28, 0.16),   # 14 right eye
        (0.32, 0.16),   # 15 left eye
        (0.25, 0.17),   # 16 right ear
        (0.35, 0.17),   # 17 left ear
    ]

    # Person 2 — right side of frame
    person2 = [
        (0.70, 0.18),   # 0  nose
        (0.70, 0.25),   # 1  neck
        (0.64, 0.27),   # 2  right shoulder
        (0.60, 0.37),   # 3  right elbow
        (0.58, 0.45),   # 4  right wrist
        (0.76, 0.27),   # 5  left shoulder
        (0.80, 0.37),   # 6  left elbow
        (0.82, 0.45),   # 7  left wrist
        (0.66, 0.45),   # 8  right hip
        (0.65, 0.60),   # 9  right knee
        (0.65, 0.75),   # 10 right ankle
        (0.74, 0.45),   # 11 left hip
        (0.75, 0.60),   # 12 left knee
        (0.75, 0.75),   # 13 left ankle
        (0.68, 0.16),   # 14 right eye
        (0.72, 0.16),   # 15 left eye
        (0.65, 0.17),   # 16 right ear
        (0.75, 0.17),   # 17 left ear
    ]

    for person_kps in [person1, person2]:
        # Scale to image dimensions
        kps = [(int(x * width), int(y * height)) for x, y in person_kps]

        # Draw limbs
        for idx, (start, end) in enumerate(POSE_CONNECTIONS):
            color = LIMB_COLORS[idx % len(LIMB_COLORS)]
            draw.line([kps[start], kps[end]], fill=color, width=4)

        # Draw keypoints
        for x, y in kps:
            r = 5
            draw.ellipse([x - r, y - r, x + r, y + r], fill=KEYPOINT_COLOR)

    return img


def extract_pose_from_image(image_path: str) -> Image.Image:
    """
    Extract OpenPose skeleton from a real photo using controlnet_aux.
    Requires: pip install controlnet_aux
    """
    from controlnet_aux import OpenposeDetector

    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    input_image = Image.open(image_path).convert("RGB")
    pose_image = detector(input_image)

    return pose_image


def main():
    parser = argparse.ArgumentParser(description="Generate pose skeletons for ControlNet")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a synthetic 2-person pose skeleton")
    parser.add_argument("--from-image", type=str, default=None,
                        help="Extract pose from a real photo")
    parser.add_argument("--output", type=str, default="data/pose_images/two_person_pose.png",
                        help="Output path")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.from_image:
        print(f"[pose] Extracting pose from: {args.from_image}")
        pose_img = extract_pose_from_image(args.from_image)
    elif args.synthetic:
        print(f"[pose] Generating synthetic 2-person pose ({args.width}x{args.height})")
        pose_img = make_synthetic_two_person_pose(args.width, args.height)
    else:
        print("Specify --synthetic or --from-image <path>")
        print("  python src/generate_pose.py --synthetic")
        print("  python src/generate_pose.py --from-image photo_of_two_people.jpg")
        return

    pose_img.save(args.output)
    print(f"[pose] Saved to {args.output}")


if __name__ == "__main__":
    main()
