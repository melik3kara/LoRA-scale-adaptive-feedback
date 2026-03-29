# src/extract_reference.py
"""
Generate a single-identity image from each LoRA and save as reference face.

Usage:
    python src/extract_reference.py

Generates one image per identity using only that LoRA (no other LoRA active),
detects and crops the face, aligns it, and saves as the reference for ArcFace scoring.

Saves both:
  - data/reference_faces/{identity_id}_full.png   (full 1024x1024 portrait)
  - data/reference_faces/{identity_id}.png         (cropped + aligned 160x160 face)
"""

import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from insightface.app import FaceAnalysis

from pipeline import load_identities, get_device, get_dtype


def build_face_detector(device: str = "cuda") -> FaceAnalysis:
    """Initialize InsightFace detector (includes RetinaFace + ArcFace)."""
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def crop_and_align_face(image: Image.Image, face_app: FaceAnalysis) -> Image.Image | None:
    """
    Detect the largest face in the image, crop and align it to 160x160.
    Returns None if no face is detected.
    """
    img_array = np.array(image)
    # InsightFace expects BGR
    img_bgr = img_array[:, :, ::-1]

    faces = face_app.get(img_bgr)

    if len(faces) == 0:
        return None

    # Pick the largest face by bounding box area
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # Extract the aligned face (normed_embedding exists, but we want the crop)
    # InsightFace stores the aligned 112x112 face in face.aimg if available,
    # but we can also crop manually from bbox for a larger crop

    x1, y1, x2, y2 = [int(v) for v in largest.bbox]

    # Add padding around the face (20%) for better ArcFace results
    h, w = img_array.shape[:2]
    face_w, face_h = x2 - x1, y2 - y1
    pad_x = int(face_w * 0.2)
    pad_y = int(face_h * 0.2)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    face_crop = image.crop((x1, y1, x2, y2))
    face_crop = face_crop.resize((160, 160), Image.LANCZOS)

    return face_crop


def generate_references(
    output_dir: str = "data/reference_faces",
    seed: int = 42,
):
    """Generate clean single-person portraits and extract aligned face crops."""
    device = get_device()
    dtype = get_dtype(device)
    identities = load_identities()

    os.makedirs(output_dir, exist_ok=True)

    # ── Load base model ONCE ─────────────────────────────────────────
    print("[reference] Loading base SDXL model...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # ── Load face detector ───────────────────────────────────────────
    print("[reference] Loading face detector...")
    face_app = build_face_detector(device)

    # ── Generate per identity ────────────────────────────────────────
    for identity_id, identity in identities.items():
        print(f"\n{'='*50}")
        print(f"Generating reference for: {identity_id}")
        print(f"{'='*50}")

        # Load this identity's LoRA
        pipe.load_lora_weights(
            identity["lora_path"],
            adapter_name=identity_id,
        )
        pipe.set_adapters([identity_id], adapter_weights=[0.9])

        trigger = identity["lora_trigger"]
        prompt = (
            f"close-up portrait photo of {trigger}, facing camera, "
            f"neutral background, soft lighting, high quality"
        )
        negative = "blurry, low quality, deformed, extra limbs, multiple people"

        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            generator=generator,
            num_inference_steps=30,
            width=1024,
            height=1024,
        ).images[0]

        # Save full portrait
        full_path = os.path.join(output_dir, f"{identity_id}_full.png")
        image.save(full_path)
        print(f"[reference] Full portrait → {full_path}")

        # Crop and align face
        face_crop = crop_and_align_face(image, face_app)

        if face_crop is not None:
            crop_path = os.path.join(output_dir, f"{identity_id}.png")
            face_crop.save(crop_path)
            print(f"[reference] Aligned face (160x160) → {crop_path}")
        else:
            print(f"[reference] WARNING: No face detected for {identity_id}!")
            print(f"[reference] Saving full image as fallback — ArcFace scores will be unreliable.")
            fallback_path = os.path.join(output_dir, f"{identity_id}.png")
            image.resize((160, 160), Image.LANCZOS).save(fallback_path)

        # Unload this LoRA before loading the next one
        pipe.unload_lora_weights()

    print(f"\nDone! Reference faces saved to {output_dir}/")
    print("These will be used by ArcFace scorer for identity comparison.")


if __name__ == "__main__":
    generate_references()
