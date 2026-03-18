# src/extract_reference.py
"""
Generate a single-identity image from each LoRA and save as reference face.

Usage:
    python src/extract_reference.py

This generates one image per identity using only that LoRA (no other LoRA active),
then saves it as the reference face image for ArcFace scoring.
"""

import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from pipeline import load_identities, get_device, get_dtype


def generate_single_identity_reference(
    identity_id: str,
    identity: dict,
    output_dir: str = "data/reference_faces",
    seed: int = 42,
):
    """Generate a clean single-person portrait using one LoRA only."""
    device = get_device()
    dtype = get_dtype(device)

    # Load base SDXL (no ControlNet needed for reference)
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

    # Load only this identity's LoRA
    pipe.load_lora_weights(
        identity["lora_path"],
        adapter_name=identity_id,
    )
    pipe.set_adapters([identity_id], adapter_weights=[0.9])

    trigger = identity["lora_trigger"]
    prompt = f"close-up portrait photo of {trigger}, facing camera, neutral background, soft lighting, high quality"
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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{identity_id}.png")
    image.save(out_path)
    print(f"[reference] Saved {identity_id} reference → {out_path}")

    # Clean up to free VRAM
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def main():
    identities = load_identities()

    for identity_id, identity in identities.items():
        print(f"\n{'='*50}")
        print(f"Generating reference for: {identity_id}")
        print(f"{'='*50}")
        generate_single_identity_reference(identity_id, identity)

    print("\nDone! Reference faces saved to data/reference_faces/")
    print("These will be used by ArcFace scorer for identity comparison.")


if __name__ == "__main__":
    main()
