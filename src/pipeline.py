# src/pipeline.py
"""
Multi-LoRA + ControlNet (OpenPose) pipeline for SDXL.

Usage:
    from src.pipeline import load_identities, build_pipeline, generate

    identities = load_identities()
    pipe = build_pipeline(identities)
    image = generate(pipe, identities, pose_image, lora_scales={"hermione": 0.7, "daenerys": 0.7})
"""

import yaml
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from attention import create_identity_masks, get_trigger_token_indices, set_regional_attention, remove_regional_attention


# ── helpers ──────────────────────────────────────────────────────────

def get_device():
    """Pick best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str):
    """float16 for cuda, float32 for mps/cpu (mps has limited fp16 support)."""
    if device == "cuda":
        return torch.float16
    return torch.float32


# ── config ───────────────────────────────────────────────────────────

def load_identities(config_path: str = "configs/identities.yaml") -> dict:
    """Load identity definitions from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["identities"]


# ── pipeline construction ────────────────────────────────────────────

def build_pipeline(
    identities: dict,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet_model: str = "thibaud/controlnet-openpose-sdxl-1.0",
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
    device: str | None = None,
) -> StableDiffusionXLControlNetPipeline:
    """
    Build an SDXL pipeline with:
      - ControlNet (OpenPose) for pose conditioning
      - Multiple LoRA adapters (one per identity)

    Returns a ready-to-use pipeline on the best available device.
    """
    device = device or get_device()
    dtype = get_dtype(device)

    print(f"[pipeline] device={device}  dtype={dtype}")

    # ControlNet
    print(f"[pipeline] Loading ControlNet: {controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )

    # VAE (fp16-fix avoids black-image issues on SDXL)
    print(f"[pipeline] Loading VAE: {vae_model}")
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=dtype,
    )

    # Base SDXL + ControlNet
    print(f"[pipeline] Loading base model: {base_model}")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Memory optimisation
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Load each identity LoRA as a named adapter
    for identity_id, identity in identities.items():
        lora_path = identity["lora_path"]
        print(f"[pipeline] Loading LoRA '{identity_id}' from {lora_path}")
        pipe.load_lora_weights(
            lora_path,
            adapter_name=identity_id,
        )

    # Activate all adapters with default scale 1.0
    adapter_names = list(identities.keys())
    pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))
    print(f"[pipeline] Active adapters: {adapter_names}")

    return pipe


# ── generation ───────────────────────────────────────────────────────

def generate(
    pipe: StableDiffusionXLControlNetPipeline,
    identities: dict,
    pose_image: Image.Image,
    lora_scales: dict[str, float] | None = None,
    ctrl_scale: float = 0.7,
    prompt: str | None = None,
    negative_prompt: str = "blurry, low quality, deformed face, extra limbs",
    num_inference_steps: int = 30,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    use_regional_attention: bool = False,
    identity_regions: dict | None = None,
) -> Image.Image:
    """
    Generate a multi-identity image.

    Args:
        pipe:                    The built SDXL pipeline.
        identities:              Identity config dict from load_identities().
        pose_image:              OpenPose skeleton image for ControlNet.
        lora_scales:             Per-identity LoRA scales, e.g. {"hermione": 0.7, "daenerys": 0.5}.
                                 Defaults to 0.8 for all if not provided.
        ctrl_scale:              ControlNet conditioning scale.
        prompt:                  Text prompt. Auto-generated from trigger words if None.
        negative_prompt:         Negative prompt.
        seed:                    Random seed for reproducibility.
        use_regional_attention:  Apply regional attention masking to reduce identity leakage.
        identity_regions:        Bounding boxes per identity {id: (x1, y1, x2, y2)}.
                                 Defaults to left/right split when None.
    """
    adapter_names = list(identities.keys())

    # Set per-identity LoRA scales
    if lora_scales is None:
        lora_scales = {k: 0.8 for k in adapter_names}
    adapter_weights = [lora_scales.get(k, 0.8) for k in adapter_names]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # Resolve identity regions early so prompt construction can use them
    if use_regional_attention and identity_regions is None:
        mid = width // 2
        identity_regions = {
            adapter_names[0]: (0,    0, mid,   height),
            adapter_names[1]: (mid,  0, width, height),
        }

    # Build prompt from trigger words if not provided
    if prompt is None:
        if identity_regions is not None:
            # Sort identities by x-position so prompt matches spatial layout
            sorted_ids = sorted(
                identity_regions.keys(),
                key=lambda k: identity_regions[k][0],  # sort by x1
            )
            trigger_parts = [
                f"{identities[k]['lora_trigger']} on the {'left' if i == 0 else 'right'}"
                for i, k in enumerate(sorted_ids)
            ]
            triggers = " and ".join(trigger_parts)
        else:
            triggers = " and ".join(
                identities[k]["lora_trigger"] for k in adapter_names
            )
        prompt = f"portrait photo of {triggers}, two people, soft natural lighting, high quality"

    # Resize pose image to match output dimensions
    pose_image_resized = pose_image.resize((width, height))

    # Apply regional attention masking if requested
    if use_regional_attention:
        spatial_masks = create_identity_masks(width, height, identity_regions)
        trigger_words = {k: identities[k]["lora_trigger"] for k in adapter_names}
        token_assignments = get_trigger_token_indices(
            pipe.tokenizer, prompt, trigger_words
        )
        set_regional_attention(pipe, spatial_masks, token_assignments)
        print(f"[pipeline] Regional attention enabled: {identity_regions}")

    print(f"[pipeline] Prompt: {prompt}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_image_resized,
            controlnet_conditioning_scale=ctrl_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        ).images[0]
    finally:
        if use_regional_attention:
            remove_regional_attention(pipe)

    return image


# ── quick test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    identities = load_identities()
    pipe = build_pipeline(identities)

    # Check for a pose image
    pose_dir = "data/pose_images"
    if os.path.exists(pose_dir) and os.listdir(pose_dir):
        pose_path = os.path.join(pose_dir, os.listdir(pose_dir)[0])
        pose_img = Image.open(pose_path).convert("RGB")
    else:
        print("[pipeline] No pose image found in data/pose_images/")
        print("[pipeline] Run: python src/generate_pose.py  to create one")
        exit(1)

    img = generate(pipe, identities, pose_img)

    os.makedirs("data/results", exist_ok=True)
    img.save("data/results/test_output.png")
    print("[pipeline] Saved test output to data/results/test_output.png")
