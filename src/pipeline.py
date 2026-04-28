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


def _find_phrase_token_indices(tokenizer, prompt: str, phrase: str) -> list[int]:
    """Return the token indices (1-based — +1 for BOS) of every occurrence of `phrase`."""
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
    if not phrase_tokens:
        return []
    plen = len(phrase_tokens)
    indices: list[int] = []
    for i in range(len(full_tokens) - plen + 1):
        if full_tokens[i:i + plen] == phrase_tokens:
            indices.extend(range(i + 1, i + plen + 1))  # +1 for BOS
    return indices


def _to_peft_scale(scale):
    """
    Convert a user-facing LoRA scale into the format diffusers/PEFT
    `set_adapters` expects per adapter.

    Accepts:
      - float            → returned as-is
      - {"down": 0.1, "mid": 0.2, "up": 0.3, "text_encoder": 0.0}
                          (short keys; mapped to UNet block prefixes)
      - {"down_blocks": ..., "mid_block": ..., "up_blocks": ..., "text_encoder": ...}
                          (long keys; passed through)

    Short keys are translated so the user-facing config stays compact while
    we still hand PEFT the regex-matchable prefixes it requires.
    """
    if not isinstance(scale, dict):
        return float(scale)

    KEY_MAP = {
        "down": "down_blocks",
        "mid":  "mid_block",
        "up":   "up_blocks",
        "te":   "text_encoder",
    }
    out = {}
    for k, v in scale.items():
        out[KEY_MAP.get(k, k)] = float(v)
    return out


def _build_attention_token_assignments(
    tokenizer,
    prompt: str,
    identities: dict,
) -> dict:
    """
    For each identity, collect token indices of all its attention_phrases
    (falling back to lora_trigger) in the full tokenized prompt.

    Expanding the mask beyond the trigger keeps the descriptive tokens (hair,
    attire, etc.) bound to their identity's region and prevents cross-region
    feature bleed — the main failure mode with rich prompts.
    """
    assignments: dict = {}
    for k, meta in identities.items():
        phrases = list(meta.get("attention_phrases") or [])
        trigger = meta.get("lora_trigger", "")
        if trigger and trigger not in phrases:
            phrases.append(trigger)

        seen: set[int] = set()
        ordered: list[int] = []
        for phrase in phrases:
            for idx in _find_phrase_token_indices(tokenizer, prompt, phrase):
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
        assignments[k] = ordered
    return assignments


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

_BASE_NEGATIVE = (
    "blurry, low quality, deformed face, extra limbs, extra person, three people, "
    "crowd, group of people, duplicate, clone, twins, identical faces, same face, "
    "merged bodies, conjoined, disfigured, asymmetric eyes"
)


def _build_structured_prompt(identities: dict, identity_regions: dict | None) -> str:
    """
    Build a richly-described spatially-ordered prompt from identity metadata.

    Uses visual_description from identities.yaml when available so CLIP has
    concrete physical tokens to latch onto — this is the main defence against
    one LoRA dominating the scene visually.
    """
    if identity_regions is not None:
        ordered = sorted(identity_regions.keys(), key=lambda k: identity_regions[k][0])
    else:
        ordered = list(identities.keys())

    pos_labels = ["left", "right", "center"]
    parts = []
    for i, k in enumerate(ordered):
        trigger = identities[k].get("lora_trigger", "")
        desc    = identities[k].get("visual_description", "")
        pos     = pos_labels[i] if i < len(pos_labels) else f"position {i+1}"
        parts.append(
            f"on the {pos}, {desc}, {trigger}" if desc else f"{trigger} on the {pos}"
        )

    body = ". ".join(parts)
    # Keep the full prompt under ~77 CLIP tokens so nothing critical is
    # truncated. Identity clauses stay at the front; polish is minimal.
    return (
        f"Two women side by side. {body}. "
        f"Different faces, different hair colors, photorealistic"
    )


def _build_combined_negative(identities: dict, base_negative: str) -> str:
    """
    Append each identity's negative_features to the base negative prompt.
    Each identity's negative_features typically lists the OTHER identity's
    distinctive traits, which helps suppress feature-bleed globally.
    """
    extras = []
    for k, meta in identities.items():
        nf = meta.get("negative_features")
        if nf:
            extras.append(nf)
    if not extras:
        return base_negative
    return base_negative + ", " + ", ".join(extras)


def generate(
    pipe: StableDiffusionXLControlNetPipeline,
    identities: dict,
    pose_image: Image.Image,
    lora_scales: dict[str, float] | None = None,
    ctrl_scale: float = 0.7,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    num_inference_steps: int = 30,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    use_regional_attention: bool = False,
    identity_regions: dict | None = None,
    guidance_scale: float = 7.5,
    use_spatial_lora_gate: bool = False,
    spatial_gate_block_floor: dict | None = None,
    spatial_gate_floor: float = 0.05,
    spatial_gate_feather_ratio: float = 0.05,
    use_bg_lock: bool = False,
    bg_lock_ratio: float = 0.35,
    bg_lock_padding: int = 0,
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

    # Set per-identity LoRA scales — supports two formats:
    #   1) Flat scalar: {"hermione": 0.7, "daenerys": 0.5}
    #   2) Block-wise dict per identity:
    #        {"hermione": {"down": 0.1, "mid": 0.2, "up": 0.35},
    #         "daenerys": {"down": 0.5, "mid": 1.2, "up": 1.5}}
    # Block-wise scales let the loop weaken a dominant LoRA in early UNet
    # blocks (which control composition) while keeping it active in later
    # blocks (which control face/texture detail).
    if lora_scales is None:
        lora_scales = {k: 0.8 for k in adapter_names}
    adapter_weights = [_to_peft_scale(lora_scales.get(k, 0.8)) for k in adapter_names]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # Resolve identity regions early so prompt construction can use them
    if use_regional_attention and identity_regions is None:
        mid = width // 2
        identity_regions = {
            adapter_names[0]: (0,    0, mid,   height),
            adapter_names[1]: (mid,  0, width, height),
        }

    # Build prompt from identity metadata if not provided
    if prompt is None:
        prompt = _build_structured_prompt(identities, identity_regions)

    if negative_prompt is None:
        negative_prompt = _build_combined_negative(identities, _BASE_NEGATIVE)

    # Resize pose image to match output dimensions
    pose_image_resized = pose_image.resize((width, height))

    # Apply regional attention masking if requested
    if use_regional_attention:
        spatial_masks = create_identity_masks(width, height, identity_regions)
        # Expanded assignments: mask trigger AND descriptive attention_phrases
        # so identity features can't leak across regions.
        token_assignments = _build_attention_token_assignments(
            pipe.tokenizer, prompt, identities
        )
        # Only cross-attention masking — self-attention masking causes "head+body" artefacts
        set_regional_attention(pipe, spatial_masks, token_assignments, mask_self_attention=False)
        print(f"[pipeline] Regional attention enabled (cross-attention only): {identity_regions}")
        for k, idxs in token_assignments.items():
            print(f"[pipeline]   {k} attention tokens: {idxs}")

    print(f"[pipeline] Prompt: {prompt}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Spatially gate LoRA residuals so each identity's contribution is
    # restricted to its own region throughout the UNet — this is the
    # mechanism, not just attention masking, that keeps identities apart.
    gate_handles = None
    if use_spatial_lora_gate and identity_regions is not None:
        from lora_gate import set_spatially_gated_lora
        gate_masks = create_identity_masks(width, height, identity_regions)
        gate_handles = set_spatially_gated_lora(
            pipe, gate_masks,
            latent_size=height // 8,
            feather_ratio=spatial_gate_feather_ratio,
            floor=spatial_gate_floor,
            block_floor=spatial_gate_block_floor,
        )

    # Background latent lock — freeze background of latent for first
    # bg_lock_ratio of denoising steps to prevent third-person hallucination.
    bg_callback = None
    bg_callback_inputs = None
    if use_bg_lock and identity_regions is not None:
        from bg_latent_lock import make_bg_lock_callback, regions_to_union_mask
        union = regions_to_union_mask(identity_regions, width, height, padding=bg_lock_padding)
        bg_callback = make_bg_lock_callback(
            union, total_steps=num_inference_steps, lock_ratio=bg_lock_ratio,
        )
        bg_callback_inputs = ["latents"]
        print(f"[pipeline] BG latent lock: first {int(num_inference_steps * bg_lock_ratio)}/{num_inference_steps} steps")

    pipe_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pose_image_resized,
        controlnet_conditioning_scale=ctrl_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
    )
    if bg_callback is not None:
        pipe_kwargs["callback_on_step_end"] = bg_callback
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = bg_callback_inputs

    try:
        image = pipe(**pipe_kwargs).images[0]
    finally:
        if use_regional_attention:
            remove_regional_attention(pipe)
        if gate_handles is not None:
            from lora_gate import remove_spatially_gated_lora
            remove_spatially_gated_lora(gate_handles)

    return image


# ── two-stage generation: layout-first, then identity ────────────────

def generate_two_stage(
    pipe: StableDiffusionXLControlNetPipeline,
    identities: dict,
    pose_image: Image.Image,
    pose_image_open: Image.Image | None = None,
    layout_lora_scales: dict | None = None,
    identity_lora_scales: dict | None = None,
    layout_ctrl_scale: float = 1.0,
    identity_ctrl_scale: float = 0.7,
    refine_strength: float = 0.55,
    refine_steps: int = 28,
    layout_steps: int = 28,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    use_regional_attention: bool = True,
    use_spatial_lora_gate: bool = True,
    identity_regions: dict | None = None,
    guidance_scale: float = 7.5,
    spatial_gate_block_floor: dict | None = None,
    spatial_gate_floor: float = 0.05,
    spatial_gate_feather_ratio: float = 0.05,
    use_bg_lock: bool = False,
    bg_lock_ratio: float = 0.35,
    bg_lock_padding: int = 0,
    bg_lock_in_layout: bool = True,
    bg_lock_in_identity: bool = False,
    **_unused,
) -> tuple[Image.Image, Image.Image]:
    """
    Two-stage generation:
      Stage 1 (layout): low LoRA scales, high ControlNet — locks pose,
                         body placement, left/right separation.
      Stage 2 (identity): re-runs ControlNet+img2img on stage-1 output with
                          high (asymmetric) LoRA scales, lower ControlNet,
                          and partial denoising — refines faces/attire
                          without re-deciding the composition.

    Returns (stage1_image, stage2_image).
    """
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

    # Stage 1 — layout locking
    if layout_lora_scales is None:
        layout_lora_scales = {k: 0.2 for k in identities}

    print("[pipeline] Two-stage: STAGE 1 (layout)")
    stage1_img = generate(
        pipe, identities, pose_image,
        lora_scales=layout_lora_scales,
        ctrl_scale=layout_ctrl_scale,
        num_inference_steps=layout_steps,
        seed=seed,
        width=width,
        height=height,
        use_regional_attention=use_regional_attention,
        # No spatial gate at layout stage — we want layout signal to spread
        use_spatial_lora_gate=False,
        identity_regions=identity_regions,
        guidance_scale=guidance_scale,
        # BG lock at layout stage: kills third-person hallucination before
        # any identity LoRA gets meaningful weight.
        use_bg_lock=(use_bg_lock and bg_lock_in_layout),
        bg_lock_ratio=bg_lock_ratio,
        bg_lock_padding=bg_lock_padding,
    )

    # Stage 2 — identity refinement via ControlNet img2img on stage-1 output.
    # The img2img pipeline shares weights/adapters with `pipe`, so all the
    # adapter scale changes we make propagate.
    img2img = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        controlnet=pipe.controlnet,
        scheduler=pipe.scheduler,
    )

    if identity_lora_scales is None:
        identity_lora_scales = {k: 0.8 for k in identities}
    adapter_names = list(identities.keys())
    adapter_weights = [_to_peft_scale(identity_lora_scales.get(k, 0.8)) for k in adapter_names]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    if identity_regions is None and use_regional_attention:
        mid = width // 2
        identity_regions = {
            adapter_names[0]: (0,    0, mid,   height),
            adapter_names[1]: (mid,  0, width, height),
        }

    prompt = _build_structured_prompt(identities, identity_regions)
    negative_prompt = _build_combined_negative(identities, _BASE_NEGATIVE)
    pose_image_resized = pose_image.resize((width, height))

    gate_handles = None
    if use_spatial_lora_gate and identity_regions is not None:
        from lora_gate import set_spatially_gated_lora
        gate_masks = create_identity_masks(width, height, identity_regions)
        gate_handles = set_spatially_gated_lora(
            pipe, gate_masks,
            latent_size=height // 8,
            feather_ratio=spatial_gate_feather_ratio,
            floor=spatial_gate_floor,
            block_floor=spatial_gate_block_floor,
        )

    if use_regional_attention:
        spatial_masks = create_identity_masks(width, height, identity_regions)
        token_assignments = _build_attention_token_assignments(
            pipe.tokenizer, prompt, identities
        )
        set_regional_attention(pipe, spatial_masks, token_assignments, mask_self_attention=False)

    # BG lock for the identity refinement stage (img2img). Optional — by
    # default disabled because at refinement stage the foreground is already
    # well-formed and the lock can blur edges.
    stage2_callback = None
    stage2_callback_inputs = None
    if use_bg_lock and bg_lock_in_identity and identity_regions is not None:
        from bg_latent_lock import make_bg_lock_callback, regions_to_union_mask
        union = regions_to_union_mask(identity_regions, width, height, padding=bg_lock_padding)
        stage2_callback = make_bg_lock_callback(
            union, total_steps=refine_steps, lock_ratio=bg_lock_ratio,
        )
        stage2_callback_inputs = ["latents"]

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    print("[pipeline] Two-stage: STAGE 2 (identity refinement)")
    print(f"[pipeline]   strength={refine_strength}  ctrl={identity_ctrl_scale}")
    img2img_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=stage1_img,
        control_image=pose_image_resized,
        controlnet_conditioning_scale=identity_ctrl_scale,
        strength=refine_strength,
        num_inference_steps=refine_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
    )
    if stage2_callback is not None:
        img2img_kwargs["callback_on_step_end"] = stage2_callback
        img2img_kwargs["callback_on_step_end_tensor_inputs"] = stage2_callback_inputs
    try:
        stage2_img = img2img(**img2img_kwargs).images[0]
    finally:
        if use_regional_attention:
            remove_regional_attention(pipe)
        if gate_handles is not None:
            from lora_gate import remove_spatially_gated_lora
            remove_spatially_gated_lora(gate_handles)

    return stage1_img, stage2_img


# ── face-local identity refinement ───────────────────────────────────

def _expand_bbox(bbox, pad_ratio: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * (1.0 + pad_ratio)
    nx1 = max(0, int(cx - side / 2)); ny1 = max(0, int(cy - side / 2))
    nx2 = min(w, int(cx + side / 2)); ny2 = min(h, int(cy + side / 2))
    return nx1, ny1, nx2, ny2


def _feather_mask(size: tuple[int, int], feather: int = 24) -> Image.Image:
    """Build a soft-edged white mask the same size as a crop."""
    from PIL import ImageFilter
    w, h = size
    m = Image.new("L", (w, h), 255)
    if feather <= 0:
        return m
    # Shrink slightly then blur, so edges fall off inside the crop
    inset = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(inset).rectangle(
        [feather, feather, w - feather, h - feather], fill=255
    )
    return inset.filter(ImageFilter.GaussianBlur(radius=feather))


def face_local_refine(
    pipe: StableDiffusionXLControlNetPipeline,
    identities: dict,
    image: Image.Image,
    face_scorer,
    identity_regions: dict,
    refine_strength: float = 0.45,
    crop_pad_ratio: float = 0.5,
    refine_steps: int = 25,
    seed: int = 42,
    feather: int = 24,
) -> Image.Image:
    """
    For each detected face, crop, upscale to 1024 if needed, run SDXL img2img
    using ONLY that identity's LoRA at high scale, downscale back, and
    feather-blend onto the original image.

    Final cleanup pass: when full-image generation has only-just-distinct
    faces, this stage replaces each face with a much more faithful version
    of the right identity, since at the crop level the LoRA is competing
    with no one.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline

    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )

    adapter_names = list(identities.keys())

    faces = face_scorer.detect_faces(image)
    assignments = face_scorer.assign_faces_to_identities(faces, identity_regions)

    out = image.copy()
    for identity_id, face in assignments.items():
        if face is None:
            print(f"[face_refine] skip {identity_id} — no face detected")
            continue

        x1, y1, x2, y2 = _expand_bbox(face["bbox"], crop_pad_ratio, image.width, image.height)
        crop = out.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        crop_hr = crop.resize((1024, 1024), Image.LANCZOS)

        # Activate ONLY this identity's LoRA at high scale
        weights = [1.4 if k == identity_id else 0.0 for k in adapter_names]
        pipe.set_adapters(adapter_names, adapter_weights=weights)

        meta = identities[identity_id]
        prompt = (
            f"close-up portrait of {meta.get('visual_description', '')}, "
            f"{meta.get('lora_trigger', '')}, sharp focus, photorealistic"
        ).strip().strip(",")
        negative = _BASE_NEGATIVE
        nf = meta.get("negative_features")
        if nf:
            negative = negative + ", " + nf

        generator = torch.Generator(device="cpu").manual_seed(seed + hash(identity_id) % 10_000)
        print(f"[face_refine] refining {identity_id}: bbox=({x1},{y1},{x2},{y2}) → 1024x1024")
        new_hr = img2img(
            prompt=prompt,
            negative_prompt=negative,
            image=crop_hr,
            strength=refine_strength,
            num_inference_steps=refine_steps,
            generator=generator,
        ).images[0]

        new_crop = new_hr.resize((cw, ch), Image.LANCZOS)
        mask = _feather_mask(new_crop.size, feather=feather)
        out.paste(new_crop, (x1, y1), mask)

    return out


# ── hair-specific refinement (DARF v6) ───────────────────────────────

def _expand_bbox_for_hair(
    bbox, image_w, image_h,
    horiz_pad: float = 0.6,
    up_pad: float = 1.6,
    down_pad: float = 0.2,
) -> tuple[int, int, int, int]:
    """
    Asymmetrically expand a face bbox to capture hair: extend MORE upward
    (where hair sits) and a bit to the sides, less downward. Default ratios
    grow up by 1.6x bbox height (typical hair extends well above the head)
    and laterally by 0.6x face width.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    nx1 = max(0, int(x1 - bw * horiz_pad))
    nx2 = min(image_w, int(x2 + bw * horiz_pad))
    ny1 = max(0, int(y1 - bh * up_pad))     # ← much more upward
    ny2 = min(image_h, int(y2 + bh * down_pad))
    return nx1, ny1, nx2, ny2


def hair_local_refine(
    pipe: StableDiffusionXLControlNetPipeline,
    identities: dict,
    image: Image.Image,
    face_scorer,
    identity_regions: dict,
    refine_strength: float = 0.55,
    refine_steps: int = 28,
    seed: int = 42,
    feather: int = 36,
    horiz_pad: float = 0.6,
    up_pad: float = 1.6,
    down_pad: float = 0.2,
    lora_alpha: float = 1.6,
) -> Image.Image:
    """
    Same idea as face_local_refine but the crop is asymmetrically expanded
    upward and laterally to include the hair, and the prompt + LoRA scaling
    are tuned to refine HAIR rather than face geometry.

    Useful when face_local_refine has already restored the right identity's
    facial features but the hair colour/texture still inherits from the
    rival LoRA's leakage (e.g. Daenerys keeping a brownish tint).
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline

    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )

    adapter_names = list(identities.keys())
    faces = face_scorer.detect_faces(image)
    assignments = face_scorer.assign_faces_to_identities(faces, identity_regions)

    out = image.copy()
    for identity_id, face in assignments.items():
        if face is None:
            print(f"[hair_refine] skip {identity_id} — no face detected")
            continue

        x1, y1, x2, y2 = _expand_bbox_for_hair(
            face["bbox"], image.width, image.height,
            horiz_pad=horiz_pad, up_pad=up_pad, down_pad=down_pad,
        )
        crop = out.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        crop_hr = crop.resize((1024, 1024), Image.LANCZOS)

        # Sole LoRA at very high alpha for hair texture
        weights = [lora_alpha if k == identity_id else 0.0 for k in adapter_names]
        pipe.set_adapters(adapter_names, adapter_weights=weights)

        meta = identities[identity_id]
        # Hair-focused prompt — emphasise hair attributes, then identity
        hair_phrases = []
        for ph in meta.get("attention_phrases", []):
            if "hair" in ph.lower():
                hair_phrases.append(ph)
        if not hair_phrases:
            hair_phrases = [meta.get("visual_description", "")]
        hair_text = ", ".join(hair_phrases)

        prompt = (
            f"close-up portrait emphasising hair texture, {hair_text}, "
            f"{meta.get('lora_trigger', '')}, "
            f"natural lighting, sharp focus, photorealistic"
        ).strip().strip(",")

        negative = _BASE_NEGATIVE
        nf = meta.get("negative_features")
        if nf:
            negative = negative + ", " + nf

        generator = torch.Generator(device="cpu").manual_seed(seed + hash(identity_id) % 10_000)
        print(
            f"[hair_refine] {identity_id}: bbox=({x1},{y1},{x2},{y2}) "
            f"→ 1024x1024  α={lora_alpha}  strength={refine_strength}"
        )
        new_hr = img2img(
            prompt=prompt,
            negative_prompt=negative,
            image=crop_hr,
            strength=refine_strength,
            num_inference_steps=refine_steps,
            generator=generator,
        ).images[0]

        new_crop = new_hr.resize((cw, ch), Image.LANCZOS)
        mask = _feather_mask(new_crop.size, feather=feather)
        out.paste(new_crop, (x1, y1), mask)

    return out


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
