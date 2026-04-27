# src/lora_gate.py
"""
Spatially Gated LoRA Residual Injection (DARF core).

Standard multi-LoRA inference applies each identity's LoRA contribution
globally over the whole UNet output:

    h_l(x) = W_l x + sum_i  alpha_{i,l} * Delta_i,l(x)

This is the failure mode that drives "two Hermiones": both LoRAs touch
every spatial location, so the dominant LoRA wins everywhere — including
inside the other identity's region.

This module monkey-patches every PEFT LoraLayer in the UNet so the LoRA
residual for identity i is multiplied by a spatial mask M_i corresponding
to that identity's region, downsampled to the layer's spatial resolution:

    h_l(x) = W_l x + sum_i  alpha_{i,l} * M_i^{(l)} * Delta_i,l(x)

For attention layers, activations are (B, H*W, C) flattened tokens. For
convolutional layers they are (B, C, H, W). Both shapes are handled.

Usage:
    from lora_gate import set_spatially_gated_lora, remove_spatially_gated_lora

    handles = set_spatially_gated_lora(pipe, identity_masks)   # 2D PIL or tensor masks
    try:
        image = pipe(...).images[0]
    finally:
        remove_spatially_gated_lora(handles)
"""

from typing import Optional
import math
import torch
import torch.nn.functional as F


def _build_mask_pyramid(
    identity_masks: dict,
    latent_size: int = 128,
    feather_ratio: float = 0.06,
    floor: float = 0.15,
) -> dict:
    """
    Build a multi-resolution pyramid of each identity's spatial mask.
    Keys are (h, w) tuples; values are float tensors with shape (1, 1, h, w).

    Hard 0/1 masks at half-image boundaries cause a "diptych" artifact —
    two regions appear as glued-together separate images with mismatched
    lighting/background. Two softeners:
      - feather_ratio: Gaussian-blur the mask with sigma = ratio * latent_size
        so the boundary is a smooth transition rather than a step.
      - floor: minimum mask value so the rival LoRA still has a small
        contribution everywhere, helping the two halves blend stylistically.
    """
    pyramid = {}
    for ident, mask in identity_masks.items():
        # Accept tensor (H, W) or (1, H, W) or (1, 1, H, W); also accept PIL.
        if hasattr(mask, "convert"):  # PIL
            import numpy as np
            arr = np.array(mask.convert("L"), dtype="float32") / 255.0
            t = torch.from_numpy(arr)
        else:
            t = torch.as_tensor(mask, dtype=torch.float32)
        while t.dim() < 4:
            t = t.unsqueeze(0)
        # Resize to latent_size base
        if t.shape[-2:] != (latent_size, latent_size):
            t = F.interpolate(t, size=(latent_size, latent_size), mode="bilinear", align_corners=False)

        # Feather the boundary with a Gaussian blur kernel
        if feather_ratio > 0:
            sigma = max(1.0, feather_ratio * latent_size)
            ksize = int(2 * round(2 * sigma) + 1)
            # Build separable 1D Gaussian kernel and apply twice (h then w)
            xs = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
            g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
            g = (g / g.sum()).view(1, 1, 1, ksize)
            pad = ksize // 2
            t = F.conv2d(F.pad(t, (pad, pad, 0, 0), mode="replicate"), g)
            t = F.conv2d(F.pad(t, (0, 0, pad, pad), mode="replicate"), g.transpose(2, 3))

        # Apply floor — never fully zero out, blend stylistically
        if floor > 0:
            t = floor + (1.0 - floor) * t.clamp(0.0, 1.0)

        pyramid[ident] = t
    return pyramid


def _resolve_hw(spatial_tokens: int) -> tuple[int, int]:
    """Best-effort recovery of (H, W) from a flattened token count."""
    sq = int(math.sqrt(spatial_tokens))
    if sq * sq == spatial_tokens:
        return sq, sq
    # Fall back: assume aspect ratio 1:1 nearest (common for SDXL pipelines)
    for h in range(1, spatial_tokens + 1):
        if spatial_tokens % h == 0:
            w = spatial_tokens // h
            if abs(h - w) <= 1:
                return h, w
    return sq, sq  # give up


def _mask_for_shape(
    pyramid_per_id: dict,
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[dict]:
    """
    Given a layer output shape, produce a dict of identity -> mask tensor that
    can be broadcast-multiplied onto that output.

    Returns None for shapes we don't recognise (non-spatial layers — e.g. norm,
    text-encoder feed-forward) so the caller can fall through to ungated behaviour.
    """
    if len(shape) == 4:
        # Conv2d activation (B, C, H, W)
        h, w = shape[-2], shape[-1]
        out = {}
        for ident, base in pyramid_per_id.items():
            m = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
            out[ident] = m.to(device=device, dtype=dtype)
        return out

    if len(shape) == 3:
        # Attention activation (B, H*W, C)
        n_tokens = shape[1]
        # Skip cross-attention sequence outputs (77 tokens) and other non-spatial
        if n_tokens < 64 or n_tokens > 32768:
            return None
        h, w = _resolve_hw(n_tokens)
        if h * w != n_tokens:
            return None
        out = {}
        for ident, base in pyramid_per_id.items():
            m = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
            # Reshape to (1, H*W, 1) for broadcast over (B, H*W, C)
            m = m.reshape(1, h * w, 1)
            out[ident] = m.to(device=device, dtype=dtype)
        return out

    return None


def _patched_lora_forward(original_forward, lora_layer, mask_pyramid):
    """
    Replacement forward for a PEFT LoraLayer.

    Calls the base layer normally to get the frozen output, then for each
    active adapter computes its LoRA delta separately, applies the spatial
    mask matching that adapter's identity, and sums everything.
    """
    def forward(x, *args, **kwargs):
        # Base output (frozen)
        base_out = lora_layer.base_layer(x, *args, **kwargs)

        active = getattr(lora_layer, "active_adapters", None)
        if not active or not hasattr(lora_layer, "lora_A"):
            return base_out

        # Try to compute per-adapter deltas
        try:
            mask_dict = _mask_for_shape(
                mask_pyramid, base_out.shape, base_out.device, base_out.dtype
            )
        except Exception:
            mask_dict = None

        result = base_out
        for adapter in active:
            if adapter not in lora_layer.lora_A:
                continue

            lora_A   = lora_layer.lora_A[adapter]
            lora_B   = lora_layer.lora_B[adapter]
            # lora_dropout is an nn.ModuleDict (no .get); scaling is a plain dict
            dropout  = (
                lora_layer.lora_dropout[adapter]
                if adapter in lora_layer.lora_dropout
                else torch.nn.Identity()
            )
            scaling  = lora_layer.scaling.get(adapter, 1.0) if isinstance(
                lora_layer.scaling, dict
            ) else 1.0

            try:
                delta = lora_B(lora_A(dropout(x))) * scaling
            except Exception:
                # Fallback — ungated behaviour for this layer/adapter combination
                return original_forward(x, *args, **kwargs)

            if mask_dict is not None and adapter in mask_dict:
                m = mask_dict[adapter]
                if m.shape[-2:] == delta.shape[-2:] or (
                    delta.dim() == 3 and m.shape[1] == delta.shape[1]
                ):
                    delta = delta * m
            result = result + delta

        return result

    return forward


def set_spatially_gated_lora(
    pipe,
    identity_masks: dict,
    latent_size: int = 128,
    feather_ratio: float = 0.06,
    floor: float = 0.15,
) -> list:
    """
    Patch every PEFT LoRA layer in the UNet so its residual is multiplied by
    each identity's spatial mask. Returns a list of (module, original_forward)
    handles that must be passed to remove_spatially_gated_lora to restore
    the original forwards.

    If PEFT internals don't match expectations on a given layer the patch
    falls through to ungated behaviour for that layer, so this is safe to
    apply broadly even when only some layers are LoRA-tuned.
    """
    pyramid = _build_mask_pyramid(identity_masks, latent_size=latent_size)

    handles = []
    n_patched = 0
    for name, module in pipe.unet.named_modules():
        # Detect PEFT LoraLayer instances by attribute signature — avoids
        # importing peft directly so this works across diffusers versions.
        if hasattr(module, "lora_A") and hasattr(module, "base_layer") \
           and hasattr(module, "active_adapters"):
            original = module.forward
            module.forward = _patched_lora_forward(original, module, pyramid)
            handles.append((module, original))
            n_patched += 1

    print(f"[lora_gate] Spatially gated {n_patched} LoRA layers "
          f"(masks @ {latent_size}x{latent_size} latent)")
    return handles


def remove_spatially_gated_lora(handles: list) -> None:
    """Restore original forwards on every patched LoRA layer."""
    for module, original in handles:
        module.forward = original
    print(f"[lora_gate] Restored {len(handles)} LoRA layers to ungated forward")
