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

DARF v2 additions:
  - Per-block floor (down/mid/up): high floor in early/composition layers
    keeps lighting and background coherent; zero floor in late/face-detail
    layers prevents identity feature leakage.
  - Optional timestep decay γ(t): scale the floor over diffusion steps so
    early steps share style globally and late steps isolate identity.
"""

from typing import Optional, Union
import math
import torch
import torch.nn.functional as F


# ── module-level state for timestep decay ──────────────────────────────
# A pipeline callback can update this per step; the patched forward reads it.
_CURRENT_DECAY = 1.0   # 1.0 = no decay


def set_timestep_decay(gamma: float) -> None:
    """Update the global γ(t) multiplier on per-block floor (called from a
    pipeline callback). γ=1.0 means full floor; γ=0.0 disables floor entirely
    (hard mask)."""
    global _CURRENT_DECAY
    _CURRENT_DECAY = float(gamma)


def reset_timestep_decay() -> None:
    set_timestep_decay(1.0)


# ── mask helpers ───────────────────────────────────────────────────────

def _build_mask_pyramid(
    identity_masks: dict,
    latent_size: int = 128,
    feather_ratio: float = 0.06,
) -> dict:
    """
    Build the base spatial mask per identity at `latent_size` resolution.
    Floor is NOT baked in here — it is applied per-layer at forward time so
    it can vary by UNet block (early/mid/late) and by diffusion timestep.
    """
    pyramid = {}
    for ident, mask in identity_masks.items():
        if hasattr(mask, "convert"):  # PIL
            import numpy as np
            arr = np.array(mask.convert("L"), dtype="float32") / 255.0
            t = torch.from_numpy(arr)
        else:
            t = torch.as_tensor(mask, dtype=torch.float32)
        while t.dim() < 4:
            t = t.unsqueeze(0)
        if t.shape[-2:] != (latent_size, latent_size):
            t = F.interpolate(t, size=(latent_size, latent_size),
                              mode="bilinear", align_corners=False)

        if feather_ratio > 0:
            sigma = max(1.0, feather_ratio * latent_size)
            ksize = int(2 * round(2 * sigma) + 1)
            xs = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
            g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
            g = (g / g.sum()).view(1, 1, 1, ksize)
            pad = ksize // 2
            t = F.conv2d(F.pad(t, (pad, pad, 0, 0), mode="replicate"), g)
            t = F.conv2d(F.pad(t, (0, 0, pad, pad), mode="replicate"), g.transpose(2, 3))

        pyramid[ident] = t.clamp(0.0, 1.0)
    return pyramid


def _resolve_hw(spatial_tokens: int) -> tuple[int, int]:
    """Best-effort recovery of (H, W) from a flattened token count."""
    sq = int(math.sqrt(spatial_tokens))
    if sq * sq == spatial_tokens:
        return sq, sq
    for h in range(1, spatial_tokens + 1):
        if spatial_tokens % h == 0:
            w = spatial_tokens // h
            if abs(h - w) <= 1:
                return h, w
    return sq, sq


def _mask_for_shape(
    pyramid_per_id: dict,
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[dict]:
    """Resize the base mask to match this layer's spatial shape."""
    if len(shape) == 4:
        h, w = shape[-2], shape[-1]
        out = {}
        for ident, base in pyramid_per_id.items():
            m = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
            out[ident] = m.to(device=device, dtype=dtype)
        return out

    if len(shape) == 3:
        n_tokens = shape[1]
        if n_tokens < 64 or n_tokens > 32768:
            return None
        h, w = _resolve_hw(n_tokens)
        if h * w != n_tokens:
            return None
        out = {}
        for ident, base in pyramid_per_id.items():
            m = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
            m = m.reshape(1, h * w, 1)
            out[ident] = m.to(device=device, dtype=dtype)
        return out

    return None


# ── per-block floor classification ─────────────────────────────────────

def _block_kind(layer_name: str) -> str:
    """Return one of: 'down', 'mid', 'up', 'other' based on UNet path."""
    if "down_blocks" in layer_name:
        return "down"
    if "mid_block" in layer_name:
        return "mid"
    if "up_blocks" in layer_name:
        return "up"
    return "other"


def _resolve_floor(block_floor: Union[float, dict], kind: str) -> float:
    if isinstance(block_floor, dict):
        return float(block_floor.get(kind, block_floor.get("other", 0.0)))
    return float(block_floor)


# ── patched forward ────────────────────────────────────────────────────

def _patched_lora_forward(original_forward, lora_layer, mask_pyramid, floor_value: float):
    """
    Per-LoRA-layer replacement forward. The closure captures the floor for
    THIS layer (already classified by block) so different UNet depths can
    use different leak amounts.
    """
    def forward(x, *args, **kwargs):
        base_out = lora_layer.base_layer(x, *args, **kwargs)

        active = getattr(lora_layer, "active_adapters", None)
        if not active or not hasattr(lora_layer, "lora_A"):
            return base_out

        try:
            mask_dict = _mask_for_shape(
                mask_pyramid, base_out.shape, base_out.device, base_out.dtype
            )
        except Exception:
            mask_dict = None

        # Effective floor at this step, this layer
        effective_floor = max(0.0, min(1.0, floor_value * _CURRENT_DECAY))

        result = base_out
        for adapter in active:
            if adapter not in lora_layer.lora_A:
                continue

            lora_A = lora_layer.lora_A[adapter]
            lora_B = lora_layer.lora_B[adapter]
            dropout = (
                lora_layer.lora_dropout[adapter]
                if adapter in lora_layer.lora_dropout
                else torch.nn.Identity()
            )
            scaling = lora_layer.scaling.get(adapter, 1.0) if isinstance(
                lora_layer.scaling, dict
            ) else 1.0

            try:
                delta = lora_B(lora_A(dropout(x))) * scaling
            except Exception:
                return original_forward(x, *args, **kwargs)

            if mask_dict is not None and adapter in mask_dict:
                m = mask_dict[adapter]
                # Apply per-layer effective floor: m_eff = floor + (1-floor) * m
                if effective_floor > 0:
                    m = effective_floor + (1.0 - effective_floor) * m
                if m.shape[-2:] == delta.shape[-2:] or (
                    delta.dim() == 3 and m.shape[1] == delta.shape[1]
                ):
                    delta = delta * m
            result = result + delta

        return result

    return forward


# ── public API ─────────────────────────────────────────────────────────

def set_spatially_gated_lora(
    pipe,
    identity_masks: dict,
    latent_size: int = 128,
    feather_ratio: float = 0.06,
    floor: Union[float, dict] = 0.05,
    block_floor: Optional[dict] = None,
) -> list:
    """
    Patch every PEFT LoRA layer in the UNet so its residual is multiplied by
    each identity's spatial mask, with a per-block floor controlling how much
    of the rival adapter still contributes.

    Args:
        pipe:           The diffusers pipeline whose `.unet` will be patched.
        identity_masks: {identity_id: PIL.Image | tensor} foreground masks.
        latent_size:    Resolution of the base mask cache (typically height/8).
        feather_ratio:  Gaussian blur sigma as ratio of `latent_size`.
        floor:          Default minimum mask value per identity (rival's
                        contribution outside its region). Used when
                        `block_floor` does not specify a value for a block.
        block_floor:    Per-block floor mapping, e.g.
                        {"down": 0.15, "mid": 0.08, "up": 0.00}.
                        High floor in early/down layers preserves global
                        composition (lighting, background); zero floor in
                        late/up layers cleanly isolates identity features.

    Returns:
        Handles list — pass to `remove_spatially_gated_lora` to restore.
    """
    if block_floor is None:
        block_floor = {"down": floor, "mid": floor, "up": floor, "other": floor}
    else:
        # Fill missing keys with the scalar default
        block_floor = {
            "down":  block_floor.get("down",  floor if isinstance(floor, float) else 0.0),
            "mid":   block_floor.get("mid",   floor if isinstance(floor, float) else 0.0),
            "up":    block_floor.get("up",    floor if isinstance(floor, float) else 0.0),
            "other": block_floor.get("other", floor if isinstance(floor, float) else 0.0),
        }

    pyramid = _build_mask_pyramid(
        identity_masks, latent_size=latent_size, feather_ratio=feather_ratio
    )

    handles = []
    counts = {"down": 0, "mid": 0, "up": 0, "other": 0}
    for name, module in pipe.unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer") \
           and hasattr(module, "active_adapters"):
            kind = _block_kind(name)
            this_floor = _resolve_floor(block_floor, kind)
            original = module.forward
            module.forward = _patched_lora_forward(original, module, pyramid, this_floor)
            handles.append((module, original))
            counts[kind] += 1

    n_total = sum(counts.values())
    print(
        f"[lora_gate] Spatially gated {n_total} LoRA layers "
        f"(down={counts['down']} floor={block_floor['down']:.2f} | "
        f"mid={counts['mid']} floor={block_floor['mid']:.2f} | "
        f"up={counts['up']} floor={block_floor['up']:.2f}) "
        f"@ {latent_size}x{latent_size} latent"
    )
    return handles


def remove_spatially_gated_lora(handles: list) -> None:
    """Restore original forwards on every patched LoRA layer."""
    for module, original in handles:
        module.forward = original
    reset_timestep_decay()
    print(f"[lora_gate] Restored {len(handles)} LoRA layers to ungated forward")
