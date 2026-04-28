# src/bg_latent_lock.py
"""
Background Latent Lock for multi-person diffusion.

Multi-LoRA two-person generation has a recurring failure mode where the
model hallucinates a third figure in regions outside the pose skeletons.
Negative prompts and high ControlNet scale only partially suppress this.

Mechanism: during the first K denoising steps, we replace the background
region of the latent (i.e. outside the union of identity foreground masks)
with the latent it had at step 0. The foreground regions evolve normally.
This freezes the model's freedom to "decide what to draw" in the empty
parts of the canvas during the early/composition phase.

Hooks into diffusers' `callback_on_step_end` API (>= 0.27).

Usage:
    from bg_latent_lock import make_bg_lock_callback

    cb = make_bg_lock_callback(
        foreground_mask=union_pil_mask,
        total_steps=30,
        lock_ratio=0.35,
    )
    image = pipe(
        ...,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
"""

from typing import Callable, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_mask_tensor(mask: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Normalize input mask into (1, 1, H, W) float tensor in [0, 1]."""
    if hasattr(mask, "convert"):  # PIL
        arr = np.array(mask.convert("L"), dtype="float32") / 255.0
        t = torch.from_numpy(arr)
    elif isinstance(mask, np.ndarray):
        t = torch.as_tensor(mask, dtype=torch.float32)
        if t.max() > 1.5:
            t = t / 255.0
    else:
        t = torch.as_tensor(mask, dtype=torch.float32)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return t.clamp(0.0, 1.0)


def union_mask(masks: list, width: int, height: int) -> Image.Image:
    """
    Build a single foreground mask = union of per-identity region masks.
    Accepts either PIL masks or (x1, y1, x2, y2) bounding boxes.
    """
    out = np.zeros((height, width), dtype="float32")
    for m in masks:
        if isinstance(m, tuple) and len(m) == 4:
            x1, y1, x2, y2 = m
            out[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1.0
        elif hasattr(m, "convert"):
            arr = np.array(m.convert("L").resize((width, height)), dtype="float32") / 255.0
            out = np.maximum(out, arr)
        else:
            arr = np.asarray(m, dtype="float32")
            if arr.max() > 1.5:
                arr = arr / 255.0
            if arr.shape != (height, width):
                # PIL resize via tensor
                t = torch.as_tensor(arr).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
                arr = t[0, 0].numpy()
            out = np.maximum(out, arr)
    return Image.fromarray((out * 255).astype("uint8"))


def make_bg_lock_callback(
    foreground_mask: Union[Image.Image, torch.Tensor],
    total_steps: int,
    lock_ratio: float = 0.35,
    feather: int = 4,
) -> Callable:
    """
    Build a SDXL-pipeline callback that locks the background latent for the
    first `lock_ratio * total_steps` denoising steps.

    Args:
        foreground_mask: PIL mask (any size) — white where the model is
                         allowed to evolve, black where the latent must
                         stay at its step-0 value.
        total_steps:     Total denoising steps the pipeline will run.
        lock_ratio:      Fraction of total_steps during which the lock is
                         active (default 0.35 — first ~third of denoising).
        feather:         Pixel-radius Gaussian blur on the mask for a soft
                         transition between locked and free regions.
    """
    base_mask = _to_mask_tensor(foreground_mask)

    if feather > 0:
        sigma = max(1.0, float(feather))
        ksize = int(2 * round(2 * sigma) + 1)
        xs = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
        g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, 1, ksize)
        pad = ksize // 2
        base_mask = F.conv2d(F.pad(base_mask, (pad, pad, 0, 0), mode="replicate"), g)
        base_mask = F.conv2d(F.pad(base_mask, (0, 0, pad, pad), mode="replicate"), g.transpose(2, 3))
        base_mask = base_mask.clamp(0.0, 1.0)

    state = {
        "initial_latents": None,
        "lock_until": int(total_steps * lock_ratio),
        "mask_cache": None,
    }

    def _resize_mask(target_h: int, target_w: int, device, dtype):
        if state["mask_cache"] is not None and state["mask_cache"].shape[-2:] == (target_h, target_w):
            return state["mask_cache"].to(device=device, dtype=dtype)
        m = F.interpolate(base_mask, size=(target_h, target_w),
                          mode="bilinear", align_corners=False)
        state["mask_cache"] = m
        return m.to(device=device, dtype=dtype)

    def callback(pipe, step: int, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        # Capture initial latent on first call
        if state["initial_latents"] is None:
            state["initial_latents"] = latents.detach().clone()

        if step < state["lock_until"]:
            h, w = latents.shape[-2], latents.shape[-1]
            fg = _resize_mask(h, w, latents.device, latents.dtype)
            init = state["initial_latents"]
            if init.shape != latents.shape:
                # batch dim could differ on classifier-free guidance — broadcast
                init = init.expand_as(latents)
            new_latents = fg * latents + (1.0 - fg) * init
            callback_kwargs["latents"] = new_latents

        return callback_kwargs

    callback.__doc__ = (
        f"BG-lock callback: lock background for first {state['lock_until']} "
        f"of {total_steps} steps."
    )
    return callback


def regions_to_union_mask(
    identity_regions: dict,
    width: int,
    height: int,
    padding: int = 0,
) -> Image.Image:
    """
    Convenience: build a union foreground mask from {id: (x1,y1,x2,y2)}.
    `padding` widens each region by N pixels (useful so faces/hair near
    the box edge are not pulled into the locked background).
    """
    boxes = []
    for (x1, y1, x2, y2) in identity_regions.values():
        boxes.append((
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        ))
    return union_mask(boxes, width, height)
