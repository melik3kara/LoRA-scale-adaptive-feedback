# Adaptive LoRA-Scale Feedback for Identity-Preserving Generation

> **Deep Generative Networks — Course Project**
> Team: [Your Name] · İdil Bilge Öziş
> Companion team: Serdar Kara · Melike Kara · Alp Eren Köken

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How the Two Sub-Projects Connect](#how-the-two-sub-projects-connect)
3. [Our Part: LoRA-Scale Adaptive Feedback](#our-part-lora-scale-adaptive-feedback)
4. [Repository Structure](#repository-structure)
5. [Setup & Installation](#setup--installation)
6. [Implementation Guide](#implementation-guide)
   - [Phase 1 — Static Sweep (Baseline)](#phase-1--static-sweep-baseline)
   - [Phase 2 — Proxy Identity Scorer](#phase-2--proxy-identity-scorer)
   - [Phase 3 — Adaptive Feedback Loop](#phase-3--adaptive-feedback-loop)
   - [Phase 4 — Logging & Evaluation](#phase-4--logging--evaluation)
7. [Experiments](#experiments)
8. [Metrics](#metrics)
9. [Expected Results](#expected-results)
10. [Deliverables](#deliverables)
11. [References](#references)

---

## Project Overview

This project investigates **identity preservation under multi-control inference** in diffusion models. When a face LoRA (identity conditioning) and ControlNet (pose conditioning) are applied together, they compete — strong pose control causes the generated face to drift away from the reference identity. We call this **identity degradation under interference**.

Our approach is to treat this as a **closed-loop control problem**: instead of choosing a fixed LoRA scale and hoping it works, we generate an image, measure how well identity is preserved, and dynamically adjust the LoRA scale before retrying. This is done entirely at inference time, with no retraining.

The full project is split into two parallel sub-projects that solve the same problem with different mechanisms:

| Sub-project | Team | Mechanism | Tuning target |
|---|---|---|---|
| **LoRA-Scale Feedback** (this repo) | [Your Name] + İdil | CLIP proxy scorer → bump α | LoRA adapter weight at generation |
| **IP-Adapter + ArcFace Feedback** | Serdar + Melike + Alp Eren | ArcFace sim → adjust IP scale | Reference image conditioning strength |

At evaluation time, both methods are compared on the same metrics (ArcFace similarity + landmark pose error), enabling a direct comparison of generation-time vs. conditioning-time control.

---

## How the Two Sub-Projects Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                        Shared pipeline                          │
│                                                                 │
│   Face LoRA ──┐                                                 │
│   ControlNet ─┼──► Diffusion generator ──► Generated images    │
│   Prompt ─────┘                                  │              │
│                                                  │              │
│         ┌────────────────────────────────────────┘              │
│         │                                                       │
│         ▼                          ▼                            │
│  [Our part]                 [Other team]                        │
│  CLIP proxy scorer          ArcFace similarity                  │
│  → adjust LoRA scale α      → adjust IP-Adapter scale          │
│                                                                 │
│         └──────────────────────────┘                            │
│                        │                                        │
│                        ▼                                        │
│              Shared evaluation suite                            │
│         ArcFace score · Landmark error · Qualitative grids      │
└─────────────────────────────────────────────────────────────────┘
```

Our Phase 1 static sweep (fixed LoRA scale × control strength grid) generates the shared baseline dataset that both teams reference. The degradation curves from this sweep justify and motivate the adaptive approach.

---

## Our Part: LoRA-Scale Adaptive Feedback

### Core research question

> Does dynamically increasing LoRA scale during inference recover identity drift caused by strong pose conditioning — and what is the cost in pose fidelity?

### The feedback loop in plain language

1. Start with a moderate LoRA scale (e.g. α = 0.5)
2. Generate an image with the current α
3. Measure how similar the generated face is to the reference face (using a fast CLIP proxy)
4. If similarity is above threshold → accept the image, log results, done
5. If similarity is below threshold → bump α up by a fixed step Δ, go back to step 2
6. If max iterations reached → log failure, move on

This means the system self-corrects: when pose conditioning is strong and identity drifts, the loop increases identity conditioning strength to compensate.

### Why CLIP instead of ArcFace for the inner loop

ArcFace is the gold-standard face recognition metric and the other team uses it for final evaluation. We use a CLIP image embedding similarity as the **control signal** inside our loop because:

- CLIP loads once and runs in ~50ms per image on a GPU
- It does not require face detection or alignment as a preprocessing step
- It is sensitive enough to catch clear identity drift
- ArcFace scores are still computed at evaluation time for fair comparison

---

## Repository Structure

```
project/
│
├── README.md                    ← this file
│
├── configs/
│   └── sweep_config.yaml        ← LoRA scales, control strengths, thresholds
│
├── data/
│   ├── reference_faces/         ← reference identity images (input)
│   ├── pose_images/             ← ControlNet conditioning images
│   └── results/                 ← generated images + logs (output)
│
├── src/
│   ├── pipeline.py              ← diffusion pipeline setup (LoRA + ControlNet)
│   ├── scorer.py                ← CLIP proxy identity scorer
│   ├── adaptive_loop.py         ← the adaptive feedback controller
│   ├── sweep.py                 ← static grid sweep (Phase 1)
│   └── evaluate.py              ← final metric computation (ArcFace + landmarks)
│
├── notebooks/
│   ├── 01_static_sweep.ipynb    ← run and visualise Phase 1
│   ├── 02_adaptive_loop.ipynb   ← run and visualise Phase 3
│   └── 03_comparison.ipynb      ← compare fixed vs adaptive results
│
└── requirements.txt
```

---

## Setup & Installation

### Requirements

- Python 3.10+
- CUDA GPU with at least 12 GB VRAM (SDXL) or 8 GB (SD 1.5)
- Git LFS (for model weights, if stored locally)

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft
pip install controlnet-aux
pip install git+https://github.com/openai/CLIP.git
pip install insightface onnxruntime-gpu   # for ArcFace evaluation
pip install opencv-python mediapipe       # for landmark detection
pip install pandas matplotlib seaborn     # for analysis
```

### Download models

```python
from diffusers import StableDiffusionXLPipeline, ControlNetModel

# Base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

# ControlNet (OpenPose for SDXL)
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
)
```

You will also need a face LoRA fine-tuned on your reference identity. If you do not have one, train it using `kohya_ss` or `diffusers` DreamBooth scripts, or download a public face LoRA from Civitai.

---

## Implementation Guide

### Phase 1 — Static Sweep (Baseline)

**Goal:** Generate a grid of images across LoRA scale × control strength combinations. This is the dataset that proves identity degradation is a real problem and gives us the degradation curves.

**File:** `src/sweep.py`

```python
import torch
import pandas as pd
from diffusers import StableDiffusionXLPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

LORA_SCALES   = [0.4, 0.6, 0.8, 1.0]
CTRL_SCALES   = [0.3, 0.5, 0.7, 0.9]
PROMPT        = "portrait photo of a person, soft natural lighting, high quality"
SEED          = 42

def run_static_sweep(pipe, pose_image, output_dir):
    records = []

    for lora_s in LORA_SCALES:
        for ctrl_s in CTRL_SCALES:
            # Set LoRA scale
            pipe.set_adapters(["face_lora"], adapter_weights=[lora_s])

            # Generate
            generator = torch.manual_seed(SEED)
            img = pipe(
                prompt=PROMPT,
                image=pose_image,
                controlnet_conditioning_scale=ctrl_s,
                generator=generator,
                num_inference_steps=30,
            ).images[0]

            # Save
            fname = f"lora{lora_s}_ctrl{ctrl_s}.png"
            img.save(f"{output_dir}/{fname}")

            records.append({
                "lora_scale": lora_s,
                "ctrl_scale": ctrl_s,
                "filename": fname,
            })

    return pd.DataFrame(records)
```

After running the sweep, compute CLIP and ArcFace similarity for every image against the reference. Plot identity score vs. LoRA scale for each control strength — this is your Figure 1.

---

### Phase 2 — Proxy Identity Scorer

**Goal:** Build a fast similarity function that the adaptive loop can call on every iteration.

**File:** `src/scorer.py`

```python
import torch
import clip
from PIL import Image

class CLIPIdentityScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def embed(self, image: Image.Image) -> torch.Tensor:
        t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(t)
        return emb / emb.norm(dim=-1, keepdim=True)

    def similarity(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        ref_emb = self.embed(ref_image)
        gen_emb = self.embed(gen_image)
        return (ref_emb @ gen_emb.T).item()
```

**Calibrating the threshold:** Run the scorer on your static sweep images and compare CLIP similarity against ArcFace similarity. Find the CLIP value that corresponds to ArcFace ≥ 0.5 (acceptable identity) — use this as your loop threshold. Typically this lands around 0.62–0.68.

---

### Phase 3 — Adaptive Feedback Loop

**Goal:** The core contribution. Dynamically adjusts LoRA scale α until identity similarity meets the threshold, or max iterations is reached.

**File:** `src/adaptive_loop.py`

```python
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
from PIL import Image
from src.scorer import CLIPIdentityScorer

@dataclass
class IterationRecord:
    iteration: int
    alpha: float
    clip_sim: float
    accepted: bool

@dataclass
class AdaptiveResult:
    image: Image.Image
    history: List[IterationRecord]
    final_alpha: float
    final_sim: float
    status: str   # "success" | "max_iters_reached"

def adaptive_generate(
    pipe,
    scorer: CLIPIdentityScorer,
    ref_image: Image.Image,
    pose_image: Image.Image,
    prompt: str,
    ctrl_scale: float = 0.7,
    threshold: float = 0.65,
    alpha_init: float = 0.5,
    delta: float = 0.15,
    max_iters: int = 4,
    seed: int = 42,
) -> AdaptiveResult:
    """
    Adaptive LoRA-scale feedback loop.

    Starts at alpha_init and bumps alpha by delta each iteration
    until CLIP identity similarity >= threshold or max_iters is reached.
    """
    alpha = alpha_init
    history = []

    for i in range(max_iters):
        # Set LoRA adapter weight for this iteration
        pipe.set_adapters(["face_lora"], adapter_weights=[alpha])

        # Generate image with current alpha
        generator = torch.manual_seed(seed)
        img = pipe(
            prompt=prompt,
            image=pose_image,
            controlnet_conditioning_scale=ctrl_scale,
            generator=generator,
            num_inference_steps=30,
        ).images[0]

        # Measure identity similarity
        sim = scorer.similarity(ref_image, img)

        record = IterationRecord(
            iteration=i,
            alpha=alpha,
            clip_sim=sim,
            accepted=(sim >= threshold),
        )
        history.append(record)

        if sim >= threshold:
            return AdaptiveResult(
                image=img,
                history=history,
                final_alpha=alpha,
                final_sim=sim,
                status="success",
            )

        # Bump alpha, clamp to 1.0
        alpha = min(alpha + delta, 1.0)

    # Max iterations reached — return best attempt
    return AdaptiveResult(
        image=img,
        history=history,
        final_alpha=alpha,
        final_sim=sim,
        status="max_iters_reached",
    )
```

**Hyperparameters to tune:**

| Parameter | Default | What it controls |
|---|---|---|
| `alpha_init` | 0.5 | Starting LoRA scale — lower means we let the loop do more work |
| `delta` | 0.15 | Step size per iteration — larger = faster but coarser |
| `threshold` | 0.65 | CLIP sim cutoff for "acceptable" identity |
| `max_iters` | 4 | Safety cap — prevents infinite loops |
| `ctrl_scale` | 0.7 | ControlNet pose strength — the interference we are fighting |

---

### Phase 4 — Logging & Evaluation

**Goal:** Produce a structured dataset that can be compared directly against the other team's results.

**File:** `src/evaluate.py`

Every run, whether fixed or adaptive, produces a record like this:

```python
{
    "method":          "adaptive_lora",   # or "fixed_lora"
    "ref_identity":    "person_A",
    "ctrl_scale":      0.7,
    "seed":            42,

    # Adaptive-only fields (fixed method: iterations=1, final_alpha=fixed value)
    "iterations":      3,
    "alpha_trajectory": [0.5, 0.65, 0.8],
    "clip_trajectory":  [0.58, 0.62, 0.71],
    "final_alpha":     0.8,
    "status":          "success",

    # Final metrics (computed once, after loop exits)
    "final_clip_sim":    0.71,
    "final_arcface_sim": 0.68,    # ArcFace on final image vs. reference
    "pose_error_px":     4.2,     # Mean landmark deviation in pixels
    "accepted":          True,
}
```

Compute ArcFace and landmark error on the final accepted image using:

```python
import insightface
import numpy as np

app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def arcface_similarity(ref_img_np, gen_img_np):
    ref_faces = app.get(ref_img_np)
    gen_faces = app.get(gen_img_np)
    if not ref_faces or not gen_faces:
        return None
    ref_emb = ref_faces[0].normed_embedding
    gen_emb = gen_faces[0].normed_embedding
    return float(np.dot(ref_emb, gen_emb))
```

---

## Experiments

### Experiment 1 — Degradation curves (Phase 1 output)

Plot ArcFace similarity vs. LoRA scale for each control strength (0.3, 0.5, 0.7, 0.9). Expected shape: identity similarity drops as control strength increases, especially at low LoRA scales. This is the motivation for the adaptive approach.

### Experiment 2 — Fixed vs. adaptive at matched pose strength

For each control strength value, compare:
- Fixed LoRA at α = 0.8 (strong, static)
- Adaptive starting at α = 0.5 with Δ = 0.15

Metrics: final ArcFace similarity, pose landmark error, number of iterations used.

### Experiment 3 — Threshold sensitivity

Run adaptive loop with threshold ∈ {0.55, 0.60, 0.65, 0.70}. Measure: how often does each threshold lead to success, how many iterations on average, and what is the trade-off in pose error?

### Experiment 4 — Cross-method comparison (with the other team)

Same reference identity, same pose image, same control strength. Compare:
- Our adaptive LoRA method
- Other team's ArcFace + IP-Adapter method

Use the shared final metrics (ArcFace + landmark error) for fair comparison.

---

## Metrics

| Metric | Tool | What it measures |
|---|---|---|
| CLIP cosine similarity | OpenAI CLIP ViT-B/32 | Fast identity proxy for the control loop |
| ArcFace cosine similarity | InsightFace | Final identity preservation quality |
| Landmark pose error (px) | MediaPipe Face Mesh | Deviation from target pose, in pixels |
| Iterations to convergence | Logged internally | Computational cost of the adaptive loop |
| Success rate | Logged internally | Fraction of runs that meet the threshold |

---

## Expected Results

- The static sweep will show a clear trade-off: higher control strength → lower identity similarity at any fixed LoRA scale.
- The adaptive loop should recover identity (higher ArcFace) compared to a fixed low α, at the cost of 1–3 additional forward passes.
- At very high control strengths (≥ 0.9), even α = 1.0 may be insufficient — the loop will hit max_iters. This failure mode is itself informative.
- Pose error may increase slightly as α grows — this is the core identity vs. pose trade-off we are quantifying.
- Comparison with the other team: neither method will dominate on all metrics; each has a regime where it works better. This is the interesting finding.

---

## Deliverables

- [ ] Static sweep images (4 LoRA scales × 4 control strengths × N identities)
- [ ] Degradation curves (Figure 1 — shared with other team)
- [ ] Adaptive loop implementation (`src/adaptive_loop.py`)
- [ ] Structured results CSV with all metrics per run
- [ ] Qualitative grid showing identity drift at fixed α vs. recovery with adaptive α
- [ ] Comparison table: adaptive LoRA vs. ArcFace IP-Adapter method
- [ ] Notebook walkthroughs (`notebooks/`)

---

## References

1. Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
2. Zhang, L., Rao, A., Agrawala, M. *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet).* ICCV 2023.
3. Ye, H. et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.* arXiv:2308.06721, 2023.
4. Deng, J. et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* TPAMI 2022.
5. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
6. Stability AI. *Stable Diffusion XL Base 1.0.* Hugging Face model card.