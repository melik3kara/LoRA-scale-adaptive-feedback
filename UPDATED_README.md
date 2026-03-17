# Adaptive LoRA-Scale Feedback for Multi-Identity Generation

> **Deep Generative Networks — Course Project**
> **Our sub-team:** [Your Name] · İdil Bilge Öziş
> **Companion sub-team:** Serdar Kara · Melike Kara · Alp Eren Köken

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Shared Identity Set](#shared-identity-set)
3. [How the Two Sub-Projects Connect](#how-the-two-sub-projects-connect)
4. [Our Part: Multi-LoRA Adaptive Feedback](#our-part-multi-lora-adaptive-feedback)
5. [Repository Structure](#repository-structure)
6. [Setup & Installation](#setup--installation)
7. [Implementation Guide](#implementation-guide)
   - [Phase 1 — Static Sweep (Baseline)](#phase-1--static-sweep-baseline)
   - [Phase 2 — Proxy Identity Scorer (Per-Identity)](#phase-2--proxy-identity-scorer-per-identity)
   - [Phase 3 — Per-Identity Adaptive Feedback Loop](#phase-3--per-identity-adaptive-feedback-loop)
   - [Phase 4 — Leakage Analysis](#phase-4--leakage-analysis)
   - [Phase 5 — Logging & Evaluation](#phase-5--logging--evaluation)
8. [Experiments](#experiments)
9. [Metrics](#metrics)
10. [Expected Results](#expected-results)
11. [Deliverables](#deliverables)
12. [References](#references)

---

## Project Overview

This project investigates **identity preservation under multi-control inference** in diffusion models, specifically when **multiple identity LoRAs** are active simultaneously alongside a pose ControlNet.

When 2–3 face LoRAs and a ControlNet are composed together at inference time, two distinct failure modes emerge:

1. **Identity–pose conflict** — strong pose conditioning pulls faces away from their reference identities
2. **Inter-identity leakage** — LoRA feature spaces bleed into each other, causing identity blending between the two or three people in the scene

Our approach addresses both problems with a **per-identity adaptive feedback loop**: after each generation, we score each face region independently against its reference identity and selectively bump the LoRA scale of whichever identity drifted the most. This is training-free and runs entirely at inference time.

### Why multiple LoRAs make this harder

With a single LoRA, increasing α always helps identity. With multiple LoRAs active simultaneously:

- Increasing α for identity A may cause A's features to bleed into identity B's face region
- The LoRAs share the same UNet attention layers, so their contributions are not spatially isolated by default
- Bumping one α does not guarantee it leaves the other identity unaffected

This makes adaptive scaling a **joint optimisation problem per generation**, not a simple per-identity threshold check.

---

## Shared Identity Set

Both sub-teams use the **same pretrained LoRAs and the same reference face images** so that final metrics are directly comparable.

> **Action item for the full team:** Fill in the table below once LoRAs are agreed.

| ID | Character / Person | LoRA source | Trigger word | Reference image |
|----|--------------------|-------------|--------------|-----------------|
| A  | TBD                | Civitai / custom | TBD      | `data/reference_faces/A.png` |
| B  | TBD                | Civitai / custom | TBD      | `data/reference_faces/B.png` |
| C  | TBD                | Civitai / custom | TBD      | `data/reference_faces/C.png` |

The companion team's IP-Adapter reference images are sourced from the same `data/reference_faces/` folder. ArcFace evaluation embeddings are computed once from these images and shared across both sub-teams.

---

## How the Two Sub-Projects Connect

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Shared foundation                               │
│                                                                      │
│  LoRA_A + LoRA_B [+ LoRA_C] ──┐                                     │
│  ControlNet (pose) ───────────┼──► Diffusion generator              │
│  Shared prompt ───────────────┘         │                           │
│                                         │                           │
│              ┌──────────────────────────┘                           │
│              │   Same images · same reference faces                 │
│              │                                                      │
│        ┌─────┴──────┐                ┌──────────────┐              │
│        │  Our part  │                │  Other team  │              │
│        │            │                │              │              │
│  CLIP proxy score   │          ArcFace similarity   │              │
│  per identity face  │          per identity region  │              │
│  → adjust LoRA α_i  │          → adjust IP scale_i  │              │
│        └─────┬──────┘                └──────┬───────┘              │
│              │                              │                       │
│              └──────────────┬───────────────┘                       │
│                             ▼                                        │
│                  Shared evaluation suite                             │
│         ArcFace · Landmark error · Leakage score · Qual. grids      │
└──────────────────────────────────────────────────────────────────────┘
```

**Data shared between the teams:**

| What | Direction | Format |
|------|-----------|--------|
| LoRA identities + reference images | Agreed jointly | `data/reference_faces/` |
| Static sweep results (Phase 1) | Our team → shared | CSV + images |
| Degradation + leakage curves | Our team → shared | Figures |
| Final generated images | Our team → other team | PNG |
| ArcFace scores on our images | Other team → us | CSV column |
| Final comparison table | Both teams jointly | Report |

---

## Our Part: Multi-LoRA Adaptive Feedback

### Core research question

> When 2–3 identity LoRAs are active simultaneously with pose conditioning, can per-identity adaptive LoRA scaling recover degraded identities without increasing leakage into the neighbouring face regions?

### The feedback loop

```
Initialise: α_A = α_B = α_C = alpha_init

For each iteration (up to max_iters):
  1. Generate image with current [α_A, α_B, α_C]
  2. Detect and crop each face region (left-to-right order)
  3. Score each face crop against its reference with CLIP
  4. For each identity i where sim_i < threshold:
       α_i ← min(α_i + Δ, alpha_max)
  5. If all identities meet threshold → accept, log, stop
  6. If max_iters reached → log partial/full failure, stop
```

Each identity is updated **independently** — identity A might converge in iteration 1 while B needs 3 more.

### Leakage as a new failure mode

Beyond identity–pose conflict, we track **inter-identity leakage**: when a face region scores higher for the *wrong* reference identity than for its own.

```
leakage(face_B) = sim(face_region_B, reference_A) − sim(face_region_B, reference_B)
```

A positive value means the wrong identity is winning that region. We measure this across all sweep conditions to document how leakage evolves with LoRA scale.

---

## Repository Structure

```
project/
│
├── README.md
│
├── configs/
│   ├── identities.yaml          ← LoRA paths, reference images, trigger words
│   └── sweep_config.yaml        ← scales, control strengths, thresholds, seeds
│
├── data/
│   ├── reference_faces/         ← one reference image per identity (A, B, C)
│   ├── pose_images/             ← ControlNet conditioning (multi-person poses)
│   └── results/
│       ├── sweep/               ← static sweep outputs
│       ├── adaptive/            ← adaptive loop outputs
│       └── logs/                ← CSV logs for all runs
│
├── src/
│   ├── pipeline.py              ← pipeline setup: multi-LoRA + ControlNet
│   ├── scorer.py                ← CLIP proxy scorer + per-identity face crop
│   ├── adaptive_loop.py         ← per-identity adaptive feedback controller
│   ├── sweep.py                 ← static grid sweep (Phase 1)
│   ├── leakage.py               ← inter-identity leakage measurement
│   └── evaluate.py              ← final ArcFace + landmark metrics
│
├── notebooks/
│   ├── 01_static_sweep.ipynb
│   ├── 02_adaptive_loop.ipynb
│   ├── 03_leakage_analysis.ipynb
│   └── 04_comparison.ipynb
│
└── requirements.txt
```

---

## Setup & Installation

### Requirements

- Python 3.10+
- CUDA GPU with ≥ 12 GB VRAM (SDXL) or ≥ 8 GB (SD 1.5)
- Agreed LoRA `.safetensors` files placed under `data/loras/`

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft
pip install controlnet-aux
pip install git+https://github.com/openai/CLIP.git
pip install insightface onnxruntime-gpu    # ArcFace evaluation
pip install opencv-python mediapipe        # face detection + landmark error
pip install pandas matplotlib seaborn      # analysis and plotting
```

### Configure identities

`configs/identities.yaml`:

```yaml
identities:
  A:
    name: "Character A"
    lora_path: "data/loras/character_a.safetensors"
    lora_trigger: "photo of charA person"
    reference_image: "data/reference_faces/A.png"
  B:
    name: "Character B"
    lora_path: "data/loras/character_b.safetensors"
    lora_trigger: "photo of charB person"
    reference_image: "data/reference_faces/B.png"
  C:
    name: "Character C"
    lora_path: "data/loras/character_c.safetensors"
    lora_trigger: "photo of charC person"
    reference_image: "data/reference_faces/C.png"
```

---

## Implementation Guide

### Phase 1 — Static Sweep (Baseline)

**Goal:** Generate a grid of multi-identity images at fixed LoRA scale × control strength combinations. All active LoRAs share the same scale value per run. This produces the degradation curves and leakage profiles that motivate the adaptive approach.

**File:** `src/sweep.py`

```python
import torch
import pandas as pd

LORA_SCALES = [0.4, 0.6, 0.8, 1.0]   # applied equally to all active LoRAs
CTRL_SCALES = [0.3, 0.5, 0.7, 0.9]
SEEDS       = [42, 123, 777]           # 3 seeds per combo for variance

def run_static_sweep(pipe, identities, pose_image, output_dir):
    records = []

    for lora_s in LORA_SCALES:
        for ctrl_s in CTRL_SCALES:
            for seed in SEEDS:
                adapter_names   = list(identities.keys())
                adapter_weights = [lora_s] * len(adapter_names)
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

                triggers = " and ".join(
                    identities[k]["lora_trigger"] for k in adapter_names
                )
                prompt = f"portrait photo of {triggers}, soft natural lighting"

                img = pipe(
                    prompt=prompt,
                    image=pose_image,
                    controlnet_conditioning_scale=ctrl_s,
                    generator=torch.manual_seed(seed),
                    num_inference_steps=30,
                ).images[0]

                fname = f"lora{lora_s}_ctrl{ctrl_s}_seed{seed}.png"
                img.save(f"{output_dir}/{fname}")
                records.append({
                    "lora_scale": lora_s, "ctrl_scale": ctrl_s,
                    "seed": seed, "filename": fname,
                })

    return pd.DataFrame(records)
```

After running the sweep, compute CLIP and ArcFace similarity + leakage for every image. These become your Figures 1 and 2.

---

### Phase 2 — Proxy Identity Scorer (Per-Identity)

**Goal:** For a multi-person generated image, detect each face region, crop it, and score each crop against **all** reference identities. This gives both self-similarity (is face A actually A?) and cross-similarity (does face A accidentally look like B?).

**File:** `src/scorer.py`

```python
import torch
import clip
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

class PerIdentityScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def detect_faces(self, image: Image.Image):
        """Returns list of face crops sorted left-to-right."""
        img_np = np.array(image)
        results = self.detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if not results.detections:
            return []
        h, w = img_np.shape[:2]
        faces = []
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))
            faces.append({"bbox": (x1, y1, x2, y2),
                          "crop": Image.fromarray(img_np[y1:y2, x1:x2])})
        faces.sort(key=lambda f: f["bbox"][0])
        return faces

    def embed(self, image: Image.Image) -> torch.Tensor:
        t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(t)
        return emb / emb.norm(dim=-1, keepdim=True)

    def score_all(self, gen_image: Image.Image, reference_images: dict):
        """
        Returns:
          scores  — { face_idx: { identity_id: clip_sim } }
          faces   — list of detected face dicts (bbox + crop)
        """
        faces = self.detect_faces(gen_image)
        ref_embs = {k: self.embed(v) for k, v in reference_images.items()}
        scores = {}
        for i, face in enumerate(faces):
            face_emb = self.embed(face["crop"])
            scores[i] = {k: (face_emb @ e.T).item() for k, e in ref_embs.items()}
        return scores, faces
```

**Calibrating the threshold:** Run the scorer on all static sweep images. Plot CLIP self-similarity against ArcFace similarity. Find the CLIP value where ArcFace ≥ 0.5 — this is your loop threshold, typically around 0.62–0.68.

---

### Phase 3 — Per-Identity Adaptive Feedback Loop

**Goal:** The core contribution. Each identity maintains its own α, updated independently each iteration.

**File:** `src/adaptive_loop.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image
import torch

@dataclass
class IdentityState:
    identity_id: str
    alpha: float
    history: List[dict] = field(default_factory=list)
    converged: bool = False

@dataclass
class AdaptiveResult:
    image: Image.Image
    identity_states: Dict[str, IdentityState]
    total_iterations: int
    status: str   # "all_converged" | "partial" | "none_converged"

def multi_lora_adaptive_generate(
    pipe,
    scorer,
    reference_images: dict,       # { "A": PIL.Image, "B": PIL.Image, ... }
    pose_image: Image.Image,
    prompt: str,
    ctrl_scale: float = 0.7,
    threshold: float = 0.65,
    alpha_init: float = 0.5,
    delta: float = 0.15,
    alpha_max: float = 1.0,
    max_iters: int = 5,
    seed: int = 42,
) -> AdaptiveResult:

    states = {
        k: IdentityState(identity_id=k, alpha=alpha_init)
        for k in reference_images
    }
    last_image = None

    for iteration in range(max_iters):
        adapter_names   = list(states.keys())
        adapter_weights = [states[k].alpha for k in adapter_names]
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        img = pipe(
            prompt=prompt,
            image=pose_image,
            controlnet_conditioning_scale=ctrl_scale,
            generator=torch.manual_seed(seed),
            num_inference_steps=30,
        ).images[0]
        last_image = img

        scores, faces = scorer.score_all(img, reference_images)
        id_keys = list(states.keys())

        for face_idx, identity_id in enumerate(id_keys):
            if face_idx >= len(scores):
                break
            self_sim = scores[face_idx].get(identity_id, 0.0)
            state = states[identity_id]
            state.history.append({
                "iteration": iteration,
                "alpha": state.alpha,
                "self_sim": self_sim,
                "all_sims": scores[face_idx],
            })
            if self_sim >= threshold:
                state.converged = True

        if all(s.converged for s in states.values()):
            return AdaptiveResult(
                image=img,
                identity_states=states,
                total_iterations=iteration + 1,
                status="all_converged",
            )

        # Only bump unconverged identities
        for identity_id, state in states.items():
            if not state.converged:
                state.alpha = min(state.alpha + delta, alpha_max)

    n_converged = sum(1 for s in states.values() if s.converged)
    status = "partial" if n_converged > 0 else "none_converged"
    return AdaptiveResult(
        image=last_image,
        identity_states=states,
        total_iterations=max_iters,
        status=status,
    )
```

**Key hyperparameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `alpha_init` | 0.5 | Starting scale — lower lets the loop do more work |
| `delta` | 0.15 | Step per iteration — larger = faster but coarser |
| `threshold` | 0.65 | CLIP sim cutoff — calibrate from sweep data |
| `alpha_max` | 1.0 | Hard ceiling per identity |
| `max_iters` | 5 | Safety cap |
| `ctrl_scale` | 0.7 | The interference strength we are fighting |

---

### Phase 4 — Leakage Analysis

**Goal:** Measure inter-identity leakage on all generated images — both sweep and adaptive outputs.

**File:** `src/leakage.py`

```python
def compute_leakage(scores: dict, identity_assignment: dict) -> dict:
    """
    scores              — { face_idx: { identity_id: clip_sim } }
    identity_assignment — { face_idx: correct_identity_id }

    leakage_score > 0 means the wrong identity is winning that face region.
    """
    leakage = {}
    for face_idx, correct_id in identity_assignment.items():
        face_scores = scores.get(face_idx, {})
        if not face_scores:
            continue
        correct_sim = face_scores.get(correct_id, 0.0)
        other_max   = max(
            (v for k, v in face_scores.items() if k != correct_id), default=0.0
        )
        leakage[face_idx] = {
            "correct_id":       correct_id,
            "correct_sim":      correct_sim,
            "max_leakage_sim":  other_max,
            "leakage_score":    other_max - correct_sim,
        }
    return leakage
```

Plot leakage score vs. LoRA scale across all ctrl_scale values. This is your Figure 2.

---

### Phase 5 — Logging & Evaluation

Every run — fixed or adaptive — logs one record with this schema. It is compatible with the other team's log format so both CSVs can be merged for the joint comparison.

```python
{
    # Run identity
    "method":              "adaptive_lora",   # or "fixed_lora"
    "num_identities":      2,                 # 2 or 3
    "identities":          ["A", "B"],
    "ctrl_scale":          0.7,
    "seed":                42,

    # Per-identity final state
    "alpha_A":             0.8,
    "alpha_B":             0.65,
    "clip_sim_A":          0.71,
    "clip_sim_B":          0.68,
    "arcface_sim_A":       0.69,   # computed post-hoc
    "arcface_sim_B":       0.66,
    "leakage_A":          -0.04,   # negative = correct identity winning
    "leakage_B":           0.02,

    # Loop metadata
    "iterations":          3,
    "status":              "all_converged",

    # Pose
    "pose_error_px":       4.8,
}
```

ArcFace scores are computed once after the loop exits using InsightFace, and can be run by either sub-team:

```python
import insightface, numpy as np

app = insightface.app.FaceAnalysis(providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def arcface_sim(ref_np, gen_np):
    ref_faces = app.get(ref_np)
    gen_faces = app.get(gen_np)
    if not ref_faces or not gen_faces:
        return None
    return float(np.dot(
        ref_faces[0].normed_embedding,
        gen_faces[0].normed_embedding
    ))
```

---

## Experiments

### Experiment 1 — Degradation and leakage curves (static sweep)
Sweep LoRA scale × control strength with 2 identities. Plot CLIP self-similarity and leakage score vs. LoRA scale, one line per control strength. This is the shared baseline Figure 1 and 2.

### Experiment 2 — Fixed vs. adaptive (2 identities)
At ctrl_scale = 0.7, compare fixed α = 0.8 vs. adaptive starting at α = 0.5. Metrics: ArcFace per identity, pose error, leakage, iterations used.

### Experiment 3 — 3-identity composition
Repeat Experiment 2 with three LoRAs. Does the adaptive loop still converge? Does leakage increase meaningfully with a third identity?

### Experiment 4 — Asymmetric LoRA strength
Test cases where one LoRA is inherently stronger than another. Does per-identity scaling outperform a shared global scale in these cases?

### Experiment 5 — Cross-method comparison (with other team)
Same identities, poses, and prompts. Adaptive LoRA (ours) vs. ArcFace IP-Adapter (theirs). Joint evaluation on shared metrics.

---

## Metrics

| Metric | Tool | Use |
|--------|------|-----|
| CLIP cosine similarity | OpenAI CLIP ViT-B/32 | Control signal inside loop |
| ArcFace cosine similarity | InsightFace | Final identity quality |
| Inter-identity leakage score | `src/leakage.py` | New phenomenon, per face region |
| Pose landmark error (px) | MediaPipe Face Mesh | Pose fidelity |
| Iterations to convergence | Logged | Computational cost |
| Convergence rate | Logged | Reliability across conditions |

---

## Expected Results

- Static sweep shows clear degradation: ctrl_scale ↑ → identity sim ↓ at any fixed LoRA scale. Leakage increases as LoRA scales approach 1.0.
- Adaptive loop converges for most 2-identity runs at ctrl_scale ≤ 0.7, using 2–3 iterations on average.
- 3-identity composition yields lower convergence rates and higher leakage — this is an expected and reportable finding.
- At ctrl_scale = 0.9, some runs will hit max_iters regardless. Log and report these failure cases.
- Cross-method comparison: our generation-time control and the other team's conditioning-time control will have different strength profiles. Neither dominates on all metrics — that contrast is the interesting finding.

---

## Deliverables

- [ ] `configs/identities.yaml` — agreed LoRA set (shared with other team)
- [ ] Static sweep images + degradation and leakage curve plots
- [ ] `src/adaptive_loop.py` — per-identity adaptive controller
- [ ] Full results CSV (compatible with other team's log schema)
- [ ] Qualitative grids: fixed vs. adaptive, 2-identity and 3-identity
- [ ] Leakage analysis plots
- [ ] Joint comparison table with companion sub-team

---

## References

1. Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
2. Zhang, L., Rao, A., Agrawala, M. *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet).* ICCV 2023.
3. Ye, H. et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.* arXiv:2308.06721, 2023.
4. Deng, J. et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* TPAMI 2022.
5. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
6. Stability AI. *Stable Diffusion XL Base 1.0.* Hugging Face model card.