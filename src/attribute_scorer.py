# src/attribute_scorer.py
"""
CLIP-based visual attribute scorer.

Crops each identity region from a generated image and computes a margin
between its similarity to the identity's positive_attributes (hair, attire)
and negative_attributes (the OTHER identity's distinctive traits).

A positive margin means the attribute prompt is satisfied — the region looks
like the right person *visually*, even if ArcFace identity matching is still
weak. This catches the "right hair color, wrong face geometry" intermediate
state earlier than ArcFace does.

Usage:
    scorer = AttributeScorer(identities)
    scores = scorer.score_image(image, identity_regions)
    # scores = {
    #     "hermione": {"margin": 0.07, "pos": 0.28, "neg": 0.21},
    #     "daenerys": {"margin": -0.04, ...},
    # }
"""

from typing import Optional
import numpy as np
import torch
from PIL import Image


class AttributeScorer:

    def __init__(
        self,
        identities: dict,
        model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[attr_scorer] Loading CLIP model: {model_id}")
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # Pre-compute text embeddings for each identity's pos/neg attributes
        self.identities = identities
        self.pos_embeds = {}
        self.neg_embeds = {}
        for k, meta in identities.items():
            pos = meta.get("positive_attributes") or []
            neg = meta.get("negative_attributes") or []
            self.pos_embeds[k] = self._encode_texts(pos) if pos else None
            self.neg_embeds[k] = self._encode_texts(neg) if neg else None

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embeds = self.model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds  # [N, D]

    @torch.no_grad()
    def _encode_image_crop(self, crop: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0]  # [D]

    def score_image(self, image: Image.Image, identity_regions: dict) -> dict:
        """
        Returns {identity_id: {"margin": float, "pos": float, "neg": float}}.

        margin = mean(cos(crop, pos)) - mean(cos(crop, neg)).
        Higher = visual attributes match the assigned identity better.
        """
        scores = {}
        for k, (x1, y1, x2, y2) in identity_regions.items():
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(image.width, int(x2)); y2 = min(image.height, int(y2))
            if x2 <= x1 or y2 <= y1:
                scores[k] = {"margin": 0.0, "pos": 0.0, "neg": 0.0}
                continue

            crop = image.crop((x1, y1, x2, y2))
            img_emb = self._encode_image_crop(crop)

            pos_e = self.pos_embeds.get(k)
            neg_e = self.neg_embeds.get(k)

            pos_sim = float((pos_e @ img_emb).mean().item()) if pos_e is not None else 0.0
            neg_sim = float((neg_e @ img_emb).mean().item()) if neg_e is not None else 0.0

            scores[k] = {
                "margin": pos_sim - neg_sim,
                "pos": pos_sim,
                "neg": neg_sim,
            }
        return scores
