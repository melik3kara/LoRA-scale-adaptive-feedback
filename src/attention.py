"""
Usage:
    from attention import RegionalAttnProcessor, set_regional_attention
 
    # Define which spatial region belongs to which identity
    # Masks are binary: 1 = this region belongs to this identity, 0 = doesn't
    # Shape: (height, width) matching the generated image resolution
    identity_masks = {
        "hermione": torch.tensor(...),  # left half = 1, right half = 0
        "daenerys": torch.tensor(...),  # left half = 0, right half = 1
    }
 
    # Define which text token indices correspond to which identity's trigger words
    token_assignments = {
        "hermione": [3, 4],   # indices of "Hermione_Granger" in the tokenized prompt
        "daenerys": [7, 8],   # indices of "dae" "woman" in the tokenized prompt
    }
 
    # Apply to the pipeline
    set_regional_attention(pipe, identity_masks, token_assignments)
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class RegionalAttnProcessor:

    def __init__(
        self,
        identity_masks: dict,
        token_assignments: dict,
        total_tokens: int,
        mask_self_attention: bool = True,
        mask_weight: float = -10000.0,
    ):
        self.identity_masks = identity_masks
        self.token_assignments = token_assignments
        self.total_tokens = total_tokens
        self.mask_self_attention = mask_self_attention
        self.mask_weight = mask_weight
        self._mask_cache = {}
        self._warned_seq_lens = set()

    def _get_spatial_masks(self, height, width, device):
        cache_key = (height, width)

        if cache_key not in self._mask_cache:
            resized = {}
            for identity_id, mask in self.identity_masks.items():
                resized_mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="nearest",
                ).squeeze().flatten().to(device)
                resized[identity_id] = resized_mask > 0.5
            self._mask_cache[cache_key] = resized

        return self._mask_cache[cache_key]

    def _build_cross_attention_mask(self, seq_len, num_tokens, height, width, device):
        spatial_masks = self._get_spatial_masks(height, width, device)
        mask = torch.zeros(seq_len, num_tokens, device=device)

        for identity_id, token_indices in self.token_assignments.items():
            my_region = spatial_masks[identity_id]
            outside_region = ~my_region

            for token_idx in token_indices:
                if token_idx < num_tokens:
                    mask[outside_region, token_idx] = self.mask_weight

        return mask

    def _build_self_attention_mask(self, seq_len, height, width, device):
        spatial_masks = self._get_spatial_masks(height, width, device)
        mask = torch.zeros(seq_len, seq_len, device=device)
        identity_ids = list(spatial_masks.keys())

        for i in range(len(identity_ids)):
            for j in range(len(identity_ids)):
                if i == j:
                    continue
                region_i = spatial_masks[identity_ids[i]]
                region_j = spatial_masks[identity_ids[j]]
                cross_mask = region_i.unsqueeze(1) & region_j.unsqueeze(0)
                mask[cross_mask] = self.mask_weight

        return mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_len, inner_dim = hidden_states.shape

        if not is_cross_attention and not self.mask_self_attention:
            return self._default_attention(attn, hidden_states, encoder_hidden_states, attention_mask)

        spatial_size = int(seq_len ** 0.5)
        if spatial_size * spatial_size != seq_len:
            # Warn once per unexpected seq_len — small seq_lens are expected
            # (non-spatial layers like mid-block), but large ones likely
            # indicate a spatial layer where regional masking silently failed
            if seq_len > 256 and seq_len not in self._warned_seq_lens:
                print(
                    f"[attention] WARNING: seq_len={seq_len} is not a perfect square — "
                    f"skipping regional mask. This may indicate non-square image generation "
                    f"or an unexpected feature map shape."
                )
                self._warned_seq_lens.add(seq_len)
            return self._default_attention(attn, hidden_states, encoder_hidden_states, attention_mask)

        query = attn.to_q(hidden_states)
        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if is_cross_attention:
            num_tokens = key.shape[2]
            regional_mask = self._build_cross_attention_mask(
                seq_len, num_tokens, spatial_size, spatial_size, hidden_states.device
            )
            attn_scores = attn_scores + regional_mask.unsqueeze(0).unsqueeze(0)
        elif self.mask_self_attention:
            regional_mask = self._build_self_attention_mask(
                seq_len, spatial_size, spatial_size, hidden_states.device
            )
            attn_scores = attn_scores + regional_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states_out = torch.matmul(attn_probs, value)

        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, seq_len, inner_dim)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        return hidden_states_out

    def _default_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask):
        batch_size, seq_len, inner_dim = hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is not None:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states_out = torch.matmul(attn_probs, value)

        hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, seq_len, inner_dim)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        return hidden_states_out


def create_identity_masks(image_width, image_height, regions: dict) -> dict:
    masks = {}
    for identity_id, (x1, y1, x2, y2) in regions.items():
        mask = torch.zeros(image_height, image_width)
        mask[y1:y2, x1:x2] = 1.0
        masks[identity_id] = mask
    return masks


def get_trigger_token_indices(tokenizer, prompt: str, trigger_words: dict) -> dict:
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    assignments = {}
    for identity_id, trigger in trigger_words.items():
        trigger_tokens = tokenizer.encode(trigger, add_special_tokens=False)
        trigger_len = len(trigger_tokens)

        indices = []
        for i in range(len(full_tokens) - trigger_len + 1):
            if full_tokens[i:i + trigger_len] == trigger_tokens:
                indices.extend(range(i, i + trigger_len))
                break

        assignments[identity_id] = [idx + 1 for idx in indices]

    return assignments


def set_regional_attention(
    pipe,
    identity_masks: dict,
    token_assignments: dict,
    total_tokens: int = 77,
    mask_self_attention: bool = True,
):
    processor = RegionalAttnProcessor(
        identity_masks=identity_masks,
        token_assignments=token_assignments,
        total_tokens=total_tokens,
        mask_self_attention=mask_self_attention,
    )

    attn_processors = {}
    for name in pipe.unet.attn_processors.keys():
        attn_processors[name] = processor

    pipe.unet.set_attn_processor(attn_processors)
    print(f"Regional attention set: {len(attn_processors)} layers masked")


def remove_regional_attention(pipe):
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    print("Regional attention removed, default processors restored")
