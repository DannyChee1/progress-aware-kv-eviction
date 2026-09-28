"""Per-layer, per-KV-head token masking for baselines that evict different tokens in each head.

Physical compaction removes the same rows in every head, so per-head methods (SnapKV, TOVA)
are emulated by attention masks; memory is accounted as the unmasked token count.
"""
from dataclasses import dataclass, field

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

NAME = 'orderkv_headmask'


@dataclass
class HeadMaskState:
    prompt_length: int
    eligible: torch.Tensor  # bool [prompt_length]: source rows a policy may evict
    blocked: dict = field(default_factory=dict)  # layer -> bool [kv_heads, prompt_length]
    tova_target: int | None = None  # TOVA: evictable rows each head may keep after this step
    per_layer: bool = False  # TOVA's default: one choice per layer from attention averaged over all heads

    def mean_blocked(self, layers: int, kv_heads: int) -> float:
        return sum(int(b.sum()) for b in self.blocked.values()) / (layers * kv_heads)


ACTIVE: HeadMaskState | None = None


def _grouped(tensor, groups):
    return tensor.repeat_interleave(groups, dim=0)


def headmask_attention(module, query, key, value, attention_mask, **kwargs):
    state = ACTIVE
    if state is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    kv_heads, key_length = key.shape[1], key.shape[-2]
    groups = module.num_key_value_groups
    blocked = state.blocked.get(module.layer_idx)
    mask = attention_mask[..., :key_length] if attention_mask is not None else None
    if blocked is not None and bool(blocked.any()):
        extra = torch.zeros((kv_heads, key_length), dtype=query.dtype, device=query.device)
        extra[:, :state.prompt_length].masked_fill_(blocked, torch.finfo(query.dtype).min)
        extra = _grouped(extra, groups)[None, :, None, :]
        mask = extra if mask is None else mask + extra
    output = sdpa_attention_forward(module, query, key, value, mask, **kwargs)
    if state.tova_target is not None and query.shape[2] == 1:
        # TOVA: drop the evictable rows the current query attends to least, per KV head.
        keys = _grouped(key[0], groups)
        scores = (query[0, :, 0, None, :].float() @ keys.float().transpose(-1, -2))[:, 0] * module.scaling
        if mask is not None:
            scores = scores + mask[0, :, 0, :].float()
        weights = scores.softmax(dim=-1).view(kv_heads, groups, key_length).mean(dim=1)[:, :state.prompt_length]
        if blocked is None:
            blocked = torch.zeros((kv_heads, state.prompt_length), dtype=torch.bool, device=query.device)
            state.blocked[module.layer_idx] = blocked
        if state.per_layer:
            live = state.eligible & ~blocked[0]
            excess = int(live.sum()) - state.tova_target
            if excess > 0:
                chosen = weights.mean(dim=0).masked_fill(~live, float('inf')).topk(excess, largest=False).indices
                blocked[:, chosen] = True
            return output
        for head in range(kv_heads):
            live = state.eligible & ~blocked[head]
            excess = int(live.sum()) - state.tova_target
            if excess > 0:
                candidates = weights[head].masked_fill(~live, float('inf'))
                blocked[head, candidates.topk(excess, largest=False).indices] = True
    return output


ALL_ATTENTION_FUNCTIONS[NAME] = headmask_attention


def snapkv_head_blocks(attentions, split, eligible, drop, groups, pool):
    """Per-layer, per-KV-head SnapKV: pooled window attention, drop the `drop` lowest eligible rows."""
    blocked = {}
    for layer, weights in enumerate(attentions):
        heads = weights[0, :, :, :split].float()  # [query heads, window, split]
        kv_heads = heads.shape[0] // groups
        scores = heads.view(kv_heads, groups, heads.shape[1], split).mean(dim=(1, 2))
        pooled = torch.nn.functional.max_pool1d(scores[:, None], pool, stride=1, padding=pool // 2)[:, 0, :split]
        mask = torch.zeros((kv_heads, eligible.numel()), dtype=torch.bool, device=weights.device)
        if drop > 0:
            candidates = pooled.masked_fill(~eligible[:split], float('inf'))
            mask[:, :split].scatter_(1, candidates.topk(drop, dim=1, largest=False).indices, True)
        blocked[layer] = mask
    return blocked
