"""The per-head masking attention function against independent references: an eager model given an explicit
per-query-head mask, physical compaction, and TOVA's choice recomputed from eager attention weights."""
import unittest

import torch
from transformers import DynamicCache

from orderkv import headmask
from orderkv.cache import compact_cache
from orderkv.positions import LogicalPositions, causal_mask
from orderkv.spans import SourceMap
from test_positions import tiny_model

PROMPT = 32
IDS = torch.arange(3, 3 + PROMPT)[None]


def prefill(model):
    cache = DynamicCache()
    model(IDS, use_cache=True, past_key_values=cache)
    return cache


def decode(model, cache, keys, mask=None, blocked=frozenset(), **kwargs):
    if mask is None:
        mask = causal_mask((PROMPT,), keys, dtype=torch.float32, device='cpu', blocked=blocked)
    return model(torch.tensor([[5]]), past_key_values=cache, use_cache=True, position_ids=torch.tensor([[PROMPT]]),
                 cache_position=torch.tensor([PROMPT]), attention_mask=mask, **kwargs)


def with_state(model, state, run):
    model.config._attn_implementation = headmask.NAME
    headmask.ACTIVE = state
    try:
        return run()
    finally:
        headmask.ACTIVE = None
        model.config._attn_implementation = 'sdpa'


def per_query_head_mask(blocked_kv, groups):
    mask = torch.zeros(1, blocked_kv.shape[0] * groups, 1, PROMPT + 1)
    mask[0, :, 0, :PROMPT] = blocked_kv.repeat_interleave(groups, dim=0).float() * torch.finfo(torch.float32).min
    return mask


class HeadMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)
        cls.model, cls.eager = tiny_model('sdpa'), tiny_model('eager')
        cls.groups = cls.model.config.num_attention_heads // cls.model.config.num_key_value_heads

    def headmask_logits(self, blocked_kv):
        state = headmask.HeadMaskState(PROMPT, torch.ones(PROMPT, dtype=torch.bool))
        state.blocked = {layer: blocked_kv.clone() for layer in range(self.model.config.num_hidden_layers)}
        cache = prefill(self.model)
        return with_state(self.model, state, lambda: decode(self.model, cache, tuple(range(PROMPT + 1)))).logits[0, -1]

    def test_per_head_mask_matches_an_explicit_eager_mask(self):
        blocked = torch.zeros(2, PROMPT, dtype=torch.bool)
        blocked[0, 4:12], blocked[1, 16:28] = True, True
        actual = self.headmask_logits(blocked)
        reference = decode(self.eager, prefill(self.eager), None, mask=per_query_head_mask(blocked, self.groups))
        torch.testing.assert_close(actual, reference.logits[0, -1], atol=1e-5, rtol=1e-4)
        swapped = decode(self.eager, prefill(self.eager), None, mask=per_query_head_mask(blocked.flip(0), self.groups))
        self.assertGreater((actual - swapped.logits[0, -1]).abs().max().item(), 1e-3)

    def test_uniform_mask_matches_compaction(self):
        hidden = set(range(4, 12))
        blocked = torch.zeros(2, PROMPT, dtype=torch.bool)
        blocked[:, sorted(hidden)] = True
        actual = self.headmask_logits(blocked)
        cache = prefill(self.model)
        state = LogicalPositions.from_prompt(PROMPT, SourceMap(tuple(range(PROMPT)), (0,) * PROMPT, frozenset()))
        result = compact_cache(cache, [i for i in range(PROMPT) if i not in hidden], state)
        compacted = decode(self.model, cache, result.positions.kept + (PROMPT,)).logits[0, -1]
        torch.testing.assert_close(actual, compacted, atol=1e-5, rtol=1e-4)
        unmasked = decode(self.model, prefill(self.model), tuple(range(PROMPT + 1))).logits[0, -1]
        self.assertGreater((actual - unmasked).abs().max().item(), 1e-3)

    def tova_choice(self, per_layer, drop=6):
        eligible = torch.zeros(PROMPT, dtype=torch.bool)
        eligible[2:30] = True
        state = headmask.HeadMaskState(PROMPT, eligible, per_layer=per_layer)
        state.tova_target = int(eligible.sum()) - drop
        cache = prefill(self.model)
        with_state(self.model, state, lambda: decode(self.model, cache, tuple(range(PROMPT + 1))))
        weights = decode(self.eager, prefill(self.eager), tuple(range(PROMPT + 1)), output_attentions=True).attentions
        return state, eligible, weights, drop

    def test_per_layer_tova_evicts_the_least_attended_rows(self):
        state, eligible, weights, drop = self.tova_choice(per_layer=True)
        for layer, attention in enumerate(weights):
            scores = attention[0, :, 0, :PROMPT].mean(dim=0).masked_fill(~eligible, float('inf'))
            expected = set(scores.topk(drop, largest=False).indices.tolist())
            for head in range(state.blocked[layer].shape[0]):
                self.assertEqual(set(state.blocked[layer][head].nonzero().flatten().tolist()), expected)

    def test_per_head_tova_evicts_each_heads_least_attended_rows(self):
        state, eligible, weights, drop = self.tova_choice(per_layer=False)
        for layer, attention in enumerate(weights):
            per_kv = attention[0, :, 0, :PROMPT].view(-1, self.groups, PROMPT).mean(dim=1)
            for head, scores in enumerate(per_kv):
                expected = set(scores.masked_fill(~eligible, float('inf')).topk(drop, largest=False).indices.tolist())
                self.assertEqual(set(state.blocked[layer][head].nonzero().flatten().tolist()), expected)


if __name__ == '__main__':
    unittest.main()
