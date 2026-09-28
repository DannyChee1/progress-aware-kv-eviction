import unittest

import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from orderkv.cache import compact_cache
from orderkv.positions import LogicalPositions, causal_mask
from orderkv.spans import SourceMap


def tiny_model(backend):
    torch.manual_seed(41)
    config = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=256, attention_dropout=0.0,
                        use_sliding_window=False, sliding_window=None,
                        bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = backend
    return Qwen2ForCausalLM(config).eval()


def prefill(model):
    ids = torch.arange(3, 35).reshape(1, -1)
    output = model(ids, use_cache=True, past_key_values=DynamicCache())
    cache, logits = output.past_key_values, output.logits[:, -1].clone()
    state = LogicalPositions.from_prompt(32, SourceMap(tuple(range(4, 28)),
                                                      tuple(i // 4 for i in range(24)), frozenset()))
    return cache, logits, state


def forward_token(model, cache, state, token, blocked=frozenset()):
    logical = state.next_position
    mask = causal_mask((logical,), state.kept + (logical,), dtype=torch.float32,
                       device="cpu", blocked=blocked)
    output = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True,
                   attention_mask=mask, position_ids=torch.tensor([[logical]]),
                   cache_position=torch.tensor([logical]))
    return output.logits[:, -1].clone(), state.append()


class PositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @torch.inference_mode()
    def test_masked_reference_twenty_four_steps_three_compactions(self):
        for backend in ("eager", "sdpa"):
            model = tiny_model(backend)
            full, _, full_state = prefill(model)
            compact, _, state = prefill(model)
            blocked = set()
            max_error = 0.0
            events = {0: set(range(4, 8)), 8: set(range(16, 20)), 16: set(range(24, 28))}
            for step in range(24):
                if step in events:
                    blocked.update(events[step])
                    result = compact_cache(compact, [i for i, p in enumerate(state.kept) if p not in blocked], state)
                    state = result.positions
                    self.assertLess(result.after.storage, result.before.storage)
                    self.assertEqual(result.after.storage, result.after.tensors)
                token = (step * 7 + 11) % 97
                reference, full_state = forward_token(model, full, full_state, token, blocked)
                actual, state = forward_token(model, compact, state, token)
                max_error = max(max_error, (actual - reference).abs().max().item())
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                self.assertEqual(actual.argmax().item(), reference.argmax().item())
                self.assertEqual(state.next_position, full_state.next_position)
                self.assertTrue(set(range(32, state.next_position)) <= set(state.kept))
                for tensor in (*compact.key_cache, *compact.value_cache):
                    self.assertEqual(tensor.shape[-2], len(state.kept))
            print(f"{backend}: 24 steps, 3 compactions, maximum logit error={max_error:.9g}")

    @torch.inference_mode()
    def test_keep_all_logits_and_stock_greedy_generation(self):
        for backend in ("eager", "sdpa"):
            model = tiny_model(backend)
            cache, logits, state = prefill(model)
            reference_cache, _, reference_state = prefill(model)
            result = compact_cache(cache, range(32), state)
            actual, _ = forward_token(model, cache, result.positions, 7)
            expected, _ = forward_token(model, reference_cache, reference_state, 7)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            cache, logits, state = prefill(model)
            tokens = []
            for _ in range(20):
                token = logits.argmax().item()
                tokens.append(token)
                logits, state = forward_token(model, cache, state, token)
            ids = torch.arange(3, 35).reshape(1, -1)
            stock = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20,
                                   do_sample=False, use_cache=True)
            self.assertEqual(tokens, stock[0, 32:].tolist())

    def test_chunk_mask_uses_original_positions(self):
        mask = causal_mask((10, 12), (0, 4, 10, 12), dtype=torch.float32, device="cpu")
        self.assertEqual((mask[0, 0] == 0).tolist(), [[True, True, True, False], [True] * 4])
        with self.assertRaises(ValueError):
            causal_mask((0,), (1,), dtype=torch.float32, device="cpu")
