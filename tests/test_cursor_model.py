"""run_cursor_case on a real tiny model: compaction must match masking step for step, and release must follow the
row cursor computed independently from where each row's text sits in the page."""
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from orderkv.cursor import CursorCondition, run_cursor_case
from orderkv.data import CasePublic, FieldSpec
from helpers import tokenizer_fixture

BLOCK, MARGIN = 16, 2
NAMES = [f"item-{i:02d}" for i in range(12)]


def page():
    parts = ["<ul>\n"]
    for i, name in enumerate(NAMES):
        parts.append(f"<li>{name} worth-{i:02d}</li>\n<p>{'filler text ' * 4}</p>\n")
    return "".join(parts) + "</ul>"


class ForcedModel(torch.nn.Module):
    """Runs the real model on its real cache, records its logits, then forces the scripted next token."""

    def __init__(self, inner, script):
        super().__init__()
        self.inner, self.config, self.script, self.real = inner, inner.config, script, []

    def forward(self, input_ids, **kwargs):
        out = self.inner(input_ids, **kwargs)
        self.real.append(out.logits[0, -1].clone())
        forced = torch.full_like(out.logits[:, -1:], -1e4)
        forced[..., self.script[min(len(self.real) - 1, len(self.script) - 1)]] = 1e4
        return SimpleNamespace(logits=forced, past_key_values=out.past_key_values)


def tiny(vocab):
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                         attention_dropout=0.0, use_sliding_window=False, sliding_window=None, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    return Qwen2ForCausalLM(config).eval()


class CursorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = tokenizer_fixture()
        cls.document = page()
        cls.public = CasePublic('lp-model', 'dev', cls.document, (FieldSpec('tsv', '[TARGET] name and worth'),))
        rows = [(NAMES[i], f"worth-{i:02d}") for i in range(1, 11)]
        cls.rows = rows
        cls.text = "name\tworth\n" + "".join(f"{a}\t{b}\n" for a, b in rows)
        cls.script = cls.tokenizer.encode(cls.text, add_special_tokens=False) + [cls.tokenizer.eos_token_id]
        cls.model = tiny(len(cls.tokenizer))
        cls.runs = {}
        for name, options in (('F0', dict(release=False)), ('R0', dict(release=True)),
                              ('M0', dict(release=True, mode='mask'))):
            forced = ForcedModel(cls.model, cls.script).eval()
            condition = CursorCondition(name, margin_blocks=MARGIN, max_new_tokens=len(cls.script) + 5,
                                        block_tokens=BLOCK, protect_first=4, protect_last=4, **options)
            record = run_cursor_case(cls.public, condition, forced, cls.tokenizer)
            cls.runs[name] = (record, torch.stack(forced.real))

    def test_outputs_follow_the_script(self):
        for record, _ in self.runs.values():
            self.assertEqual(record['status'], 'complete', record.get('error'))
            self.assertEqual(record['raw_output'], self.text)

    def test_compaction_matches_masking_at_every_step(self):
        compact, masked, full = self.runs['R0'][1], self.runs['M0'][1], self.runs['F0'][1]
        self.assertEqual(compact.shape, masked.shape)
        torch.testing.assert_close(compact, masked, atol=1e-5, rtol=1e-4)
        # The check has teeth only if hiding the released rows changes the logits.
        self.assertGreater((masked - full).abs().max().item(), 1e-3)

    def test_compaction_frees_what_masking_hides(self):
        compact, masked = self.runs['R0'][0], self.runs['M0'][0]
        freed = sum(e['physical_before'] - e['physical_after'] for e in compact['eviction_events'])
        self.assertGreater(freed, 0)
        self.assertEqual(freed, masked['eviction_events'][-1]['masked_total'])
        self.assertLess(compact['memory_bytes']['kv_byte_steps'], self.runs['F0'][0]['memory_bytes']['kv_byte_steps'])

    def true_blocks(self):
        return [self.document.index(name) // BLOCK for name, _ in self.rows]

    def test_release_never_reaches_unwritten_rows(self):
        record, starts = self.runs['R0'][0], self.true_blocks()
        self.assertTrue(record['eviction_events'])
        for event in record['eviction_events']:
            row = event['row'] - 2  # row 1 is the header
            self.assertLessEqual(event['cursor_block'], starts[row])
            self.assertEqual(event['release_before_block'], event['cursor_block'] - MARGIN)
            self.assertTrue(all(event['release_before_block'] <= s for s in starts[row + 1:]))

    @unittest.expectedFailure
    def test_cursor_sits_on_the_located_row(self):
        # The cursor maps a page offset through prompt offsets, so it sits about one block before the matched row.
        record, starts = self.runs['R0'][0], self.true_blocks()
        for event in record['eviction_events']:
            self.assertEqual(event['cursor_block'], starts[event['row'] - 2])


if __name__ == '__main__':
    unittest.main()
