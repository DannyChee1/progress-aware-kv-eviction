from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from orderkv.config import Config
from orderkv.data import CasePublic, FieldSpec
from helpers import ScriptedModel, tokenizer_fixture
from test_positions import tiny_model


class CursorReleaseTests(unittest.TestCase):
    def test_locator_and_tracker(self):
        from orderkv.cursor import RowLocator, RowTracker
        page = "<ul><li>Alpha&nbsp;One <b>10</b></li><li>Beta Two 20</li><li>Gamma Three 30</li></ul>"
        locator = RowLocator(page, min_anchor=4)
        first = locator.locate(["Alpha One", "10"])
        second = locator.locate(["Gamma Three", "30"])
        self.assertIsNotNone(first)
        self.assertLess(first[0], second[0])
        self.assertIsNone(locator.locate(["Beta Two", "20"]))  # behind the cursor now
        self.assertIsNone(locator.locate(["zz", "1"]))  # anchor too short
        tracker = RowTracker()
        self.assertEqual(tracker.update("name\tvalue\nAlp"), [["name", "value"]])
        self.assertEqual(tracker.update("name\tvalue\nAlpha One\t10\n\nBeta"), [["Alpha One", "10"]])
        self.assertEqual(tracker.rows, 2)
        fenced = RowTracker()
        self.assertEqual(fenced.update("```tsv\nname\tvalue\nA\t1\n"), [["name", "value"], ["A", "1"]])

    def test_run_cursor_case_releases_behind_rows(self):
        from orderkv.cursor import CursorCondition, run_cursor_case
        tokenizer = tokenizer_fixture()
        items = "".join(f"<li>item-{i} value-{i} filler words here</li>\n" for i in range(12))
        public = CasePublic("lp-fixture", "dev", f"<ul>\n{items}</ul>", (FieldSpec("tsv", "[TARGET] name and value"),))
        text = "name\tvalue\nitem-2\tvalue-2\nitem-9\tvalue-9\nitem-11\tvalue-11\n"
        tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        released = run_cursor_case(public, CursorCondition("R0", release=True, margin_blocks=0, max_new_tokens=200,
                                                           block_tokens=8, protect_first=1, protect_last=1),
                                   ScriptedModel(tokens, len(tokenizer)).eval(), tokenizer)
        baseline = run_cursor_case(public, CursorCondition("F0", release=False, max_new_tokens=200, block_tokens=8,
                                                           protect_first=1, protect_last=1),
                                   ScriptedModel(tokens, len(tokenizer)).eval(), tokenizer)
        self.assertEqual(released["status"], "complete", released)
        self.assertEqual(released["raw_output"], text)
        self.assertEqual(released["rows_completed"], 4)
        self.assertEqual(released["rows_located"], 3)
        self.assertGreaterEqual(len(released["eviction_events"]), 1)
        self.assertEqual(baseline["eviction_events"], [])
        self.assertLess(released["memory_bytes"]["kv_byte_steps"], baseline["memory_bytes"]["kv_byte_steps"])
        for event in released["eviction_events"]:
            self.assertLess(event["physical_after"], event["physical_before"])


class LongProcTests(unittest.TestCase):
    def test_adapter_task_text_and_case(self):
        from orderkv.longproc import build_case, task_text
        template = "[TASK]\n...\n```html\n{html_str}\n```\n\n[TARGET INFORMATION]\nAbout {task_topic}: {task_description}{filtering_instruction}\n\n[OUTPUT FORMAT]\n{tsv_header}\n"
        record = {'task_id': 'W1_0.5k_1', 'website_id': 'W1', 'html_path': 'page.html', 'task_topic': 'Books',
                  'task_description': '(1) title; (2) price', 'filtering_instruction': '', 'tsv_header': 'title\tprice',
                  'gt': 'title\tprice\nA\t1\nB\t2', 'output_length_group': '0.5k'}
        text = task_text(template, record)
        self.assertTrue(text.startswith('[TARGET INFORMATION]'))
        self.assertIn('About Books: (1) title; (2) price', text)
        self.assertIn('title\tprice', text)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'page.html').write_text('<ul><li>A 1</li><li>B 2</li></ul>')
            public, gold = build_case(record, Path(directory), template, 'dev')
        self.assertEqual(public.case_id, 'dev-W1_0.5k_1')
        self.assertEqual(gold.values['tsv'], 'title\tprice\nA\t1\nB\t2\n')

    def test_row_f1_matches_longproc_semantics(self):
        from orderkv.metrics import row_f1
        gold = 'a\tb\nAlpha One\t10\nBeta, Two\t20\nGamma\t30\n'
        exact = row_f1('a\tb\nalpha one\t10\nbeta two\t20\ngamma\t30\n', gold)
        self.assertEqual(exact['f1'], 1.0)
        partial = row_f1('a\tb\nalpha one\t10\nDelta\t40\ntrailing text', gold)
        self.assertAlmostEqual(partial['precision'], 0.5)
        self.assertAlmostEqual(partial['recall'], 1 / 3)
        self.assertEqual(row_f1('', gold)['f1'], 0.0)
        fenced = row_f1('```tsv\na\tb\nAlpha One\t10\nBeta, Two\t20\nGamma\t30\n```', gold)
        self.assertEqual(fenced['f1'], 1.0)

    def test_tsv_config_mode(self):
        base = replace(Config(), model_revision="a" * 40,
                       conditions=('F0', 'R0'), context_targets=(16384,), context_tolerance=0.5)
        base.validate()
        for bad in ({'conditions': ('F0', 'Z9')}, {'conditions': ('F0', 'F0')}, {'cursor_margin_blocks': -1}):
            with self.assertRaises(ValueError):
                replace(base, **bad).validate()

    def test_cursor_study_runs_paired_conditions(self):
        from unittest.mock import patch
        from orderkv.cursor import run_cursor_study
        from dataclasses import asdict
        from orderkv.data import GoldPrivate
        tokenizer = tokenizer_fixture()
        public = CasePublic('dev-W1', 'dev', '<ul><li>item-1 value-1</li></ul>', (FieldSpec('tsv', '[TARGET] rows'),))
        gold = GoldPrivate({'tsv': 'name\tvalue\nitem-1\tvalue-1\n'}, {'tsv': ()}, {'tsv': ()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dev_public.jsonl').write_text(json.dumps(asdict(public)) + '\n')
            (root / 'dev_private.jsonl').write_text(json.dumps({'case_id': public.case_id, **asdict(gold)}) + '\n')
            config = replace(Config(), model_revision='a' * 40,
                             conditions=('F0', 'R0'), context_targets=(2048,), context_tolerance=0.95,
                             case_limit=1, dataset=str(root / 'dev_public.jsonl'), private_labels=str(root / 'dev_private.jsonl'))
            def fake_run(public, condition, model, tokenizer, deadline=None):
                return {'case_id': public.case_id, 'condition': condition.name,
                        'status': 'capped' if condition.name == 'R0' else 'complete',
                        'raw_output': 'name\tvalue\nitem-1\tvalue-1\n', 'timings_seconds': {'end_to_end': 1},
                        'eviction_events': [{'row': 2}] if condition.release else []}
            with patch('orderkv.cursor.run_cursor_case', side_effect=fake_run) as mocked:
                records = run_cursor_study(config, ScriptedModel([], len(tokenizer)).eval(), tokenizer, root / 'runs')
                skipped = run_cursor_study(replace(config, context_tolerance=0.05), ScriptedModel([], len(tokenizer)).eval(),
                                           tokenizer, root / 'runs-skip')
            self.assertEqual(skipped, [])
            self.assertIn('skipped_length', (root / 'runs-skip' / 'runs.jsonl').read_text())
            self.assertEqual({r['condition'] for r in records}, {'F0', 'R0'})
            self.assertTrue(all(r['quality']['f1'] == 1.0 for r in records))
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(run_cursor_study(config, ScriptedModel([], len(tokenizer)).eval(), tokenizer, root / 'runs', resume=True), [])
            # A code-hash change must not trigger re-runs when case, condition, config and data match.
            runs = root / 'runs' / 'runs.jsonl'
            rows = [json.loads(l) for l in runs.read_text().splitlines()]
            for row in rows:
                row['code_hash'] = 'changed'
            runs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            for marker in (root / 'runs').glob('*.complete'):
                marker.unlink()
            with patch('orderkv.cursor.run_cursor_case', side_effect=fake_run) as again:
                self.assertEqual(run_cursor_study(config, ScriptedModel([], len(tokenizer)).eval(), tokenizer, root / 'runs', resume=True), [])
            self.assertEqual(again.call_count, 0)
            # A condition whose forecast would cross the shard deadline is not started.
            with patch('orderkv.cursor.run_cursor_case', side_effect=fake_run) as late:
                self.assertEqual(run_cursor_study(config, ScriptedModel([], len(tokenizer)).eval(), tokenizer,
                                                  root / 'runs-late', max_seconds=10), [])
            self.assertEqual(late.call_count, 0)
            from orderkv.cursor import condition_forecast_seconds
            self.assertAlmostEqual(condition_forecast_seconds(5120), 30 + 0.16 * 5120)


class CursorDiagnosticTests(unittest.TestCase):
    def test_mask_mode_hides_rows_without_compacting(self):
        from orderkv.cursor import CursorCondition, run_cursor_case
        tokenizer = tokenizer_fixture()
        items = "".join(f"<li>item-{i} value-{i} filler words here</li>\n" for i in range(12))
        public = CasePublic("lp-fixture", "dev", f"<ul>\n{items}</ul>", (FieldSpec("tsv", "[TARGET] name and value"),))
        text = "name\tvalue\nitem-2\tvalue-2\nitem-9\tvalue-9\nitem-11\tvalue-11\n"
        tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        common = dict(release=True, margin_blocks=0, max_new_tokens=200, block_tokens=8, protect_first=1, protect_last=1)
        masked = run_cursor_case(public, CursorCondition("M0", mode="mask", **common), ScriptedModel(tokens, len(tokenizer)).eval(), tokenizer)
        compact = run_cursor_case(public, CursorCondition("R0", **common), ScriptedModel(tokens, len(tokenizer)).eval(), tokenizer)
        self.assertEqual(masked["raw_output"], compact["raw_output"])
        self.assertTrue(masked["eviction_events"])
        self.assertTrue(all("masked_new" in e for e in masked["eviction_events"]))
        live = [s["live"] for s in masked["forward_bytes"]]
        self.assertEqual(live, sorted(live))  # storage never shrinks under masking
        self.assertLess(masked["forward_bytes"][-1]["source"], masked["forward_bytes"][0]["source"])

    def test_tsv_condition_sets(self):
        base = replace(Config(), model_revision="a" * 40,
                       conditions=('M0', 'T1'), context_targets=(16384,), context_tolerance=0.5)
        base.validate()
        for bad in (('M0', 'X1'), ('T1', 'T1'), ()):
            with self.assertRaises(ValueError):
                replace(base, conditions=bad).validate()


class PrefillBaselineTests(unittest.TestCase):
    @torch.inference_mode()
    def test_streaming_and_snapkv_drop_the_budget_at_prefill(self):
        from orderkv.cursor import CursorCondition, prefill_drop_order, run_cursor_case
        tokenizer = tokenizer_fixture()
        model = tiny_model("sdpa")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.max_position_embeddings = 4096
        items = "".join(f"<li>item-{i} value-{i}</li>" for i in range(20))
        public = CasePublic("lp-fixture", "dev", f"<ul>{items}</ul>", (FieldSpec("tsv", "[TARGET] rows"),))
        common = dict(max_new_tokens=6, block_tokens=8, protect_first=4, protect_last=4)
        full = run_cursor_case(public, CursorCondition("F0", **common), model, tokenizer)
        for name, mode in (("L0", "streaming"), ("S0", "snapkv")):
            record = run_cursor_case(public, CursorCondition(name, mode=mode, drop_tokens=40, window=8, pool=3, **common),
                                     model, tokenizer)
            event = record["eviction_events"][0]
            self.assertEqual(event["dropped_tokens"], 40)
            self.assertEqual(event["physical_after"], record["prompt_tokens"] - 40)
            self.assertLess(record["memory_bytes"]["kv_byte_steps"], full["memory_bytes"]["kv_byte_steps"])
            self.assertEqual(model.config._attn_implementation, "sdpa")
        self.assertEqual(prefill_drop_order("streaming", [5, 6, 7]), [5, 6, 7])
        self.assertEqual(prefill_drop_order("snapkv", [5, 6, 7], {5: 0.3, 6: 0.1, 7: 0.2}), [6, 7, 5])
        with self.assertRaises(ValueError):
            prefill_drop_order("compact", [1])


class PerHeadBaselineTests(unittest.TestCase):
    @torch.inference_mode()
    def test_snapkv_head_and_tova_match_their_budgets(self):
        from orderkv import headmask
        from orderkv.cursor import CursorCondition, run_cursor_case
        tokenizer = tokenizer_fixture()
        model = tiny_model("sdpa")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.max_position_embeddings = 4096
        items = "".join(f"<li>item-{i} value-{i}</li>" for i in range(20))
        public = CasePublic("lp-fixture", "dev", f"<ul>{items}</ul>", (FieldSpec("tsv", "[TARGET] rows"),))
        common = dict(max_new_tokens=6, block_tokens=8, protect_first=4, protect_last=4)
        full = run_cursor_case(public, CursorCondition("F0", **common), model, tokenizer)
        per_token = full["forward_bytes"][0]["tensor"] // (full["prompt_tokens"] + 1)
        snap = run_cursor_case(public, CursorCondition("S1", mode="snapkv_head", drop_tokens=40, window=8, pool=3, **common),
                               model, tokenizer)
        for step in snap["forward_bytes"]:
            self.assertAlmostEqual(step["live"] - step["effective"], 40 * per_token)
        self.assertLess(snap["memory_bytes"]["effective_kv_byte_steps"], full["memory_bytes"]["kv_byte_steps"])
        source = full["forward_bytes"][0]["source"] // per_token  # live source rows before any eviction
        schedule = tuple(range(source, source - 60, -10))
        tova = run_cursor_case(public, CursorCondition("T0", mode="tova", schedule=schedule, **common), model, tokenizer)
        hidden = [(s["live"] - s["effective"]) / per_token for s in tova["forward_bytes"]]
        self.assertEqual(hidden, sorted(hidden))  # evictions only accumulate
        self.assertGreater(hidden[-1], 0)
        self.assertIsNone(headmask.ACTIVE)
        self.assertEqual(model.config._attn_implementation, "sdpa")


class RepetitionStopTests(unittest.TestCase):
    def test_three_identical_rows_end_generation(self):
        from orderkv.cursor import CursorCondition, run_cursor_case
        tokenizer = tokenizer_fixture()
        public = CasePublic("lp-fixture", "dev", "<ul><li>alpha one</li></ul>", (FieldSpec("tsv", "[TARGET] rows"),))
        text = "a\tb\nx\ty\nx\ty\nx\ty\nx\ty\nx\ty\n"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        record = run_cursor_case(public, CursorCondition("F0", max_new_tokens=200, block_tokens=8, protect_first=1, protect_last=1),
                                 ScriptedModel(tokens, len(tokenizer)).eval(), tokenizer)
        self.assertTrue(record.get("stopped_on_repetition"))
        self.assertEqual(record["raw_output"], "a\tb\nx\ty\nx\ty\nx\ty\n")


class PerLayerTovaTests(unittest.TestCase):
    def test_per_layer_tova_evicts_the_same_rows_in_every_head(self):
        from orderkv import headmask
        from transformers import DynamicCache
        from orderkv.positions import causal_mask
        model = tiny_model('sdpa')
        ids = torch.arange(3, 35)[None]
        with torch.inference_mode():
            cache = DynamicCache()
            model(ids, use_cache=True, past_key_values=cache)
            state = headmask.HeadMaskState(32, torch.ones(32, dtype=torch.bool), tova_target=20, per_layer=True)
            model.config._attn_implementation = headmask.NAME
            headmask.ACTIVE = state
            try:
                model(torch.tensor([[5]]), past_key_values=cache, use_cache=True, position_ids=torch.tensor([[32]]),
                      cache_position=torch.tensor([32]), attention_mask=causal_mask((32,), tuple(range(33)), dtype=torch.float32, device='cpu'))
            finally:
                headmask.ACTIVE = None
                model.config._attn_implementation = 'sdpa'
        for blocked in state.blocked.values():
            self.assertTrue(torch.equal(blocked[0], blocked[1]))
            self.assertEqual(int(blocked[0].sum()), 12)
