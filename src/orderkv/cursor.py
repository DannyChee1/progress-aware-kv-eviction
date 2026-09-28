"""Cursor release for document-order structured output (LongProc HTML-to-TSV).

After each completed output row, the row's cells are string-matched back into the
source text; every source block more than `margin_blocks` before the latest match
is released. No evidence prediction is involved: the model's own output marks
what has been consumed. Rows whose longest cell is shorter than `min_anchor`
characters are not trusted as anchors.
"""
import bisect
from dataclasses import dataclass
import hashlib
import html as htmlmod
import json
import time

import torch
from transformers import DynamicCache

from orderkv.cache import cache_bytes, compact_cache
from orderkv.positions import LogicalPositions, causal_mask
from orderkv.spans import map_source


def squash(text: str) -> tuple[str, list[int]]:
    """Lowercase, entity-decoded, whitespace-free text plus a map back to original offsets."""
    squashed, back = [], []
    i = 0
    while i < len(text):
        if text[i] == '&':
            end = text.find(';', i, i + 10)
            if end > 0:
                piece = htmlmod.unescape(text[i:end + 1])
                if len(piece) == 1:
                    if not piece.isspace():
                        squashed.append(piece.lower())
                        back.append(i)
                    i = end + 1
                    continue
        if not text[i].isspace():
            squashed.append(text[i].lower())
            back.append(i)
        i += 1
    back.append(len(text))
    return ''.join(squashed), back


class RowLocator:
    """Finds the source character span of an emitted TSV row in whitespace-free, entity-decoded page text."""

    def __init__(self, source_text: str, window: int = 1500, min_anchor: int = 6):
        self.squashed, self.back = squash(source_text)
        self.window = window
        self.min_anchor = min_anchor
        self.cursor = 0  # squashed index of the latest located row start

    def locate(self, cells: list[str]) -> tuple[int, int] | None:
        cells = [''.join(c.lower().split()) for c in cells if c.strip() and c.strip().upper() != 'N/A']
        cells = [c for c in cells if c]
        # A dotted cell that never appears verbatim is matched by its last part.
        cells = [c.rsplit('.', 1)[1] if '.' in c and c not in self.squashed and len(c.rsplit('.', 1)[1]) >= 2 else c
                 for c in cells]
        if not cells:
            return None
        anchor = max(cells, key=len)
        if len(anchor) < self.min_anchor:
            return None
        start_at = self.cursor
        while True:
            index = self.squashed.find(anchor, start_at, len(self.squashed) + len(anchor))
            if index < 0:
                return None
            lo, hi = max(0, index - self.window), index + self.window
            found = []
            for cell in cells:
                at = self.squashed.find(cell, lo, hi)
                if at < 0:
                    found = None
                    break
                found.append((self.back[at], self.back[min(at + len(cell), len(self.back) - 1)]))
            if found:
                start = min(f[0] for f in found)
                self.cursor = max(self.cursor, bisect.bisect_left(self.back, start))
                return start, max(f[1] for f in found)
            start_at = index + 1


class RowTracker:
    """Yields completed TSV rows (header included) as newline-terminated lines arrive."""

    def __init__(self):
        self.consumed = 0
        self.rows = 0

    def update(self, text: str) -> list[list[str]]:
        completed = []
        while True:
            newline = text.find('\n', self.consumed)
            if newline < 0:
                return completed
            line = text[self.consumed:newline]
            self.consumed = newline + 1
            if line.strip() and not line.strip().startswith('```'):  # code fences are not rows
                self.rows += 1
                completed.append(line.split('\t'))


@dataclass(frozen=True)
class CursorCondition:
    name: str
    release: bool = False
    margin_blocks: int = 8
    max_new_tokens: int = 1024
    block_tokens: int = 128
    protect_first: int = 32
    protect_last: int = 64
    mode: str = 'compact'  # 'compact' | 'mask' (cursor release); 'streaming' | 'snapkv' | 'snapkv_head' | 'tova'
    tova_per_layer: bool = False  # TOVA's default granularity: same evictions for every head in a layer
    drop_tokens: int = 0  # prefill baselines: source tokens removed once, after prefill (per head for S1)
    schedule: tuple = ()  # T0: live source rows per decode step to follow (cursor release's measured curve)
    window: int = 32  # SnapKV observation window: the last prompt tokens (task instructions)
    pool: int = 7  # SnapKV max-pool kernel over token scores


CONDITIONS = {'F0': dict(release=False), 'R0': dict(release=True), 'M0': dict(release=True, mode='mask'),
              'L0': dict(release=False, mode='streaming'), 'S0': dict(release=False, mode='snapkv'),
              'S1': dict(release=False, mode='snapkv_head'), 'T0': dict(release=False, mode='tova'),
              'T1': dict(release=False, mode='tova', tova_per_layer=True)}


def prefill_drop_order(mode, eligible, scores=None):
    """Order in which a prefill baseline drops eligible source positions (first dropped first)."""
    if mode == 'streaming':
        return list(eligible)  # StreamingLLM keeps sinks and the most recent context: drop the oldest source
    if mode == 'snapkv':
        return sorted(eligible, key=lambda p: (scores[p], p))  # lowest pooled attention first
    raise ValueError(f'Not a prefill baseline: {mode}')


def render_longproc_prompt(public, apply_chat_template):
    """The LongProc user prompt: task framing, the HTML page, then the public task text."""
    content = ("[TASK]\nYour task is to extract specific information from an HTML webpage and output the extracted "
               "information in a tsv file. You will be first given an HTML webpage. Then, you should follow the specific "
               "instruction provided later and output the tsv file following the format provided in the instruction.\n\n"
               f"[INPUT WEBPAGE]\n```html\n{public.document}\n```\n\n{public.fields[0].description}")
    prompt = apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if prompt.count(public.document) != 1:
        raise ValueError('Source must occur exactly once in rendered prompt')
    start = prompt.index(public.document)
    return prompt, (start, start + len(public.document))


def condition_forecast_seconds(max_new_tokens, seconds_per_token=0.16, overhead=30):
    """Worst-case wall time of one condition; long-output pages must not start past the shard deadline."""
    return overhead + seconds_per_token * max_new_tokens


def run_cursor_study(config, model, tokenizer, output, *, resume=False, max_seconds=None, max_cases=None,
                     checkpoint=lambda: None, group_forecast_seconds=None, identity_config=None):
    """Runs each LongProc case under every configured condition, resumable by case, condition, config and data."""
    import importlib.metadata
    import random
    from dataclasses import asdict
    from pathlib import Path
    from orderkv.experiment import RunStore, digest, load_cases
    from orderkv.metrics import row_f1
    config.validate()
    identity_config = identity_config or config
    cases = load_cases(config)
    budget_file = json.loads(Path(config.baseline_budget_path).read_text()) if config.baseline_budget_path else {}
    budgets = budget_file.get('drop_tokens', {})
    schedules = budget_file.get('source_schedule', {})
    seconds = min(config.shard_seconds, max_seconds) if max_seconds is not None else config.shard_seconds
    deadline = time.perf_counter() + seconds
    store = RunStore(output)
    # Resume by case, condition, config and data: scoring or selection-code edits between
    # shards must not re-run finished generations (the code hash still lands in each record).
    finished = set()
    runs_file = Path(output) / 'runs.jsonl'
    if resume and runs_file.exists():
        for line in runs_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                # Greedy decoding is deterministic, so a capped generation is a finished result too.
                if row.get('status') in ('complete', 'capped') and row.get('config_hash') == identity_config.digest():
                    finished.add((row['case_id'], row['condition'], row.get('data_hash')))
    source_hash = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    environment_hash = digest(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
    records = []
    attempted = 0
    forecast = group_forecast_seconds if group_forecast_seconds is not None else condition_forecast_seconds(config.max_new_tokens)
    for public, gold in cases:
        if time.perf_counter() + forecast >= deadline:
            break
        if max_cases is not None and attempted >= max_cases:
            break
        prompt, _ = render_longproc_prompt(public, tokenizer.apply_chat_template)
        size = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        if not any(abs(size - target) <= target * config.context_tolerance for target in config.context_targets):
            store.save(digest({'case': public.case_id, 'skip': 'length', 'config': identity_config.digest()}),
                       {'case_id': public.case_id, 'condition': 'skipped', 'status': 'skipped_length', 'prompt_tokens': size})
            continue
        attempted += 1
        conditions = list(config.conditions)
        random.Random(f'{config.seed}:{public.case_id}').shuffle(conditions)
        for name in conditions:
            key = digest({'revision': config.model_revision, 'code': source_hash, 'config': identity_config.digest(),
                          'data': digest([asdict(public), asdict(gold)]), 'case': public.case_id,
                          'condition': name, 'repetition': 0, 'environment': environment_hash})
            if resume and (store.completed(key)
                           or (public.case_id, name, digest([asdict(public), asdict(gold)])) in finished):
                continue
            if time.perf_counter() + forecast >= deadline:
                return records  # the next shard resumes this page's remaining condition
            options = dict(CONDITIONS[name])
            if options.get('mode') in ('streaming', 'snapkv', 'snapkv_head', 'tova'):
                if public.case_id not in budgets:
                    continue  # baselines run only where a matched memory budget exists
                options['drop_tokens'] = int(budgets[public.case_id])
                if options['mode'] == 'tova':
                    options['schedule'] = tuple(schedules[public.case_id])
            condition = CursorCondition(name, margin_blocks=config.cursor_margin_blocks, max_new_tokens=config.max_new_tokens, block_tokens=config.block_tokens,
                                        protect_first=config.protect_prompt_first,
                                        protect_last=config.protect_prompt_last, **options)
            try:
                record = run_cursor_case(public, condition, model, tokenizer, deadline=deadline)
                record['quality'] = row_f1(record['raw_output'], gold.values['tsv'])
            except Exception as error:
                record = {'case_id': public.case_id, 'condition': name, 'status': 'failed',
                          'error': f'{type(error).__name__}: {error}'}
            record.update(model_revision=config.model_revision, model_id=config.model_id, code_hash=source_hash,
                          config_hash=identity_config.digest(), data_hash=digest([asdict(public), asdict(gold)]),
                          seed=config.seed, environment_hash=environment_hash)
            store.save(key, record)
            checkpoint()
            records.append(record)
            if record['status'] in ('deadline', 'failed'):
                return records
    return records


@torch.inference_mode()
def run_cursor_case(case_public, condition: CursorCondition, model, tokenizer, *, deadline=None):
    """Greedy free-text decode of a TSV; with `release`, evict source blocks behind the row cursor."""
    if condition.max_new_tokens < 1 or model.training or condition.margin_blocks < 0:
        raise ValueError('Positive token cap, eval mode, and nonnegative margin required')
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    sync()
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    prompt, span = render_longproc_prompt(case_public, tokenizer.apply_chat_template)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    history = list(encoded['input_ids'])
    prompt_length = len(history)
    offsets = encoded['offset_mapping']
    source = map_source(offsets, span, condition.block_tokens, condition.protect_first, condition.protect_last)
    state = LogicalPositions.from_prompt(prompt_length, source)
    locator = RowLocator(case_public.document)
    tracker = RowTracker()
    blocked = set()  # mask mode: logical positions hidden from attention but still stored
    source_starts = [offsets[p][0] for p in source.positions]
    record = {'case_id': case_public.case_id, 'split': case_public.split, 'condition': condition.name,
              'field_order': ['tsv'], 'prompt_tokens': prompt_length,
              'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest(), 'status': 'pending', 'raw_output': '',
              'generated_tokens': 0, 'eviction_events': [], 'forward_bytes': [], 'timings_seconds': {},
              'rows_completed': 0, 'rows_located': 0, 'margin_blocks': condition.margin_blocks,
              'deferred_effect': 'first token after compaction uses pre-compaction logits'}
    record['timings_seconds']['setup'] = time.perf_counter() - started
    cache = DynamicCache()
    prefill_start = time.perf_counter()
    head_state = None
    if condition.mode in ('snapkv', 'snapkv_head'):
        split = prompt_length - condition.window
        model(torch.tensor([history[:split]], device=device), use_cache=True, past_key_values=cache, logits_to_keep=1)
        positions = tuple(range(split, prompt_length))
        window_mask = causal_mask(positions, tuple(range(prompt_length)), dtype=dtype, device=device)
        original = model.config._attn_implementation
        try:
            model.config._attn_implementation = 'eager'
            out = model(torch.tensor([history[split:]], device=device), past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([positions], device=device),
                        cache_position=torch.tensor(positions, device=device), attention_mask=window_mask,
                        output_attentions=True, logits_to_keep=1)
        finally:
            model.config._attn_implementation = original
        if condition.mode == 'snapkv_head':
            from orderkv import headmask
            eligible_mask = torch.zeros(prompt_length, dtype=torch.bool, device=device)
            eligible_mask[[p for p in range(split) if p not in state.protected and state.source_blocks[p] is not None]] = True
            head_state = headmask.HeadMaskState(prompt_length, eligible_mask, headmask.snapkv_head_blocks(
                out.attentions, split, eligible_mask, condition.drop_tokens, model.config.num_attention_heads
                // model.config.num_key_value_heads, condition.pool))
        attention = torch.stack([layer[0, :, :, :split].float().mean(dim=(0, 1)) for layer in out.attentions]).mean(dim=0)
        pooled = torch.nn.functional.max_pool1d(attention[None, None], condition.pool, stride=1,
                                                padding=condition.pool // 2)[0, 0, :split]
        scores = pooled.tolist()
    else:
        out = model(torch.tensor([history], device=device), use_cache=True, past_key_values=cache, logits_to_keep=1)
        scores = None
    logits = out.logits[:, -1].clone()
    del out
    sync()
    record['timings_seconds']['prefill'] = time.perf_counter() - prefill_start
    if condition.mode in ('streaming', 'snapkv') and condition.drop_tokens > 0:
        limit = prompt_length - condition.window if condition.mode == 'snapkv' else prompt_length
        eligible = [p for p in state.kept if p < limit and p not in state.protected and state.source_blocks[p] is not None]
        count = condition.drop_tokens
        dropped = set(prefill_drop_order(condition.mode, eligible, scores)[:count])
        keep = tuple(i for i, pos in enumerate(state.kept) if pos not in dropped)
        result = compact_cache(cache, keep, state)
        state = result.positions
        record['eviction_events'].append({'row': 0, 'prefill_baseline': condition.mode, 'dropped_tokens': len(dropped),
                                          'requested_tokens': count, 'eligible_tokens': len(eligible),
                                          'physical_before': prompt_length,
                                          'physical_after': len(state.kept),
                                          'bytes_released': result.before.storage - result.after.storage,
                                          'first_affected_forward_step': 0})
    record['drop_tokens'] = condition.drop_tokens
    protected_source = sum(1 for p in source.positions if p in state.protected)
    if condition.mode == 'tova':
        from orderkv import headmask
        eligible_mask = torch.zeros(prompt_length, dtype=torch.bool, device=device)
        eligible_mask[[p for p in range(prompt_length) if p not in state.protected and state.source_blocks[p] is not None]] = True
        head_state = headmask.HeadMaskState(prompt_length, eligible_mask, per_layer=condition.tova_per_layer)
    original_attention = None
    layers = kv_heads = 1
    if head_state is not None:
        original_attention = model.config._attn_implementation
        layers, kv_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
        headmask.ACTIVE = head_state
        model.config._attn_implementation = headmask.NAME
    loop_start = time.perf_counter()
    generated = []
    cursor_block = 0
    recent_rows = []
    stop_repeating = False
    try:
        for step in range(condition.max_new_tokens):
            if deadline is not None and time.perf_counter() >= deadline:
                record['status'] = 'deadline'
                break
            if stop_repeating:
                record['status'] = 'complete'
                record['stopped_on_repetition'] = True
                break
            token = logits[0].argmax().item()
            if token == tokenizer.eos_token_id:
                record['status'] = 'complete'
                break
            history.append(token)
            generated.append(token)
            logical = state.next_position
            mask = causal_mask((logical,), state.kept + (logical,), dtype=dtype, device=device,
                               blocked=frozenset(blocked))
            if condition.mode == 'tova':
                target = condition.schedule[min(step, len(condition.schedule) - 1)] if condition.schedule else prompt_length
                head_state.tova_target = max(0, int(target) - protected_source)
            # RoPE takes the original position: kept keys were rotated at their original positions, so the
            # new query must be too, however many rows compaction removed.
            out = model(torch.tensor([[token]], device=device), past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[logical]], device=device),
                        cache_position=torch.tensor([logical], device=device), attention_mask=mask)
            logits = out.logits[:, -1].clone()
            del out
            state = state.append()
            sizes = cache_bytes(cache)
            per_token = sizes.tensors // len(state.kept)
            source_rows = sum(p < prompt_length and state.source_blocks[p] is not None and p not in blocked
                              for p in state.kept)
            hidden = head_state.mean_blocked(layers, kv_heads) if head_state is not None else 0.0
            record['forward_bytes'].append({'step': step, 'logical_position': logical, 'live': sizes.storage,
                                            'tensor': sizes.tensors, 'source': source_rows * per_token,
                                            'generated': len(generated) * per_token,
                                            'effective': sizes.storage - hidden * per_token})
            text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).rstrip('�')
            record['raw_output'] = text
            for row in tracker.update(text):
                record['rows_completed'] += 1
                recent_rows.append(tuple(row))
                if len(recent_rows) >= 3 and len(set(recent_rows[-3:])) == 1:
                    stop_repeating = True  # same row three times in a row: a decoding loop
                if tracker.rows == 1:
                    continue  # the header row is not evidence
                located = locator.locate(row)
                if located is None:
                    continue
                record['rows_located'] += 1
                index = bisect.bisect_right(source_starts, located[0]) - 1
                block = source.block_ids[max(0, index)]
                cursor_block = max(cursor_block, block)
                if not condition.release:
                    continue
                release_before = cursor_block - condition.margin_blocks
                if condition.mode == 'mask':
                    newly = {pos for pos in state.kept if pos < prompt_length and pos not in state.protected
                             and state.source_blocks[pos] is not None and state.source_blocks[pos] < release_before
                             and pos not in blocked}
                    if newly:
                        blocked |= newly
                        record['eviction_events'].append({
                            'row': tracker.rows, 'cursor_block': cursor_block, 'release_before_block': release_before,
                            'logical_position': logical, 'masked_total': len(blocked), 'masked_new': len(newly),
                            'seconds': 0.0, 'first_affected_forward_step': step + 1})
                    continue
                keep = tuple(i for i, pos in enumerate(state.kept)
                             if pos >= prompt_length or pos in state.protected
                             or state.source_blocks[pos] is None or state.source_blocks[pos] >= release_before)
                if len(keep) != len(state.kept):
                    sync()
                    compact_start = time.perf_counter()
                    result = compact_cache(cache, keep, state)
                    sync()
                    record['eviction_events'].append({
                        'row': tracker.rows, 'cursor_block': cursor_block, 'release_before_block': release_before,
                        'logical_position': logical, 'physical_before': len(state.kept),
                        'physical_after': len(result.positions.kept),
                        'bytes_released': result.before.storage - result.after.storage,
                        'seconds': time.perf_counter() - compact_start, 'first_affected_forward_step': step + 1})
                    state = result.positions
        else:
            record['status'] = 'capped'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        if head_state is not None:
            headmask.ACTIVE = None
            model.config._attn_implementation = original_attention
    sync()
    record['generated_tokens'] = len(generated)
    record['timings_seconds']['decode_including_compaction'] = time.perf_counter() - loop_start
    record['timings_seconds']['end_to_end'] = time.perf_counter() - started
    steps = record['forward_bytes']
    record['memory_bytes'] = {'average_decode_kv': sum(s['live'] for s in steps) / len(steps) if steps else None,
                              'kv_byte_steps': sum(s['live'] for s in steps),
                              # Per-head baselines keep storage allocated and mask rows; count only unmasked rows.
                              'effective_kv_byte_steps': sum(s.get('effective', s['live']) for s in steps),
                              'cuda_max_allocated': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    return record
