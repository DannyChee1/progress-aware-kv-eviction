"""Effective release margin of the live cursor-release runs: after every located row, the distance in 128-token
blocks from the furthest matched row to the release point the run used. Needs full records with model outputs and
the Llama tokenizer. Usage: python scripts/effective_margin.py [--outputs outputs] [--out FILE]
"""
import argparse
import bisect
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer

from orderkv.config import load_config
from orderkv.cursor import RowLocator, RowTracker, render_longproc_prompt
from orderkv.experiment import load_cases
from orderkv.spans import map_source

RUNS = (('Qwen2.5-7B held-out', 'configs/longproc-0.5k-test-r0.yaml',
         '841279964948f4391c2ef6dc50da3b7540ff787ccb873706f3824a907d1a5175', 'tokenizer'),
        ('Llama-3.1-8B held-out', 'configs/longproc-0.5k-llama-r0.yaml',
         'ab9547cd12a952c2beb1a5937b7916b53c2c602a7dad4a3691fe9342cca95d95', 'llama-tokenizer'),
        ('Qwen2.5-7B 8K tier', 'configs/longproc-8k-test-r0.yaml',
         'a7fb36a23fe64019a60a5eaf5d9131db7c970cf10e873f90fe2604f3ed6e6b24', 'tokenizer'))


def margins(config, series, tokenizer_path, margin):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    cases = {public.case_id: public for public, _ in load_cases(config)}
    out = []
    for line in open(series / 'runs.jsonl'):
        record = json.loads(line)
        if record.get('condition') != 'R0' or record['status'] not in ('complete', 'capped'):
            continue
        public = cases[record['case_id']]
        prompt, span = render_longproc_prompt(public, tokenizer.apply_chat_template)
        offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        source = map_source(offsets, span, 128, 32, 64)
        starts = [offsets[p][0] for p in source.positions]
        block = lambda offset: source.block_ids[max(0, bisect.bisect_right(starts, offset) - 1)]
        locator, tracker, furthest, cursor = RowLocator(public.document), RowTracker(), 0, 0
        for row in tracker.update(record['raw_output']):
            located = None if tracker.rows == 1 else locator.locate(row)
            if located is None:
                continue
            furthest = max(furthest, block(located[0] + span[0]))
            # The run placed its cursor by comparing the page offset with prompt offsets, as run_cursor_case does.
            cursor = max(cursor, block(located[0]))
            out.append(furthest - (cursor - margin))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--out', type=Path, default=Path('results/effective-margin.json'))
    args = parser.parse_args()
    report = {}
    for label, config_path, series, tokenizer in RUNS:
        config = load_config(config_path)
        values = margins(config, args.outputs / series, tokenizer, config.cursor_margin_blocks)
        report[label] = {'configured_margin_blocks': config.cursor_margin_blocks, 'located_rows': len(values),
                         'mean_blocks': statistics.mean(values), 'median_blocks': statistics.median(values),
                         'min_blocks': min(values), 'max_blocks': max(values)}
        print(label, report[label])
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
