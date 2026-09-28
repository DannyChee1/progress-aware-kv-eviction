"""Choose the release margin on development pages only (CPU, no model).

Gold rows stand in for model output. Each row is located in the page (first match after the previous row), and the
decode is replayed with release behind the cursor at each candidate margin. A violation is a row whose text starts
in a block already released when the row is written. The rule: the smallest margin with at most 0.5% violations.
Usage: python scripts/margin_audit.py [--root external/longproc/data/html_to_tsv]
"""
import argparse
import bisect
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer

from orderkv.cursor import squash

MARGINS = (0, 2, 4, 8, 16)


def locate_rows(page, rows, window=1500):
    """Each row's span: the earliest anchor match after the previous row with all its cells within the window."""
    low, back = squash(page)
    cursor, spans = 0, []
    for cells in rows:
        cells = [''.join(c.lower().split()) for c in cells if c.strip() and c.strip().upper() != 'N/A']
        anchor = max(cells, key=len) if cells else None
        positions, start_at = None, cursor
        while anchor:
            index = low.find(anchor, start_at)
            if index < 0:
                break
            found = []
            for cell in cells:
                at = low.find(cell, max(0, index - window), index + window)
                if at < 0:
                    found = None
                    break
                found.append((back[at], back[min(at + len(cell), len(back) - 1)]))
            if found:
                positions = found
                break
            start_at = index + 1
        if not positions:
            spans.append(None)
            continue
        start = min(p[0] for p in positions)
        spans.append((start, max(p[1] for p in positions)))
        cursor = max(cursor, bisect.bisect_left(back, start))
    return spans


def audit(record, root, tokenizer, margin, block_tokens=128):
    page = (root / record['html_path']).read_text(errors='replace')
    lines = record['gt'].strip().split('\n')
    rows = [line.split('\t') for line in lines[1:]]
    spans = locate_rows(page, rows)
    reliable = [max((len(c) for c in r if c.strip()), default=0) >= 6 for r in rows]
    offsets = tokenizer(page, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    n, starts = len(offsets), [a for a, _ in offsets]
    block_at = lambda char: min(n - 1, bisect.bisect_right(starts, char) - 1) // block_tokens
    lengths = [max(1, len(tokenizer.encode(lines[r + 1] + '\n', add_special_tokens=False))) for r in range(len(rows))]
    prompt = n + 400  # page plus LongProc's framing, approximately
    full = sum(prompt + s for s in range(1, sum(lengths) + 1))
    area = step = cursor = violations = located = 0
    for r, span in enumerate(spans):
        released = max(0, cursor - margin)
        live = n - released * block_tokens
        for _ in range(lengths[r]):
            step += 1
            area += prompt - n + live + step
        if span is None or not reliable[r]:
            continue
        located += 1
        block = block_at(span[0])
        violations += block < released
        cursor = max(cursor, block)
    return 1 - area / full, violations, located


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('external/longproc/data/html_to_tsv'))
    parser.add_argument('--tokenizer', default='tokenizer')
    parser.add_argument('--out', type=Path, default=Path('results/margin-dev.json'))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dev = json.loads((args.root / 'html_to_tsv_0.5k.json').read_text())[:40]  # the dev split: first 40 pages
    rule = 'smallest margin in {0,2,4,8,16} with violation rate <= 0.5% on dev pages only'
    sweep = {}
    for margin in MARGINS:
        results = [audit(r, args.root, tokenizer, margin) for r in dev]
        sweep[margin] = (statistics.mean(x[0] for x in results), sum(x[1] for x in results) / max(1, sum(x[2] for x in results)))
        print(f'margin {margin:>2}: saving {sweep[margin][0]:.1%}, violations {sweep[margin][1]:.2%}')
    chosen = min(m for m, (_, v) in sweep.items() if v <= 0.005)
    args.out.write_text(json.dumps({'rule': rule, 'pages': 40, 'split': 'dev (first 40 of the 0.5k tier)',
                                    'sweep': {str(m): {'saving': s, 'violation_rate': v} for m, (s, v) in sweep.items()},
                                    'chosen_margin': chosen}, indent=1) + '\n')


if __name__ == '__main__':
    main()
