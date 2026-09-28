"""Turn full run records into their published form: no output text, LongProc's official score, and a SHA-256 of each
output so identical-output counts can still be checked. Series directories must be named by their config's hash.
Usage: python scripts/publish_records.py [--outputs outputs] [--runs runs]
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

from orderkv.config import series_id
from orderkv.metrics import row_f1

TEXT_FIELDS = ('raw_output', 'generated_token_ids')


def published(record, gold):
    out = {k: v for k, v in record.items() if k not in TEXT_FIELDS}
    text = record.get('raw_output')
    if text is not None:
        out['output_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        if record['case_id'] in gold:
            out['quality'] = row_f1(text, gold[record['case_id']])
    return out


def write(rows, path, gold):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(published(r, gold)) + '\n' for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    args = parser.parse_args()
    gold = {}
    for name in glob.glob('data/longproc-*/*_private.jsonl'):
        for line in open(name):
            row = json.loads(line)
            gold[row['case_id']] = row['values']['tsv']
    for config in sorted(args.outputs.glob('*/config.json')):
        series = config.parent
        values = json.loads(config.read_text())
        if series_id(values) != series.name:
            raise ValueError(f'{series}: directory name is not the hash of its config')
        rows = [json.loads(l) for l in (series / 'runs.jsonl').read_text().splitlines() if l.strip()]
        write(rows, args.runs / series.name / 'runs.jsonl', gold)
        (args.runs / series.name / 'config.json').write_text(config.read_text())
        print(f'{series.name}: {len(rows)} records')


if __name__ == '__main__':
    main()
