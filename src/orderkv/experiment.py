"""Result storage and case loading for the LongProc runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from orderkv.config import load_config
from orderkv.data import CasePublic, FieldSpec, GoldPrivate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RunStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def completed(self, key):
        marker = self.directory / f'{key}.complete'
        if not marker.exists():
            return False
        record = json.loads((self.directory / marker.read_text()).read_text())
        return record['status'] == 'complete' and record['completion_key'] == key

    def save(self, key, record):
        filename = f'{key}-{uuid.uuid4().hex}.json'
        payload = dict(record, completion_key=key)
        with (self.directory / filename).open('x') as file:
            json.dump(payload, file)
            file.flush()
            os.fsync(file.fileno())
        with (self.directory / 'runs.jsonl').open('a') as file:
            file.write(json.dumps(payload) + '\n')
            file.flush()
            os.fsync(file.fileno())
        if record['status'] == 'complete':
            temporary = self.directory / f'{key}.tmp'
            temporary.write_text(filename)
            os.replace(temporary, self.directory / f'{key}.complete')


def load_cases(config):
    public_rows = [json.loads(line) for line in Path(config.dataset).read_text().splitlines() if line.strip()]
    private_rows = [json.loads(line) for line in Path(config.private_labels).read_text().splitlines() if line.strip()]
    labels = {row['case_id']: row for row in private_rows}
    if len(labels) != len(private_rows) or len({r['case_id'] for r in public_rows}) != len(public_rows):
        raise ValueError('Duplicate case IDs')
    if set(labels) != {r['case_id'] for r in public_rows}:
        raise ValueError('Public/private case IDs differ')
    cases = []
    for row in public_rows:
        expected_split = Path(config.dataset).stem.removesuffix('_public')
        if set(row) != {'case_id', 'split', 'document', 'fields'} or row['split'] != expected_split:
            raise ValueError(f'Expected public-only {expected_split} input')
        public = CasePublic(row['case_id'], row['split'], row['document'], tuple(FieldSpec(**f) for f in row['fields']))
        if [f.field_id for f in public.fields] != ['tsv']:
            raise ValueError('Expected one TSV field')
        private = dict(labels[public.case_id])
        del private['case_id']
        gold = GoldPrivate(**private)
        if set(gold.values) != {f.field_id for f in public.fields}:
            raise ValueError('Gold/schema mismatch')
        cases.append((public, gold))
    return cases[:config.case_limit]


def main():
    """Validate a config and its cases; print the results directory name the run will use."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config = load_config(parser.parse_args().config)
    print(json.dumps({'config_hash': config.digest(), 'cases': len(load_cases(config)), 'conditions': config.conditions}))


if __name__ == '__main__':
    main()
