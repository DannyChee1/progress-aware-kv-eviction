"""Fetch LongProc at the pinned commit and rebuild the page and gold files listed by each committed manifest.

LongProc's HTML pages and gold tables are not redistributed here. Each data/longproc-*/<split>_manifest.json names
its cases and the SHA-256 of both files; every rebuilt file is checked against it.
Usage: python scripts/fetch_longproc.py [--root external/longproc]
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from orderkv.longproc import build_case

URL = 'https://github.com/princeton-pli/LongProc'
COMMIT = '673ec4c230876e941d674116d66d28783c186b15'


def checkout(root):
    if not (root / '.git').exists():
        subprocess.run(['git', 'clone', '--quiet', '--filter=blob:none', '--no-checkout', URL, str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--quiet', COMMIT], check=True)
    return root / 'data' / 'html_to_tsv'


def rebuild(manifest_path, tasks, source, template):
    manifest = json.loads(manifest_path.read_text())
    split = manifest_path.name.removesuffix('_manifest.json')
    public, private = [], []
    for entry in manifest['cases']:
        case, gold = build_case(tasks[entry['realized']['task_id']], source, template, split)
        if case.case_id != entry['case_id']:
            raise ValueError(f'{manifest_path}: expected {entry["case_id"]}, built {case.case_id}')
        public.append(asdict(case))
        private.append({'case_id': case.case_id, **asdict(gold)})
    for name, rows in ((f'{split}_public.jsonl', public), (f'{split}_private.jsonl', private)):
        payload = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()
        if hashlib.sha256(payload).hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'{manifest_path.parent / name}: SHA-256 differs from the manifest')
        (manifest_path.parent / name).write_bytes(payload)
    return len(public)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('external/longproc'))
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    source = checkout(args.root)
    template = yaml.safe_load((source / 'prompts.yaml').read_text())['USER_PROMPT']
    tasks = {r['task_id']: r for tier in ('0.5k', '2k', '8k')
             for r in json.loads((source / f'html_to_tsv_{tier}.json').read_text())}
    for manifest_path in sorted(args.data.glob('longproc-*/*_manifest.json')):
        print(f'{manifest_path.parent}: {rebuild(manifest_path, tasks, source, template)} cases, hashes match')


if __name__ == '__main__':
    main()
