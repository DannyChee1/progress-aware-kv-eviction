"""Paired summaries of the LongProc HTML-to-TSV runs, written to results/.

Row F1 is LongProc's official evaluator: recomputed from raw output when a record has it, otherwise the score
stored in the published record. Matched-memory intervals resample whole websites (10,000 draws, random.Random(0),
one generator per report); held-out intervals resample pages (10,000 draws, numpy seed 0).
Usage: python scripts/results.py [--runs runs] [--out results]
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np

from orderkv.metrics import row_f1

QWEN, LLAMA, QWEN_8K = 'data/longproc-0.5k', 'data/longproc-0.5k-llama', 'data/longproc-8k'
SERIES = {
    'qwen_r0': '841279964948f4391c2ef6dc50da3b7540ff787ccb873706f3824a907d1a5175',
    'qwen_uniform': '403f48a53427af6b687a234100bc8d1f75fd58eab4af8ec08c501ab9efa4a402',
    'qwen_perhead': '0700aed152079eaa602fa193b86ba7de21071b302d13ee77901a0d06717c6a46',
    'qwen_tova_layer': '3362b481f78fb708f27a7922a774d783a8557aa4cfe0fb3dacf747b3dec9b433',
    'qwen_8k_r0': 'a7fb36a23fe64019a60a5eaf5d9131db7c970cf10e873f90fe2604f3ed6e6b24',
    'llama_r0': 'ab9547cd12a952c2beb1a5937b7916b53c2c602a7dad4a3691fe9342cca95d95',
    'llama_baselines': 'ae31e8a3c4e0c97f6efb63b5d4c13afa3cb3c256244d0f447834f2576da54bb3',
    'llama_tova_layer': '2f8817cc4687135d3fba6a6b37cfaf21b8d1e4acba816194dcf8e42f83312249',
}


def output_id(record):
    text = record.get('raw_output')
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else record['output_sha256']


class Pages:
    def __init__(self, runs, data, series, statuses=('complete', 'capped')):
        private = Path(f'{data}/test_private.jsonl')
        self.gold = {json.loads(l)['case_id']: json.loads(l)['values']['tsv'] for l in open(private)} if private.exists() else {}
        self.site = {c['case_id']: c['realized']['website_id'] for c in json.load(open(f'{data}/test_manifest.json'))['cases']}
        budget = Path(f'{data}/baseline_budget.json')
        self.budget = json.load(open(budget))['drop_tokens'] if budget.exists() else {}
        self.rec = {}
        for name in series:
            for line in open(runs / SERIES[name] / 'runs.jsonl'):
                r = json.loads(line)
                if r['status'] in statuses:
                    self.rec[(r['case_id'], r['condition'])] = r
        self.f1 = {}

    def paired(self, kinds, candidates=None):
        candidates = self.budget if candidates is None else candidates
        return sorted(c for c in candidates if all((c, k) in self.rec for k in kinds))

    def score(self, c, k):
        if (c, k) not in self.f1:
            r = self.rec[(c, k)]
            self.f1[(c, k)] = row_f1(r['raw_output'], self.gold[c])['f1'] if 'raw_output' in r else r['quality']['f1']
        return self.f1[(c, k)]

    def change(self, a, b='F0'):
        return lambda c: self.score(c, a) - self.score(c, b)

    def effective(self, c, k):
        memory = self.rec[(c, k)]['memory_bytes']
        return memory.get('effective_kv_byte_steps', memory['kv_byte_steps'])

    def per_step(self, k):
        """Decode-KV saving per generated token against full cache on the same page."""
        def saving(c):
            full = self.rec[(c, 'F0')]['memory_bytes']['kv_byte_steps']
            return 1 - (self.effective(c, k) / self.rec[(c, k)]['generated_tokens']) / (full / self.rec[(c, 'F0')]['generated_tokens'])
        return saving

    def total_saving(self, k):
        return lambda c: 1 - self.effective(c, k) / self.rec[(c, 'F0')]['memory_bytes']['kv_byte_steps']

    def identical(self, pages, k, other='F0'):
        return sum(output_id(self.rec[(c, k)]) == output_id(self.rec[(c, other)]) for c in pages)

    def capped(self, pages, k):
        return sum(self.rec[(c, k)]['status'] == 'capped' for c in pages)


class Bootstrap:
    def __init__(self, pages, site):
        self.sites = defaultdict(list)
        for c in pages:
            self.sites[site[c]].append(c)
        self.keys, self.rng = list(self.sites), random.Random(0)

    def ci(self, fn):
        draws = sorted(statistics.mean(fn(c) for c in [c for s in (self.rng.choice(self.keys) for _ in self.keys)
                                                          for c in self.sites[s]]) for _ in range(10000))
        return [draws[250], draws[9750]]


def mean(fn, pages):
    return statistics.mean(fn(c) for c in pages)


def matched_memory(p, kinds, names, total_saving):
    pages = p.paired(kinds)
    boot = Bootstrap(pages, p.site)
    out = {'pages': len(pages), 'sites': len(boot.sites), 'conditions': {}}
    for k in kinds:
        d = p.change(k)
        row = {'name': names[k], 'mean_f1': mean(lambda c: p.score(c, k), pages), 'f1_change': mean(d, pages),
               'ci95_site': boot.ci(d)}
        if total_saving:
            row['kv_saving_mean_ratio'] = mean(p.total_saving(k), pages)
        row.update(kv_per_step_saving=mean(p.per_step(k), pages), capped=p.capped(pages, k),
                   pages_losing_5pp=sum(d(c) < -0.05 for c in pages))
        out['conditions'][k] = row
    return pages, out


def qwen_baselines(runs):
    p = Pages(runs, QWEN, ('qwen_r0', 'qwen_uniform', 'qwen_perhead'))
    names = {'F0': 'full cache', 'R0': 'cursor release', 'T0': 'TOVA per KV head', 'L0': 'StreamingLLM-style',
             'S1': 'SnapKV per-head', 'S0': 'SnapKV head-uniform'}
    _, out = matched_memory(p, ('F0', 'R0', 'T0', 'L0', 'S1', 'S0'), names, total_saving=True)
    return 'qwen-baselines.json', {'model': 'Qwen/Qwen2.5-7B-Instruct', **out}


def llama_baselines(runs):
    p = Pages(runs, LLAMA, ('llama_r0', 'llama_baselines'))
    names = {'F0': 'full cache', 'R0': 'cursor release', 'T0': 'TOVA per KV head', 'L0': 'StreamingLLM-style',
             'S1': 'SnapKV per-head'}
    pages, out = matched_memory(p, ('F0', 'R0', 'T0', 'L0', 'S1'), names, total_saving=False)
    return 'llama-baselines.json', {'model': 'meta-llama/Llama-3.1-8B-Instruct', **out,
                                    'missing_pages': sorted(set(p.budget) - set(pages))}


def tova(p, model, cursor_in_same_draws):
    kinds = ('F0', 'R0', 'T0', 'T1')
    names = {'F0': 'full cache', 'R0': 'cursor release', 'T0': 'TOVA per KV head', 'T1': 'TOVA per layer (default)'}
    pages = p.paired(kinds)
    boot = Bootstrap(pages, p.site)
    out = {'model': model, 'pages': len(pages), 'sites': len(boot.sites), 'conditions': {}}
    for k in kinds:
        d = p.change(k)
        out['conditions'][k] = {'name': names[k], 'mean_f1': mean(lambda c: p.score(c, k), pages),
                                'f1_change': mean(d, pages), 'ci95_site': boot.ci(d),
                                'kv_per_step': mean(p.per_step(k), pages), 'lose_5pp': sum(d(c) < -0.05 for c in pages),
                                'capped': p.capped(pages, k)}
    fn = p.change('T1', 'T0')
    out['T1_minus_T0'] = {'mean': mean(fn, pages), 'ci95_site': boot.ci(fn)}
    fn = p.change('R0', 'T1')
    if cursor_in_same_draws:
        out['R0_minus_T1'] = {'mean': mean(fn, pages), 'ci95_site': boot.ci(fn)}
    else:
        own = p.paired(('F0', 'R0', 'T1'))
        out['R0_minus_T1'] = {'mean': mean(fn, own), 'ci95_site': Bootstrap(own, p.site).ci(fn)}
    return out


def qwen_tova(runs):
    p = Pages(runs, QWEN, ('qwen_r0', 'qwen_perhead', 'qwen_tova_layer'))
    return 'qwen-tova.json', tova(p, 'Qwen/Qwen2.5-7B-Instruct', cursor_in_same_draws=False)


def llama_tova(runs):
    p = Pages(runs, LLAMA, ('llama_r0', 'llama_baselines', 'llama_tova_layer'))
    return 'llama-tova.json', tova(p, 'meta-llama/Llama-3.1-8B-Instruct', cursor_in_same_draws=True)


def page_bootstrap(values, statistic, resamples=10000):
    indices = np.random.default_rng(0).integers(0, len(values), size=(resamples, len(values)))
    draws = [statistic(values[i]) for i in indices]
    return {'estimate': float(statistic(values)), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def held_out(runs):
    """Cursor release against full cache on every held-out page where both runs finished before the token cap."""
    out = {}
    for label, data, series in (('Qwen2.5-7B-Instruct, 0.5K', QWEN, 'qwen_r0'),
                                ('Llama-3.1-8B-Instruct, 0.5K', LLAMA, 'llama_r0'),
                                ('Qwen2.5-7B-Instruct, 8K', QWEN_8K, 'qwen_8k_r0')):
        p = Pages(runs, data, (series,), statuses=('complete',))
        pages = p.paired(('F0', 'R0'), {c for c, _ in p.rec})
        f1 = np.array([[p.score(c, 'R0'), p.score(c, 'F0')] for c in pages])
        kv = np.array([[p.rec[(c, k)]['memory_bytes']['kv_byte_steps'] for k in ('R0', 'F0')] for c in pages], dtype=float)
        out[label] = {'pages': len(pages), 'full_cache_f1': float(f1[:, 1].mean()), 'cursor_f1': float(f1[:, 0].mean()),
                      'f1_change': page_bootstrap(f1, lambda v: float(np.mean(v[:, 0] - v[:, 1]))),
                      'kv_saving': page_bootstrap(kv, lambda v: float(1 - v[:, 0].sum() / v[:, 1].sum())),
                      'identical_outputs': p.identical(pages, 'R0')}
    return 'held-out.json', out


REPORTS = (qwen_baselines, llama_baselines, qwen_tova, llama_tova, held_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for report in REPORTS:
        filename, content = report(args.runs)
        (args.out / filename).write_text(json.dumps(content, indent=1) + '\n')
        print(f'wrote {args.out / filename}')


if __name__ == '__main__':
    main()
