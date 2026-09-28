"""README charts from the published run records, each in a light and a dark variant.

accuracy: row F1 at equal decode memory on the held-out pages where cursor release frees memory, both models.
memory: KV cache held at each decode step, full cache vs cursor release, on the held-out Qwen page with the
median saving. Usage: python scripts/make_charts.py [--runs runs] [--out figures]
"""
import argparse
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from results import LLAMA, QWEN, Pages  # noqa: E402

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', secondary='#52514e', muted='#898781', grid='#e1e0d9',
                  axis='#c3c2b7', cursor='#2a78d6', full='#eb6834', other='#c3c2b7'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', secondary='#c3c2b7', muted='#898781', grid='#2c2c2a',
                 axis='#383835', cursor='#3987e5', full='#d95926', other='#52514e'),
}
METHODS = (('R0', 'Cursor release'), ('T1', 'TOVA'), ('L0', 'StreamingLLM'), ('S1', 'SnapKV'))
MODELS = (('Qwen2.5-7B-Instruct', QWEN, ('qwen_r0', 'qwen_uniform', 'qwen_perhead', 'qwen_tova_layer')),
          ('Llama-3.1-8B-Instruct', LLAMA, ('llama_r0', 'llama_baselines', 'llama_tova_layer')))


def style(ax, t):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], labelcolor=t['secondary'], length=0)
    ax.grid(axis='x', color=t['grid'], linewidth=1)
    ax.set_axisbelow(True)


def accuracy(runs, out, theme):
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), dpi=200, facecolor=t['surface'])
    for ax, (name, data, series) in zip(axes, MODELS):
        p = Pages(runs, data, series)
        pages = p.paired(('F0',) + tuple(k for k, _ in METHODS))
        full = statistics.mean(p.score(c, 'F0') for c in pages)
        style(ax, t)
        for y, (kind, label) in enumerate(METHODS):
            value = statistics.mean(p.score(c, kind) for c in pages)
            ax.barh(y, value, height=0.42, color=t['cursor'] if kind == 'R0' else t['other'])
            # Values sit inside the bar end so they never collide with the full-cache line.
            ax.text(value - 0.015, y, f'{value:.2f}', ha='right', va='center', fontsize=8.5,
                    color='#ffffff' if kind == 'R0' else t['ink'])
        ax.axvline(full, color=t['full'], linewidth=2)
        ax.text(full + 0.012, -0.62, f'full cache {full:.2f}', ha='left', va='bottom', fontsize=8.5, color=t['secondary'])
        ax.set_yticks(range(len(METHODS)), [label for _, label in METHODS])
        ax.set_ylim(len(METHODS) - 0.5, -0.8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(f'{name}, {len(pages)} pages', loc='left', fontsize=10, color=t['ink'])
        ax.set_xlabel('Row F1', fontsize=8.5, color=t['secondary'])
    fig.suptitle('Accuracy when every method keeps the same decode memory', x=0.01, ha='left', fontsize=11.5,
                 color=t['ink'])
    fig.tight_layout()
    fig.savefig(out / f'accuracy-{theme}.png', facecolor=t['surface'])
    plt.close(fig)


def memory(runs, out, theme):
    t = THEMES[theme]
    p = Pages(runs, QWEN, ('qwen_r0',))
    saving = p.total_saving('R0')
    engaged = sorted((saving(c), c) for c in p.paired(('F0', 'R0'), {c for c, _ in p.rec}) if saving(c) > 0.001)
    share, page = engaged[len(engaged) // 2]
    fig, ax = plt.subplots(figsize=(9, 3.1), dpi=200, facecolor=t['surface'])
    style(ax, t)
    ax.grid(axis='y', color=t['grid'], linewidth=1)
    ax.grid(axis='x', visible=False)
    for kind, label, color in (('F0', 'Full cache', t['full']), ('R0', 'Cursor release', t['cursor'])):
        steps = p.rec[(page, kind)]['forward_bytes']
        x, y = [s['step'] for s in steps], [s['live'] / 1e9 for s in steps]
        ax.plot(x, y, color=color, linewidth=2, solid_joinstyle='round', solid_capstyle='round', label=label)
        ax.plot(x[-1], y[-1], 'o', markersize=5, color=color, markeredgecolor=t['surface'], markeredgewidth=2)
        ax.text(x[-1] + 8, y[-1], f'{label} {y[-1]:.2f} GB', va='center', fontsize=8.5, color=t['secondary'])
    ax.set_ylim(0, None)
    ax.set_xlim(0, x[-1] * 1.32)
    ax.set_xlabel('Output tokens generated', fontsize=8.5, color=t['secondary'])
    ax.set_ylabel('KV cache held (GB)', fontsize=8.5, color=t['secondary'])
    legend = ax.legend(loc='lower left', frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(t['secondary'])
    ax.set_title(f"KV cache held during decoding: Qwen2.5-7B on held-out page {page.split('-', 1)[1]} ({share:.0%} saved)",
                 loc='left', fontsize=11, color=t['ink'])
    fig.tight_layout()
    fig.savefig(out / f'memory-{theme}.png', facecolor=t['surface'])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        accuracy(args.runs, args.out, theme)
        memory(args.runs, args.out, theme)
    print(f'wrote {sorted(p.name for p in args.out.glob("*.png"))}')


if __name__ == '__main__':
    main()
