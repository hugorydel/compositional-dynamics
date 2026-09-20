"""Full-run event times for all 16 laws in the first 200 complete worlds."""
import json
from pathlib import Path

import _paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from relspec.config import DEFAULT
from relspec.measure import detect_emergence


def load_times():
    records = {}
    for depth in (1, 2, 3):
        records[depth] = {int(p.stem.split('_')[0][1:]): p
                          for p in Path(_paths.RESULTS, 'f1').glob(f'w*_d{depth}.json')}
    seeds = sorted(set.intersection(*(set(r) for r in records.values())))[:200]
    if len(seeds) != 200:
        raise ValueError(f'Expected 200 complete worlds; found {len(seeds)}')
    result = {}
    for depth in (1, 2, 3):
        times = {src: [] for src in ('net', 'pred')}
        for seed in seeds:
            r = json.loads(records[depth][seed].read_text())
            laws = sorted(r['net']['retrieval'])
            for src in times:
                rec = r[src]
                row = []
                for law in laws:
                    index = detect_emergence(rec['retrieval'][law], rec['geometric'][law],
                                             DEFAULT.hold, DEFAULT.tau)
                    row.append(rec['epochs'][int(index)] if np.isfinite(index) else np.nan)
                times[src].append(row)
        result[depth] = {src: np.asarray(v) for src, v in times.items()}
        print(f'Depth {depth}: loaded 200 worlds', flush=True)
    return laws, result


def main(from_summary=False):
    if from_summary:
        with np.load(Path(_paths.RESULTS, '_timing_summary.npz')) as saved:
            laws = saved['laws'].tolist()
            data = {d: {src: saved[f'd{d}_{src}'] for src in ('net', 'pred')}
                    for d in (1, 2, 3)}
    else:
        laws, data = load_times()
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 7.3), constrained_layout=True)
    fmt = FuncFormatter(lambda x, pos: f'{x/1000:g}k' if x >= 1000 else f'{x:g}')
    for col, depth in enumerate((1, 2, 3)):
        obs, pred = data[depth]['net'], data[depth]['pred']
        if not np.all(np.isfinite(obs) & np.isfinite(pred)):
            raise ValueError('This figure assumes all law-worlds attain the criterion')
        ax = axes[0, col]
        lo = 0.85 * min(obs.min(), pred.min())
        hi = 1.15 * max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color='#a0a0a0', lw=1, zorder=1)
        ax.scatter(obs.ravel(), pred.ravel(), s=9, color='#1b6ca8', alpha=.22,
                   edgecolors='none', rasterized=True, zorder=2)
        ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi),
               xlabel='Observed epoch', title=f'Depth {depth} · n = 3,200')
        if col == 0:
            ax.set_ylabel('Predicted epoch')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ticks = {1: [2000, 10000, 50000], 2: [750, 1500, 3000],
                 3: [750, 1000, 1500, 2000]}[depth]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_formatter(NullFormatter())
        ax = axes[1, col]
        q25, med, q75 = np.percentile(obs, [25, 50, 75], axis=0)
        ax.errorbar(np.arange(len(laws)), med, yerr=[med-q25, q75-med], fmt='o',
                    color='#1b6ca8', markersize=4, elinewidth=1.2, capsize=2)
        ax.set_xticks(np.arange(len(laws)))
        ax.set_xticklabels(laws, rotation=90, fontsize=11)
        ax.set_yscale('log')
        ax.set_xlabel('Law identity')
        if col == 0:
            ax.set_ylabel('Observed emergence epoch\nmedian and interquartile range')
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_locator(FixedLocator({1: [4000, 10000, 20000],
                                                2: [1000, 1500, 2000],
                                                3: [1100, 1300, 1500]}[depth]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f'Medians: {med.min()/1000:g}k–{med.max()/1000:.3g}k', fontsize=12)
        for row in (0, 1):
            panel = axes[row, col]
            panel.spines[['top', 'right']].set_visible(False)
            panel.text(-.12, 1.06, 'abcdef'[row*3+col], transform=panel.transAxes,
                       weight='bold', fontsize=12)
    for extension in ('pdf', 'png'):
        output = Path(_paths.FIGURES, 'fig4_timing.' + extension)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print('Wrote', output)
    np.savez_compressed(Path(_paths.RESULTS, '_timing_summary.npz'), laws=np.asarray(laws),
                        **{f'd{d}_{src}': data[d][src] for d in data for src in ('net', 'pred')})
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-summary', action='store_true',
                        help='Redraw the previously generated event-time summary')
    main(parser.parse_args().from_summary)
