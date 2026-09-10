"""Figure 1: compositional abstraction emerges at a time specific to each law.

Two by three.  Columns are depths, sharing one logarithmic epoch axis so the
acceleration with depth is visible rather than hidden by per-column scaling.
Top row is the behavioural measure, bottom row the geometric one, with the
parameter-free prediction overlaid on every curve.

The behavioural measure is the percentage of a law's held-out composites that
have STABLY resolved, counted from the first evaluation after each item's last
failure.  Instantaneous top-1 accuracy is the wrong object for a panel about
emergence: with a modest number of held-out items, single items cross and
recross a nearest-neighbour boundary, so the curve steps and reverses while
learning is monotone underneath.

Four ladder rungs are drawn rather than all sixteen laws.  Sixteen curves plus
sixteen predictions cannot be read, and at depth the laws compress into one
bundle.  The other twelve appear in panel g, predicted against observed
emergence, which is where the full set carries information.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence, resolved
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
NLAW = 4
XLIM = (3e2, 3e4)


def cells(depth):
    return sorted(glob.glob(os.path.join(RESULTS, "f1", "w*_d%d.json" % depth)))


def emergence(ep, rec):
    return {k: (float(ep[int(i)]) if np.isfinite(i) else np.inf)
            for k, i in ((k, detect_emergence(rec["retrieval"][k],
                                              rec["geometric"][k],
                                              S.hold, S.tau))
                         for k in rec["retrieval"])}


def curve(recs, src, key, law, ep):
    if key == "resolved":
        return np.mean([resolved(ep, np.array(r[src]["hits"][law], bool))
                        for r in recs], axis=0)
    return np.mean([r[src][key][law] for r in recs], axis=0)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    cols = plt.cm.viridis(np.linspace(0.08, 0.85, NLAW))
    nw = 0

    for col, depth in enumerate(DEPTHS):
        files = cells(depth)
        nw = max(nw, len(files))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        ep = np.array(recs[0]["net"]["epochs"], float)
        lrt = recs[0]["lr_target"]
        order = sorted(emergence(ep, recs[0]["net"]).items(), key=lambda kv: kv[1])
        names = [k for k, _ in order]
        pick = [names[int(round(v))] for v in np.linspace(0, len(names) - 1, NLAW)]

        for row, key, ylab in (
                (0, "resolved", "Held-out composites\nstably resolved  (%)"),
                (1, "geometric", "Composition error\n"
                                 r"$\|z-(x{+}y)\|\,/\,\|x{+}y\|$")):
            ax = axes[row][col]
            for k, law in enumerate(pick):
                ax.plot(ep, curve(recs, "net", key, law, ep), color=cols[k],
                        lw=1.7, zorder=3)
                ax.plot(ep, curve(recs, "pred", key, law, ep), **PRED)
            ax.set_xscale("log")
            ax.set_xlim(*XLIM)
            if row == 0:
                ax.set_ylim(-4, 106)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                        % (depth, lrt), transform=ax.transAxes, fontsize=9,
                        ha="center", va="bottom", color=DARK)
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 4)
                ax.axhline(S.tau, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                ax.set_xlabel("Training epoch")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    axes[0][0].plot([], [], color=cols[0], lw=1.7, label="Earliest rung")
    axes[0][0].plot([], [], color=cols[-1], lw=1.7, label="Latest rung")
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    p = figure("fig1_emergence.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s  (%d world%s)" % (p, nw, "" if nw == 1 else "s"))


if __name__ == "__main__":
    main()
