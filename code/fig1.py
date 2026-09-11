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
NLAW = 3               # Law A, B, C
LAW_COLOURS = ("#1b6ca8", "#b8860b", "#c0392b")  # matches F2/F3
XLIM = (0, 5000)       # standardised across all three figures
XTICKS = ([0, 1000, 2000, 3000, 4000, 5000],
          ["0", "1k", "2k", "3k", "4k", "5k"])
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}



def cells(depth, worlds=None):
    """Every world's cell for this depth, or only the listed ones.

    Restricting the set is how the effect of adding a world is checked: the
    same code path draws one world and several, so a difference between the
    two renders is the data and not the plotting.
    """
    f = sorted(glob.glob(os.path.join(RESULTS, "f1", "w*_d%d.json" % depth)))
    if worlds is None:
        return f
    keep = {"w%02d_d%d.json" % (w, depth) for w in worlds}
    return [p for p in f if os.path.basename(p) in keep]


def emergence(ep, rec):
    return {k: (float(ep[int(i)]) if np.isfinite(i) else np.inf)
            for k, i in ((k, detect_emergence(rec["retrieval"][k],
                                              rec["geometric"][k],
                                              S.hold, S.tau))
                         for k in rec["retrieval"])}


def stack(recs, src, key, law, ep):
    """One row per world, so the mean and the spread come from the same array.

    Drawing only the mean of a handful of worlds hides whether the laws are
    separated in every world or only on average, which is the whole claim.
    """
    if key == "resolved":
        H = [np.array(r[src]["hits"][law], bool) for r in recs]
        # zero means "no better than answering the same entity every time"
        return np.array([resolved(ep, h, chance=1.0 / h.shape[1]) for h in H])
    return np.array([r[src][key][law] for r in recs], float)


def draw(ax, ep, V, **kw):
    """Mean across worlds."""
    ax.plot(ep, V.mean(axis=0), **kw)


def main(worlds=None, out="fig1_emergence.png"):
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    cols = LAW_COLOURS
    nw = 0

    for col, depth in enumerate(DEPTHS):
        files = cells(depth, worlds)
        nw = max(nw, len(files))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        ep = np.array(recs[0]["net"]["epochs"], float)
        lrt = recs[0]["lr_target"]
        # order by MEAN emergence across worlds, not world 0's alone, so the
        # three drawn laws do not change identity when a world is added
        em = [emergence(ep, r["net"]) for r in recs]
        order = sorted(em[0], key=lambda k: np.mean([e[k] for e in em]))
        names = list(order)
        pick = [names[int(round(v))] for v in np.linspace(0, len(names) - 1, NLAW)]

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for k, law in enumerate(pick):
                draw(ax, ep, stack(recs, "net", key, law, ep),
                     color=cols[k], lw=1.7, zorder=3)
                draw(ax, ep, stack(recs, "pred", key, law, ep), **PRED)
            ax.set_xlim(*XLIM)
            ax.set_xticks(XTICKS[0])
            ax.set_xticklabels(XTICKS[1])
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

    for k, nm in enumerate("ABC"[:NLAW]):
        axes[0][0].plot([], [], color=cols[k], lw=1.7, label="Law %s" % nm)
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s  (%d world%s)" % (p, nw, "" if nw == 1 else "s"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, nargs="+", default=None,
                    help="world seeds to include; default is every cell found")
    ap.add_argument("--out", default="fig1_emergence.png")
    ns = ap.parse_args()
    main(ns.worlds, ns.out)
