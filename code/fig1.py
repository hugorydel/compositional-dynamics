"""Figure 1: compositional abstraction emerges at a time specific to each law.

Two by three.  Columns are depths.  Top row is the behavioural measure, held-out
composite retrieval; bottom row is the geometric one, composition error.  The
parameter-free predicted trajectory is overlaid on every curve.

Sixteen laws differing only in the spectral structure of their lattices, each
withholding most of its composites.  Colour runs from the law the theory places
first to the one it places last, so the ordering is legible without a sixteen
entry legend.

The x axis is logarithmic and per column, because the budgets differ by nearly
an order of magnitude: depth accelerates emergence in epochs, so depths 2 and 3
finish in a fraction of depth 1's window.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.7, ls=(0, (2.2, 2.0)), zorder=6)


def law_colours(n):
    """A sequential ramp over the sixteen laws, ordered by predicted
    emergence, so the ordering reads off the colour without a legend of
    sixteen entries."""
    return plt.cm.viridis(np.linspace(0.08, 0.92, n))


def cells(depth):
    return sorted(glob.glob(os.path.join(RESULTS, "f1", "w*_d%d.json" % depth)))


def emergence(ep, rec):
    out = {}
    for k in rec["retrieval"]:
        i = detect_emergence(rec["retrieval"][k], rec["geometric"][k], S.hold, S.tau)
        out[k] = float(ep[int(i)]) if np.isfinite(i) else np.inf
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.42)
    n_worlds = 0

    for col, depth in enumerate(DEPTHS):
        files = cells(depth)
        n_worlds = max(n_worlds, len(files))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        ep = np.array(recs[0]["net"]["epochs"], float)
        lrt = recs[0]["lr_target"]
        order = sorted(emergence(ep, recs[0]["net"]).items(), key=lambda kv: kv[1])
        names = [k for k, _ in order]
        cols = dict(zip(names, law_colours(len(names))))

        for row, key, ylab in ((0, "retrieval",
                                "Held-out composite\nretrieval  (%)"),
                               (1, "geometric",
                                "Composition error\n"
                                r"$\|z-(x{+}y)\|\,/\,\|x{+}y\|$")):
            ax = axes[row][col]
            for nm in names:
                obs = np.mean([r["net"][key][nm] for r in recs], axis=0)
                pre = np.mean([r["pred"][key][nm] for r in recs], axis=0)
                ax.plot(ep, obs, color=cols[nm], lw=1.25, zorder=3)
                ax.plot(ep, pre, **PRED)
            ax.set_xscale("log")
            ax.set_xlim(max(ep[1], 100), ep[-1])
            if row == 0:
                ax.set_ylim(-4, 106)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                        % (depth, lrt), transform=ax.transAxes,
                        fontsize=9, ha="center", va="bottom", color=DARK)
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 4)
                ax.axhline(S.tau, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                ax.set_xlabel("Training epoch")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    axes[0][0].plot([], [], color=law_colours(2)[0], lw=1.4, label="Earliest law")
    axes[0][0].plot([], [], color=law_colours(2)[-1], lw=1.4, label="Latest law")
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    p = figure("fig1_emergence.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s  (%d world%s)" % (p, n_worlds, "" if n_worlds == 1 else "s"))


if __name__ == "__main__":
    main()
