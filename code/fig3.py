"""Figure 3: one fact makes many cross-structure comparisons available.

Two by three.  Columns are depths, sharing one logarithmic axis of epochs since
the linking fact.  Nothing before the intervention is drawn.  Both arms leave
identical pre-switch weights; only one then receives the fact.

Top row is the percentage of the eighty withheld cross-structure comparisons
that have STABLY resolved, counted from the first evaluation after each one's
last failure.  Instantaneous top-1 accuracy is not that quantity and can fall,
which is why it cannot carry a panel labelled as comparisons resolved.

Bottom row is the share of the offset error lying along the coordinate the
linking fact removes.  The unlinked world has a one-dimensional null space, one
global offset of the copy that the facts do not fix, and every pair is equally
exposed to it.  Measuring the total distance to one arbitrary ground-truth
alignment instead lets a control that is converging perfectly well appear to
get worse, because the free coordinate settles at the minimum-norm point, which
sits further from that alignment than the initialisation does.  Normalised so
an undetermined arm sits at one.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec.measure import resolved
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
ARMS = (("hold", "#c0392b", "No linking fact"),
        ("insert", "#1b6ca8", "One linking fact"))
XLIM = (2e1, 1e5)


def after(rec, arm, src, key):
    d = rec["arms"][arm][src]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    m = ep > 0
    if key == "resolved":
        v = resolved(np.array(d["epochs"], float),
                     np.array(d["hits"]["cross"], bool))
    else:
        v = np.array(d[key]["cross"], float)
    return ep[m], v[m]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    npairs = 80

    for col, depth in enumerate(DEPTHS):
        files = sorted(glob.glob(os.path.join(RESULTS, "f3",
                                              "w*_d%d.json" % depth)))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        npairs, lrt = recs[0]["n_pairs"], recs[0]["lr_target"]

        for row, key, ylab in (
                (0, "resolved", "Cross-structure comparisons\n"
                                "stably resolved  (%)"),
                (1, "geometric", "Offset error along the\n"
                                 "freed coordinate")):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                for src, kw in (("net", dict(color=c, lw=1.8, zorder=3)),
                                ("pred", PRED)):
                    ser = [after(r, arm, src, key) for r in recs]
                    ax.plot(ser[0][0], np.mean([v for _, v in ser], axis=0), **kw)
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
                ax.set_ylim(1e-4, 3)
                ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                ax.set_xlabel("Epochs since the linking fact")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    axes[0][0].set_ylabel("Cross-structure comparisons\nstably resolved  "
                          "(%% of %d)" % npairs, linespacing=1.6)
    for arm, c, lab in ARMS:
        axes[0][0].plot([], [], color=c, lw=1.8, label=lab)
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    p = figure("fig3_integration.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
