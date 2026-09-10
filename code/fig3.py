"""Figure 3: one fact makes many cross-structure comparisons available.

Two by three.  Columns are depths.  Top row is the behavioural measure, how many
of the eighty withheld cross-lattice comparisons are resolved; bottom row is the
geometric one, the offset error in units of one relation step.  Prediction
overlaid on every curve.

One lattice duplicated, the copy unanchored, so no comparison across the two is
determined.  Both arms share one trajectory up to the dashed line; only then
does one arm receive the single linking fact.  The x axis is epochs since that
fact.

The two rows disagree about timing on purpose, and that disagreement is the
result.  A rank test is invariant to a global scale the offset error is not, so
the comparisons become behaviourally available long before the geometry is
metrically right.  At depth 1 the offset error does not converge at any
affordable budget, and that panel is meant to be read as unconverged.

The control arm has no geometric plateau either.  Its offset error drifts upward
as the free coordinate creeps toward the minimum-norm solution.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
ARMS = (("hold", "#c0392b", "No linking fact"),
        ("insert", "#1b6ca8", "One linking fact"))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.42)
    npairs = 80

    for col, depth in enumerate(DEPTHS):
        files = sorted(glob.glob(os.path.join(RESULTS, "f3",
                                              "w*_d%d.json" % depth)))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        npairs = recs[0]["n_pairs"]
        sw = recs[0]["t_switch"]
        lrt = recs[0]["lr_target"]
        ep = np.array(recs[0]["arms"]["insert"]["net"]["epochs"], float) - sw

        for row, key in ((0, "retrieval"), (1, "geometric")):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                obs = np.mean([r["arms"][arm]["net"][key]["cross"]
                               for r in recs], axis=0)
                pre = np.mean([r["arms"][arm]["pred"][key]["cross"]
                               for r in recs], axis=0)
                if key == "retrieval":
                    obs, pre = obs * npairs / 100.0, pre * npairs / 100.0
                ax.plot(ep, obs, color=c, lw=1.7, zorder=3)
                ax.plot(ep, pre, **PRED)
            ax.axvline(0, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(ep[0], ep[-1])
            if row == 0:
                ax.set_ylim(-4, npairs + 5)
                ax.set_yticks([0, npairs // 4, npairs // 2, npairs])
                ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                        % (depth, lrt), transform=ax.transAxes,
                        fontsize=9, ha="center", va="bottom", color=DARK)
                if col == 0:
                    ax.set_ylabel("Cross-structure comparisons\nresolved  (of %d)"
                                  % npairs, linespacing=1.6)
            else:
                ax.set_yscale("log")
                ax.set_ylim(3e-2, 8)
                ax.set_xlabel("Epochs since the linking fact")
                if col == 0:
                    ax.set_ylabel("Offset error\n(units of one $x$ step)",
                                  linespacing=1.6)
            if col:
                ax.set_yticklabels([])
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.24 if col == 0 else -0.08, dy=1.16)

    for arm, c, lab in ARMS:
        axes[0][0].plot([], [], color=c, lw=1.7, label=lab)
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=1.8, handlelength=1.8)
    p = figure("fig3_integration.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
