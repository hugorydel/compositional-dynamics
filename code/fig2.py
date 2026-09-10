"""Figure 2: identifiability is where the evidence sits, not whether it exists.

Two by three.  Columns are depths.  Top row is held-out composite retrieval,
bottom row is composition error.  Prediction overlaid on every curve.

Two structurally identical blocks with matched premise and composite counts.
They differ only in placement: the closed block's composites close a triangle on
triads that carry their premises, the open block's sit on ungrounded pairs.  One
bridging fact is inserted at the dashed line, and the x axis is epochs since
that fact, so the three columns align at zero despite different budgets.

Both counterbalance arms are drawn.  If the effect is structural it must follow
the open placement rather than the block label, so the two arms overlap.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec.config import DEFAULT as S
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
CLOSED, OPEN = "#1b6ca8", "#c0392b"
YLAB = {
    "retrieval": "Held-out composite\nretrieval  (%)",
    "geometric": "Composition error\n" r"$\|z-(x{+}y)\|\,/\,\|x{+}y\|$",
}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.42)

    for col, depth in enumerate(DEPTHS):
        files = sorted(glob.glob(os.path.join(RESULTS, "f2",
                                              "w*_d%d_*.json" % depth)))
        if not files:
            continue
        lrt = json.load(open(files[0]))["lr_target"]
        for row, key in ((0, "retrieval"), (1, "geometric")):
            ax = axes[row][col]
            lim = None
            for f in files:
                r = json.load(open(f))
                ep = np.array(r["net"]["epochs"], float) - r["t_switch"]
                lim = (ep[0], ep[-1])
                for law in r["net"][key]:
                    c = OPEN if law == r["open"] else CLOSED
                    ax.plot(ep, r["net"][key][law], color=c, lw=1.6,
                            alpha=0.85, zorder=3)
                    ax.plot(ep, r["pred"][key][law], **PRED)
            ax.axvline(0, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(*lim)
            if row == 0:
                ax.set_ylim(-4, 106)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                        % (depth, lrt), transform=ax.transAxes,
                        fontsize=9, ha="center", va="bottom", color=DARK)
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 6)
                ax.axhline(S.tau, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                ax.set_xlabel("Epochs since the bridging fact")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(YLAB[key], linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.24 if col == 0 else -0.08, dy=1.16)

    axes[0][0].plot([], [], color=CLOSED, lw=1.6,
                    label=r"Closed placement   $\rho = 0$")
    axes[0][0].plot([], [], color=OPEN, lw=1.6,
                    label=r"Open placement   $\rho = 0.258$")
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=1.8, handlelength=1.8)
    p = figure("fig2_identifiability.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
