"""How the post-link trajectory changes with the step size.

The mean dynamics depend on the product of learning rate and epochs, so on an
axis of rescaled epochs the prediction is one curve, the same at every rate.
Only the discrete process moves.  Drawing the prediction once and overlaying
the networks makes the approach visible: the lighter the line, the coarser the
step, and the gap that opens at depth 3 closes as the step shrinks.

Writes `figures/fig3_rates.png`.  Reads whatever rate directories exist, so a
rate still running is simply absent rather than an error.
"""

import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from style import DARK, panel

import fig3

RATES = ((0.03, "#a8cbe3"), (0.01, "#6aa6cd"),
         (0.006, "#2e7ab0"), (0.003, "#0b3d62"))
DEPTHS = (1, 2, 3)
SEED = 0
XT = fig3.ticks_for(fig3.WINDOW)


def sub_for(rate):
    return "f3" if rate == 0.03 else "f3_lr%s" % ("%g" % rate).replace(".", "p")


def load(rate, depth):
    p = os.path.join(RESULTS, sub_for(rate), "w%02d_d%d.json" % (SEED, depth))
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    have = []

    for col, depth in enumerate(DEPTHS):
        for row, key, ylab in ((0, "resolved", "Compositional Accuracy\n(held-out)"),
                               (1, "geometric", "Geometric Error")):
            ax = axes[row][col]
            drawn = False
            for rate, c in RATES:
                rec = load(rate, depth)
                if rec is None:
                    continue
                e, v = fig3.after(rec, "insert", "net", key, rescale=True)
                ax.plot(e, v, color=c, lw=1.6, zorder=3)
                if not drawn:          # the prediction is rate-independent here
                    ep, q = fig3.after(rec, "insert", "pred", key,
                                       rescale=True)
                    ax.plot(ep, q, **fig3.PRED)
                    drawn = True
                if rate not in have:
                    have.append(rate)
            ax.set_xlim(0, fig3.WINDOW)
            ax.set_xticks(XT[0])
            ax.set_xticklabels(XT[1])
            if row == 0:
                ax.set_ylim(-4, 106)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.text(0.5, 1.05, r"$N = %d$" % depth, transform=ax.transAxes,
                        fontsize=9, ha="center", va="bottom", color=DARK)
            else:
                ax.set_yscale("log")
                ax.set_ylim(2e-3, 20)
                ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)), zorder=1)
                ax.set_xlabel("Epochs since the linking fact\n"
                              + r"(rescaled to $\eta = 0.03$)")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    axes[0][0].set_ylabel("Cross-structure comparisons\nstably resolved  (%% of 80)",
                          linespacing=1.6)
    for rate, c in RATES:
        if rate in have:
            axes[0][0].plot([], [], color=c, lw=1.6,
                            label=r"$\eta_{\mathrm{target}} = %g$" % rate)
    axes[0][0].plot([], [], label="Prediction (one curve, all rates)", **fig3.PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=len(h), handletextpad=0.6, columnspacing=1.8, handlelength=1.8)
    p = figure("fig3_rates.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s  (rates drawn: %s)"
          % (p, ", ".join("%g" % r for r in have)))


if __name__ == "__main__":
    main()
