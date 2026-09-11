"""Figure 3: one fact makes many cross-structure comparisons available.

Two by three.  Columns are depths, sharing one logarithmic axis of epochs since
the linking fact.  Nothing before the intervention is drawn.  Both arms leave
identical pre-switch weights; only one then receives the fact.

Top row is the percentage of the eighty withheld cross-structure comparisons
that have STABLY resolved, counted from the first evaluation after each one's
last failure.  Instantaneous top-1 accuracy is not that quantity and can fall,
which is why it cannot carry a panel labelled as comparisons resolved.

Bottom row is the distance from the predicted point to the entity it should
have retrieved, in units of the median spacing between candidates in the same
learned embedding.  No ground-truth alignment enters, which matters here: the
unlinked world has one global offset of the copy that no fact fixes, so any
measure taken against a chosen ground truth lets a control that is converging
perfectly well appear to get worse, as the free coordinate settles at the
minimum-norm point rather than at the chosen one.
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
XLIM = (0, 3000)       # everything happens inside this; the runs go to 10k
BASE_RATE = 0.03       # the locked step size, and the unit of the epoch axis.
# The mean dynamics depend on the product of rate and epochs, so a record run
# at another rate is drawn against epochs rescaled by BASE_RATE / its rate.
# The prediction is then literally the same curve and only the discrete process
# differs, which is what makes a rate comparison readable.  See
# tests/check_lr3.py.
XTICKS = ([0, 1000, 2000, 3000], ["0", "1k", "2k", "3k"])
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}


def after(rec, arm, src, key):
    """One arm's series, on an axis of epochs since the linking fact.

    The behavioural curve is chance-corrected against WHAT THE ARM SCORES AT
    THE SWITCH.  At that instant no fact in the world has constrained the
    alignment between the two copies, both arms hold identical weights, and the
    residual is a rigid translation of one copy: measured per item, its spread
    is under 0.05 spacings while its size is six or seven.  Anything retrieved
    correctly there is retrieved without the information the question is about,
    so it is the baseline by construction.

    The best CONSTANT answer, 9 of 80, is the right baseline only when that
    translation is big enough to collapse every query onto one corner entity.
    Depths 1 and 3 land there and score exactly 11.2%; the depth-2 translation
    points elsewhere, projects queries onto a whole face of the target block
    and scores 31.2%, which a constant-answer correction leaves as 22.5% of
    apparent compositional accuracy before the fact arrives.  Measured by
    tests/check_readout.py.
    """
    d = rec["arms"][arm][src]
    sc = BASE_RATE / rec.get("lr_target", BASE_RATE)
    ep = (np.array(d["epochs"], float) - rec["t_switch"]) / sc
    m = ep >= 0
    if key == "resolved":
        h = np.array(d["hits"]["cross"], bool)
        k = int(np.argmin(np.abs(ep)))
        v = resolved(np.array(d["epochs"], float), h, chance=float(h[k].mean()))
    else:
        v = np.array(d[key]["cross"], float)
    return ep[m], v[m]


def main(sub="f3", out="fig3_integration.png"):
    """`sub` is the results sub-directory, so a run at another step size can be
    drawn with the same code into its own file."""
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    npairs, lrt = 80, BASE_RATE

    for col, depth in enumerate(DEPTHS):
        files = sorted(glob.glob(os.path.join(RESULTS, sub,
                                              "w*_d%d.json" % depth)))
        if not files:
            for row in (0, 1):
                axes[row][col].set_visible(False)
            continue
        recs = [json.load(open(f)) for f in files]
        npairs, lrt = recs[0]["n_pairs"], recs[0]["lr_target"]

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                for src, kw in (("net", dict(color=c, lw=1.8, zorder=3)),
                                ("pred", PRED)):
                    ser = [after(r, arm, src, key) for r in recs]
                    ax.plot(ser[0][0], np.mean([v for _, v in ser], axis=0), **kw)
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
                # the depth 3 control spikes to about 100 spacings early on,
                # so the ceiling has to clear that; the floor is the smallest
                # value any arm reaches.  The reference is ONE candidate
                # spacing, the measured retrieval boundary: pooled over every
                # item and evaluation, retrieval is 100% below it and falls
                # away above it, and the epoch at which the mean error crosses
                # it is the epoch the row above reaches ceiling, at all three
                # depths.  That is what links the two rows.
                ax.set_ylim(2e-3, 150)
                ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                lab = "Epochs since the linking fact"
                if lrt != BASE_RATE:
                    lab += "\n" + r"(rescaled to $\eta = %g$)" % BASE_RATE
                ax.set_xlabel(lab)
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
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
