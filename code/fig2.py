"""Figure 2: one bridging fact makes an already-experienced composite learnable.

Two by three.  Columns are depths, sharing one logarithmic axis of epochs since
the bridge.  Nothing before the intervention is drawn; the pre-switch phase
established that the premises and the composite were experienced, and that
belongs in the caption rather than in negative x.

Exactly two curves per panel, and they are the same law: the one whose evidence
sits on a detached pair, branched at the switch into receiving the bridging
fact or not.  Both branches leave identical pre-switch weights.  An earlier
version compared that law against the already-identifiable one and gave the
bridge to both arms, which is no control at all, since both are learnable
afterwards and both errors fall.

The two counterbalance arms are averaged, because they are the same condition
with the block labels swapped.  Whether the effect follows the placement rather
than the label is a separate check and belongs in a supplementary panel.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec.config import DEFAULT as S
from relspec.measure import resolved
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
ARMS = (("hold", "#c0392b", "No bridging fact"),
        ("insert", "#1b6ca8", "One bridging fact"))
XLIM = (0, 4000)       # the depth 3 budget, so every column spans its panel;
XTICKS = ([0, 1000, 2000, 3000, 4000],   # depths 1 and 2 run on to 40k and 8k
          ["0", "1k", "2k", "3k", "4k"])
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}


def after(rec, arm, src, key, law):
    """One arm's post-switch series, with the switch at zero."""
    d = rec["arms"][arm][src]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    m = ep >= 0
    if key == "resolved":
        h = np.array(d["hits"][law], bool)
        # the record carries the best-constant-answer rate of its own
        # evaluation set.  The reciprocal of the item count coincides with it
        # only while every held-out target is distinct, which is a property of
        # the world rather than something the figure should rely on.
        v = resolved(np.array(d["epochs"], float), h,
                     chance=rec["chance"][law])
    else:
        v = np.array(d[key][law], float)
    return ep[m], v[m]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)

    for col, depth in enumerate(DEPTHS):
        # exactly the two counterbalance arms.  A dose series over `n_bridge`
        # lives in the same directory and answers a different question, so it
        # must not be swept up by a wildcard.
        files = sorted(f for a in ("A", "B")
                       for f in glob.glob(os.path.join(
                           RESULTS, "f2", "w*_d%d_%s.json" % (depth, a))))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        lrt = recs[0]["lr_target"]

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                for src, kw in (("net", dict(color=c, lw=1.8, zorder=3)),
                                ("pred", PRED)):
                    ser = [after(r, arm, src, key, r["open"]) for r in recs]
                    ep = ser[0][0]
                    ax.plot(ep, np.mean([v for _, v in ser], axis=0), **kw)
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
                ax.set_ylim(1e-3, 6)
                ax.axhline(S.tau, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)),
                           zorder=1)
                ax.set_xlabel("Epochs since the bridging fact")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    for arm, c, lab in ARMS:
        axes[0][0].plot([], [], color=c, lw=1.8, label=lab)
    axes[0][0].plot([], [], label="Prediction (no fitted parameters)", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    p = figure("fig2_identifiability.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
