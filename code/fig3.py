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
WINDOW = 3000          # width of the axis, in epochs at BASE_RATE
BASE_RATE = 0.03       # the rate WINDOW is quoted in
# The axis is drawn in REAL epochs, so the numbers under it are the numbers a
# run used.  WINDOW is quoted at BASE_RATE only so that one constant keeps the
# same amount of learning in view whatever rate a record was produced at: the
# mean dynamics depend on the product of rate and epochs, so at a tenth the
# rate the same window is ten times the epochs.
#
# The comparison figure in tests/fig_rates.py does rescale, because four rates
# have to share one axis there and the prediction is a single curve on it.
# That is the only place the transformation earns its keep, and even there the
# axis says so.  A panel showing one rate should not: it would print epochs
# nobody ran and invite a reader to reproduce it with the wrong budget.
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}


BAND = "sem"           # "sem", a (lo, hi) percentile pair, or None for min-max
# What the shading is for decides which of these is right.  A spread band
# answers "how much do worlds differ", and over twenty worlds min-to-max
# answers it with two draws and widens as worlds are added.  A standard-error
# band answers "how well is the plotted curve pinned down", which is the
# question a curve drawn against a prediction raises, and it narrows as worlds
# are added.  The spread is still worth quoting in the text; it does not belong
# on a panel whose subject is agreement.


def spread(V, key):
    """Lower and upper edges of the shading, in the space the axis uses.

    The geometric row is logarithmic, so its centre is a geometric mean and its
    error is taken on the logarithms before being mapped back.  Doing it in the
    linear space would put the band off-centre on the drawn curve.  The
    behavioural row is a bounded percentage and is clipped to its range.
    """
    n = len(V)
    if n < 2:
        return None
    if BAND is None:
        return V.min(0), V.max(0)
    if BAND != "sem":
        return (np.percentile(V, BAND[0], axis=0),
                np.percentile(V, BAND[1], axis=0))
    W = np.log(np.maximum(V, 1e-12)) if key == "geometric" else V
    m, se = W.mean(0), W.std(0, ddof=1) / np.sqrt(n)
    if key == "geometric":
        return np.exp(m - se), np.exp(m + se)
    return np.clip(m - se, 0.0, 100.0), np.clip(m + se, 0.0, 100.0)


def summarise(V, key):
    """Across-world centre for one series.

    The geometric row is drawn on a logarithmic axis, where an arithmetic mean
    is the wrong centre: it is carried by whichever world happens to have the
    largest error, so a set of worlds that each sit a factor of two above the
    prediction can average to a curve that sits well under a factor of two.
    The geometric mean is the arithmetic mean of the logarithms, which is what
    the axis shows, and it reproduces the typical world.  The behavioural row
    is a percentage on a linear axis and is averaged as one.
    """
    if key == "geometric":
        return np.exp(np.mean(np.log(np.maximum(V, 1e-12)), axis=0))
    return np.mean(V, axis=0)


def scale_of(rec):
    """Epochs at this record's rate per epoch at BASE_RATE."""
    return BASE_RATE / rec.get("lr_target", BASE_RATE)


def ticks_for(hi):
    """Four ticks across the axis, labelled in thousands where that is shorter."""
    vals = [0.0, hi / 3.0, 2.0 * hi / 3.0, hi]
    return vals, ["0" if v == 0 else
                  ("%gk" % (v / 1000.0) if v >= 1000 else "%g" % v)
                  for v in vals]


def after(rec, arm, src, key, rescale=False):
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
    sc = scale_of(rec) if rescale else 1.0
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
        xhi = WINDOW * scale_of(recs[0])
        xt = ticks_for(xhi)

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                for src, kw in (("net", dict(color=c, lw=1.8, zorder=3)),
                                ("pred", PRED)):
                    ser = [after(r, arm, src, key) for r in recs]
                    x = ser[0][0]
                    V = np.array([v for _, v in ser])
                    ax.plot(x, summarise(V, key), **kw)
                    if src == "net" and len(V) > 1:
                        band = spread(V, key)
                        if band is not None:
                            ax.fill_between(x, band[0], band[1], color=c,
                                            alpha=0.25, lw=0, zorder=2)
            ax.set_xlim(0, xhi)
            ax.set_xticks(xt[0])
            ax.set_xticklabels(xt[1])
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
    # "Prediction" alone.  That it carries no fitted parameters is the whole
    # claim of the figure and needs a sentence, so it belongs in the caption; a
    # legend key can only assert it in passing.
    axes[0][0].plot([], [], label="Prediction", **PRED)
    # The shading carries no legend entry.  What a band means -- which
    # statistic, over how many worlds, computed on which scale -- is a sentence,
    # and a sentence belongs in the caption, where a reader looks for it.  A
    # key can only name it, and naming it badly is worse than leaving it out.
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               handletextpad=0.6, columnspacing=2.4, handlelength=1.8)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
