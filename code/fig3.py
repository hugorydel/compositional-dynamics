"""Figure 3: one fact makes many cross-structure comparisons available.

Two by three.  Columns are depths, sharing one logarithmic axis of epochs since
the linking fact.  Nothing before the intervention is drawn.  Both arms leave
identical pre-switch weights; only one then receives the fact.

Top row is the percentage of the eighty withheld cross-structure comparisons
answered correctly at each evaluation, rescaled against what the arm scores at
the switch.  It depends only on the state at that evaluation, so both arms
start at the same point.  At depth 3 it dips and recovers after the fact, and
the prediction dips in the same worlds (tests/check_reversals.py).

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
from relspec.measure import resolved, instantaneous
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


MEASURE = "instant"    # behavioural row: "instant" (causal) or "stable" (retrospective)
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


def scorable(rec):
    """Has this world enough wrong at the switch to score anything?

    The behavioural curve is corrected against what the arm scores at the
    switch, so a world where the undetermined offset already happens to answer
    every comparison correctly divides by zero and contributes a curve of NaN.
    One world in a hundred does exactly that at depth 3, and because a NaN
    propagates through the mean it removed the whole depth-3 curve rather than
    one world's contribution to it.

    Two items, not one: over a single remaining query the correction is again
    degenerate.  The same rule is applied in Figure 2 for the same reason.
    Nothing else comes close here -- the next world has 53 of 80 still wrong.
    """
    d = rec["arms"]["insert"]["net"]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    h = np.array(d["hits"]["cross"], bool)[int(np.argmin(np.abs(ep)))]
    return int((~h).sum()) >= 2


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
        if MEASURE == "instant":
            v = instantaneous(h, chance=float(h[k].mean()))
        else:
            v = resolved(np.array(d["epochs"], float), h,
                         chance=float(h[k].mean()))
    else:
        v = np.array(d[key]["cross"], float)
    return ep[m], v[m]


N_WORLDS = 200         # every column averages exactly this many worlds


def seed_of(path):
    return int(os.path.basename(path).split("_")[0][1:])


def world_set(sub):
    """The first `N_WORLDS` seeds, in seed order, scorable at EVERY depth.

    One set for the whole figure rather than one per column.  Filtering each
    depth on its own would let the columns average different worlds -- depth 3
    dropping world 91 and depths 1 and 2 keeping it -- so a difference between
    columns could be a difference in sample.  A seed unscorable at any depth is
    dropped from all of them and the next seed takes its place, which is why
    runs are extended past `N_WORLDS` rather than stopped at it.

    Files beyond the set are ignored, not deleted: a spare run that wrote a
    few extra cells changes nothing about what is drawn.
    """
    have = {d: {seed_of(f): f for f in glob.glob(
                os.path.join(RESULTS, sub, "w*_d%d.json" % d))} for d in DEPTHS}
    common = sorted(set.intersection(*(set(v) for v in have.values())))
    good, bad = [], []
    for s in common:
        if len(good) == N_WORLDS:
            break
        ok = all(scorable(json.load(open(have[d][s]))) for d in DEPTHS)
        (good if ok else bad).append(s)
    return good, bad, have


def main(sub="f3_lr0p003", out="fig3_eta0p003.png"):
    """`sub` is the results sub-directory, so a run at another step size can be
    drawn with the same code into its own file.

    The default is the published set.  The locked-rate directory it used to
    point at is gone, and pointing a default at a directory that no longer
    exists produces an empty figure rather than an error, which is the sort of
    thing that survives to a draft.
    """
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    lrt = BASE_RATE
    seeds, dropped, have = world_set(sub)
    print("  %d worlds in every column%s%s"
          % (len(seeds),
             "" if not dropped else ", dropped as unscorable: %s" % dropped,
             "" if len(seeds) == N_WORLDS or not seeds
             else "  -- SHORT of %d, extend the run" % N_WORLDS))

    for col, depth in enumerate(DEPTHS):
        files = [have[depth][s] for s in seeds]
        if not files:
            for row in (0, 1):
                axes[row][col].set_visible(False)
            continue
        recs = [json.load(open(f)) for f in files]
        lrt = recs[0]["lr_target"]
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
                # value any arm reaches.  No reference line: retrieval is 100%
                # below about one candidate spacing and falls away above it
                # (tests/check_readout.py), but where exactly to draw the line
                # inside that transition is a choice, and a dashed rule invites
                # a reader to treat the choice as a result.
                ax.set_ylim(2e-3, 150)
                ax.set_xlabel("Epochs since the linking fact")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

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
