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

The behavioural row is instantaneous accuracy over all fifteen held-out
composites, rescaled so the best constant answer reads as zero.  The control
therefore sits above zero at depth 1: before the bridge, the undetermined
composite passes through a region where some worlds already retrieve every
target, and the theory predicts the same arms (tests/check_degenerate.py).
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
ARMS = (("hold", "#c0392b", "No bridging fact"),
        ("insert", "#1b6ca8", "One bridging fact"))
XLIM = (0, 2000)       # the runs go to 40k / 8k / 4k since the bridge
XTICKS = ([0, 500, 1000, 1500, 2000],
          ["0", "500", "1k", "1.5k", "2k"])
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}


def summarise(V, key):
    """Centre of a stack of curves.

    The geometric row is drawn on a logarithmic axis, where an arithmetic mean
    is the wrong centre: it is carried by whichever world happens to have the
    largest error.  The geometric mean is the arithmetic mean of the logarithms,
    which is what the axis shows, and it reproduces the typical world.  The
    behavioural row is a percentage on a linear axis and is averaged as one.
    """
    if key == "geometric":
        return np.exp(np.mean(np.log(np.maximum(V, 1e-12)), axis=0))
    return np.mean(V, axis=0)


def at_risk(rec, law):
    """Which held-out items were still WRONG when the bridge arrived.

    Both branches leave the switch holding identical weights, so they score
    identical hits there; this is checked across every arm by
    `tests/check_atrisk.py`.  An item already correct at that instant cannot
    demonstrate anything about the intervention, and scoring it does active
    harm, because the resolution measure is retrospective: credit runs from the
    first evaluation after an item's LAST failure.  A pre-correct item that the
    bridge then locks in is credited from epoch zero, while the same item in
    the control eventually drifts wrong and is not, so the two arms separate at
    the switch despite being in the same state.  That gap was 4.9 points at
    depth 1.

    Restricting to the items at risk removes it.  The restriction cannot bias
    the comparison, because it is defined by the shared switch state and is
    therefore the same set in both arms.  It removes 7.9 per cent of items at
    depth 1, 3.3 at depth 2 and 1.1 at depth 3, and it empties three depth-1
    arms entirely, in worlds 22, 28 and 32, where all fifteen were already
    correct.  Those arms are dropped rather than counted as zero.
    """
    d = rec["arms"]["hold"]["net"]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    return ~np.array(d["hits"][law], bool)[ep >= 0][0]


def scorable(rec):
    """Has this arm enough left to score once the pre-correct ones go?

    Two items, not one.  The behavioural curve is corrected against the best
    constant answer, and over a single query some constant answer always scores
    it, so the correction divides by zero and the arm carries no information
    either way.  Five depth-1 arms fall out: three where all fifteen items were
    already correct at the switch, and two where fourteen were.
    """
    return int(at_risk(rec, rec["open"]).sum()) >= 2


MEASURE = "instant"    # behavioural row: "instant" (causal) or "stable" (retrospective)
BAND = "sem"           # "sem", a (lo, hi) percentile pair, or None for min-max
# What the shading is for decides which of these is right.  A spread band
# answers "how much do worlds differ", and each world's behavioural curve here
# is a step, so a spread band covers the whole panel wherever the worlds
# disagree about having resolved yet and reports only that some have and some
# have not.  A standard-error band answers "how well is the plotted curve
# pinned down", which is the question a curve drawn against a prediction
# raises, and it narrows as worlds are added.  The spread is still worth
# quoting in the text: 250 to 2,960 epochs to resolution at depth 1, 45 to 210
# at depth 2, 30 to 80 at depth 3, tenth to ninetieth percentile.


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


def by_world(recs, keep_all=False):
    """{seed: [its scorable counterbalance records]}.

    The two arms of a world share a seed, hence its premise geometry and its
    spectrum, and differ only in which block carries the closed composites.
    They are not independent draws, so they are pooled INTO a world before
    anything is computed across worlds.  Treating all the arms as one sample
    would halve the apparent spread by counting each world twice.  A world
    whose arms are all unscorable drops out; one with a single scorable arm is
    kept on that arm.
    """
    out = {}
    for r in recs:
        if keep_all or scorable(r):
            out.setdefault(r["seed"], []).append(r)
    return dict(sorted(out.items()))


def after(rec, arm, src, key, law, restrict=True):
    """One arm's post-switch series, with the switch at zero."""
    d = rec["arms"][arm][src]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    m = ep >= 0
    if key == "resolved" and MEASURE == "instant":
        # every item, against the best constant answer; no at-risk restriction,
        # which existed only to cancel the look-ahead in `resolved`
        h = np.array(d["hits"][law], bool)
        v = instantaneous(h, chance=1.0 / h.shape[1])
    elif key == "resolved" and not restrict:
        # every item in every arm, against the best constant answer over all
        # fifteen.  Shown only to make visible what the restriction removes;
        # see `main(exclude=False)`.
        h = np.array(d["hits"][law], bool)
        v = resolved(np.array(d["epochs"], float), h, chance=1.0 / h.shape[1])
    elif key == "resolved":
        keep = at_risk(rec, law)
        h = np.array(d["hits"][law], bool)[:, keep]
        # the best constant answer scores one item, since every held-out target
        # in this world is distinct; `tests/check_atrisk.py` verifies that the
        # rate stored in the record is one over the item count, which is what
        # makes the same form exact after the restriction.
        v = resolved(np.array(d["epochs"], float), h, chance=1.0 / keep.sum())
    else:
        v = np.array(d[key][law], float)
    return ep[m], v[m]


N_WORLDS = 200         # every column averages exactly this many worlds


def world_set(keep_all=False):
    """The first `N_WORLDS` seeds, in seed order, usable at EVERY depth.

    A seed is usable at a depth when both counterbalance arms are on disk and
    at least one of them is scorable, which is the rule `by_world` already
    applies within a column.  Applying it per column let the columns average
    different worlds: world 126 lost both arms at depth 1 and was kept at
    depths 2 and 3.  A seed that fails at any depth is dropped from all of
    them and the next seed takes its place.

    Only the two counterbalance arms are read.  A dose series over `n_bridge`
    lives in the same directory and answers a different question, so it must
    not be swept up by a wildcard.  Files beyond the set are ignored.
    """
    arms = {}
    for d in DEPTHS:
        arms[d] = {}
        for a in ("A", "B"):
            for f in glob.glob(os.path.join(RESULTS, "f2",
                                            "w*_d%d_%s.json" % (d, a))):
                seed = int(os.path.basename(f).split("_")[0][1:])
                arms[d].setdefault(seed, []).append(f)
    complete = [s for s in sorted(set.intersection(*(set(v) for v in arms.values())))
                if all(len(arms[d][s]) == 2 for d in DEPTHS)]
    good, bad = [], []
    for s in complete:
        if len(good) == N_WORLDS:
            break
        ok = keep_all or all(any(scorable(json.load(open(f))) for f in arms[d][s])
                             for d in DEPTHS)
        (good if ok else bad).append(s)
    return good, bad, arms


def main(exclude=None, out="fig2_identifiability.png"):
    """`exclude` applies the at-risk restriction and the scorability filter.

    Both exist only for the retrospective measure, whose look-ahead made two
    arms with identical switch weights score differently; restricting to the
    items still wrong at the switch cancelled that, at the price of dropping
    arms with nothing left to score.  The instantaneous measure has no
    look-ahead, so by default nothing is excluded and every arm and item is
    scored.  The arms that used to be dropped are a transient in the
    undetermined composite that the theory itself predicts
    (tests/check_degenerate.py).
    """
    if exclude is None:
        exclude = MEASURE == "stable"
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    seeds, dropped, arms = world_set(keep_all=not exclude)
    print("  %d worlds in every column%s%s"
          % (len(seeds),
             "" if not dropped else ", dropped as unscorable: %s" % dropped,
             "" if len(seeds) == N_WORLDS or not seeds
             else "  -- SHORT of %d, extend the run" % N_WORLDS))

    for col, depth in enumerate(DEPTHS):
        files = [f for s in seeds for f in sorted(arms[depth][s])]
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        lrt = recs[0]["lr_target"]
        worlds_ = by_world(recs, keep_all=not exclude)
        nw = len(worlds_)
        kept = sum(len(v) for v in worlds_.values())
        print("  N=%d  %d worlds, %d of %d arms scorable"
              % (depth, nw, kept, len(recs)))

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for arm, c, _ in ARMS:
                for src, kw in (("net", dict(color=c, lw=1.8, zorder=3)),
                                ("pred", PRED)):
                    ep = after(recs[0], arm, src, key, recs[0]["open"],
                               restrict=exclude)[0]
                    W = np.array([
                        summarise(np.array([after(r, arm, src, key, r["open"],
                                                  restrict=exclude)[1]
                                            for r in rs]), key)
                        for rs in worlds_.values()])
                    ax.plot(ep, summarise(W, key), **kw)
                    if src == "net":
                        band = spread(W, key)
                        if band is not None:
                            ax.fill_between(ep, band[0], band[1], color=c,
                                            alpha=0.25, lw=0, zorder=2)
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
                # No reference line.  Retrieval is reliable well below one
                # candidate spacing and falls away above it, but where inside
                # that transition to draw a rule is a choice, and a dashed line
                # invites a reader to treat the choice as a result.
                ax.set_ylim(1e-3, 6)
                ax.set_xlabel("Epochs since the bridging fact")
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
    # one column per entry, so they sit on a single line whatever the count
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(l), handletextpad=0.6, columnspacing=2.4,
               handlelength=1.8)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
