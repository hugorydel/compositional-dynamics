"""Figure 3: one fact makes many cross-structure comparisons available.

Two by three.  Columns are depths sharing one linear axis of epochs since the
linking fact.  Nothing before the intervention is drawn.  Both arms leave
identical pre-switch weights; only one then receives the fact.

Top row counts correct novel compositions: how many of the eighty withheld
cross-structure comparisons a world answers correctly at each evaluation,
averaged across worlds.  Retrieval is scored against ALL EIGHTEEN entities,
not against the nine in the destination structure, which is what makes the
count a count of inferences.  With nine candidates an unresolved offset lands
9.4, 12.6 and 13.1 of the eighty queries on the right target by luck at depths
1, 2 and 3, a baseline that differs by depth and had to be subtracted away.
Against the full pool the same networks answer 1.0, 0.2 and 0.4 of 80 at the
intervention -- zero in 177, 198 and 198 of 200 worlds -- so nothing is
subtracted, the axis counts correct answers, and the three depths share one
true ceiling of 80.

Those counts come from the replay pass (`analysis/controls.py`), which re-runs
each stored cell from its own seed and presentation order, checks every
held-out hit and geometric score against the published record before adding
anything to it, and scores the same states against the wider pool.  The
prediction is the same integration scored the same way
(`analysis/controls_pred.py`); nothing is fitted to the network.

Bottom row is the geometric error the paper defines: the distance from the
predicted point to the entity it should have retrieved, in units of the median
nearest-neighbour spacing among the nine destination entities in the same
learned embedding.  The two rows deliberately use different sets -- a query
competes against all eighteen entities, the unit of distance is the local scale
of the destination structure -- and that normalization is unchanged here; only
its resolution is, since it too is read from the replay.  No ground-truth alignment enters, which matters: the unlinked world has
one global offset of the copy that no fact fixes, so any measure taken against
a chosen ground truth lets a control that is converging perfectly well appear
to get worse, as the free coordinate settles at the minimum-norm point rather
than at the chosen one.

  usage:  python fig3.py
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from style import DARK, panel

SUB = "f3"             # the published run: records and controls, results/f3
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
# rate the same window is ten times the epochs.  The published run is at 0.003,
# so the panels span 30,000 epochs.
#
# The comparison figure in tests/fig_rates.py does rescale, because four rates
# have to share one axis there and the prediction is a single curve on it.
# That is the only place the transformation earns its keep, and even there the
# axis says so.  A panel showing one rate should not: it would print epochs
# nobody ran and invite a reader to reproduce it with the wrong budget.


MEASURE = "raw"        # behavioural row: plain accuracy, no chance correction
# Plain accuracy is the honest measure once the candidate pool is the whole
# world.  The chance corrections this constant used to select existed because
# the nine-candidate baseline was large and depth-dependent; against eighteen
# candidates it is not, so there is nothing to correct and no future to read.
# Other modules still read this constant to know what is drawn
# (tests/check_reversals.py).


ADJUST = "correct"     # correct of 80, not gained since the switch
# "correct", not "count".  Subtracting each world's own count at the
# intervention removed a depth-dependent luck baseline that the full candidate
# pool has already removed (see the docstring), and it cost the panel its
# ceiling: a curve of gains plateaus at 80 minus each world's lucky hits, so
# the three depths ended at different heights for a reason no reader could see.


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

    Only needed for `MEASURE = "instant"`, where the behavioural curve is
    corrected against what the arm scores at the switch, so a world where the
    undetermined offset already happens to answer
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

    By default each world's behavioural curve is plain accuracy, and `main`
    turns it into inferences gained since the switch.  With `MEASURE =
    "instant"` it is instead chance-corrected against WHAT THE ARM SCORES AT
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
        if MEASURE == "raw":
            v = instantaneous(h, chance=0.0)
        elif MEASURE == "instant":
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
                os.path.join(glob.escape(RESULTS), sub, "w*_d%d.json" % d))} for d in DEPTHS}
    common = sorted(set.intersection(*(set(v) for v in have.values())))
    good, bad = [], []
    for s in common:
        if len(good) == N_WORLDS:
            break
        # plain accuracy never divides by zero, so nothing needs dropping
        ok = MEASURE == "raw" or all(scorable(json.load(open(have[d][s])))
                                     for d in DEPTHS)
        (good if ok else bad).append(s)
    return good, bad, have

def sem(V):
    """Standard error of the mean across worlds."""
    return V.std(axis=0, ddof=1) / np.sqrt(len(V))


def band(ax, x, V, colour):
    """Across-world mean of a count, shaded by its standard error.

    The behavioural row is a count on a linear axis, so both are taken in the
    space the axis shows.  The shading answers "how well is the plotted curve
    pinned down", which is the question a curve drawn against a parameter-free
    prediction raises, and it narrows as worlds are added.  The spread across
    worlds is worth quoting in the text; it does not belong on a panel whose
    subject is agreement.
    """
    m, s = V.mean(axis=0), sem(V)
    ax.plot(x, m, color=colour, lw=1.8, zorder=3)
    ax.fill_between(x, m - s, m + s, color=colour, alpha=0.25, lw=0, zorder=2)


def control(sub, stem, seed, depth):
    """One cell of the replay pass, as a dictionary of arrays."""
    path = os.path.join(RESULTS, sub, "%s_w%02d_d%d.npz" % (stem, seed, depth))
    if not os.path.exists(path):
        raise SystemExit("no control pass for world %d at depth %d (%s).\n"
                         "Run:  python analysis/controls.py --cells f3:0-199:%d"
                         % (seed, depth, path, depth))
    with np.load(path) as z:
        return {k: z[k] for k in z.files if not k.startswith("pre.")}


def series(sub, depth, seeds):
    """Everything the panels draw, for the network and for the prediction.

    Correct compositions of 80 against all eighteen candidates, and the
    geometric error the paper defines on the destination pool.  Both come from
    the replay rather than from the stored records.  The replay reproduces each
    record at every epoch the two grids share -- `analysis/controls.py` and
    `analysis/controls_pred.py` fail the cell otherwise -- and evaluates
    between them as well, which is what the geometric row needs: at depth 1 the
    record's first post-intervention evaluation is at 2,500 epochs, so a curve
    drawn from it crosses the one interval where it bends as a straight line.

    The network and the prediction are on different grids, and each is shared
    by every world at a depth, which is checked here rather than assumed: a
    world on a different grid would be averaged into the wrong epochs and
    nothing downstream would notice.
    """
    grids, cnt, geo = {}, {}, {}
    for seed in seeds:
        cells = (("net", control(sub, "controls", seed, depth), "cross18"),
                 ("pred", control(sub, "controls_pred", seed, depth), "pred18"))
        for name, z, key in cells:
            if name not in grids:
                grids[name] = z["epochs"]
            elif not np.array_equal(grids[name], z["epochs"]):
                raise SystemExit("world %d at depth %d is on a different %s grid"
                                 % (seed, depth, name))
            for arm, _, _ in ARMS:
                cnt.setdefault((name, arm), []).append(z["%s.%s" % (arm, key)].sum(axis=1))
                geo.setdefault((name, arm), []).append(z["%s.geo9" % arm])
    return (grids, {k: np.array(v, float) for k, v in cnt.items()},
            {k: np.array(v, float) for k, v in geo.items()})


def log_band(ax, x, V, colour, style=None):
    """A geometric row: centre and error taken on the logarithms.

    On a logarithmic axis an arithmetic mean is the wrong centre -- it is
    carried by whichever world happens to have the largest error, so a set of
    worlds that each sit a factor of two above the prediction can average to a
    curve that sits well under a factor of two -- and a linear error band sits
    off-centre on the drawn curve.  With `style`, the prediction: one curve,
    no band.
    """
    L = np.log10(np.maximum(V, 1e-12))
    m, s = L.mean(axis=0), sem(L)
    if style is not None:
        ax.plot(x, 10.0 ** m, **style)
        return
    ax.plot(x, 10.0 ** m, color=colour, lw=1.8, zorder=3)
    ax.fill_between(x, 10.0 ** (m - s), 10.0 ** (m + s), color=colour,
                    alpha=0.25, lw=0, zorder=2)


def main(sub=SUB, out="fig3_integration.png"):
    """`sub` is the results sub-directory holding both the records and the
    replay pass, so another run can be drawn with the same code into its own
    file."""
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    seeds, dropped, have = world_set(sub)
    print("  %d worlds in every column%s%s"
          % (len(seeds),
             "" if not dropped else ", dropped as unscorable: %s" % dropped,
             "" if len(seeds) == N_WORLDS or not seeds
             else "  -- SHORT of %d, extend the run" % N_WORLDS))

    for col, depth in enumerate(DEPTHS):
        recs = [json.load(open(have[depth][s])) for s in seeds]
        lrt, n_items = recs[0]["lr_target"], recs[0]["n_pairs"]
        xhi = WINDOW * scale_of(recs[0])
        grids, cnt, geo = series(sub, depth, seeds)
        m, pm = grids["net"] <= xhi, grids["pred"] <= xhi

        ax = axes[0][col]
        for arm, colour, _ in ARMS:
            band(ax, grids["net"][m], cnt[("net", arm)][:, m], colour)
            ax.plot(grids["pred"][pm], cnt[("pred", arm)].mean(axis=0)[pm], **PRED)
        print("  N=%d  linked arm answers %.1f of %d at the end of the axis, "
              "control %.1f"
              % (depth, cnt[("net", "insert")][:, m][:, -1].mean(), n_items,
                 cnt[("net", "hold")][:, m][:, -1].mean()))
        ax.set_ylim(-0.04 * n_items, 1.06 * n_items)
        ax.set_yticks(list(range(0, n_items + 1, 20)))
        ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                % (depth, lrt), transform=ax.transAxes, fontsize=9,
                ha="center", va="bottom", color=DARK)
        if col == 0:
            ax.set_ylabel("Correct Novel Compositions\n(out of %d)"
                          % n_items, linespacing=1.6, fontsize=9)

        ax = axes[1][col]
        for arm, colour, _ in ARMS:
            log_band(ax, grids["net"][m], geo[("net", arm)][:, m], colour)
            log_band(ax, grids["pred"][pm], geo[("pred", arm)][:, pm], colour,
                     style=PRED)
        ax.set_yscale("log")
        # the same range in all three columns, 10 down to half a decade past
        # 10^-2, which holds every curve drawn here.  No reference line:
        # retrieval is 100% below about one candidate spacing and falls away
        # above it (tests/check_readout.py), but where exactly to draw the line
        # inside that transition is a choice, and a dashed rule invites a
        # reader to treat the choice as a result.
        ax.set_ylim(5e-3, 10)
        ax.set_xlabel("Epochs since the linking fact")
        if col == 0:
            ax.set_ylabel("Geometric Error", linespacing=1.6)

        for row in (0, 1):
            ax = axes[row][col]
            ax.set_xlim(0, xhi)
            ax.set_xticks(ticks_for(xhi)[0])
            ax.set_xticklabels(ticks_for(xhi)[1])
            if col:
                ax.set_yticklabels([])
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.24 if col == 0 else -0.08, dy=1.16)

    for _, colour, label in ARMS:
        axes[0][0].plot([], [], color=colour, lw=1.8, label=label)
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
    plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
