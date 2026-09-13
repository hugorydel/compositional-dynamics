"""Figure 1: compositional abstraction emerges at a time specific to each law.

Two by three.  Columns are depths, sharing one logarithmic epoch axis so the
acceleration with depth is visible rather than hidden by per-column scaling.
Top row is the behavioural measure, bottom row the geometric one, with the
parameter-free prediction overlaid on every curve.

The behavioural measure is the percentage of a law's held-out composites that
have STABLY resolved, counted from the first evaluation after each item's last
failure.  Instantaneous top-1 accuracy is the wrong object for a panel about
emergence: with a modest number of held-out items, single items cross and
recross a nearest-neighbour boundary, so the curve steps and reverses while
learning is monotone underneath.

Three laws are drawn rather than all sixteen.  Sixteen curves plus sixteen
predictions cannot be read, and at depth the laws compress into one bundle.
WHICH three is fixed by the ladder design in `worlds.F1_LAWS` and never by the
measured times: ranking the sixteen by observed emergence and drawing the
fastest, median and slowest selects on the dependent variable, so the drawn
law changes identity from world to world and the average across worlds mixes
conditions.  The remaining thirteen stay in every record and belong in a
predicted-against-observed panel, where the full set carries information.

Each line is the mean over worlds and the band is the range the worlds
actually covered.  A mean alone reads as one crisp run; the spread is a
property of the environment, since every world redraws the relation vectors,
the lattice origins and which composites are withheld.
"""

import glob
import json
import os

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS, figure
from relspec import worlds
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence, resolved
from style import DARK, panel

DEPTHS = (1, 2, 3)
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
LAWS = worlds.F1_LAWS  # Law A, B, C -- fixed by design, see worlds.F1_LAWS
LAW_COLOURS = ("#1b6ca8", "#b8860b", "#c0392b")  # matches F2/F3
BAND = "sem"           # "sem", a (lo, hi) percentile pair, or None for min-max
# Matches fig3.  What the shading is for decides which of these is right.  A
# spread band answers "how much do worlds differ", and it does not narrow as
# worlds are added: with fifty worlds the 10-90 band still covered most of the
# depth-1 accuracy panel, because a per-world accuracy curve is a step function
# over 7 to 9 held-out items and the percentiles jump between discrete levels.
# A standard-error band answers "how well is the plotted curve pinned down",
# which is the question a curve drawn against a prediction raises, and it
# narrows with the number of worlds.  The spread belongs in the text, not on a
# panel whose subject is agreement.
XLIM = (0, 3000)       # standardised across all three figures
XTICKS = ([0, 1000, 2000, 3000], ["0", "1k", "2k", "3k"])
YLAB = {"top": "Compositional Accuracy\n(held-out)",
        "bot": "Geometric Error"}



def cells(depth, worlds=None):
    """Every world's cell for this depth, or only the listed ones.

    Restricting the set is how the effect of adding a world is checked: the
    same code path draws one world and several, so a difference between the
    two renders is the data and not the plotting.
    """
    f = sorted(glob.glob(os.path.join(RESULTS, "f1", "w*_d%d.json" % depth)))
    if worlds is None:
        return f
    keep = {"w%02d_d%d.json" % (w, depth) for w in worlds}
    return [p for p in f if os.path.basename(p) in keep]


def emergence(ep, rec):
    return {k: (float(ep[int(i)]) if np.isfinite(i) else np.inf)
            for k, i in ((k, detect_emergence(rec["retrieval"][k],
                                              rec["geometric"][k],
                                              S.hold, S.tau))
                         for k in rec["retrieval"])}


def stack(recs, src, key, law, ep):
    """One row per world, so the mean and the spread come from the same array.

    Drawing only the mean of a handful of worlds hides whether the laws are
    separated in every world or only on average, which is the whole claim.
    """
    if key == "resolved":
        H = [np.array(r[src]["hits"][law], bool) for r in recs]
        # zero means "no better than answering the same entity every time"
        return np.array([resolved(ep, h, chance=1.0 / h.shape[1]) for h in H])
    return np.array([r[src][key][law] for r in recs], float)


def spread(V, key):
    """Lower and upper edges of the shading, in the space the axis uses.

    The geometric row is logarithmic, so its error is taken on the logarithms
    and mapped back.  Doing it linearly would put the band off-centre on the
    drawn curve.  The behavioural row is a bounded percentage and is clipped.
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
    largest error, so worlds that each sit a factor of two above the prediction
    can average to a curve well under a factor of two.  The geometric mean is
    the arithmetic mean of the logarithms, which is what the axis shows.  The
    behavioural row is a percentage on a linear axis and is averaged as one.
    """
    if key == "geometric":
        return np.exp(np.mean(np.log(np.maximum(V, 1e-12)), axis=0))
    return np.mean(V, axis=0)


def draw(ax, ep, V, key, band=True, **kw):
    """Across-world centre, with the standard-error band for the network only.

    Adding the band to the predictions as well would put six overlapping fills
    in a panel.
    """
    ax.plot(ep, summarise(V, key), **kw)
    if band and len(V) > 1:
        lo, hi = spread(V, key)
        ax.fill_between(ep, lo, hi, color=kw.get("color", DARK), alpha=0.25,
                        lw=0, zorder=2)


def main(worlds=None, out="fig1_emergence.png", laws=None, band=True):
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    cols = LAW_COLOURS
    nw = 0

    for col, depth in enumerate(DEPTHS):
        files = cells(depth, worlds)
        nw = max(nw, len(files))
        if not files:
            continue
        recs = [json.load(open(f)) for f in files]
        ep = np.array(recs[0]["net"]["epochs"], float)
        lrt = recs[0]["lr_target"]
        pick = laws or LAWS

        for row, key, ylab in ((0, "resolved", YLAB["top"]),
                               (1, "geometric", YLAB["bot"])):
            ax = axes[row][col]
            for k, law in enumerate(pick):
                draw(ax, ep, stack(recs, "net", key, law, ep), key,
                     band=band, color=cols[k], lw=1.7, zorder=3)
                draw(ax, ep, stack(recs, "pred", key, law, ep), key,
                     band=False, **PRED)
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
                ax.set_ylim(1e-3, 4)
                ax.set_xlabel("Training epoch")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(ylab, linespacing=1.6)
            panel(ax, "abcdef"[row * 3 + col],
                  dx=-0.22 if col == 0 else -0.08, dy=1.16)

    for k, nm in enumerate("ABC"[:len(laws or LAWS)]):
        axes[0][0].plot([], [], color=cols[k], lw=1.7, label="Law %s" % nm)
    # "Prediction" alone.  That it carries no fitted parameters is the whole
    # claim of the figure and needs a sentence, so it belongs in the caption; a
    # legend key can only assert it in passing.
    axes[0][0].plot([], [], label="Prediction", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    # one column per entry, so the four sit on a single line.  With ncol=3 and
    # four entries matplotlib fills column-major and wraps Law B underneath.
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=len(l), handletextpad=0.6, columnspacing=1.6,
               handlelength=1.8)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s  (%d world%s)" % (p, nw, "" if nw == 1 else "s"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, nargs="+", default=None,
                    help="world seeds to include; default is every cell found")
    ap.add_argument("--out", default="fig1_emergence.png")
    ap.add_argument("--laws", nargs=3, default=None, metavar="LAW",
                    help="the three laws to draw, e.g. L6_0 L2_0 L0_0; "
                         "default is worlds.F1_LAWS.  Every cell stores all "
                         "sixteen, so changing this recomputes nothing")
    ap.add_argument("--no-band", dest="band", action="store_false",
                    help="draw the across-world centre only, with no band")
    ns = ap.parse_args()
    main(ns.worlds, ns.out, ns.laws, ns.band)
