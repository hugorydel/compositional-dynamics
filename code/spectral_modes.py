"""Section 4.3 figure: a fast and a slow law differ in their spectral support.

  a  each spectral mode's contribution to the initial error of two of
     Figure 1's laws, |a_jk| ||zeta_k(0)|| from Proposition 2, normalized to
     each law's largest contribution, against the mode's rate
     lambda_k / lambda_max
  b  the theory's predicted geometric error for the same two laws, read from
     the stored Experiment 1 record: the curves Figure 1 averages, for one world

The fast and slow laws are Figure 1's laws A and C (`worlds.F1_LAWS`), in
Figure 1's colours, in world 0 at depth 1: weight-averaged log10 mode rate
-1.54 and -1.82, and predicted geometric error below 0.5 from epoch 2,700 and
13,775.  Law B, between them on both measures, is left out so the contrast
reads at a glance.  The two panels sit side by side at 0.95 of the text width, as Figures 4-6,
at the size the manuscript sets them, so the font sizes here are the sizes on
the page.  The axis titles sit a few points further from their axes than
matplotlib's default, so the panels do not read as cramped.

  usage:  python spectral_modes.py
"""

import json
import os

import _paths
import matplotlib.pyplot as plt
import numpy as np
from _paths import RESULTS
from relspec import System, models, worlds
from relspec.config import override
from style import DARK

WORLD, DEPTH = 0, 1
LAWS = (("L3_1", "#1b6ca8", "Fast law"), ("L5_0", "#c0392b", "Slow law"))
WIDTH, HEIGHT = 5.22, 1.65          # inches: 0.95 of the text width, as Figures 4-6
PAD = 5                             # points between each axis title and its tick labels
FS = 7.0

plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.minor.width": 0.4, "xtick.major.size": 2.4,
                     "ytick.major.size": 2.4, "xtick.minor.size": 1.4})


def weights():
    """Per-mode weights of each law, and the modes' relative rates."""
    S = override(init_seed=1000 + WORLD, eval_every=worlds.F1_EVERY[DEPTH])
    w = worlds.emergence_world(WORLD)
    s = System.build(w, settings=S)
    E0 = models.make_model(w, DEPTH, S).embedding()
    rate = s.evals_M / s.evals_M[0]
    return rate, {law: s.gating_weights(law, E0) for law, _, _ in LAWS}


def predicted():
    """The stored prediction for this world: epochs and geometric error per law."""
    with open(os.path.join(RESULTS, "f1", "w%02d_d%d.json" % (WORLD, DEPTH))) as f:
        pred = json.load(f)["pred"]
    return np.asarray(pred["epochs"]), {law: np.asarray(pred["geometric"][law])
                                        for law, _, _ in LAWS}


def letter(fig, ax, text, dx):
    """Panel letter above the axes, `dx` (figure fraction) left of their edge,
    so it clears the axis title."""
    box = ax.get_position()
    fig.text(box.x0 - dx, box.y1 + 0.06, text, weight="bold", fontsize=9, va="bottom")


def main():
    rate, weight = weights()
    epochs, geo = predicted()

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.1, right=0.99, top=0.86,
                          bottom=0.27)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    keep = epochs >= 100
    for law, colour, name in LAWS:
        g = weight[law]
        k = np.flatnonzero(g > 0.02 * g.max())
        ax_a.vlines(rate[k], 0, g[k] / g.max(), color=colour, lw=0.9)
        ax_a.scatter(rate[k], g[k] / g.max(), s=6, color=colour, zorder=3, linewidth=0)
        ax_b.plot(epochs[keep], geo[law][keep], color=colour, lw=1.3)
    # named in panel b's empty upper right: in panel a the longest name would
    # sit on the fast law's tallest stems
    for (law, colour, name), y in zip(LAWS, (0.98, 0.84)):
        ax_b.text(0.99, y, name, color=colour, transform=ax_b.transAxes, ha="right",
                  va="top", fontsize=FS)

    ax_a.set_xscale("log")
    ax_a.set_ylim(0, 1.08)
    ax_a.set_yticks([0, 1])
    ax_a.set_xlabel(r"mode rate $\lambda_k/\lambda_{\max}$", labelpad=PAD)
    ax_a.set_ylabel("normalized\ncontribution", labelpad=PAD, linespacing=1.0)

    ax_b.set_xscale("log")
    ax_b.set_xlim(100, epochs[-1])
    ax_b.set_ylim(0, None)
    ax_b.set_xticks([1e2, 1e3, 1e4])
    ax_b.set_xlabel("epoch", labelpad=PAD)
    ax_b.set_ylabel("geometric error", labelpad=PAD)
    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=DARK)

    letter(fig, ax_a, "a", dx=0.085)
    letter(fig, ax_b, "b", dx=0.075)
    # tight, so no empty margin sits between the axis labels and the caption;
    # no creation date, so a redraw is byte-identical
    crop = dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(_paths.figure("spectral_modes.png"), dpi=400, **crop)
    fig.savefig(_paths.figure("spectral_modes.pdf"), metadata={"CreationDate": None},
                **crop)
    plt.close(fig)
    print("wrote %s" % _paths.figure("spectral_modes.pdf"))


if __name__ == "__main__":
    main()
