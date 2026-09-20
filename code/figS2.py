"""Figure S2: the trained facts, while the structure is being remodelled.

The linking fact reorganizes an embedding that was already fitting every fact
it had been trained on, and for a while that reorganization costs some of them.
Each panel is the fraction of the world's own training premises the network
still retrieves, averaged across the same 200 worlds per depth that Figure 3
draws, shaded by the standard error across worlds.  The loss is transient: the
mean returns to 100%, and world by world every affected world recovers except
one at depth 1 (`tests/pat_stage1_summary.py` reports the per-world counts).

The prediction is the integrated theory from the same initialization, scored on
the same premises by the same rule (`analysis/controls_pred.py`), so the dip is
something the dynamics imply rather than an accident of one training run.

The control arm, which never receives the linking fact, never loses a trained
fact at any depth, so the two curves separate only where the intervention acts.
That is why the axis has to include the intervention itself: at epoch 0 the two
arms are the same network holding the same weights, and an axis that starts at
the first evaluation after it leaves a gap that looks like a difference between
arms.  It is linear below ten epochs and logarithmic above, over the same
30,000 epochs Figure 3 spans: the dip is at +50 epochs at depth 3 and at
+2,680 at depth 1, and on a linear axis the deep panels collapse onto the
vertical axis.  The cost is that the logarithm widens the early epochs, so the
three dips are not drawn at comparable widths; the numbers are in
`tests/pat_stage1_summary.py`.

Scores come from the replay pass (`analysis/controls.py`), which re-runs each
stored cell from its own seed and presentation order, checks every held-out hit
and geometric score against the published record, and then scores the trained
facts as well.  Each premise is scored inside its own entity's block, the same
retrieval rule the rest of the paper uses.

  usage:  python figS2.py
"""

import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS, figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import fig3  # noqa: E402
from style import DARK, panel  # noqa: E402

OUT = Path(RESULTS) / fig3.SUB
DEPTHS = fig3.DEPTHS
ARMS = fig3.ARMS
PRED = fig3.PRED
LINTHRESH = 10         # epochs; linear below, logarithmic above


def sem(V):
    return V.std(axis=0, ddof=1) / np.sqrt(len(V))


def band(ax, x, V, colour):
    m, s = V.mean(axis=0), sem(V)
    ax.plot(x, m, color=colour, lw=1.8, zorder=3)
    ax.fill_between(x, m - s, m + s, color=colour, alpha=0.25, lw=0, zorder=2)


def premises(stem, depth):
    """Each world's retrieval of its trained premises, per arm, in per cent."""
    rows, epochs = {arm: [] for arm, _, _ in ARMS}, None
    paths = sorted(OUT.glob("%s_w*_d%d.npz" % (stem, depth)))
    if not paths:
        raise SystemExit("no %s cells at depth %d in %s; run analysis/controls.py"
                         % (stem, depth, OUT))
    for path in paths:
        with np.load(path) as z:
            if epochs is None:
                epochs = z["epochs"]
            elif not np.array_equal(epochs, z["epochs"]):
                raise SystemExit("%s is on a different grid" % path.name)
            for arm, _, _ in ARMS:
                rows[arm].append(100.0 * z["%s.premise" % arm].mean(axis=1))
    return epochs, {k: np.array(v, float) for k, v in rows.items()}, paths


def window(path):
    """The axis Figure 3 uses, in this run's own epochs."""
    record = json.loads((path.parent / path.name.split("_", 1)[1]).with_suffix(".json")
                        .read_text())
    return fig3.WINDOW * fig3.scale_of(record)


def main(out="figS2_retention.png"):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    fig.subplots_adjust(wspace=0.14, bottom=0.34)
    for col, depth in enumerate(DEPTHS):
        ep, net, paths = premises("controls", depth)
        pep, pred, _ = premises("controls_pred", depth)
        n_premise = json.loads(paths[0].with_suffix(".json").read_text())["n_premise"]
        xhi = window(paths[0])
        ax, m, pm = axes[col], ep <= xhi, pep <= xhi
        for arm, colour, _ in ARMS:
            band(ax, ep[m], net[arm][:, m], colour)
            ax.plot(pep[pm], pred[arm].mean(axis=0)[pm], **PRED)
        lo = net["insert"].mean(axis=0)
        print("  N=%d  %d worlds, mean retention falls to %.1f%% at +%s, ends at "
              "%.1f%%; prediction falls to %.1f%%"
              % (depth, len(net["insert"]), lo[m].min(),
                 "{:,.0f}".format(ep[m][int(lo[m].argmin())]), lo[m][-1],
                 pred["insert"].mean(axis=0)[pm].min()))
        ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.4)
        ax.set_xlim(0, xhi)
        ax.set_ylim(92, 100.6)
        ax.set_xlabel("Epochs since the linking fact")
        ax.text(0.5, 1.05, r"$N = %d$" % depth, transform=ax.transAxes, fontsize=9,
                ha="center", va="bottom", color=DARK)
        if col == 0:
            ax.set_ylabel("Retrieval of trained facts\n(%%, %d premises)" % n_premise,
                          linespacing=1.6, fontsize=9)
        else:
            ax.set_yticklabels([])
        panel(ax, "abc"[col], dx=-0.24 if col == 0 else -0.08, dy=1.14)
    for _, colour, label in ARMS:
        axes[0].plot([], [], color=colour, lw=1.8, label=label)
    axes[0].plot([], [], label="Prediction", **PRED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3,
               handletextpad=0.6, columnspacing=2.4, handlelength=1.8)
    p = figure(out)
    fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
