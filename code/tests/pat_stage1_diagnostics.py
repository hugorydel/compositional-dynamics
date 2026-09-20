"""Five questions about the Stage 1 replays, answered from the saved cells.

Nothing is trained here; `pat_stage1_controls.py` produces the cells and this
reads them.

  D1  the retention dip is a staircase.  Is that only the item count, and does
      the continuous residual underneath it move smoothly?
  D2  Experiment 2 retention never moves.  Do its facts move at all, or is the
      measure blind?  The facts touching the intervened triad are the test.
  D3  the two arms' geometric errors look further apart at greater depth at the
      left edge of a log axis.  They hold identical weights at the
      intervention, so they must start equal; how far does each move before the
      first drawn point?
  D4  Experiment 2's geometric error rises between 1,000 and 10,000 epochs at
      N=1.  The error is a distance divided by candidate spacing, so the rise
      can come from either.  Which?
  D5  is that rise a property of the experiment or of world 0?

  usage:  python pat_stage1_diagnostics.py
"""

import json
from pathlib import Path

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from style import DARK, panel  # noqa: E402

OUT = Path(RESULTS) / "_pat_stage1"
ARMS = (("hold", "#c0392b", "No linking fact"), ("insert", "#1b6ca8", "One linking fact"))
DASHED = (0, (2.2, 2.0))


def load(stem):
    with np.load(OUT / (stem + ".npz")) as z:
        return {k: z[k] for k in z.files}


def f3(seed, depth):
    return "f3_lr0p003_w%02d_d%d" % (seed, depth)


def f2(seed, depth):
    return "f2_w%02d_d%d_A" % (seed, depth)


def have(stem):
    return (OUT / (stem + ".npz")).exists()


def acc(a, arm, key):
    return 100.0 * a["%s.%s" % (arm, key)].mean(axis=1)


def d1_discreteness():
    """Is the staircase just the item count, with a smooth residual under it?"""
    print("D1  Experiment 3, world 0: what the retention staircase is made of")
    print("    %-6s %8s %8s %10s %12s %12s %12s"
          % ("depth", "items", "ever", "step size", "worst resid", "flip above",
             "safe below"))
    for depth in (1, 2, 3):
        a = load(f3(0, depth))
        hits, norm = a["insert.premise"], a["insert.premise_norm"]
        ever = ~hits.all(axis=0)
        flipped = norm[~hits]
        held = norm[hits]
        print("    %-6d %8d %8d %9.2f%% %12.3f %12.3f %12.3f"
              % (depth, hits.shape[1], int(ever.sum()), 100.0 / hits.shape[1],
                 float(norm[:, ever].max()), float(flipped.min()), float(held.max())))
    print("    step size is one item; 'flip above' is the smallest residual at which an")
    print("    item was actually wrong, in candidate spacings, and 'safe below' the")
    print("    largest at which one was still right")
    print()


def d2_experiment2_movement():
    """Do Experiment 2's trained facts move at all after the intervention?"""
    print("D2  Experiment 2, world 0: do the trained facts move, or is the measure blind?")
    print("    %-6s %14s %14s %14s %10s"
          % ("depth", "triad resid", "other resid", "rise (triad)", "flip needs"))
    for depth in (1, 2, 3):
        stem = f2(0, depth)
        if not have(stem):
            continue
        a = load(stem)
        meta = json.loads((OUT / (stem + ".json")).read_text())
        norm = a["insert.premise_norm"]
        # the intervention closes the first premise triad of the open block, so
        # its two premise facts are the ones whose geometry must change most
        spread = norm.max(axis=0) - norm[0]
        touched = np.argsort(spread)[-2:]
        rest = np.setdiff1d(np.arange(norm.shape[1]), touched)
        print("    %-6d %14.3f %14.3f %13.3f%% %10s"
              % (depth, float(norm[:, touched].max()), float(norm[:, rest].max()),
                 100.0 * float(spread[touched].mean() / max(norm[0, touched].mean(), 1e-9)),
                 "> %.2f" % float(norm.max()) if a["insert.premise"].all() else "reached"))
        _ = meta
    print("    residuals are in candidate spacings; every fact stays retrievable, so the")
    print("    flat line means no flip, not an inert network")
    print()


def d3_starting_point():
    """Both arms hold the same weights at the intervention, so they start equal."""
    print("D3  do the arms start apart, or is that the log axis?")
    print("    %-14s %6s %14s %14s %14s"
          % ("experiment", "depth", "geo at +0", "difference", "by +10 epochs"))
    for label, stem_of, key in (("Experiment 3", f3, "geo9"), ("Experiment 2", f2, "geo_B")):
        for depth in (1, 2, 3):
            stem = stem_of(0, depth)
            if not have(stem):
                continue
            a = load(stem)
            ep = a["epochs"]
            k = int(np.argmin(np.abs(ep - 10)))
            h, i = a["hold.%s" % key], a["insert.%s" % key]
            print("    %-14s %6d %14.6f %14.2e %14s"
                  % (label, depth, float(h[0]), abs(float(h[0] - i[0])),
                     "%.4f vs %.4f" % (float(h[k]), float(i[k]))))
    print("    the arms are identical at the intervention; the visible gap at the left")
    print("    edge is movement during the first ten epochs, which is faster at depth")
    print()


def d4_decomposition():
    """Is the rise in the numerator or in the spacing it is divided by?"""
    print("D4  Experiment 2, world 0: where the rise between 1,000 and 10,000 epochs is")
    print("    %-6s %-7s %26s %26s %20s"
          % ("depth", "arm", "geometric error", "query-target distance", "spacing"))
    for depth in (1, 2, 3):
        stem = f2(0, depth)
        if not have(stem):
            continue
        a = load(stem)
        ep = a["epochs"]
        lo, hi = int(np.argmin(np.abs(ep - 1000))), len(ep) - 1
        for arm, _, _ in ARMS:
            g, d, s = (a["%s.%s_B" % (arm, k)] for k in ("geo", "distance", "spacing"))
            print("    %-6d %-7s %11.4f -> %11.4f %11.4f -> %11.4f %8.3f -> %8.3f"
                  % (depth, arm, g[lo], g[hi], d[lo], d[hi], s[lo], s[hi]))
    print("    a normalized error can rise while the raw distance falls, if the")
    print("    representation contracts and the candidates move closer together")
    print()


def d5_across_worlds():
    """Is the rise a property of the experiment or of one world?"""
    print("D5  Experiment 2: the same window in other worlds")
    print("    %-6s %6s %16s %16s %14s"
          % ("depth", "world", "geo at +1,000", "geo at the end", "change"))
    for depth in (1, 3):
        for seed in (0, 1, 2):
            stem = f2(seed, depth)
            if not have(stem):
                continue
            a = load(stem)
            ep = a["epochs"]
            lo = int(np.argmin(np.abs(ep - 1000)))
            g = a["insert.geo_B"]
            print("    %-6d %6d %16.4f %16.4f %13.1f%%"
                  % (depth, seed, g[lo], g[-1], 100.0 * (g[-1] / g[lo] - 1.0)))
    print()


def figure_d1(seed=0):
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.38, bottom=0.16)
    for col, depth in enumerate((1, 2, 3)):
        a = load(f3(seed, depth))
        ep = a["epochs"]
        m = ep > 0
        hits, norm = a["insert.premise"], a["insert.premise_norm"]
        ever = ~hits.all(axis=0)
        ax = axes[0][col]
        ax.plot(ep[m], 100.0 * hits.mean(axis=1)[m], color="#1b6ca8", lw=1.7)
        ax.set_ylim(88, 101.5)
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("Premise retrieval (%)\nof 24 trained facts", linespacing=1.6,
                          fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.text(0.5, 1.05, "$N = %d$" % depth, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color=DARK)
        panel(ax, "abc"[col], dx=-0.26 if col == 0 else -0.08, dy=1.12)

        ax = axes[1][col]
        for j in np.flatnonzero(~ever):
            ax.plot(ep[m], norm[m, j], color="#b8b8b8", lw=0.7, zorder=2)
        for j in np.flatnonzero(ever):
            ax.plot(ep[m], norm[m, j], color="#c0392b", lw=1.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs since the linking fact")
        if col == 0:
            ax.set_ylabel("Residual per fact\n(candidate spacings)", linespacing=1.6,
                          fontsize=8.5)
        else:
            ax.set_yticklabels([])
        panel(ax, "def"[col], dx=-0.26 if col == 0 else -0.08, dy=1.12)
    handles = [Line2D([], [], color="#c0392b", lw=1.6, label="facts that briefly fail"),
               Line2D([], [], color="#b8b8b8", lw=0.7, label="facts that never fail")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.075), ncol=2,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    path = OUT / "diag_d1_staircase.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def figure_d345(seed=0):
    rows = (("geo", "Geometric error\n(held-out, normalized)", True),
            ("distance", "Query-to-target\ndistance (raw)", True),
            ("spacing", "Candidate spacing\n(raw)", False))
    fig, axes = plt.subplots(3, 3, figsize=(9.8, 8.0))
    fig.subplots_adjust(wspace=0.18, hspace=0.34, bottom=0.12)
    for col, depth in enumerate((1, 2, 3)):
        a = load(f2(seed, depth))
        ep = a["epochs"]
        for row, (key, label, logy) in enumerate(rows):
            ax = axes[row][col]
            for arm, colour, _ in ARMS:
                ax.plot(ep, a["%s.%s_B" % (arm, key)], color=colour,
                        lw=2.4 if arm == "hold" else 1.4)
            ax.set_xscale("symlog", linthresh=10)
            if logy:
                ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(label, linespacing=1.6, fontsize=8.5)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.text(0.5, 1.05, "$N = %d$" % depth, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, color=DARK)
            if row == 2:
                ax.set_xlabel("Epochs since the linking fact")
            panel(ax, "abcdefghi"[row * 3 + col], dx=-0.26 if col == 0 else -0.08, dy=1.1)
    handles = [Line2D([], [], color=c, lw=2.4 if n == "hold" else 1.4, label=l)
               for n, c, l in ARMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07), ncol=2,
               handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
    path = OUT / "diag_d34_decomposition.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def figure_d5():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    fig.subplots_adjust(wspace=0.28, bottom=0.3)
    shades = ("#1b6ca8", "#4c9ad4", "#9ec9e8")
    for col, depth in enumerate((1, 3)):
        ax = axes[col]
        drawn = 0
        for seed in (0, 1, 2):
            stem = f2(seed, depth)
            if not have(stem):
                continue
            a = load(stem)
            ep = a["epochs"]
            m = ep > 0
            ax.plot(ep[m], a["insert.geo_B"][m], color=shades[seed], lw=1.6,
                    label="world %d" % seed)
            drawn += 1
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs since the linking fact")
        ax.set_ylabel("Geometric error" if col == 0 else None)
        ax.text(0.5, 1.05, "Experiment 2, $N = %d$ (%d worlds)" % (depth, drawn),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=DARK)
        panel(ax, "ab"[col], dx=-0.2, dy=1.12)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    path = OUT / "diag_d5_worlds.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def main():
    d1_discreteness()
    d2_experiment2_movement()
    d3_starting_point()
    d4_decomposition()
    d5_across_worlds()
    for path in (figure_d1(), figure_d345(), figure_d5()):
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
