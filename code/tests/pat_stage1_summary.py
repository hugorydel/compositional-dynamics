"""E1 and E2 across all 200 Experiment 3 worlds, from the completed control pass.

`analysis/controls.py` replayed every cell and stored per-item scores; this
turns them into the two answers the revision needs.

  E1  retrieval against all 18 entities beside the nine destination entities:
      what the baseline becomes, whether the linked networks still resolve
      every comparison, and how much later they do it
  E2  retrieval of the facts the network was trained on, while it reorganizes:
      how often a transient loss happens, how many facts, how long, and
      whether every world recovers

Worlds are the unit throughout: each cell contributes one value, summarized
across the 200 of them with a 95% t interval, as the manuscript does.

  usage:  python pat_stage1_summary.py
"""

import json
from pathlib import Path

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from scipy import stats as ss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from style import DARK, panel  # noqa: E402

OUT = Path(RESULTS) / "f3"
DEPTHS = (1, 2, 3)
ARMS = (("hold", "#c0392b", "No linking fact"), ("insert", "#1b6ca8", "One linking fact"))
DASHED = (0, (2.2, 2.0))


def load_depth(depth):
    """Every world's scores at one depth, stacked on a shared epoch grid."""
    cells, epochs = {}, None
    for path in sorted(OUT.glob("controls_w*_d%d.npz" % depth)):
        seed = int(path.stem.split("_")[-2][1:])
        with np.load(path) as z:
            data = {k: z[k] for k in z.files if not k.startswith("pre.")}
        if epochs is None:
            epochs = data["epochs"]
        elif not np.array_equal(epochs, data["epochs"]):
            raise SystemExit("world %d at depth %d is on a different grid" % (seed, depth))
        cells[seed] = data
    return epochs, cells


def stack(cells, arm, key):
    """(worlds x evaluations) accuracy in percent."""
    return np.array([100.0 * cells[s]["%s.%s" % (arm, key)].mean(axis=1)
                     for s in sorted(cells)])


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float(x.mean()), float(x.mean()), float(x.mean())
    h = ss.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def first_at(epochs, series, level):
    k = np.flatnonzero(series >= level - 1e-9)
    return epochs[k[0]] if len(k) else np.inf


def e1(data):
    print("E1  candidate pool, Experiment 3, 200 worlds per depth")
    print("    %-6s %-9s %22s %22s" % ("depth", "candidates", "at the fact", "at 30,000 epochs"))
    for depth in DEPTHS:
        epochs, cells = data[depth]
        k30 = int(np.argmin(np.abs(epochs - 30000)))
        for key, label in (("cross9", "nine"), ("cross18", "all 18")):
            link, hold = stack(cells, "insert", key), stack(cells, "hold", key)
            m0, _, _ = ci(link[:, 0])
            m30, lo30, hi30 = ci(link[:, k30])
            h30, hlo, hhi = ci(hold[:, k30])
            print("    %-6d %-9s %10.1f%% of 80 %8s %8.1f%% [%.1f, %.1f]  no link %.1f%% [%.1f, %.1f]"
                  % (depth, label, m0, "", m30, lo30, hi30, h30, hlo, hhi))
    print()
    print("    %-6s %-9s %14s %16s %26s" % ("depth", "candidates", "all 80 at 30k",
                                            "all 80 at 40k", "epochs to 72 of 80"))
    for depth in DEPTHS:
        epochs, cells = data[depth]
        k30 = int(np.argmin(np.abs(epochs - 30000)))
        for key, label in (("cross9", "nine"), ("cross18", "all 18")):
            link = stack(cells, "insert", key)
            t = np.array([first_at(epochs, v, 100.0 * 72 / 80) for v in link])
            fin = t[np.isfinite(t)]
            print("    %-6d %-9s %10d/200 %12d/200 %14s (IQR %s-%s), reached in %d"
                  % (depth, label, int((link[:, k30] >= 100 - 1e-9).sum()),
                     int((link[:, -1] >= 100 - 1e-9).sum()),
                     "{:,.0f}".format(np.median(fin)),
                     "{:,.0f}".format(np.percentile(fin, 25)),
                     "{:,.0f}".format(np.percentile(fin, 75)), len(fin)))
    print()
    hold18 = [stack(data[d][1], "hold", "cross18").max() for d in DEPTHS]
    print("    no-link accuracy against all 18 candidates never exceeds %.1f%% at any depth,"
          % max(hold18))
    print("    any world, any evaluation")
    print()


def e2(data):
    print("E2  retention of trained facts, Experiment 3, linked arm, 200 worlds per depth")
    print("    %-6s %12s %12s %12s %16s %14s %12s"
          % ("depth", "worlds hit", "facts lost", "lowest", "onset", "duration", "recovered"))
    summary = {}
    for depth in DEPTHS:
        epochs, cells = data[depth]
        prem = stack(cells, "insert", "premise")
        n_items = cells[sorted(cells)[0]]["insert.premise"].shape[1]
        dipped = prem.min(axis=1) < 100 - 1e-9
        lost = np.round((100 - prem.min(axis=1)) / 100 * n_items).astype(int)
        onset, dur = [], []
        for row in prem[dipped]:
            bad = epochs[row < 100 - 1e-9]
            onset.append(bad.min())
            dur.append(bad.max() - bad.min())
        recovered = int((prem[dipped][:, -1] >= 100 - 1e-9).sum())
        summary[depth] = dict(dipped=int(dipped.sum()), lost=lost[dipped],
                              onset=np.array(onset), dur=np.array(dur),
                              recovered=recovered, prem=prem, epochs=epochs)
        print("    %-6d %8d/200 %12s %11.1f%% %16s %14s %9d/%d"
              % (depth, int(dipped.sum()),
                 "%d (max %d)" % (int(np.median(lost[dipped])), int(lost.max())) if dipped.any() else "-",
                 prem.min(axis=1).min(),
                 "{:,.0f}".format(np.median(onset)) if onset else "-",
                 "{:,.0f}".format(np.median(dur)) if dur else "-",
                 recovered, int(dipped.sum())))
    print()
    print("    %-6s %26s %28s" % ("depth", "all trained facts", "worlds that never dip"))
    for depth in DEPTHS:
        epochs, cells = data[depth]
        facts = stack(cells, "insert", "facts")
        prem = summary[depth]["prem"]
        safe = prem.min(axis=1) >= 100 - 1e-9
        worst = [cells[s]["insert.premise_norm"].max() for s in sorted(cells)]
        worst = np.array(worst)[safe]
        print("    %-6d %14d/200 hit, lowest %5.1f%% %18s"
              % (depth, int((facts.min(axis=1) < 100 - 1e-9).sum()), facts.min(),
                 "worst residual %.2f spacings" % worst.max() if len(worst) else "-"))
    print()
    hold = {d: stack(data[d][1], "hold", "premise") for d in DEPTHS}
    print("    no-link arm: %s"
          % "; ".join("N=%d lowest %.1f%%" % (d, hold[d].min()) for d in DEPTHS))
    print()
    return summary


def figures(data, summary):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3))
    fig.subplots_adjust(wspace=0.16, bottom=0.34)
    for col, depth in enumerate(DEPTHS):
        epochs, cells = data[depth]
        m = epochs > 0
        ax = axes[col]
        for arm, colour, _ in ARMS:
            for key, ls, lw in (("cross9", "-", 1.8), ("cross18", DASHED, 1.2)):
                v = stack(cells, arm, key)
                ax.plot(epochs[m], v.mean(axis=0)[m], color=colour, lw=lw, ls=ls)
        ax.set_xscale("log")
        ax.set_ylim(-4, 104)
        ax.set_xlabel("Epochs since the linking fact")
        ax.text(0.5, 1.05, "$N = %d$" % depth, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color=DARK)
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Cross-structure\naccuracy (%)", linespacing=1.6)
        panel(ax, "abc"[col], dx=-0.22 if col == 0 else -0.08, dy=1.12)
    handles = [Line2D([], [], color=c, lw=1.8, label=l) for _, c, l in ARMS] + [
        Line2D([], [], color=DARK, lw=1.8, label="Nine destination candidates"),
        Line2D([], [], color=DARK, lw=1.2, ls=DASHED, label="All 18 candidates")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.12), ncol=4,
               handletextpad=0.6, columnspacing=1.8, handlelength=1.8)
    p1 = OUT / "diagnostic_e1_candidate_pool.png"
    fig.savefig(p1, bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.42, bottom=0.14)
    for col, depth in enumerate(DEPTHS):
        epochs, cells = data[depth]
        m = epochs > 0
        prem = summary[depth]["prem"]
        ax = axes[0][col]
        ax.plot(epochs[m], prem.mean(axis=0)[m], color="#1b6ca8", lw=1.8)
        sem = prem.std(axis=0, ddof=1) / np.sqrt(len(prem))
        ax.fill_between(epochs[m], (prem.mean(axis=0) - sem)[m],
                        (prem.mean(axis=0) + sem)[m], color="#1b6ca8", alpha=0.25, lw=0)
        ax.set_ylim(min(90.0, prem.mean(axis=0).min() - 1.0), 100.4)
        ax.text(0.5, 1.05, "$N = %d$" % depth, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color=DARK)
        if col == 0:
            ax.set_ylabel("Mean retention\nacross worlds (%)", linespacing=1.6, fontsize=9)
        else:
            ax.set_yticklabels([])
        panel(ax, "abc"[col], dx=-0.24 if col == 0 else -0.08, dy=1.12)

        ax = axes[1][col]
        share = 100.0 * (prem < 100 - 1e-9).mean(axis=0)
        ax.plot(epochs[m], share[m], color="#c0392b", lw=1.8)
        ax.set_ylim(-2, max(12, share.max() * 1.2))
        ax.set_xlabel("Epochs since the linking fact")
        if col == 0:
            ax.set_ylabel("Worlds with a fact\nunretrieved (%)", linespacing=1.6, fontsize=9)
        panel(ax, "def"[col], dx=-0.24 if col == 0 else -0.08, dy=1.12)
        for row in (0, 1):
            axes[row][col].set_xscale("log")
    p2 = OUT / "diagnostic_e2_retention.png"
    fig.savefig(p2, bbox_inches="tight", dpi=200)
    plt.close(fig)
    for p in (p1, p2):
        print("wrote %s" % p)


if __name__ == "__main__":
    data = {d: load_depth(d) for d in DEPTHS}
    for d in DEPTHS:
        if len(data[d][1]) != 200:
            print("warning: depth %d has %d worlds" % (d, len(data[d][1])))
    e1(data)
    figures(data, e2(data))
