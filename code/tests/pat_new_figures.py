"""Figure 3 rebuilt on the full candidate pool, and the retention figure.

F3   the paper's Figure 3, with cross-structure retrieval scored against all
     eighteen entities instead of the nine destination entities, and the
     prediction recomputed and scored the same way (`pat_pred18.py`).  The
     geometric row is unchanged: it is a distance normalized by candidate
     spacing, and the paper's own definition uses the destination pool.

F4   loss and recovery of trained facts while the structure is remodelled:
     mean retention across worlds, linked arm against its control.

Both average the same 200 worlds per depth as the published figures, with
shading for the standard error across worlds.  Drawn into the diagnostics
directory, not into `figures/`, since neither is adopted yet.

  usage:  python pat_new_figures.py
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
from style import DARK, panel  # noqa: E402

OUT = Path(RESULTS) / "_pat_stage1"
SUB = "f3_lr0p003"
DEPTHS = (1, 2, 3)
ARMS = (("hold", "#c0392b", "No linking fact"), ("insert", "#1b6ca8", "One linking fact"))
PRED = dict(color=DARK, lw=0.8, ls=(0, (2.2, 2.0)), zorder=6)
WINDOW = 30000


def sem(v):
    return v.std(axis=0, ddof=1) / np.sqrt(len(v))


def band(ax, x, v, colour):
    m, s = v.mean(axis=0), sem(v)
    ax.plot(x, m, color=colour, lw=1.8, zorder=3)
    ax.fill_between(x, m - s, m + s, color=colour, alpha=0.25, lw=0, zorder=2)


def counts_18(depth):
    """Network and predicted counts of 80 under the full candidate pool."""
    net, epochs = {}, None
    for path in sorted(OUT.glob("f3_lr0p003_w*_d%d.npz" % depth)):
        with np.load(path) as z:
            if epochs is None:
                epochs = z["epochs"]
            for arm, _, _ in ARMS:
                net.setdefault(arm, []).append(z["%s.cross18" % arm].sum(axis=1))
    pred, pred_epochs = {}, None
    for path in sorted(OUT.glob("pred18_%s_w*_d%d.npz" % (SUB, depth))):
        with np.load(path) as z:
            if pred_epochs is None:
                pred_epochs = z["epochs"]
            for arm, _, _ in ARMS:
                pred.setdefault(arm, []).append(z["%s.pred18" % arm].sum(axis=1))
    return (epochs, {k: np.array(v, float) for k, v in net.items()},
            pred_epochs, {k: np.array(v, float) for k, v in pred.items()})


def geometry(depth):
    """The published geometric row, straight from the records."""
    out, epochs = {}, None
    for path in sorted((Path(RESULTS) / SUB).glob("w*_d%d.json" % depth)):
        rec = json.loads(path.read_text())
        t1 = rec["t_switch"]
        for arm, _, _ in ARMS:
            for src in ("net", "pred"):
                d = rec["arms"][arm][src]
                ep = np.asarray(d["epochs"], float) - t1
                m = ep >= 0
                epochs = ep[m]
                out.setdefault((arm, src), []).append(
                    np.asarray(d["geometric"]["cross"], float)[m])
    return epochs, {k: np.array(v) for k, v in out.items()}


def ticks(hi):
    step = 10000 if hi > 20000 else 1000
    vals = list(range(0, int(hi) + 1, step))
    return vals, ["0" if v == 0 else "%gk" % (v / 1000.0) for v in vals]


def figure_f3():
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))
    fig.subplots_adjust(wspace=0.14, hspace=0.44)
    for col, depth in enumerate(DEPTHS):
        ep, net, pep, pred = counts_18(depth)
        gep, geo = geometry(depth)
        ax = axes[0][col]
        for arm, colour, _ in ARMS:
            m = ep <= WINDOW
            band(ax, ep[m], net[arm][:, m], colour)
            if arm in pred:
                pm = pep <= WINDOW
                ax.plot(pep[pm], pred[arm].mean(axis=0)[pm], **PRED)
        ax.set_ylim(-3.2, 84.8)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.text(0.5, 1.05, r"$N = %d$      $\eta_{\mathrm{target}} = 0.003$" % depth,
                transform=ax.transAxes, fontsize=9, ha="center", va="bottom", color=DARK)
        if col == 0:
            ax.set_ylabel("Correct Novel Compositions\n(of 80, all 18 candidates)",
                          linespacing=1.6, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax = axes[1][col]
        m = gep <= WINDOW
        for arm, colour, _ in ARMS:
            v = np.log10(np.maximum(geo[(arm, "net")][:, m], 1e-12))
            mu, s = v.mean(axis=0), sem(v)
            ax.plot(gep[m], 10 ** mu, color=colour, lw=1.8, zorder=3)
            ax.fill_between(gep[m], 10 ** (mu - s), 10 ** (mu + s), color=colour,
                            alpha=0.25, lw=0, zorder=2)
            p = np.log10(np.maximum(geo[(arm, "pred")][:, m], 1e-12))
            ax.plot(gep[m], 10 ** p.mean(axis=0), **PRED)
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 10)
        ax.set_xlabel("Epochs since the linking fact")
        if col == 0:
            ax.set_ylabel("Geometric Error", linespacing=1.6)
        else:
            ax.set_yticklabels([])
        for row in (0, 1):
            axes[row][col].set_xlim(0, WINDOW)
            axes[row][col].set_xticks(ticks(WINDOW)[0])
            axes[row][col].set_xticklabels(ticks(WINDOW)[1])
            panel(axes[row][col], "abcdef"[row * 3 + col],
                  dx=-0.24 if col == 0 else -0.08, dy=1.16)
    for _, colour, label in ARMS:
        axes[0][0].plot([], [], color=colour, lw=1.8, label=label)
    axes[0][0].plot([], [], label="Prediction", **PRED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               handletextpad=0.6, columnspacing=2.4, handlelength=1.8)
    path = OUT / "F3_all18_candidates.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def figure_f4():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    fig.subplots_adjust(wspace=0.14, bottom=0.34)
    for col, depth in enumerate(DEPTHS):
        rows = {arm: [] for arm, _, _ in ARMS}
        epochs = None
        for path in sorted(OUT.glob("f3_lr0p003_w*_d%d.npz" % depth)):
            with np.load(path) as z:
                if epochs is None:
                    epochs = z["epochs"]
                for arm, _, _ in ARMS:
                    rows[arm].append(100.0 * z["%s.premise" % arm].mean(axis=1))
        ax = axes[col]
        m = epochs > 0
        for arm, colour, _ in ARMS:
            band(ax, epochs[m], np.array(rows[arm])[:, m], colour)
        ax.set_xscale("log")
        ax.set_ylim(92, 100.6)
        ax.set_xlabel("Epochs since the linking fact")
        ax.text(0.5, 1.05, r"$N = %d$" % depth, transform=ax.transAxes, fontsize=9,
                ha="center", va="bottom", color=DARK)
        if col == 0:
            ax.set_ylabel("Retrieval of trained facts\n(%, 24 premises)",
                          linespacing=1.6, fontsize=9)
        else:
            ax.set_yticklabels([])
        panel(ax, "abc"[col], dx=-0.24 if col == 0 else -0.08, dy=1.14)
    for _, colour, label in ARMS:
        axes[0].plot([], [], color=colour, lw=1.8, label=label)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=2,
               handletextpad=0.6, columnspacing=2.4, handlelength=1.8)
    path = OUT / "F4_retention.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (figure_f3(), figure_f4()):
        print("wrote %s" % p)
