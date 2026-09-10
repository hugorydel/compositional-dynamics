"""Toy figure 2 -- the two proposed measures side by side, at depth.

Layout of the paper's Figure 6: one column per depth, one row per measure.

    a b c   held-out compositional accuracy: for a held-out pair `a -> b`, is
            `b` ranked first among all 54 entities for the query `a + (x+y)`?
    d e f   compositional geometric error: ||z - (x+y)|| / ||x+y||.

Two laws in every panel: one whose composite relation is shown once, which is
structurally identifiable (rho = 0), and one whose composite is never shown,
which is not (rho = 0.577).  The dashed dark line is the prospective
prediction, integrated from the same initialisation and scored by the same two
tests, with no fitted parameters.

The rows disagree, and that is the finding.  Accuracy puts the two laws on top
of each other at 100%, because withholding the composite fact leaves the pair
joined by an observed two-step path.  The geometric error separates them by
four orders of magnitude, because it interrogates the composite RELATION rather
than a path between two entities.

The error is normalised by ||x + y||, not by ||z||.  When the composite is
never shown, z lies in the null space and its length is free, so dividing by it
sends the ratio to 10^1 - 10^2 and off any axis the identifiable law can share.
||x + y|| is determined by the observed facts, so the ratio is bounded and 1.0
reads as "the composite is entirely wrong".

Data: toy_d{1,2,3}_link.json.  Prediction: toy_theory_d{1,2,3}_link.json.
"""
from toy_common import (DEPTHS, IDC, NOC, PRED, panel, data, pred, series,
                        gmean, head, plt, np, save)

XMAX = {0: 500, 1: 4000}      # the accuracy row saturates inside 500
XTICK = {0: ([0, 250, 500], ["0", "250", "500"]),
         1: ([0, 2000, 4000], ["0", "2k", "4k"])}
LAWS = [("E2_id", IDC, r"Composite seen once  ($\rho = 0$)"),
        ("E2_no", NOC, r"Composite never seen  ($\rho = 0.58$)")]
ROWS = [("acc_xy", "Held-out compositional\naccuracy (%)", False),
        ("geo_xy", r"Compositional geometric error" "\n"
                   r"$\|z-(x{+}y)\|\,/\,\|x{+}y\|$", True)]

fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.9))
fig.subplots_adjust(wspace=0.14, hspace=0.45)

for row, (key, ylab, logy) in enumerate(ROWS):
    for col, dep in enumerate(DEPTHS):
        ax = axes[row][col]
        d, p = data(dep, True), pred(dep, True)
        ep = np.array(d["runs"][0]["epochs"], float)
        m = ep <= XMAX[row]
        for law, colr, lab in LAWS:
            O = series(d["runs"], "%s.%s" % (key, law))[:, m]
            P = series(p["runs"], "%s.%s" % (key, law))[:, m]
            o = gmean(O) if logy else O.mean(axis=0)
            q = gmean(P) if logy else P.mean(axis=0)
            ax.fill_between(ep[m], O.min(axis=0), O.max(axis=0), color=colr,
                            alpha=0.15, lw=0, zorder=2)
            ax.plot(ep[m], o, color=colr, lw=1.9, zorder=3, label=lab)
            ax.plot(ep[m], q, zorder=4, **PRED)
            print("  %-6s N = %d  %-6s  observed %9.4f -> %9.4f | predicted "
                  "%9.4f -> %9.4f" % (key, dep, law, o[0], o[-1], q[0], q[-1]))
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 4)
            ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=(0, (3, 3)), zorder=1)
        else:
            ax.set_ylim(-4, 106)
            ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, XMAX[row])
        ax.set_xticks(XTICK[row][0])
        ax.set_xticklabels(XTICK[row][1])
        ax.set_xlabel("Training epoch")
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(ylab, linespacing=1.7)
        if row == 0:
            head(ax, dep, d["lr_target"], dy=1.13)
        panel(ax, "abcdef"[row * 3 + col],
              dx=-0.22 if col == 0 else -0.08, dy=1.26 if row == 0 else 1.14)

axes[0][0].plot([], [], label="Prediction", **PRED)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3,
           handletextpad=0.6, columnspacing=1.8, handlelength=1.5)

save(fig, "toy_fig2_measures.png")
