"""Toy figure 3 -- one linking fact, many inferences, at depth.

Layout of the paper's Figure 6.  An anchored 3 x 3 lattice and an unanchored
copy that reuses its relations; every cross-lattice comparison is undetermined
until one fact joins them.

    a b c   geometric error on the 81 held-out cross-lattice comparisons, in
            units of one `p` step, geometric mean over pairs with a 10-90 band
            across pairs.
    d e f   how many of those 81 the network has permanently resolved, against
            epoch.  This replaces the earlier distance ordering, whose rank
            correlation flipped sign across depth and so was not measuring a
            property of the world.

The dashed dark line is the prospective prediction: the mean dynamics
integrated from the same initialisation at the same learning rate, scored by
the same two tests, with no fitted parameters.

THE EPOCH BUDGET DIFFERS BY COLUMN AND THE CAPTION MUST SAY SO.  Every column
runs at the frozen lr_target = 0.3 and `System.lr` is depth-independent, so
nothing but depth separates them.  The unanchored copy is the slowest mode in
the world and depth preconditions exactly that mode: N = 1 needs 120k epochs to
reach the error N = 2 reaches in 4k.  The columns are therefore not comparable
epoch for epoch.

Data: toy_d{1,2,3}_{link,nolink}.json.  Prediction: toy_theory_*.json.
"""
from toy_common import (DEPTHS, NPAIRS, NONE, BRIDGE, PRED, panel, data, pred,
                        gmean, unlocked, head, plt, np, save)

XB = {1: 120000, 2: 4000, 3: 4000}
TICKS = {1: ([0, 60000, 120000], ["0", "60k", "120k"]),
         2: ([0, 2000, 4000], ["0", "2k", "4k"]),
         3: ([0, 2000, 4000], ["0", "2k", "4k"])}
ARMS = ((False, NONE, "No linking fact", 3), (True, BRIDGE, "One linking fact", 5))

fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.9))
fig.subplots_adjust(wspace=0.16, hspace=0.45)

# ------------------------------------------- top row: the geometry it unlocks
for col, dep in enumerate(DEPTHS):
    ax = axes[0][col]
    for link, colr, lab, z in ARMS:
        d, p = data(dep, link), pred(dep, link)
        ep = np.array(d["runs"][0]["epochs"], float)
        m = ep <= XB[dep]
        O = np.array([r["e3_err"] for r in d["runs"]], float).mean(axis=0)[m]
        P = np.array([r["e3_err"] for r in p["runs"]], float).mean(axis=0)[m]
        ax.fill_between(ep[m], np.percentile(O, 10, axis=1),
                        np.percentile(O, 90, axis=1), color=colr, alpha=0.15,
                        lw=0, zorder=z)
        ax.plot(ep[m], gmean(O, axis=1), color=colr, lw=1.9, label=lab,
                zorder=z + 1)
        ax.plot(ep[m], gmean(P, axis=1), zorder=7, **PRED)
        print("  N = %d  %-9s  geometric error observed %7.3f -> %.3e | "
              "predicted %7.3f -> %.3e"
              % (dep, "link" if link else "no link", gmean(O, axis=1)[0],
                 gmean(O, axis=1)[-1], gmean(P, axis=1)[0],
                 gmean(P, axis=1)[-1]))
    ax.set_yscale("log")
    ax.set_xlim(0, XB[dep])
    ax.set_ylim(1e-3, 40)
    ax.set_xticks(TICKS[dep][0])
    ax.set_xticklabels(TICKS[dep][1])
    ax.set_xlabel("Training epoch")
    if col:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Geometric error  (units of $p$)")
    head(ax, dep, data(dep, True)["lr_target"], dy=1.13)
    panel(ax, "abc"[col], dx=-0.17 if col == 0 else -0.08, dy=1.26)

XU, XUTICK = 800, [0, 400, 800]   # every arm has finished unlocking by here

# ------------------------------------ bottom row: how many inferences unlock
for col, dep in enumerate(DEPTHS):
    ax = axes[1][col]
    for link, colr, lab, z in ARMS:
        d, p = data(dep, link), pred(dep, link)
        ep = np.array(d["runs"][0]["epochs"], float)
        m = ep <= XU
        O = unlocked(d["runs"], ep)[:, m]
        P = unlocked(p["runs"], ep)[:, m]
        ax.fill_between(ep[m], O.min(axis=0), O.max(axis=0), color=colr,
                        alpha=0.15, lw=0, zorder=z)
        ax.plot(ep[m], O.mean(axis=0), color=colr, lw=1.9, label=lab,
                zorder=z + 1)
        ax.plot(ep[m], P.mean(axis=0), zorder=7, **PRED)
        print("  N = %d  %-9s  resolved of %d: observed %.1f | predicted %.1f"
              % (dep, "link" if link else "no link", NPAIRS,
                 O.mean(axis=0)[-1], P.mean(axis=0)[-1]))
    ax.set_xlim(0, XU)
    ax.set_ylim(-4, NPAIRS + 5)
    ax.set_yticks([0, 27, 54, 81])
    ax.set_xticks(XUTICK)
    ax.set_xlabel("Training epoch")
    if col:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Held-out cross-lattice\ninferences resolved (of %d)"
                      % NPAIRS, linespacing=1.7)
    panel(ax, "def"[col], dx=-0.17 if col == 0 else -0.08, dy=1.14)

axes[0][0].plot([], [], label="Prediction", **PRED)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3,
           handletextpad=0.6, columnspacing=1.8, handlelength=1.5)

save(fig, "toy_fig3_cross_lattice.png")
