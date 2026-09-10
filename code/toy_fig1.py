"""Toy figure 1 -- held-out compositional accuracy, at depth.

Layout of the paper's Figure 1: one panel per depth, one curve per law,
accuracy against training epoch, with the prospective prediction overlaid as a
dashed dark line.  The prediction is the integrated mean dynamics from the same
initialisation, scored by the same rank-1 test, with no fitted parameters.

The measure is the proposed one: for a held-out pair `a -> b`, is `b` ranked
first among all 54 entities for the query point `a + (x + y)`?

Laws A and B each show their composite on one of four lattice cells; law C
shows its composite on none, and is provably non-identifiable (rho = 0.577).
The curves are indistinguishable.  Withholding the composite FACT does not
withhold the composite: `a`'s and `b`'s endpoints are still joined by an
observed x-step and then an observed y-step, so the query point is pinned by
training facts whatever the law's identifiability.  The prediction tracks the
curves for the same reason, which is why an overlay agreeing here is not
evidence about composition.

Law A pools the two anchored lattices, which share their evidence condition.

Data: toy_d{1,2,3}_link.json.  Prediction: toy_theory_d{1,2,3}_link.json.
"""
from toy_common import (DEPTHS, PRED, panel, ramp, data, pred, series,
                        head, plt, np, save)

XMAX = 500
LAWS = [("E1", "Law A", 6, 1), ("E2_id", "Law B", 3, 1),
        ("E2_no", "Law C", 4, 0)]
COLOUR = dict(zip([k for k, _, _, _ in LAWS], ramp(3)))

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
fig.subplots_adjust(wspace=0.09)

for ax, dep, letter in zip(axes, DEPTHS, "abc"):
    d, p = data(dep, True), pred(dep, True)
    ep = np.array(d["runs"][0]["epochs"], float)
    m = ep <= XMAX
    for key, name, nheld, nex in LAWS:
        obs = series(d["runs"], "acc_xy." + key).mean(axis=0)
        pre = series(p["runs"], "acc_xy." + key).mean(axis=0)
        ax.plot(ep[m], obs[m], color=COLOUR[key], lw=1.9, zorder=3,
                label="%s  (%d composite example%s, %d held out)"
                      % (name, nex, "" if nex == 1 else "s", nheld))
        ax.plot(ep[m], pre[m], zorder=4, **PRED)
        print("  N = %d  %-6s  observed %5.1f -> %5.1f | predicted %5.1f -> "
              "%5.1f | max |gap| %4.1f pts"
              % (dep, name, obs[0], obs[-1], pre[0], pre[-1],
                 np.abs(obs - pre).max()))

    ax.set_xlim(0, XMAX)
    ax.set_ylim(-4, 106)
    ax.set_xticks([0, 250, 500])
    ax.set_xlabel("Training epoch")
    head(ax, dep, d["lr_target"])
    panel(ax, letter, dx=-0.11 if letter == "a" else -0.05, dy=1.15)

axes[0].set_ylabel("Held-out compositional accuracy (%)")
axes[0].set_yticks([0, 25, 50, 75, 100])
axes[0].plot([], [], label="Prediction", **PRED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
           handletextpad=0.6, columnspacing=2.2)

save(fig, "toy_fig1_accuracy.png")
