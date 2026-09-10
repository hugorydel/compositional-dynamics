"""Where does the theory-to-network gap at depth come from?

Three trajectories of the SAME quantity, from the SAME initialisation, on the
same world:

  sgd     per-fact stochastic descent -- what the figures plot
  full    epoch-averaged full-batch descent -- the discrete process the
          continuous theory is the limit of
  theory  the integrated mean dynamics -- what the overlay plots

That isolates the two candidates.  If `full` sits on `theory`, the gap is
stochastic descent, and no amount of integration accuracy will close it.  If
`full` departs from `theory` too, the gap is the ODE integration itself, i.e.
the per-depth substep counts, which were tuned on the paper's worlds.

Each cell is run at the frozen lr_target = 0.3 and again at 0.03 with ten
times the epochs, so the two rates cover the same effective time
`epochs x lr_target` and can be read on one axis.  That is the direct test of
whether the step size is responsible.

Quantity plotted: cross-lattice geometric error, geometric mean over the 81
held-out pairs, in units of one `p` step.
"""
import json
import time

import numpy as np

import _paths                                               # noqa: F401
from _paths import figure, result                           # noqa: E402
import matplotlib.pyplot as plt                             # noqa: E402
from style import DARK, panel                               # noqa: E402
from toy import build, M                                   # noqa: E402
from relspec import System, models, theory, train           # noqa: E402
from relspec.config import override                         # noqa: E402

SEED = 0
CROSS = [(i, j, k, l) for i in range(M) for j in range(M)
         for k in range(M) for l in range(M)]
# (lr_target, epochs, eval_every).  epochs x lr_target is matched.
RATES = [(0.3, 4000, 20), (0.03, 40000, 200)]
DEPTHS = (2, 3)
COL = dict(sgd="#2c5985", full="#c98b34", theory="#9d3b39")


def geo(world, Es):
    ti, gt = world.tok_index, world.meta["gt_ent"]
    rn = np.linalg.norm(world.meta["gt_rel"]["p"])
    out = np.zeros((len(Es), len(CROSS)))
    for t, E in enumerate(Es):
        for c, (i, j, k, l) in enumerate(CROSS):
            a, b = "A_%d_%d" % (i, j), "Acopy_%d_%d" % (k, l)
            out[t, c] = np.linalg.norm((E[ti[b]] - E[ti[a]])
                                       - (gt[b] - gt[a])) / rn
    return np.exp(np.log(np.maximum(out, 1e-16)).mean(axis=1))


def cell(depth, lrt, epochs, every):
    st = override(lr_target=lrt, eval_every=every)
    w = build(True, SEED)
    s = System.build(w, settings=st)
    lr = s.lr(depth, settings=st)
    probes = dict(E=np.eye(w.P))
    out = {}
    for mode in ("sgd", "full"):
        m = models.make_model(w, depth, st)
        tr = train.train(m, s, lr, epochs, st, order_seed=7, mode=mode,
                         eval_every=every, probes=probes)
        out[mode] = (np.array(tr.epochs, float), geo(w, tr.probes["E"]))
    tr, _ = theory.predict(s, depth, lr, epochs, models.make_model(w, depth, st),
                           settings=st, eval_every=every, probes=probes)
    out["theory"] = (np.array(tr.epochs, float), geo(w, tr.probes["E"]))
    return out


def dev(a, b):
    """Median absolute log10 ratio between two curves, over the window where
    either has begun to move (below 95% of its starting value), so the shared
    flat head does not dilute the statistic."""
    x, y = a[1], b[1]
    n = min(len(x), len(y))
    x, y = np.maximum(x[:n], 1e-16), np.maximum(y[:n], 1e-16)
    m = (x < 0.95 * x[0]) | (y < 0.95 * y[0])
    if not m.any():
        m = np.ones(n, bool)
    return float(np.median(np.abs(np.log10(x[m] / y[m]))))


def main():
    CACHE = {}
    fig, axes = plt.subplots(len(DEPTHS), len(RATES), figsize=(7.6, 5.6))
    fig.subplots_adjust(wspace=0.16, hspace=0.5)
    print("median |log10 ratio| over the whole trajectory")
    print("  %-5s %-6s %10s %10s %10s %8s"
          % ("N", "lr", "sgd/theory", "full/theory", "sgd/full", "secs"))

    for r, depth in enumerate(DEPTHS):
        for c, (lrt, epochs, every) in enumerate(RATES):
            t0 = time.time()
            o = cell(depth, lrt, epochs, every)
            CACHE["N%d_lr%g" % (depth, lrt)] = {
                k: dict(epochs=v[0].tolist(), geo=v[1].tolist())
                for k, v in o.items()}
            ax = axes[r][c]
            for k in ("sgd", "full", "theory"):
                x, y = o[k]
                ax.plot(x * lrt, y, color=COL[k], lw=1.7 if k != "theory" else 1.2,
                        ls="-" if k != "theory" else (0, (2.4, 2.2)), label=k,
                        zorder=3)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_ylim(1e-3, 40)
            ax.set_xlabel(r"Effective time  (epochs $\times\ \eta_{\mathrm{target}}$)")
            if c == 0:
                ax.set_ylabel("Geometric error\n(units of $p$)", linespacing=1.6)
            ax.text(0.5, 1.04, r"$N = %d$      $\eta_{\mathrm{target}} = %g$"
                    % (depth, lrt), transform=ax.transAxes, fontsize=8.5,
                    ha="center", va="bottom", color=DARK)
            panel(ax, "abcd"[r * 2 + c], dx=-0.24 if c == 0 else -0.10, dy=1.2)
            print("  %-5d %-6g %10.3f %10.3f %10.3f %8.0f"
                  % (depth, lrt, dev(o["sgd"], o["theory"]),
                     dev(o["full"], o["theory"]), dev(o["sgd"], o["full"]),
                     time.time() - t0), flush=True)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, ["Per-fact SGD (what the figures plot)",
                   "Full-batch GD (the theory's discrete process)",
                   "Integrated mean dynamics (the overlay)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=1,
               handletextpad=0.6)
    with open(result("toy_diag.json"), "w") as f:
        json.dump(CACHE, f)
    p = figure("toy_diag.png")
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print("wrote %s" % p)



if __name__ == "__main__":
    main()