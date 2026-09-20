"""Does Experiment 2's depth-3 rebound survive a ten-times smaller step?

The linked arm's geometric error at depth 3 reaches a minimum and then climbs
again inside its window, in the network and in the prediction alike.  Two
explanations fit that: the mean flow genuinely turns around, or per-row SGD at
eta_target = 0.03 departs from the flow it approximates, as it did at depth 3
in Experiment 3 before that experiment was moved to 0.003.

The mean dynamics depend on the product of rate and epochs, so dividing the
rate by ten and multiplying every budget by ten leaves the predicted curve
unchanged and only makes the discrete process finer.  If the rebound is a
step-size artefact it should shrink; if it is the flow, it should reappear at
the same rescaled epoch.

One cell only: world 0, depth 3, closed block A, the same initialization and
presentation orders as the stored run.  Writes beside the other Stage 1
diagnostics and touches no published record.

  usage:  python pat_rate_check.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import _boot  # noqa: F401, E402
import _paths  # noqa: F401, E402
import numpy as np  # noqa: E402
from _paths import RESULTS  # noqa: E402
from relspec import System, worlds  # noqa: E402
from relspec.config import DEFAULT, override  # noqa: E402
from relspec.measure import law_plan  # noqa: E402
from run_experiments import staged  # noqa: E402

OUT = Path(RESULTS) / "f2"
BASE, RATE = 0.03, 0.003
SEED, DEPTH, CLOSED = 0, 3, "A"


def run():
    scale = int(round(BASE / RATE))
    t1 = worlds.F2_SWITCH[DEPTH] * scale
    t2 = worlds.F2_AFTER[DEPTH] * scale
    every = worlds.F2_EVERY[DEPTH] * scale
    S = override(init_seed=1000 + SEED, eval_every=every, lr_target=RATE,
                 ode_substeps={DEPTH: max(1, int(round(DEFAULT.substeps(DEPTH) * RATE / BASE)))})
    w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=CLOSED)
    w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block=CLOSED, n_bridge=1)
    s0, s1 = (System.build(w, settings=S) for w in (w0, w1))
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    lr = s0.lr(DEPTH, target=RATE, settings=S)
    extra = dict(closed=CLOSED, open="B" if CLOSED == "A" else "A", rate=RATE,
                 scale=scale, substeps=S.substeps(DEPTH))
    stem = "ratecheck_w%02d_d%d_%s" % (SEED, DEPTH, CLOSED)
    print("running %s: %s pre-intervention epochs, then 2 x %s, evaluated every %s"
          % (stem, "{:,}".format(t1), "{:,}".format(t2), "{:,}".format(every)), flush=True)
    t0 = time.time()
    rec = staged("f2", SEED, DEPTH, S, w0, s0, s1, plan, lr, RATE, t1, t2, False,
                 extra, str(OUT / (stem + ".json")), str(OUT / (stem + ".npz")),
                 arm=CLOSED)
    print("  done in %.0f s" % (time.time() - t0), flush=True)
    return rec, scale


def curve(record, arm, src):
    """Post-intervention geometric error of the underdetermined law."""
    d = record["arms"][arm][src]
    ep = np.asarray(d["epochs"], float) - record["t_switch"]
    m = ep >= 0
    return ep[m], np.asarray(d["geometric"][record["open"]], float)[m]


def compare(rec, scale):
    old = json.loads((Path(RESULTS) / "f2" / ("w%02d_d%d_%s.json" % (SEED, DEPTH, CLOSED))).read_text())
    print()
    print("linked arm, geometric error of the underdetermined law")
    print("  %-14s %-5s %10s %10s %9s %14s"
          % ("rate", "src", "min", "end", "rise", "min at epoch"))
    rows = {}
    for label, record, factor in (("0.03 (stored)", old, 1.0), ("0.003 (new)", rec, 1.0 / scale)):
        for src in ("net", "pred"):
            ep, g = curve(record, "insert", src)
            k = int(g.argmin())
            rows[(label, src)] = (ep * factor, g)
            print("  %-14s %-5s %10.4f %10.4f %8.0f%% %14s"
                  % (label, src, g[k], g[-1], 100 * (g[-1] / g[k] - 1),
                     "{:,.0f}".format(ep[k] * factor)))
    print("  epochs for the 0.003 run are divided by %d, so both rows are on the same"
          % scale)
    print("  amount of learning; a step-size artefact would shrink, the flow would not")
    return rows


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from style import DARK, panel

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    styles = {("0.03 (stored)", "net"): dict(color="#1b6ca8", lw=1.8),
              ("0.03 (stored)", "pred"): dict(color=DARK, lw=0.9, ls=(0, (2.2, 2.0))),
              ("0.003 (new)", "net"): dict(color="#c0392b", lw=1.8),
              ("0.003 (new)", "pred"): dict(color=DARK, lw=0.9, ls=(0, (1, 1.6)))}
    for key, (ep, g) in rows.items():
        m = ep > 0
        ax.plot(ep[m], g[m], label="%s, %s" % key, **styles[key])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs since the linking fact, rescaled to $\\eta_{\\mathrm{target}}=0.03$")
    ax.set_ylabel("Geometric error")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    panel(ax, "a", dx=-0.16, dy=1.06)
    path = OUT / "diagnostic_ratecheck_d3.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote %s" % path)


if __name__ == "__main__":
    record, scale_used = run()
    draw(compare(record, scale_used))
