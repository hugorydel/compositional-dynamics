"""How often does instantaneous accuracy fall, and does the theory fall with it?

The behavioural row switched from stable resolution, which reads the future,
to instantaneous chance-corrected accuracy, which does not.  The cost of the
switch is that a curve can fall when a single held-out item crosses back over
the retrieval boundary.  This measures that cost where it could matter:

  per world     the largest peak-to-trough fall of each world's curve
  drawn curve   the same for the across-world mean, which is what a panel shows
  theory        whether the prediction falls in the same worlds, which
                separates dynamics the theory expects from noise it does not

Reads the records the figures read, on the world sets the figures use.
"""

import json

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec.measure import instantaneous

import fig1
import fig2
import fig3

DEPTHS = (1, 2, 3)
FALL = 5.0          # points; a fall smaller than this is not counted


def fall(V):
    V = np.asarray(V, float)
    return np.max(np.maximum.accumulate(V, axis=-1) - V, axis=-1)


def row(tag, net, pred):
    fn, fp = fall(net), fall(pred)
    print("  %-22s >%g pts: network %3d  prediction %3d  both %3d  of %3d"
          " | drawn curve falls %.2f pts"
          % (tag, FALL, (fn > FALL).sum(), (fp > FALL).sum(),
             ((fn > FALL) & (fp > FALL)).sum(), len(fn),
             float(fall(np.mean(net, axis=0)))))


def chance_const(h):
    return 1.0 / h.shape[1]


def figure1():
    print("Figure 1, from initialisation, chance = best constant answer")
    for d in DEPTHS:
        recs = [json.load(open(f)) for f in fig1.cells(d)]
        for lab, law in zip("ABC", fig1.LAWS):
            c = {}
            for src in ("net", "pred"):
                c[src] = []
                for r in recs:
                    h = np.array(r[src]["hits"][law], bool)
                    c[src].append(instantaneous(h, chance=chance_const(h)))
            row("N=%d law %s" % (d, lab), c["net"], c["pred"])


def figure2():
    print("Figure 2, open law, after the bridge, every arm and item")
    seeds, _, arms = fig2.world_set(keep_all=True)
    for d in DEPTHS:
        c = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        for seed in seeds:
            for f in arms[d][seed]:
                r = json.load(open(f))
                for arm, src in c:
                    dd = r["arms"][arm][src]
                    ep = np.array(dd["epochs"], float) - r["t_switch"]
                    h = np.array(dd["hits"][r["open"]], bool)[ep >= 0]
                    c[arm, src].append(instantaneous(h, chance=chance_const(h)))
        for arm in ("hold", "insert"):
            row("N=%d %s" % (d, arm), c[arm, "net"], c[arm, "pred"])


def figure3():
    print("Figure 3, after the link, chance = score at the switch")
    seeds, _, have = fig3.world_set("f3_lr0p003")
    for d in DEPTHS:
        c = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        for seed in seeds:
            r = json.load(open(have[d][seed]))
            for arm, src in c:
                dd = r["arms"][arm][src]
                ep = np.array(dd["epochs"], float) - r["t_switch"]
                h = np.array(dd["hits"]["cross"], bool)
                k0 = int(np.argmin(np.abs(ep)))
                c[arm, src].append(instantaneous(h[ep >= 0],
                                                 chance=float(h[k0].mean())))
        for arm in ("hold", "insert"):
            row("N=%d %s" % (d, arm), c[arm, "net"], c[arm, "pred"])


if __name__ == "__main__":
    figure1()
    print()
    figure2()
    print()
    figure3()
