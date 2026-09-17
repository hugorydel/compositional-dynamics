"""Does continuing from a checkpoint reproduce an uninterrupted run exactly?

Every runner now saves, with each record, the state needed to continue it
(`relspec.checkpoint`).  That is only safe if a continued run is the same run:
the same presentation orders, the same evaluations and the same predicted
curve, to the last bit.  Each case runs a cell straight through and then again
in pieces, into a temporary directory, and compares the records and the final
checkpoints for exact equality.

  F1, depths 1 and 2   straight to 100 epochs; in chunks of 25; and to 50,
                       then with the budget raised to 100
  F3, depths 1 and 2   straight (switch 50, window 50); the window raised from
                       25 to 50; and the switch moved from 25 to 50
  F2, depth 1          straight (switch 40, window 40); the window raised from
                       20 to 40

Nothing is written under results/.
"""

import json
import os
import tempfile

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from relspec import System, checkpoint, worlds
from relspec.measure import cross_plan, law_plan

import run_experiments as rx

SEED = 0


def same(a, b):
    """'identical', or where two cells differ."""
    with open(a[0]) as f:
        ra = json.load(f)
    with open(b[0]) as f:
        rb = json.load(f)
    if ra != rb:
        return "RECORDS DIFFER"
    pa, ma = checkpoint.load(a[1])
    pb, mb = checkpoint.load(b[1])
    if ma != mb or set(pa) != set(pb):
        return "CHECKPOINT SETTINGS DIFFER"
    for n in pa:
        if pa[n]["epochs"] != pb[n]["epochs"] or pa[n]["rng"] != pb[n]["rng"]:
            return "CHECKPOINT %s DIFFERS IN EPOCHS OR GENERATOR" % n
        for src in ("net", "pred"):
            if (len(pa[n][src]) != len(pb[n][src])
                    or not all(np.array_equal(x, y) for x, y in zip(pa[n][src], pb[n][src]))):
                return "CHECKPOINT %s.%s DIFFERS" % (n, src)
    return "identical"


def case(label, tmp, run, schedules):
    """Run each schedule of budgets in turn into its own files and compare
    every one with the first."""
    out = {}
    for name, steps in schedules:
        stem = os.path.join(tmp, "%s_%s" % (label, name)).replace(" ", "_").replace("=", "")
        p = (stem + ".json", stem + ".npz")
        for step in steps:
            run(*step, *p)
        out[name] = p
    ref = schedules[0][0]
    for name, _ in schedules[1:]:
        print("  %-7s %-22s against %s: %s" % (label, name, ref, same(out[ref], out[name])),
              flush=True)


def f1(depth):
    def run(epochs, chunk, rec, ck):
        rx.run_f1(SEED, depth, epochs=epochs, chunk=chunk, rec_path=rec, ck=ck)
    return run


def f3(depth):
    S = rx.settings_for(SEED, 25)
    w0 = worlds.integration_world(SEED, link=False)
    w1 = worlds.integration_world(SEED, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(depth, settings=S)

    def run(t1, t2, rec, ck):
        rx.staged("f3", SEED, depth, S, w0, s0, s1, plan, lr, S.lr_target, t1, t2,
                  True, dict(n_pairs=len(plan["a"])), rec, ck)
    return run


def f2():
    S = rx.settings_for(SEED, 10)
    w0 = worlds.identifiability_world(SEED, K=16, k=4, closed_block="A")
    w1 = worlds.identifiability_world(SEED, K=16, k=4, closed_block="A", n_bridge=1)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    lr = s0.lr(1, settings=S)

    def run(t1, t2, rec, ck):
        rx.staged("f2", SEED, 1, S, w0, s0, s1, plan, lr, S.lr_target, t1, t2,
                  False, dict(closed="A"), rec, ck, arm="A")
    return run


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for depth in (1, 2):
            case("F1 N=%d" % depth, tmp, f1(depth), [
                ("straight", [(100, 100)]),
                ("chunks of 25", [(100, 25)]),
                ("50, then 100", [(50, 50), (100, 100)])])
        for depth in (1, 2):
            case("F3 N=%d" % depth, tmp, f3(depth), [
                ("straight", [(50, 50)]),
                ("window 25, then 50", [(50, 25), (50, 50)]),
                ("switch 25, then 50", [(25, 25), (50, 50)])])
        case("F2 N=1", tmp, f2(), [
            ("straight", [(40, 40)]),
            ("window 20, then 40", [(40, 20), (40, 40)])])


if __name__ == "__main__":
    main()
