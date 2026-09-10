"""Produce the data for all three figures, one file per cell, resumably.

A cell is one (figure, world seed, depth) and, for Figure 2, one counterbalance
arm.  Each cell writes its own file and is skipped if that file already exists,
so a run can be stopped and restarted at any point and costs at most the cell
in flight.  Writes go to a temporary name and are then renamed, so an interrupt
during writing cannot leave a truncated file that a later run would mistake for
a finished one.

Everything is deterministic given the seed: the world, the initialisation and
the presentation order are all fixed by it, so a recomputed cell is identical
to the one it replaces and resuming cannot change a result.

Each world gets its own initialisation, `init_seed = 1000 + seed`.  Sharing one
initialisation across worlds would make the laws within a world non-independent
and shrink the effective sample to the number of initialisations rather than
the number of worlds.

  usage:  python run_experiments.py plan            what would run
          python run_experiments.py run 0           world 0, all figures
          python run_experiments.py run 0 1 2       worlds 0, 1, 2
          python run_experiments.py run 0 --only f1
"""

import copy
import json
import os
import sys
import time

import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from relspec import System, models, theory, train, worlds
from relspec.config import override
from relspec.measure import cross_plan, law_plan

DEPTHS = (1, 2, 3)
ORDER_SEED = 7


def settings_for(seed, every):
    return override(init_seed=1000 + seed, eval_every=every)


def cell_path(fig, seed, depth, arm=None):
    d = os.path.join(RESULTS, fig)
    os.makedirs(d, exist_ok=True)
    name = "w%02d_d%d" % (seed, depth) + ("_%s" % arm if arm else "") + ".json"
    return os.path.join(d, name)


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def series(traj, want_items=True):
    """Per-item `hits` are always kept.  A monotone curve of items stably
    resolved cannot be reconstructed from the aggregate percentage afterwards,
    and that curve, not instantaneous top-1 accuracy, is what an emergence
    panel should show."""
    out = dict(
        epochs=[float(e) for e in traj.epochs],
        retrieval={k: list(map(float, v)) for k, v in traj.retrieval.items()},
        rank={k: list(map(float, v)) for k, v in traj.rank.items()},
        geometric={k: list(map(float, v)) for k, v in traj.geometric.items()},
        hits={k: v.astype(int).tolist() for k, v in traj.hits.items()},
    )
    if want_items:
        out["errs"] = {k: (None if v is None else v.tolist())
                       for k, v in traj.errs.items()}
    return out


def join(a, b, t1):
    """Concatenate two phases, dropping b's duplicated first evaluation."""
    out = dict(epochs=a["epochs"] + [t1 + e for e in b["epochs"][1:]])
    for key in ("retrieval", "rank", "geometric", "hits"):
        if key in a:
            out[key] = {k: a[key][k] + b[key][k][1:] for k in a[key]}
    if "errs" in a:
        out["errs"] = {k: (None if a["errs"][k] is None
                           else a["errs"][k] + b["errs"][k][1:]) for k in a["errs"]}
    return out


# --------------------------------------------------------------------------- #
#  Figure 1: unstaged, one run per depth                                       #
# --------------------------------------------------------------------------- #

def run_f1(seed, depth):
    S = settings_for(seed, worlds.F1_EVERY[depth])
    ep = worlds.F1_EPOCHS[depth]
    w = worlds.emergence_world(seed)
    s = System.build(w, settings=S)
    plan = law_plan(w, worlds.held_composites(w))
    lr = s.lr(depth, settings=S)
    obs = train.train(models.make_model(w, depth, S), s, lr, ep, S,
                      order_seed=ORDER_SEED, eval_every=S.eval_every, plan=plan)
    pre, _ = theory.predict(s, depth, lr, ep, models.make_model(w, depth, S),
                            settings=S, eval_every=S.eval_every, plan=plan)
    return dict(figure="f1", seed=seed, depth=depth, epochs_max=ep,
                lr_target=S.lr_target, init_seed=S.init_seed,
                order_seed=ORDER_SEED, net=series(obs), pred=series(pre))


# --------------------------------------------------------------------------- #
#  Figure 2: staged, one run per counterbalance arm                            #
# --------------------------------------------------------------------------- #

def run_f2(seed, depth, closed):
    """One counterbalance arm, branched at the switch.

    The comparison the figure needs is the SAME initially non-identifiable law
    with and without the bridge, from the same pre-switch weights.  An earlier
    version gave the bridge to every arm and used the already-identifiable law
    as the control, which is no control at all: both laws are learnable after
    the switch, so both errors fall and the causal claim disappears.  The
    closed law is still recorded, but its job is to show that premise and
    composite knowledge were acquired beforehand, not to stand in for the
    counterfactual.
    """
    S = settings_for(seed, worlds.F2_EVERY[depth])
    t1, t2 = worlds.F2_SWITCH[depth], worlds.F2_AFTER[depth]
    w0 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed)
    w1 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed,
                                      n_bridge=1)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    # evaluation set from the POST-intervention world, so it cannot move
    plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
    lr = s0.lr(depth, settings=S)

    m = models.make_model(w0, depth, S)
    a = train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED,
                    eval_every=S.eval_every, plan=plan)
    ta, st = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                            settings=S, eval_every=S.eval_every, plan=plan)

    arms = {}
    for name, sy in (("hold", s0), ("insert", s1)):
        mm = copy.deepcopy(m)  # both branches leave the SAME pre-switch state
        b = train.train(mm, sy, lr, t2, S, order_seed=ORDER_SEED + 1,
                        eval_every=S.eval_every, plan=plan)
        tb, _ = theory.predict(sy, depth, lr, t2, st, settings=S,
                               eval_every=S.eval_every, plan=plan)
        arms[name] = dict(net=join(series(a, False), series(b, False), t1),
                          pred=join(series(ta, False), series(tb, False), t1))
    return dict(figure="f2", seed=seed, depth=depth, closed=closed,
                open="B" if closed == "A" else "A", t_switch=t1,
                epochs_max=t1 + t2, lr_target=S.lr_target,
                init_seed=S.init_seed, order_seed=ORDER_SEED,
                rho_before={l.name: float(s0.identifiability(l, S)["rho"])
                            for l in w0.laws},
                arms=arms)


# --------------------------------------------------------------------------- #
#  Figure 3: staged, two branches sharing one pre-switch trajectory            #
# --------------------------------------------------------------------------- #

def run_f3(seed, depth):
    S = settings_for(seed, worlds.F3_EVERY[depth])
    t1, t2 = worlds.F3_SWITCH[depth], worlds.F3_AFTER[depth]
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    pairs = worlds.held_cross(w0, reference=w1)
    ti, gt = w0.tok_index, w0.meta["gt_ent"]
    sc = float(np.linalg.norm(w0.meta["gt_rel"]["x"]))
    Emn = s0.Estar
    base = np.array([np.linalg.norm((Emn[ti[b]] - Emn[ti[a]])
                                    - (gt[b] - gt[a])) / sc
                     for a, b, _, _ in pairs])
    plan = cross_plan(w1, pairs, baseline=base,
                      freed=worlds.freed_coefficients(s0, pairs))
    lr = s0.lr(depth, settings=S)

    m = models.make_model(w0, depth, S)
    a = train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED,
                    eval_every=S.eval_every, plan=plan)
    ta, st = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                            settings=S, eval_every=S.eval_every, plan=plan)

    arms = {}
    for name, sy in (("hold", s0), ("insert", s1)):
        mm = copy.deepcopy(m)  # both branches leave the SAME pre-switch state
        b = train.train(mm, sy, lr, t2, S, order_seed=ORDER_SEED + 1,
                        eval_every=S.eval_every, plan=plan)
        tb, _ = theory.predict(sy, depth, lr, t2, st, settings=S,
                               eval_every=S.eval_every, plan=plan)
        arms[name] = dict(
            net=join(series(a, True), series(b, True), t1),
            pred=join(series(ta, True), series(tb, True), t1))
    return dict(figure="f3", seed=seed, depth=depth, t_switch=t1,
                epochs_max=t1 + t2, lr_target=S.lr_target,
                init_seed=S.init_seed, order_seed=ORDER_SEED,
                n_pairs=len(plan["a"]), arms=arms)


# --------------------------------------------------------------------------- #

def cells(seeds, only=None):
    for seed in seeds:
        for depth in DEPTHS:
            if only in (None, "f1"):
                yield ("f1", seed, depth, None)
            if only in (None, "f2"):
                for arm in ("A", "B"):
                    yield ("f2", seed, depth, arm)
            if only in (None, "f3"):
                yield ("f3", seed, depth, None)


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "plan"
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]
        args = args[: args.index("--only")]
    seeds = [int(a) for a in args[1:]] or [0]

    todo = list(cells(seeds, only))
    if mode == "plan":
        for fig, seed, depth, arm in todo:
            p = cell_path(fig, seed, depth, arm)
            print("  %-4s w%02d N=%d %-4s %s"
                  % (fig, seed, depth, arm or "", "done" if os.path.exists(p)
                     else "TO RUN"))
        n = sum(1 for c in todo if not os.path.exists(cell_path(*c)))
        print("%d of %d cells to run" % (n, len(todo)))
        return

    t_all = time.time()
    for fig, seed, depth, arm in todo:
        p = cell_path(fig, seed, depth, arm)
        if os.path.exists(p):
            print("  skip %-4s w%02d N=%d %s" % (fig, seed, depth, arm or ""),
                  flush=True)
            continue
        t0 = time.time()
        if fig == "f1":
            rec = run_f1(seed, depth)
        elif fig == "f2":
            rec = run_f2(seed, depth, arm)
        else:
            rec = run_f3(seed, depth)
        save(p, rec)
        print("  wrote %-4s w%02d N=%d %-2s (%.0fs)"
              % (fig, seed, depth, arm or "", time.time() - t0), flush=True)
    print("all cells done in %.1f min" % ((time.time() - t_all) / 60.0))


if __name__ == "__main__":
    main()
