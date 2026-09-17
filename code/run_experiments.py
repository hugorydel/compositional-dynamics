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

Every cell also saves a checkpoint, the weights and generator state it ended
with, under `results/checkpoints/` (`relspec.checkpoint`).  A cell whose budget
is raised continues from its checkpoint instead of starting again: Figure 1
from the end of its run, Figures 2 and 3 from the end of their branches, or
from the switch when the switch moves later.  Figure 1 cells also save every
10,000 epochs, so an interrupted long run loses at most that much.  A
continued cell is identical to one run straight through
(tests/check_checkpoint.py).

  usage:  python run_experiments.py plan            what would run
          python run_experiments.py run 0           world 0, all figures
          python run_experiments.py run 0 1 2       worlds 0, 1, 2
          python run_experiments.py run 0 --only f1
"""

import os

# Cells are parallelised across PROCESSES, and each one multiplies many small
# matrices.  Letting BLAS also spread a single matmul over threads oversubscribes
# the machine badly, so pin it to one thread before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import _paths  # noqa: F401, E402
import numpy as np  # noqa: E402
from _paths import RESULTS
from relspec import System, checkpoint, models, theory, train, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import cross_plan, law_plan  # noqa: E402

DEPTHS = (1, 2, 3)
ORDER_SEED = 7
NPROC = int(os.environ.get("RELSPEC_NPROC", "12"))


def f1_order(seed):
    """Figure 1 draws a fresh presentation order per world.

    Holding one order fixed across worlds makes the SGD order noise perfectly
    correlated between them, so a band drawn across worlds understates its own
    spread.  Figures 2 and 3 keep the fixed order for now: they are staged, and
    changing it would invalidate their stored cells for no gain.
    """
    return 2000 + seed


def settings_for(seed, every):
    return override(init_seed=1000 + seed, eval_every=every)


def cell_path(fig, seed, depth, arm=None):
    d = os.path.join(RESULTS, fig)
    os.makedirs(d, exist_ok=True)
    name = "w%02d_d%d" % (seed, depth) + ("_%s" % arm if arm else "") + ".json"
    return os.path.join(d, name)


REQUIRED = ("epochs", "retrieval", "rank", "geometric", "hits")


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def stale(r):
    """Was this cell produced under settings that no longer apply?

    Checking the record FORMAT is not enough.  A budget change, a world change
    or a new presentation order all leave the format intact while making the
    stored curve incomparable with a fresh one, and resume would keep it.  The
    run then silently mixes settings across cells.  Everything compared below
    is stored in the record itself, so this needs no bookkeeping elsewhere.
    """
    fig, depth = r.get("figure"), r.get("depth")
    if fig == "f1":
        return (r.get("epochs_max") != worlds.F1_EPOCHS[depth]
                or r.get("order_seed") != f1_order(r.get("seed")))
    if fig == "f2":
        return (r.get("t_switch") != worlds.F2_SWITCH[depth]
                or r.get("epochs_max") != worlds.F2_SWITCH[depth]
                + worlds.F2_AFTER[depth])
    if fig == "f3":
        return (r.get("t_switch") != worlds.F3_SWITCH[depth]
                or r.get("epochs_max") != worlds.F3_SWITCH[depth]
                + worlds.F3_AFTER[depth])
    return True


def usable(path):
    """A cell counts as done only if it carries everything the analysis needs
    AND was produced under the settings in force now.

    Resuming on the presence of a file alone silently accepts cells written
    under an older record format, which is how a run once produced Figure 1
    depths 1 and 2 without the per-item hit histories the stable-resolution
    curve is built from.  A stale cell is treated as missing and recomputed.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            r = json.load(f)
    except (ValueError, OSError):
        return False
    d = r["arms"][sorted(r["arms"])[0]]["net"] if "arms" in r else r.get("net", {})
    return all(k in d for k in REQUIRED) and not stale(r)


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

F1_CHUNK = 10000   # epochs between saves of a Figure 1 cell


def run_f1(seed, depth, epochs=None, chunk=F1_CHUNK, rec_path=None, ck=None):
    """One Figure 1 cell, saved with its checkpoint every `chunk` epochs.

    A cell whose record and checkpoint stop short of the budget, under the
    same settings, continues from where they stop: raising the budget costs
    only the new epochs, and an interrupted cell loses at most one chunk.
    `epochs`, `rec_path` and `ck` default to the figure's budget and paths.
    """
    S = settings_for(seed, worlds.F1_EVERY[depth])
    target = worlds.F1_EPOCHS[depth] if epochs is None else epochs
    rec_path = rec_path or cell_path("f1", seed, depth)
    ck = ck or ckpt_path("f1", seed, depth)
    w = worlds.emergence_world(seed)
    s = System.build(w, settings=S)
    plan = law_plan(w, worlds.held_composites(w))
    lr = s.lr(depth, settings=S)
    order, every = f1_order(seed), S.eval_every
    meta = dict(figure="f1", seed=seed, depth=depth, arm=None, lr_target=S.lr_target,
                init_seed=S.init_seed, order_seed=order, eval_every=every,
                substeps=S.substeps(depth))

    net = models.make_model(w, depth, S)
    # depth 1: the closed form's starting state; deeper: the integrator's state
    pred = [net.embedding().copy()] if depth == 1 else [x.copy() for x in net.W]
    start, rng, net_ser, pred_ser = 0, None, None, None
    rec, ph = resumable(rec_path, ck, meta)
    if (rec is not None and ph["run"]["epochs"] == rec["epochs_max"] <= target
            and rec["epochs_max"] % every == 0):
        start, rng = rec["epochs_max"], ph["run"]["rng"]
        checkpoint.restore(net, ph["run"]["net"])
        pred, net_ser, pred_ser = ph["run"]["pred"], rec["net"], rec["pred"]

    while start < target:
        end = min(target, start + chunk)
        obs = train.train(net, s, lr, end, S, order_seed=order, eval_every=every,
                          plan=plan, start=start, rng_state=rng)
        tp, st = theory.predict(s, depth, lr, end, pred[0] if depth == 1 else pred,
                                settings=S, eval_every=every, plan=plan, start=start)
        net_ser = series(obs) if net_ser is None else join(net_ser, series(obs), 0)
        pred_ser = series(tp) if pred_ser is None else join(pred_ser, series(tp), 0)
        rng, start = obs.rng_state, end
        if depth > 1:
            pred = st
        rec = dict(figure="f1", seed=seed, depth=depth, epochs_max=end,
                   lr_target=S.lr_target, init_seed=S.init_seed,
                   order_seed=order, net=net_ser, pred=pred_ser)
        checkpoint.save(ck, {"run": dict(epochs=end, net=checkpoint.weights(net),
                                         rng=rng, pred=pred)}, meta)
        save(rec_path, rec)
    return rec


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
    extra = dict(
        closed=closed, open="B" if closed == "A" else "A",
        # accuracy of the best constant answer, per law.  The figure
        # rescales by it so an undetermined law cannot be credited for
        # the items whose target happens to be the entity it returns to
        # everything.  Stored rather than recomputed, because it is a
        # property of the evaluation set this cell actually used.
        chance={n: float(plan[n]["chance"]) for n in plan["names"]},
        rho_before={l.name: float(s0.identifiability(l, S)["rho"]) for l in w0.laws})
    return staged("f2", seed, depth, S, w0, s0, s1, plan, lr, S.lr_target, t1, t2,
                  False, extra, cell_path("f2", seed, depth, closed),
                  ckpt_path("f2", seed, depth, closed), arm=closed)


# --------------------------------------------------------------------------- #
#  Figure 3: staged, two branches sharing one pre-switch trajectory            #
# --------------------------------------------------------------------------- #

def run_f3(seed, depth):
    S = settings_for(seed, worlds.F3_EVERY[depth])
    t1, t2 = worlds.F3_SWITCH[depth], worlds.F3_AFTER[depth]
    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    plan = cross_plan(w1, worlds.held_cross(w0, reference=w1))
    lr = s0.lr(depth, settings=S)
    return staged("f3", seed, depth, S, w0, s0, s1, plan, lr, S.lr_target, t1, t2,
                  True, dict(n_pairs=len(plan["a"])), cell_path("f3", seed, depth),
                  ckpt_path("f3", seed, depth))


# --------------------------------------------------------------------------- #
#  Checkpoints, and the staged run Figures 2 and 3 share                       #
# --------------------------------------------------------------------------- #

CKPT = os.path.join(RESULTS, "checkpoints")


def ckpt_path(sub, seed, depth, arm=None):
    """A cell's checkpoint: `results/checkpoints/<sub>/`, named as its record."""
    name = "w%02d_d%d" % (seed, depth) + ("_%s" % arm if arm else "") + ".npz"
    return os.path.join(CKPT, sub, name)


def resumable(rec_path, ck, meta):
    """The stored record and checkpoint of a cell, when both exist and were
    made under exactly `meta`; otherwise `(None, None)`, and the cell starts
    again from epoch 0."""
    if not (os.path.exists(rec_path) and os.path.exists(ck)):
        return None, None
    phases, stored = checkpoint.load(ck)
    if phases is None or stored != meta:
        return None, None
    try:
        with open(rec_path) as f:
            return json.load(f), phases
    except (ValueError, OSError):
        return None, None


def head(ser, t):
    """A stored series up to and including epoch `t`: the pre-switch part of a
    staged arm."""
    k = int(np.searchsorted(np.asarray(ser["epochs"], float), t, side="right"))
    out = dict(epochs=ser["epochs"][:k])
    for key in ("retrieval", "rank", "geometric", "hits"):
        if key in ser:
            out[key] = {n: v[:k] for n, v in ser[key].items()}
    if "errs" in ser:
        out["errs"] = {n: (None if v is None else v[:k]) for n, v in ser["errs"].items()}
    return out


def staged(fig, seed, depth, S, w0, s0, s1, plan, lr, lr_target, t1, t2, items,
           extra, rec_path, ck, arm=None):
    """One staged cell: a pre-switch run on `s0` to epoch `t1`, then two
    branches from its state for `t2` epochs, `hold` on `s0` and `insert` on
    `s1`.  Saves the record and its checkpoint.

    Continues from the cell's record and checkpoint when they were made under
    the same settings.  A later switch continues the pre-switch run and runs
    both branches afresh from the new switch, since they must leave the new
    state; the same switch with a longer window continues each branch.
    Anything else starts again.  `items` keeps per-item errors in the record;
    `extra` holds the figure's own record fields.
    """
    every = S.eval_every
    assert t1 % every == 0 and t2 % every == 0, "budgets must sit on the evaluation grid"
    meta = dict(figure=fig, seed=seed, depth=depth, arm=arm, lr_target=lr_target,
                init_seed=S.init_seed, order_seed=ORDER_SEED, eval_every=every,
                substeps=S.substeps(depth))

    m = models.make_model(w0, depth, S)
    pre_pred = [m.embedding().copy()] if depth == 1 else [x.copy() for x in m.W]
    start, rng, net0, pred0 = 0, None, None, None
    rec, ph = resumable(rec_path, ck, meta)
    if rec is not None and not ph["pre"]["epochs"] == rec["t_switch"] <= t1:
        rec = None
    if rec is not None:
        start, rng, pre_pred = rec["t_switch"], ph["pre"]["rng"], ph["pre"]["pred"]
        checkpoint.restore(m, ph["pre"]["net"])
        net0 = head(rec["arms"]["hold"]["net"], start)
        pred0 = head(rec["arms"]["hold"]["pred"], start)

    if start < t1:
        a = train.train(m, s0, lr, t1, S, order_seed=ORDER_SEED, eval_every=every,
                        plan=plan, start=start, rng_state=rng)
        ta, st = theory.predict(s0, depth, lr, t1, pre_pred[0] if depth == 1 else pre_pred,
                                settings=S, eval_every=every, plan=plan, start=start)
        net0 = series(a, items) if net0 is None else join(net0, series(a, items), 0)
        pred0 = series(ta, items) if pred0 is None else join(pred0, series(ta, items), 0)
        rng = a.rng_state
        if depth > 1:
            pre_pred = st
        rec = None                  # the branches must leave the new state
    else:
        st = theory.traj_end_embedding(s0, lr, t1, pre_pred[0]) if depth == 1 else pre_pred

    phases = {"pre": dict(epochs=t1, net=checkpoint.weights(m), rng=rng, pred=pre_pred)}
    arms = {}
    for name, sy in (("hold", s0), ("insert", s1)):
        old = None if rec is None else rec["epochs_max"] - rec["t_switch"]
        if old is not None and old <= t2 and ph[name]["epochs"] == old:
            mm = checkpoint.restore(models.make_model(w0, depth, S), ph[name]["net"])
            b0, rng_b, arm_pred = old, ph[name]["rng"], ph[name]["pred"]
            net_b, pred_b = rec["arms"][name]["net"], rec["arms"][name]["pred"]
        else:
            mm = copy.deepcopy(m)   # both branches leave the SAME pre-switch state
            b0, rng_b = 0, None
            arm_pred = [st] if depth == 1 else [x.copy() for x in st]
            net_b, pred_b = net0, pred0
        if b0 < t2:
            b = train.train(mm, sy, lr, t2, S, order_seed=ORDER_SEED + 1,
                            eval_every=every, plan=plan, start=b0, rng_state=rng_b)
            tb, wend = theory.predict(sy, depth, lr, t2,
                                      arm_pred[0] if depth == 1 else arm_pred,
                                      settings=S, eval_every=every, plan=plan, start=b0)
            net_b = join(net_b, series(b, items), t1)
            pred_b = join(pred_b, series(tb, items), t1)
            rng_b = b.rng_state
            if depth > 1:
                arm_pred = wend
        arms[name] = dict(net=net_b, pred=pred_b)
        phases[name] = dict(epochs=t2, net=checkpoint.weights(mm), rng=rng_b, pred=arm_pred)

    out = dict(figure=fig, seed=seed, depth=depth, t_switch=t1, epochs_max=t1 + t2,
               lr_target=lr_target, init_seed=S.init_seed, order_seed=ORDER_SEED,
               **extra, arms=arms)
    checkpoint.save(ck, phases, meta)
    save(rec_path, out)
    return out


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


def one(cell):
    """Run one cell and write it, or skip it if it is already usable.

    Module level and taking a single picklable argument, because Windows starts
    pool workers with spawn: the worker re-imports this module and calls this
    function by name, so a closure or a bound method would not survive.
    """
    fig, seed, depth, arm = cell
    p = cell_path(fig, seed, depth, arm)
    tag = "%-4s w%02d N=%d %-2s" % (fig, seed, depth, arm or "")
    if usable(p):
        return ("skip", tag, 0.0)
    t0 = time.time()
    if fig == "f1":
        rec = run_f1(seed, depth)
    elif fig == "f2":
        rec = run_f2(seed, depth, arm)
    else:
        rec = run_f3(seed, depth)
    save(p, rec)
    return ("wrote", tag, time.time() - t0)


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "plan"
    only, nproc = None, NPROC
    for flag in ("--only", "--nproc"):
        if flag in args:
            v = args[args.index(flag) + 1]
            if flag == "--only":
                only = v
            else:
                nproc = int(v)
            args = args[: args.index(flag)] + args[args.index(flag) + 2:]
    seeds = [int(a) for a in args[1:]] or [0]

    todo = list(cells(seeds, only))
    if mode == "plan":
        for fig, seed, depth, arm in todo:
            p = cell_path(fig, seed, depth, arm)
            print("  %-4s w%02d N=%d %-4s %s"
                  % (fig, seed, depth, arm or "",
                     "done" if usable(p) else "TO RUN"))
        n = sum(1 for c in todo if not usable(cell_path(*c)))
        print("%d of %d cells to run" % (n, len(todo)))
        return

    todo = [c for c in todo if not usable(cell_path(*c))]
    t_all = time.time()
    nproc = max(1, min(nproc, len(todo)))
    print("%d cells on %d process%s"
          % (len(todo), nproc, "" if nproc == 1 else "es"), flush=True)

    if nproc == 1:
        results = [one(c) for c in todo]
        for what, tag, dt in results:
            print("  %s %s (%.0fs)" % (what, tag, dt), flush=True)
    else:
        # imap_unordered so a long cell does not hold back the reporting of
        # short ones; ordering of the output carries no meaning anyway
        with Pool(nproc) as pool:
            for what, tag, dt in pool.imap_unordered(one, todo):
                print("  %s %s (%.0fs)" % (what, tag, dt), flush=True)

    wall = time.time() - t_all
    print("all cells done in %.1f min" % (wall / 60.0))


if __name__ == "__main__":
    main()
