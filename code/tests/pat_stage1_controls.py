"""Stage 1 diagnostics: can the candidate-pool and retention controls be scored
inside one replay, over the whole 200-world sample?

Experiments 2 and 3 stored scores, not embeddings, so both outstanding controls
need the training run back:

  E1  score each cross-structure query against all 18 entities as well as the
      nine destination entities
  E2  score the original premise facts, and every original fact, while the
      network reorganizes after the intervention

The pilot in `review/pat_controls.py` answered these for single cells by saving
every embedding: 10.9 MB for one Experiment 2 cell, about 13 GB over the full
sample.  This scores both controls DURING the replay and keeps only per-item
hits, which is what would make the full run affordable.  It also samples early
post-intervention epochs densely, since a transient dip between the original
evaluations is exactly what E2 is looking for.

Every replay is checked against its stored record: identical held-out hits and
geometric scores at the original evaluation epochs, or the cell fails.  A cell
therefore either reproduces the published run and adds measurements, or it is
not usable as evidence.

  usage:  python pat_stage1_controls.py --cells f3:0:1 f3:0:2 f3:0:3 f2:0:1:A
          python pat_stage1_controls.py --plot
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import _boot  # noqa: F401, E402
import _paths  # noqa: F401, E402
import numpy as np  # noqa: E402
from _paths import RESULTS  # noqa: E402
from relspec import System, checkpoint, models, train, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import (_rank_of, _spacing, apply_plan, cross_plan,  # noqa: E402
                             law_plan)

OUT = Path(RESULTS) / "_pat_stage1"

# Post-intervention sampling, in epochs since the intervention: every 10 up to
# 500, then every 50 up to 2,000, then geometrically by `TAIL` to the end of
# the window, on top of the original evaluation grid.
#
# The tail is not decoration.  A fixed dense window suits depths 2 and 3, whose
# reorganization is over within a few hundred epochs, and misses depth 1, whose
# world-0 dip sits at +2,500 -- just outside a 2,000-epoch window, leaving it
# sampled only on its original 2,500-epoch grid, which is how the pilot missed
# a dip at depth 3 in the first place.  Geometric spacing costs few points and
# is scale-free, so no depth is undersampled where its own dynamics are fast.
DENSE = ((500, 10), (2000, 50))
TAIL = 1.05


def block_of(name):
    """Entities are named by block (`A_1_2`, `BT3_1`).  The first character is
    the block, which is the candidate pool a within-block fact is scored in."""
    return name[0]


def fact_queries(world, facts):
    """Head-plus-relation queries for `facts`, grouped by candidate pool.

    Each fact is scored inside its own block, as the pilot did, so a premise
    fact competes against the entities it could plausibly be confused with
    rather than against the whole world.
    """
    pools = {}
    for e in world.entities:
        pools.setdefault(block_of(e), []).append(e)
    ti, out = world.tok_index, []
    for b, names in sorted(pools.items()):
        names = sorted(names)
        pos = {e: k for k, e in enumerate(names)}
        rows = [(a, r, t) for a, r, t in facts if block_of(a) == b and t in pos]
        if rows:
            out.append((np.array([ti[e] for e in names]),
                        np.array([ti[a] for a, _, _ in rows]),
                        np.array([ti[r] for _, r, _ in rows]),
                        np.array([pos[t] for _, _, t in rows])))
    return out


def fact_hits(queries, E):
    """Hits, raw residual distances, and residuals in candidate spacings.

    The hit is discrete and only moves when a nearest neighbour changes, which
    makes a retention curve a staircase with one step per item.  The residual
    in spacings is the continuous quantity underneath it, normalized exactly as
    the paper's geometric error is, so a dip can be read as a threshold
    crossing rather than as an event of its own.
    """
    hits, residual, normalized = [], [], []
    for cand_idx, a_idx, r_idx, target in queries:
        cand, q = E[cand_idx], E[a_idx] + E[r_idx]
        d = np.linalg.norm(q - cand[target], axis=1)
        hits.append(_rank_of(q, cand, target) == 0)
        residual.append(d)
        normalized.append(d / (_spacing(cand) + 1e-12))
    return (np.concatenate(hits), np.concatenate(residual),
            np.concatenate(normalized))


def build(sub, seed, depth, closed):
    """Everything one cell needs, reconstructed from its stored record."""
    tag = "w%02d_d%d" % (seed, depth) + ("_%s" % closed if sub == "f2" else "")
    path = Path(RESULTS) / sub / (tag + ".json")
    record = json.loads(path.read_text())
    net = record["arms"]["insert"]["net"]
    every = int(net["epochs"][1] - net["epochs"][0])
    S = override(init_seed=record["init_seed"], eval_every=every,
                 lr_target=record["lr_target"])
    if sub == "f2":
        w0 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed)
        w1 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed,
                                          n_bridge=1)
        plan = law_plan(w1, worlds.held_composites(w0, reference=w1))
        plan18 = None
    else:
        w0 = worlds.integration_world(seed, link=False)
        w1 = worlds.integration_world(seed, link=True)
        pairs = worlds.held_cross(w0, reference=w1)
        plan = cross_plan(w1, pairs)
        plan18 = cross_plan(w1, pairs, candidates=w1.entities)
    facts = list(dict.fromkeys(w0.facts))
    premise = [f for f in facts if f[1].startswith(("x", "y"))]
    s0, s1 = (System.build(w, settings=S) for w in (w0, w1))
    return dict(sub=sub, kind=("f2" if sub == "f2" else "f3"), seed=seed, depth=depth, closed=closed, tag=tag,
                record=record, S=S, every=every, w0=w0, plan=plan, plan18=plan18,
                s0=s0, s1=s1, lr=s0.lr(depth, settings=S),
                t1=record["t_switch"], t2=record["epochs_max"] - record["t_switch"],
                premise=fact_queries(w0, premise), facts=fact_queries(w0, facts),
                n_premise=len(premise), n_facts=len(facts))


def grid_for(ctx):
    """Original evaluation epochs since the intervention, plus dense early ones
    and a geometric tail to the end of the window."""
    ep = np.asarray(ctx["record"]["arms"]["insert"]["net"]["epochs"], float)
    g = {int(e - ctx["t1"]) for e in ep if e >= ctx["t1"]}
    start = 0
    for limit, step in DENSE:
        g |= set(range(start, min(limit, ctx["t2"]) + 1, step))
        start = limit
    t = float(start)
    while t < ctx["t2"]:
        t *= TAIL
        g.add(int(round(t)))
    return sorted(e for e in g if 0 <= e <= ctx["t2"])


def measure(ctx, E):
    """Every control score at one state, kept as per-item hits.

    The held-out geometric error is also split into the two quantities it is
    made of, the mean query-to-target distance and the median candidate
    spacing it is divided by, because a normalized error can rise when the
    representation contracts rather than when retrieval degrades.
    """
    out = {}
    plan = ctx["plan"]
    _, geo, hits, _, _ = apply_plan(plan, E)
    if ctx["kind"] == "f3":
        out["cross9"] = hits["cross"]
        out["geo9"] = geo["cross"]
        _, _, h18, _, _ = apply_plan(ctx["plan18"], E)
        out["cross18"] = h18["cross"]
        cand = E[plan["cand"]]
        pts = (E[plan["a"]] + plan["nx"][:, None] * E[plan["x"]][None, :]
               + plan["ny"][:, None] * E[plan["y"]][None, :])
        out["geo9_distance"] = float(np.linalg.norm(pts - cand[plan["bpos"]], axis=1).mean())
        out["geo9_spacing"] = float(_spacing(cand))
    else:
        for law in plan["names"]:
            p = plan[law]
            out["law_" + law] = hits[law]
            out["geo_" + law] = geo[law]
            cand = E[p["cand"]]
            q = E[p["a"]] + E[p["z"]][None, :]
            out["distance_" + law] = float(np.linalg.norm(q - cand[p["b"]], axis=1).mean())
            out["spacing_" + law] = float(_spacing(cand))
    for key in ("premise", "facts"):
        h, r, rn = fact_hits(ctx[key], E)
        out[key] = h
        out[key + "_norm"] = rn.astype(np.float32)
        out[key + "_residual"] = float(r.mean())
    return out


def verify(ctx, arm, epochs, scored):
    """The replay must reproduce the stored record exactly where they overlap."""
    old = ctx["record"]["arms"][arm]["net"]
    ep = np.asarray(old["epochs"], float)
    keep = np.flatnonzero(ep >= ctx["t1"])
    index = {t: i for i, t in enumerate(epochs)}
    names = ["cross"] if ctx["kind"] == "f3" else list(ctx["plan"]["names"])
    checked, worst = 0, 0.0
    for k in keep:
        i = index[int(ep[k] - ctx["t1"])]
        for name in names:
            hkey = "cross9" if ctx["kind"] == "f3" else "law_" + name
            gkey = "geo9" if ctx["kind"] == "f3" else "geo_" + name
            mismatches = int(np.sum(scored[hkey][i] != np.asarray(old["hits"][name][k], bool)))
            delta = abs(float(scored[gkey][i]) - float(old["geometric"][name][k]))
            if mismatches or delta > 1e-9:
                raise AssertionError(
                    "%s %s at +%d: %d hit mismatches, geometry off by %.3g"
                    % (ctx["tag"], arm, ep[k] - ctx["t1"], mismatches, delta))
            worst = max(worst, delta)
            checked += 1
    return checked, worst


def replay(ctx):
    """Pre-train, branch, and score both arms on the combined grid."""
    order = ctx["record"]["order_seed"]
    model = models.make_model(ctx["w0"], ctx["depth"], ctx["S"])
    t0 = time.time()
    pre = train.train(model, ctx["s0"], ctx["lr"], ctx["t1"], ctx["S"],
                      order_seed=order, eval_every=ctx["t1"])
    pre_seconds = time.time() - t0

    grid = grid_for(ctx)
    # The weights at the intervention, in full precision: a few kB that let a
    # later pass re-run either arm without repeating pre-training, which is 40
    # to 70 per cent of a cell.  Layer weights, not the product embedding,
    # since a deep model cannot be continued from its product alone.
    result = dict(epochs=np.asarray(grid), arms={},
                  pre=[w.copy() for w in checkpoint.weights(model)],
                  pre_rng=pre.rng_state,
                  timing=dict(pre_seconds=pre_seconds))
    for arm, system in (("hold", ctx["s0"]), ("insert", ctx["s1"])):
        mm = copy.deepcopy(model)
        scored, at, rng = {}, 0, None
        t0 = time.time()
        for t in grid:
            if t > at:
                traj = train.train(mm, system, ctx["lr"], t, ctx["S"],
                                   order_seed=order + 1, eval_every=max(1, t - at),
                                   start=at, rng_state=rng)
                rng, at = traj.rng_state, t
            for key, value in measure(ctx, mm.embedding()).items():
                scored.setdefault(key, []).append(value)
        scored = {k: np.asarray(v) for k, v in scored.items()}
        checked, worst = verify(ctx, arm, grid, scored)
        result["arms"][arm] = scored
        result["timing"][arm + "_seconds"] = time.time() - t0
        result["timing"][arm + "_checked"] = checked
        result["timing"][arm + "_worst_geometry_delta"] = worst
    return result


def save(ctx, result):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s" % (ctx["sub"], ctx["tag"])
    arrays = {"epochs": result["epochs"]}
    for i, w in enumerate(result.get("pre", [])):
        arrays["pre.%d" % i] = w
    for arm, scored in result["arms"].items():
        for key, value in scored.items():
            arrays["%s.%s" % (arm, key)] = value
    np.savez_compressed(OUT / (stem + ".npz"), **arrays)
    original = int(sum(1 for e in np.asarray(
        ctx["record"]["arms"]["insert"]["net"]["epochs"], float) if e >= ctx["t1"]))
    summary = dict(cell=stem, seed=ctx["seed"], depth=ctx["depth"],
                   closed=ctx["closed"], t_switch=ctx["t1"], window=ctx["t2"],
                   n_evaluations=len(result["epochs"]), original_evaluations=original,
                   n_premise=ctx["n_premise"], n_facts=ctx["n_facts"],
                   grid=signature(), timing=result["timing"],
                   pre_state=[list(w.shape) for w in result.get("pre", [])],
                   pre_rng=result.get("pre_rng"),
                   bytes_npz=(OUT / (stem + ".npz")).stat().st_size, arms={})
    for arm, scored in result["arms"].items():
        entry = {}
        for key in ("cross9", "cross18", "premise", "facts"):
            if key in scored:
                acc = 100.0 * scored[key].mean(axis=1)
                entry[key] = dict(start=float(acc[0]), min=float(acc.min()),
                                  end=float(acc[-1]),
                                  epoch_of_min=int(result["epochs"][int(acc.argmin())]))
        summary["arms"][arm] = entry
    (OUT / (stem + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def signature():
    """What the saved scores depend on besides the record itself, so a cell
    made under a different sampling grid is re-run rather than reused."""
    return dict(dense=[list(p) for p in DENSE], tail=TAIL)


def done(spec):
    """Is this cell already on disk under the grid in force here?"""
    sub, seed, depth, closed = spec
    tag = "w%02d_d%d" % (seed, depth) + ("_%s" % closed if sub == "f2" else "")
    path = OUT / ("%s_%s.json" % (sub, tag))
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("grid") == signature()
    except (ValueError, OSError):
        return False


def run(spec):
    sub, seed, depth, closed = spec
    ctx = build(sub, seed, depth, closed)
    print("%s/%s: pre-training %s epochs, then 2 x %s with %d evaluations"
          % (sub, ctx["tag"], "{:,}".format(ctx["t1"]), "{:,}".format(ctx["t2"]),
             len(grid_for(ctx))), flush=True)
    summary = save(ctx, replay(ctx))
    t = summary["timing"]
    print("  replayed in %.0f s (pre %.0f s, arms %.0f + %.0f s); %d checks passed; "
          "%.1f kB stored"
          % (t["pre_seconds"] + t["hold_seconds"] + t["insert_seconds"],
             t["pre_seconds"], t["hold_seconds"], t["insert_seconds"],
             t["hold_checked"] + t["insert_checked"], summary["bytes_npz"] / 1e3),
          flush=True)
    for arm, entry in summary["arms"].items():
        print("   %-6s %s" % (arm, "; ".join(
            "%s %.1f%% -> %.1f%% (min %.1f%% at +%s)"
            % (k, v["start"], v["end"], v["min"], "{:,}".format(v["epoch_of_min"]))
            for k, v in entry.items())), flush=True)
    return summary


def parse(cell):
    """`f3:0:2`, `f2:0:1:A`, or a seed range: `f3:0-199:3`."""
    parts = cell.split(":")
    sub = "f3_lr0p003" if parts[0] == "f3" else "f2"
    closed = parts[3] if len(parts) > 3 else ("A" if sub == "f2" else None)
    lo, _, hi = parts[1].partition("-")
    seeds = range(int(lo), int(hi) + 1) if hi else [int(lo)]
    return [(sub, s, int(parts[2]), closed) for s in seeds]


def load(stem):
    with np.load(OUT / (stem + ".npz")) as z:
        return {k: z[k] for k in z.files}


def accuracy(arrays, arm, key):
    v = arrays.get("%s.%s" % (arm, key))
    return None if v is None else 100.0 * v.mean(axis=1)


def plot(seed=0):
    """One figure per experiment: the paper's own measures, then the controls.

    The paper's held-out accuracy and geometric error are drawn first, from the
    replay itself, so a reader can see that the replayed run is the published
    run before reading anything new from it.  The added control sits underneath
    on the same axis.  The no-link arm is drawn thick and underneath, so that
    where the two conditions coincide -- which is the whole of Experiment 2's
    retention -- the linked arm is visible on top of it rather than hidden.

    The epoch axis is logarithmic because everything of interest happens early.
    The evaluation at the intervention itself has no place on a log axis and is
    reported in the saved summaries instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from style import DARK, panel

    arms = (("hold", "#c0392b", "No linking fact", 2.6),
            ("insert", "#1b6ca8", "One linking fact", 1.4))
    dashed = (0, (2.2, 2.0))
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    for kind, title in (("f3", "Experiment 3"), ("f2", "Experiment 2")):
        fig, axes = plt.subplots(3, 3, figsize=(9.8, 8.4))
        fig.subplots_adjust(wspace=0.16, hspace=0.34, bottom=0.12)
        for col, depth in enumerate((1, 2, 3)):
            stem = ("f3_lr0p003_w%02d_d%d" % (seed, depth) if kind == "f3"
                    else "f2_w%02d_d%d_A" % (seed, depth))
            a = load(stem)
            meta = json.loads((OUT / (stem + ".json")).read_text())
            ep = a["epochs"]
            m = ep > 0
            if kind == "f3":
                acc_key, geo_key, extra = "cross9", "geo9", "cross18"
            else:
                law = "B" if meta["closed"] == "A" else "A"
                acc_key, geo_key, extra = "law_" + law, "geo_" + law, None

            ax = axes[0][col]
            for arm, colour, _, lw in arms:
                ax.plot(ep[m], accuracy(a, arm, acc_key)[m], color=colour, lw=lw)
                if extra:
                    ax.plot(ep[m], accuracy(a, arm, extra)[m], color=colour,
                            lw=max(1.0, lw - 0.6), ls=dashed)
            ax.set_ylim(-4, 104)
            if col == 0:
                ax.set_ylabel("Held-out accuracy (%%)\n%s" % (
                    "solid: 9 candidates, dashed: all 18" if kind == "f3"
                    else "the underdetermined law"), linespacing=1.6, fontsize=8.5)

            ax = axes[1][col]
            for arm, colour, _, lw in arms:
                ax.plot(ep[m], a["%s.%s" % (arm, geo_key)][m], color=colour, lw=lw)
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel("Geometric error\n(held-out queries)", linespacing=1.6,
                              fontsize=8.5)

            ax = axes[2][col]
            for arm, colour, _, lw in arms:
                ax.plot(ep[m], accuracy(a, arm, "premise")[m], color=colour, lw=lw)
                ax.plot(ep[m], accuracy(a, arm, "facts")[m], color=colour,
                        lw=max(1.0, lw - 0.6), ls=dashed)
            ax.set_ylim(88, 101.5)
            if col == 0:
                ax.set_ylabel("Trained-fact retrieval (%%)\nsolid: %d premises, "
                              "dashed: all %d" % (meta["n_premise"], meta["n_facts"]),
                              linespacing=1.6, fontsize=8.5)

            for row in range(3):
                ax = axes[row][col]
                ax.set_xscale("log")
                if col:
                    ax.set_yticklabels([])
                if row == 2:
                    ax.set_xlabel("Epochs since the linking fact")
                if row == 0:
                    ax.text(0.5, 1.05, "%s, $N = %d$" % (title, depth),
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=9, color=DARK)
                panel(ax, "abcdefghi"[row * 3 + col],
                      dx=-0.26 if col == 0 else -0.08, dy=1.1)
        # one error axis for every depth, so the columns can be compared
        lo = min(min(line.get_ydata().min() for line in ax.lines) for ax in axes[1])
        hi = max(max(line.get_ydata().max() for line in ax.lines) for ax in axes[1])
        for ax in axes[1]:
            ax.set_ylim(lo * 0.7, hi * 1.4)
        handles = [Line2D([], [], color=c, lw=w, label=l) for _, c, l, w in arms]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.075),
                   ncol=2, handletextpad=0.6, columnspacing=2.0, handlelength=1.8)
        path = OUT / ("stage1_%s_checks.png" % kind)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        written.append(path)
    for p in written:
        print("wrote %s" % p, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["f3:0:1"],
                    help="experiment:seed:depth[:closed], e.g. f3:0:2 or f2:0:1:A")
    ap.add_argument("--plot", action="store_true", help="draw the saved cells only")
    ap.add_argument("--seed", type=int, default=0, help="which saved world to draw")
    ap.add_argument("--nproc", type=int, default=1, help="cells to replay at once")
    ap.add_argument("--force", action="store_true", help="replay cells already saved")
    ns = ap.parse_args()
    if ns.plot:
        plot(ns.seed)
        return
    todo = [c for spec in ns.cells for c in parse(spec)]
    skipped = 0
    if not ns.force:
        keep = [c for c in todo if not done(c)]
        skipped, todo = len(todo) - len(keep), keep
    print("%d cells to replay%s" % (len(todo), "" if not skipped
                                    else ", %d already saved" % skipped), flush=True)
    if ns.nproc > 1 and len(todo) > 1:
        t0 = time.time()
        with Pool(min(ns.nproc, len(todo))) as pool:
            for k, _ in enumerate(pool.imap_unordered(run, todo), 1):
                print("   %d of %d done, %.1f min elapsed"
                      % (k, len(todo), (time.time() - t0) / 60.0), flush=True)
        return
    for cell in todo:
        run(cell)


if __name__ == "__main__":
    main()
