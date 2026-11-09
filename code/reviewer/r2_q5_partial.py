"""Reviewer 2, W4 / Q5: a link that resolves only part of the ambiguity.

  "In a system containing multiple null directions, how does the framework
   model and predict trajectories when an intervention resolves only a proper
   subset of ambiguities?"  (Q5)

World: three copies (A, B, C) of the Experiment 3 lattice sharing x, y, z.
Only A is anchored, so B and C can each translate freely: two null directions
per embedding column, v_B and v_C (ones on that lattice's entities). In ground
truth B sits one x step beyond A's last first-axis entity (as in Experiment 3)
and C one y step beyond A's last second-axis entity. The link is the same
single fact as in Experiment 3, A -> B by x, so it removes v_B and leaves v_C.

The framework's account (App. A.6): after the link, ker A' = ker A ∩ ker g^T.
A query with contrast c = e_a + n_x e_x + n_y e_y - e_b is determined iff
P0' c = 0. Determined queries converge as the constrained modes decay; for the
rest the component along the remaining null direction stays where learning left
it (exactly, at N = 1). This script checks both parts:

  theory   null dimension before/after; |P0' c| / |c| for every A->B, A->C and
           B->C query (A->B determined, the other two not)
  dynamics network and prediction, linked and no-link arms, for each query set

Training follows Experiment 3 at the repository's current settings
(`relspec.worlds` F3_SWITCH / F3_AFTER / F3_EVERY, `lr_target` from
`relspec.config`), from the same kind of initialization (seed 1000 + s).
Per-cell arrays go to `results/reviewer_outputs/r2_q5_partial/cells/`.

  usage:  python r2_q5_partial.py                     worlds 0-49, depths 1-3
          python r2_q5_partial.py --worlds 0-199 --nproc 12
          python r2_q5_partial.py --summarize
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from common import cell_dir, first_at_least, write_outputs  # noqa: E402
from _paths import CODE  # noqa: E402

sys.path.insert(0, os.path.join(CODE, "analysis"))
from controls import fact_hits, fact_queries  # noqa: E402
from relspec import System, models, theory, train, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import apply_plan, cross_plan  # noqa: E402
from relspec.worlds import Law, World  # noqa: E402

NAME = "r2_q5_partial"
QUESTION = ("Q5, 'In a system containing multiple null directions, how does the framework "
            "model and predict trajectories when an intervention resolves only a proper subset "
            "of ambiguities?'")
M = 3
SETS = ("AB", "AC", "BC")
ORDER_SEED = 7
CRIT_FRAC = 0.9


def name(pre, i, j):
    return "%s_%d_%d" % (pre, i, j)


# lattice coordinates in units of (x, y): where each lattice's (0, 0) sits
ORIGIN = {"A": (0, 0), "B": (M, 0), "C": (0, M)}


def three_lattice_world(seed=0, link=False, d=16):
    """Three copies of the Experiment 3 lattice; A anchored; optional A->B link."""
    rng = np.random.default_rng(seed)
    gx, gy = rng.standard_normal(d), rng.standard_normal(d)
    gt_r = {"x": gx, "y": gy, "z": gx + gy}
    base = rng.standard_normal(d)
    ents, facts, anchors, gt_e = [], [], {}, {}
    for pre in ("A", "B", "C"):
        ox, oy = ORIGIN[pre]
        for i in range(M):
            for j in range(M):
                ents.append(name(pre, i, j))
                gt_e[name(pre, i, j)] = base + (ox + i) * gx + (oy + j) * gy
        for i in range(M):
            for j in range(M):
                if i + 1 < M:
                    facts.append((name(pre, i, j), "x", name(pre, i + 1, j)))
                if j + 1 < M:
                    facts.append((name(pre, i, j), "y", name(pre, i, j + 1)))
                if i + 1 < M and j + 1 < M:
                    facts.append((name(pre, i, j), "z", name(pre, i + 1, j + 1)))
    for c in (name("A", 0, 0), name("A", 1, 0), name("A", 0, 1)):
        anchors[c] = gt_e[c]
    if link:
        facts.append((name("A", M - 1, 0), "x", name("B", 0, 0)))
    return World(entities=ents, relations=list(gt_r), facts=facts, anchors=anchors,
                 laws=[Law("Lz", "x", "y", "z")], d=d,
                 meta=dict(gt_ent=gt_e, gt_rel=gt_r, m=M, link=link, blocks=("A", "B", "C")))


def pairs_for(src, dst, shown):
    out = []
    for i in range(M):
        for j in range(M):
            for k in range(M):
                for l in range(M):
                    a, b = name(src, i, j), name(dst, k, l)
                    if any((a, r, b) in shown for r in ("x", "y", "z")):
                        continue
                    nx = ORIGIN[dst][0] + k - ORIGIN[src][0] - i
                    ny = ORIGIN[dst][1] + l - ORIGIN[src][1] - j
                    out.append((a, b, nx, ny))
    return out


def contrasts(world, pairs):
    ti, C = world.tok_index, []
    for a, b, nx, ny in pairs:
        c = np.zeros(world.P)
        c[ti[a]] += 1; c[ti[b]] -= 1; c[ti["x"]] += nx; c[ti["y"]] += ny
        C.append(c)
    return np.array(C)


def null_dim(A):
    s = np.linalg.svd(A, compute_uv=False)
    return A.shape[1] - int(np.sum(s > 1e-9 * s[0]))


def unresolved(system, C):
    """|P0 c| / |c| per query contrast (rows of C)."""
    R = system.project_null(C.T).T
    return np.linalg.norm(R, axis=1) / np.linalg.norm(C, axis=1)


def score(E, plans, queries):
    out = {}
    for k, p in plans.items():
        _, geo, h, _, _ = apply_plan(p, E)
        out[k] = int(h[k].sum())
        out[k + "_geo"] = float(geo[k])
    out["premise"] = int(fact_hits(queries, E)[0].sum())
    return out


def run_cell(spec):
    seed, depth = spec
    tag = "w%02d_d%d" % (seed, depth)
    path = os.path.join(cell_dir(NAME), tag + ".npz")
    if os.path.exists(path):
        return tag, "cached"
    every = worlds.F3_EVERY[depth]
    S = override(init_seed=1000 + seed, eval_every=every)
    t1, t2 = worlds.F3_SWITCH[depth], worlds.F3_AFTER[depth]
    w0, w1 = three_lattice_world(seed, False), three_lattice_world(seed, True)
    s0, s1 = System.build(w0, settings=S), System.build(w1, settings=S)
    shown = set(w1.facts)
    pairs = {k: pairs_for(k[0], k[1], shown) for k in SETS}
    plans = {k: cross_plan(w1, p, name=k) for k, p in pairs.items()}
    facts = list(dict.fromkeys(w0.facts))
    queries = fact_queries(w0, [f for f in facts if f[1] in ("x", "y")])
    lr = s0.lr(depth, settings=S)
    arrays = dict(null_before=null_dim(s0.A), null_after=null_dim(s1.A))
    for k, p in pairs.items():
        C = contrasts(w1, p)
        arrays["unres_before." + k] = unresolved(s0, C)
        arrays["unres_after." + k] = unresolved(s1, C)

    grid = np.arange(0, t2 + 1, every)
    net = models.make_model(w0, depth, S)
    train.train(net, s0, lr, t1, S, order_seed=ORDER_SEED, eval_every=t1)
    _, pstate = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                               settings=S, eval_every=t1)
    probes = {"E": np.eye(w0.P)}
    for arm, system in (("hold", s0), ("insert", s1)):
        mm, rows = copy.deepcopy(net), []
        tr = train.train(mm, system, lr, t2, S, order_seed=ORDER_SEED + 1, eval_every=every,
                         probes=probes)
        rows = [score(E, plans, queries) for E in tr.probes["E"]]
        ptr, _ = theory.predict(system, depth, lr, t2, pstate, settings=S, eval_every=every,
                                probes=probes)
        prows = [score(E, plans, queries) for E in ptr.probes["E"]]
        for who, R in (("net", rows), ("pred", prows)):
            for key in R[0]:
                arrays["%s.%s.%s" % (arm, who, key)] = np.array([r[key] for r in R])
    arrays["epochs"] = grid
    arrays["n_queries"] = np.array([len(pairs[k]) for k in SETS])
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)
    return tag, "done"


def summarize():
    files = sorted(glob.glob(os.path.join(glob.escape(cell_dir(NAME)), "w*_d*.npz")))
    if not files:
        raise SystemExit("no cells yet; run without --summarize first")
    cells, summ = [], {}
    for f in files:
        z = np.load(f)
        seed, depth = int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5])
        nq = dict(zip(SETS, z["n_queries"].tolist()))
        ep = z["epochs"].astype(float)
        row = dict(seed=seed, depth=depth, null_before=int(z["null_before"]),
                   null_after=int(z["null_after"]))
        for k in SETS:
            row["n_" + k] = nq[k]
            row["unres_before_max_" + k] = float(z["unres_before." + k].max())
            row["unres_after_min_" + k] = float(z["unres_after." + k].min())
            row["unres_after_max_" + k] = float(z["unres_after." + k].max())
            for arm in ("hold", "insert"):
                for who in ("net", "pred"):
                    s = z["%s.%s.%s" % (arm, who, k)]
                    row["%s_%s_%s_start" % (arm, who, k)] = int(s[0])
                    row["%s_%s_%s_end" % (arm, who, k)] = int(s[-1])
                    row["%s_%s_%s_t90" % (arm, who, k)] = first_at_least(ep, s, CRIT_FRAC * nq[k])
                    g = z["%s.%s.%s_geo" % (arm, who, k)]
                    row["%s_%s_%s_geo_start" % (arm, who, k)] = float(g[0])
                    row["%s_%s_%s_geo_end" % (arm, who, k)] = float(g[-1])
        for arm in ("hold", "insert"):
            for who in ("net", "pred"):
                row["%s_%s_premise_min" % (arm, who)] = int(z["%s.%s.premise" % (arm, who)].min())
        cells.append(row)
    rep = ["%d worlds per depth. A anchored; B and C free; the link joins A and B. "
           "Counts are correct comparisons out of the set size (9 destination candidates); "
           "start = at the intervention, end = %s epochs later."
           % (len({c["seed"] for c in cells}), ", ".join(
               "N = %d: %s" % (d, "{:,}".format(worlds.F3_AFTER[d])) for d in sorted({c["depth"] for c in cells}))), "",
           "Theory check (all cells):", "",
           "| Quantity | A→B | A→C | B→C |", "|---|---|---|---|"]
    a = lambda k: np.array([c[k] for c in cells], float)
    rep.append("| Queries per set | %d | %d | %d |" % tuple(cells[0]["n_" + k] for k in SETS))
    rep.append("| max ‖P₀c‖/‖c‖ before the link | %s |" % " | ".join("%.3f" % a("unres_before_max_" + k).max() for k in SETS))
    rep.append("| ‖P₀′c‖/‖c‖ after the link, min–max | %s |" % " | ".join(
        "%.1e–%.1e" % (a("unres_after_min_" + k).min(), a("unres_after_max_" + k).max()) for k in SETS))
    rep += ["", "Null dimension per embedding column: before %s, after %s." % (
        sorted(set(a("null_before").astype(int).tolist())), sorted(set(a("null_after").astype(int).tolist()))), "",
        "| Depth | Arm | A→B net (pred), start → end | A→C net (pred), start → end | "
        "B→C net (pred), start → end | A→B worlds reaching 90%, net (pred) | "
        "A→B epochs to 90%, median net (pred) | Median abs. timing error | Lowest premise count, mean net (pred) |",
        "|" + "---|" * 9]
    for depth in sorted({c["depth"] for c in cells}):
        for arm in ("insert", "hold"):
            C = [c for c in cells if c["depth"] == depth]
            b = lambda k: np.array([c[k] for c in C], float)
            parts = []
            for k in SETS:
                parts.append("%.1f (%.1f) → %.1f (%.1f)" % (
                    b("%s_net_%s_start" % (arm, k)).mean(), b("%s_pred_%s_start" % (arm, k)).mean(),
                    b("%s_net_%s_end" % (arm, k)).mean(), b("%s_pred_%s_end" % (arm, k)).mean()))
            tn, tp = b("%s_net_AB_t90" % arm), b("%s_pred_AB_t90" % arm)
            both = np.isfinite(tn) & np.isfinite(tp)
            S = dict(reach_net=int(np.isfinite(tn).sum()), reach_pred=int(np.isfinite(tp).sum()),
                     t_net=float(np.nanmedian(tn)) if np.isfinite(tn).any() else float("nan"),
                     t_pred=float(np.nanmedian(tp)) if np.isfinite(tp).any() else float("nan"),
                     t_err=float(np.median(np.abs(tn[both] - tp[both]))) if both.any() else float("nan"))
            for k in SETS:
                for who in ("net", "pred"):
                    S["%s_%s_end" % (k, who)] = float(b("%s_%s_%s_end" % (arm, who, k)).mean())
            summ["N%d_%s" % (depth, arm)] = S
            f = lambda v: "%.0f" % v if np.isfinite(v) else "never"
            rep.append("| N = %d | %s | %s | %d/%d (%d/%d) | %s (%s) | %s | %.1f (%.1f) |" % (
                depth, "linked" if arm == "insert" else "no link", " | ".join(parts),
                S["reach_net"], len(C), S["reach_pred"], len(C), f(S["t_net"]), f(S["t_pred"]),
                f(S["t_err"]), b("%s_net_premise_min" % arm).mean(), b("%s_pred_premise_min" % arm).mean()))
    rep += ["", "Geometric error (Eq. 27, mean distance from predicted to true target), start → end, "
            "mean over worlds, net (pred). A count of 9/81 means every query in the set lands nearest "
            "the same candidate, i.e. the set is shifted as a whole.", "",
            "| Depth | Arm | A→B | A→C | B→C |", "|---|---|---|---|---|"]
    for depth in sorted({c["depth"] for c in cells}):
        C = [c for c in cells if c["depth"] == depth]
        b = lambda k: np.array([c[k] for c in C], float).mean()
        for arm in ("insert", "hold"):
            rep.append("| N = %d | %s | %s |" % (depth, "linked" if arm == "insert" else "no link", " | ".join(
                "%.2f (%.2f) → %.2f (%.2f)" % tuple(b("%s_%s_%s_geo_%s" % (arm, who, k, t))
                                                  for t in ("start", "end") for who in ("net", "pred"))
                for k in SETS)))
            for k in SETS:
                summ["N%d_%s" % (depth, arm)]["%s_geo_end_net" % k] = b("%s_net_%s_geo_end" % (arm, k))
                summ["N%d_%s" % (depth, arm)]["%s_geo_end_pred" % k] = b("%s_pred_%s_geo_end" % (arm, k))
    write_outputs(NAME, "Reviewer 2, Q5/W4: partial resolution of ambiguity", QUESTION, rep, summ,
                  cells, "new three-lattice worlds built in this script; new runs")


def parse_range(s):
    lo, _, hi = s.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="0-49")
    ap.add_argument("--depths", default="1-3")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv if argv is not None else [])
    if not args.summarize:
        specs = [(s, d) for d in parse_range(args.depths) for s in parse_range(args.worlds)]
        if args.nproc > 1:
            with Pool(args.nproc) as pool:
                for tag, status in pool.imap_unordered(run_cell, specs):
                    print(tag, status, flush=True)
        else:
            for spec in specs:
                print(*run_cell(spec), flush=True)
    summarize()


if __name__ == "__main__":
    main(sys.argv[1:])
