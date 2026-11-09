"""Reviewer 2, W5 / Q3 (first part): how much does integration depend on rehearsal?

  "How sensitive is Experiment 3 to the continuous rehearsal of background
   facts during the linking phase?"  (Q3)
  "boundary integration occurs in ~20 trials ... The model here requires
   thousands of epochs where original facts and anchors are continuously
   rehearsed."  (W5)

Each cell starts from the network state stored at the intervention
(`results/f3/controls_w{s}_d{N}.npz`, `pre.*`; where a file has no stored
state, the pre-intervention run is replayed from its seed), at the paper's
settings, and
continues under five post-link training sets:

  full            every original fact and anchor, plus the link (the paper's arm)
  replay25        a fixed 25% of the original facts, all anchors, plus the link
  anchors_link    the 3 anchors plus the link, no original facts
  link_only       the link alone
  link_only_fast  the link alone at a per-row rate with eta * ||g||^2 * kappa = 0.5,
                  where kappa = sum_l ||left_l||^2 ||right_l||^2 (spectral norms of
                  the layer products around each factor, at the intervention; 1 at
                  N = 1) bounds how much depth amplifies one step

One epoch presents the link once in every condition, so the time axis is link
presentations. Background updates per link presentation are the remaining rows.
The prediction for each condition integrates that condition's own constraint
rows from the theory's own predicted state at the intervention, as in
`analysis/controls_pred.py`.

Checks (with --verify, default on for worlds 0-4): the `full` network arm must
reproduce the stored linked arm's 9-candidate hits exactly at every stored
evaluation, and the `full` prediction must reproduce the stored prediction's
hits (`controls_pred_*.npz`) at every shared evaluation.

Per-cell arrays go to `results/reviewer_outputs/r2_q3_rehearsal/cells/`; a cell
already on disk is reused, so an interrupted run resumes.

  usage:  python r2_q3_rehearsal.py                    worlds 0-49, depths 1-3
          python r2_q3_rehearsal.py --worlds 0-199 --nproc 12
          python r2_q3_rehearsal.py --summarize        rebuild the report only
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from common import cell_dir, first_at_least, write_outputs  # noqa: E402
from _paths import CODE, RESULTS  # noqa: E402

sys.path.insert(0, os.path.join(CODE, "analysis"))
from controls import fact_hits, fact_queries  # noqa: E402
from relspec import System, checkpoint, models, theory, train, worlds  # noqa: E402
from relspec.config import override  # noqa: E402
from relspec.measure import apply_plan, cross_plan  # noqa: E402

NAME = "r2_q3_rehearsal"
QUESTION = ("Q3, 'How sensitive is Experiment 3 to the continuous rehearsal of background "
            "facts during the linking phase?' and W5, '~20 trials ... thousands of epochs "
            "where original facts and anchors are continuously rehearsed'")
CONDITIONS = ("full", "replay25", "anchors_link", "link_only", "link_only_fast")
REPLAY_FRACTION = 0.25
FAST_STEP = 0.5          # eta * ||g||^2 for link_only_fast
CRITERION = 72           # of 80 cross-structure comparisons, as in the paper
EARLY = 50               # every link presentation is evaluated up to this many


def build_systems(w0, w1, seed, S):
    link = [f for f in w1.facts if f not in set(w0.facts)]
    assert len(link) == 1, "expected exactly one linking fact"
    facts0 = list(w0.facts)
    rng = np.random.default_rng(10_000 + seed)
    k = max(1, int(round(REPLAY_FRACTION * len(facts0))))
    keep = sorted(rng.choice(len(facts0), size=k, replace=False))
    replay = [facts0[i] for i in keep] + link
    sys_ = {
        "full": System.build(w1, settings=S),
        "replay25": System.build(w1, facts=replay, settings=S),
        "anchors_link": System.build(w1, facts=link, settings=S),
        "link_only": System.build(w1, facts=link, anchors={}, settings=S),
    }
    sys_["link_only_fast"] = sys_["link_only"]
    return sys_, link[0]


def scores(E, plan9, plan18, queries):
    _, geo, h9, _, _ = apply_plan(plan9, E)
    _, _, h18, _, _ = apply_plan(plan18, E)
    return dict(cross9=int(h9["cross"].sum()), cross18=int(h18["cross"].sum()),
                geo9=float(geo["cross"]),
                premise=int(fact_hits(queries["premise"], E)[0].sum()),
                facts=int(fact_hits(queries["facts"], E)[0].sum()),
                hits9=h9["cross"].copy())


def pred_states(system, depth, lr, grid, state, S):
    """As `analysis/controls_pred.states`: predicted embeddings at `grid`.
    Returns None if the integration diverges."""
    out, prev = [], 0
    probes = {"E": np.eye(system.world.P)}
    for t in grid[1:]:
        span = int(t - prev)
        traj, state = theory.predict(system, depth, lr, span, state, settings=S,
                                     eval_every=span, probes=probes)
        if traj is None:
            return None
        if not out:
            out.append(traj.probes["E"][0])
        out.append(traj.probes["E"][-1])
        prev = t
    return out


def depth_gain(Ws):
    """kappa = sum_k ||W_0..W_{k-1}||^2 ||W_{k+1}..||^2, spectral norms; 1 at N = 1."""
    if len(Ws) == 1:
        return 1.0
    total = 0.0
    for k in range(len(Ws)):
        left = models.embed(Ws[:k]) if k else None
        right = models.embed(Ws[k + 1:]) if k + 1 < len(Ws) else None
        nl = np.linalg.norm(left, 2) ** 2 if left is not None else 1.0
        nr = np.linalg.norm(right, 2) ** 2 if right is not None else 1.0
        total += nl * nr
    return float(total)


def nan_scores(n, n_items):
    return [dict(cross9=np.nan, cross18=np.nan, geo9=np.nan, premise=np.nan, facts=np.nan,
                 hits9=np.zeros(n_items, bool)) for _ in range(n)]


def run_cell(spec):
    seed, depth, verify = spec
    tag = "w%02d_d%d" % (seed, depth)
    path = os.path.join(cell_dir(NAME), tag + ".npz")
    if os.path.exists(path):
        return tag, "cached"
    rec = json.load(open(os.path.join(RESULTS, "f3", tag + ".json")))
    ep0 = rec["arms"]["insert"]["net"]["epochs"]
    S = override(init_seed=rec["init_seed"], eval_every=int(ep0[1] - ep0[0]),
                 lr_target=rec["lr_target"], ode_substeps={depth: rec.get("substeps", 1)})
    ctrl = np.load(os.path.join(RESULTS, "f3", "controls_%s.npz" % tag))
    stored_grid = np.asarray(ctrl["epochs"], int)
    grid = np.array(sorted(set(stored_grid.tolist()) | set(range(0, EARLY + 1))), int)
    t1 = rec["t_switch"]

    w0 = worlds.integration_world(seed, link=False)
    w1 = worlds.integration_world(seed, link=True)
    s0 = System.build(w0, settings=S)
    systems, link = build_systems(w0, w1, seed, S)
    pairs = worlds.held_cross(w0, reference=w1)
    plan9 = cross_plan(w1, pairs)
    plan18 = cross_plan(w1, pairs, candidates=w1.entities)
    facts = list(dict.fromkeys(w0.facts))
    queries = {"premise": fact_queries(w0, [f for f in facts if f[1].startswith(("x", "y"))]),
               "facts": fact_queries(w0, facts)}
    lr = s0.lr(depth, target=rec["lr_target"], settings=S)
    g2 = 3.0  # ||g||^2 for a fact row (+1, +1, -1)
    rates = {c: lr for c in CONDITIONS}

    net = models.make_model(w0, depth, S)
    if "pre.0" in ctrl.files:
        checkpoint.restore(net, [ctrl["pre.%d" % i] for i in range(depth)])
    else:
        # some stored controls (all of depth 1) predate saving the state:
        # replay the pre-intervention run, which is deterministic given the seed
        train.train(net, s0, s0.lr(depth, target=rec["lr_target"], settings=S), t1, S,
                    order_seed=rec["order_seed"], eval_every=t1)
    _, pstate = theory.predict(s0, depth, lr, t1, models.make_model(w0, depth, S),
                               settings=S, eval_every=t1)
    kappa = depth_gain(checkpoint.weights(net))
    rates["link_only_fast"] = FAST_STEP / (g2 * kappa)

    arrays = dict(epochs=grid, stored=np.isin(grid, stored_grid))
    for cond in CONDITIONS:
        system, eta = systems[cond], rates[cond]
        mm, at, rng, rows = copy.deepcopy(net), 0, None, []
        for t in grid:
            if t > at:
                tr = train.train(mm, system, eta, int(t), S, order_seed=rec["order_seed"] + 1,
                                 eval_every=int(t - at), start=at, rng_state=rng)
                rng, at = tr.rng_state, int(t)
            E = mm.embedding()
            rows.append(scores(E, plan9, plan18, queries) if np.all(np.isfinite(E))
                        else nan_scores(1, len(pairs))[0])
        Es = pred_states(system, depth, eta, grid, pstate, S)
        preds = (nan_scores(len(grid), len(pairs)) if Es is None
                 else [scores(E, plan9, plan18, queries) for E in Es])
        arrays["%s.pred_diverged" % cond] = Es is None
        for who, R in (("net", rows), ("pred", preds)):
            for key in ("cross9", "cross18", "geo9", "premise", "facts"):
                arrays["%s.%s.%s" % (cond, who, key)] = np.array([r[key] for r in R])
        arrays["%s.rows" % cond] = system.A.shape[0]
        arrays["%s.lr" % cond] = eta
        if verify and cond == "full":
            mine = np.array([r["hits9"] for r in rows])[arrays["stored"]]
            if not np.array_equal(mine, ctrl["insert.cross9"]):
                raise AssertionError("%s: full-replay network differs from the stored linked arm" % tag)
            cp = np.load(os.path.join(RESULTS, "f3", "controls_pred_%s.npz" % tag))
            pg = np.asarray(cp["epochs"], float)
            common_t = np.intersect1d(pg, grid)
            ia, ib = np.searchsorted(pg, common_t), np.searchsorted(grid, common_t)
            pmine = np.array([r["hits9"] for r in preds])[ib]
            if not np.array_equal(pmine, cp["insert.pred9"][ia]):
                raise AssertionError("%s: full-replay prediction differs from the stored one" % tag)
            arrays["verified"] = True
    arrays["kappa"] = kappa
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)
    return tag, "verified" if verify else "done"


def summarize():
    files = sorted(glob.glob(os.path.join(glob.escape(cell_dir(NAME)), "w*_d*.npz")))
    if not files:
        raise SystemExit("no cells yet; run without --summarize first")
    cells, summ = [], {}
    for f in files:
        z = np.load(f)
        seed, depth = int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5])
        ep = z["epochs"].astype(float)
        for cond in CONDITIONS:
            row = dict(seed=seed, depth=depth, condition=cond, rows=int(z["%s.rows" % cond]),
                       background_per_link=int(z["%s.rows" % cond]) - 1, lr=float(z["%s.lr" % cond]))
            for who in ("net", "pred"):
                c9 = z["%s.%s.cross9" % (cond, who)]
                pr = z["%s.%s.premise" % (cond, who)]
                row.update({
                    "%s_cross9_end" % who: float(c9[-1]),
                    "%s_cross9_max" % who: float(np.max(c9)),
                    "%s_t_criterion" % who: first_at_least(ep, c9, CRITERION),
                    "%s_premise_min" % who: float(np.min(pr)),
                    "%s_premise_end" % who: float(pr[-1]),
                    "%s_cross18_end" % who: float(z["%s.%s.cross18" % (cond, who)][-1]),
                    "%s_geo9_ratio" % who: float(z["%s.%s.geo9" % (cond, who)][-1]
                                                 / z["%s.%s.geo9" % (cond, who)][0])})
            row["pred_diverged"] = bool(z["%s.pred_diverged" % cond])
            row["kappa"] = float(z["kappa"])
            cells.append(row)
    rep = ["%d worlds x %d depths; each condition continues from the stored network "
           "state at the intervention for %s link presentations. Criterion: %d of 80 "
           "cross-structure comparisons correct (9 destination candidates). Premises: 24."
           % (len({c["seed"] for c in cells}), len({c["depth"] for c in cells}),
              "{:,}".format(int(np.load(files[0])["epochs"][-1])), CRITERION), "",
           "| Depth | Condition | Background updates per link presentation | Correct at end, net (pred) | "
           "Worlds reaching criterion, net (pred) | Link presentations to criterion, median net (pred) | "
           "Median abs. timing error | Worlds with any premise loss, net (pred) | "
           "Lowest premise count, mean net (pred) | Premises correct at end, mean net (pred) |",
           "|" + "---|" * 10]
    for depth in sorted({c["depth"] for c in cells}):
        for cond in CONDITIONS:
            C = [c for c in cells if c["depth"] == depth and c["condition"] == cond]
            a = lambda k: np.array([c[k] for c in C], float)
            tn, tp = a("net_t_criterion"), a("pred_t_criterion")
            both = np.isfinite(tn) & np.isfinite(tp)
            S = dict(n=len(C), background=int(np.median(a("background_per_link"))),
                     pred_diverged=int(a("pred_diverged").sum()),
                     end_net=np.nanmean(a("net_cross9_end")), end_pred=np.nanmean(a("pred_cross9_end")),
                     reach_net=int(np.isfinite(tn).sum()), reach_pred=int(np.isfinite(tp).sum()),
                     t_net=float(np.nanmedian(tn)) if np.isfinite(tn).any() else float("nan"),
                     t_pred=float(np.nanmedian(tp)) if np.isfinite(tp).any() else float("nan"),
                     t_abs_err=float(np.median(np.abs(tn[both] - tp[both]))) if both.any() else float("nan"),
                     loss_net=int((a("net_premise_min") < 24).sum()),
                     loss_pred=int((a("pred_premise_min") < 24).sum()),
                     pmin_net=np.nanmean(a("net_premise_min")), pmin_pred=np.nanmean(a("pred_premise_min")),
                     pend_net=np.nanmean(a("net_premise_end")), pend_pred=np.nanmean(a("pred_premise_end")))
            summ["N%d_%s" % (depth, cond)] = S
            rep.append("| N = %d | %s | %d | %.1f (%.1f) | %d/%d (%d/%d) | %s (%s) | %s | %d (%d) | %.1f (%.1f) | %.1f (%.1f) |" % (
                depth, cond, S["background"], S["end_net"], S["end_pred"], S["reach_net"], S["n"],
                S["reach_pred"], S["n"], "%.0f" % S["t_net"] if np.isfinite(S["t_net"]) else "never",
                "%.0f" % S["t_pred"] if np.isfinite(S["t_pred"]) else "never",
                "%.0f" % S["t_abs_err"] if np.isfinite(S["t_abs_err"]) else "n/a",
                S["loss_net"], S["loss_pred"], S["pmin_net"], S["pmin_pred"], S["pend_net"], S["pend_pred"]))
    rep += ["", "Conditions: full = all 32 original facts + 3 anchors + link; replay25 = a fixed 25%% of "
            "the facts (8) + anchors + link; anchors_link = anchors + link; link_only = link alone at the "
            "paper's rate; link_only_fast = link alone with eta * ||g||^2 * kappa = %.1f. "
            "Predictions that diverged are excluded from the (pred) columns; count per row: %s."
            % (FAST_STEP, ", ".join("%s %d" % (k, v["pred_diverged"]) for k, v in summ.items()
                                    if v["pred_diverged"]) or "none")]
    write_outputs(NAME, "Reviewer 2, Q3/W5: dependence of integration on rehearsal", QUESTION, rep,
                  summ, cells, "stored intervention states `results/f3/controls_w*_d*.npz` and records "
                  "`results/f3/w*_d*.json`; new runs per condition")


def parse_range(s):
    lo, _, hi = s.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="0-49")
    ap.add_argument("--depths", default="1-3")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--verify-worlds", default="0-4")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv if argv is not None else [])
    if not args.summarize:
        ver = set(parse_range(args.verify_worlds))
        specs = [(s, d, s in ver) for d in parse_range(args.depths) for s in parse_range(args.worlds)]
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
