"""Results, section 1: spectral structure predicts law-specific emergence.

Reads Figure 1's records (16 laws, 200 worlds, depths 1 to 3).  The claim is
that the parameter-free prediction reproduces when each law emerges.

Main, for numbers the Results may quote:

  emergence         whether network and prediction agree that a law emerges
                    inside the budget, and where both do, the R² of log10 t*
                    and the absolute error in epochs (median, 95th percentile,
                    share within one evaluation), per depth
  laws              how far apart the laws emerge: the range of median t*
                    among laws whose median is estimable, with the laws whose
                    median lies beyond the budget counted
  trajectories      supporting: mean R² of each law's trajectory within each
                    world, for held-out accuracy and geometric error (log10),
                    one law at a time so differences between laws are not
                    credited to the prediction

Supplementary: the share of within-world variance in log10 t* that law identity
accounts for (conditional on an observed t*), the lowest single law-world R²,
and two descriptions for sentences the Results may or may not keep: whether
transitions sharpen with depth, and whether the geometric ordering of the laws
follows the behavioural one.

t* is the figure's, applied identically to network and prediction: every
held-out composite retrieved and the geometric error under `tau`, both holding
for `hold` evaluations.  A law with no t* inside the 10,000-epoch budget is not
given one.  Its median over worlds is still estimable when more than half the
worlds observe a t*, since every unobserved one lies beyond the budget; a law
short of that is counted, not placed in the range.

  usage:  python f1_emergence.py
"""

import glob
import json
import os

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from scipy import stats as ss
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence

from common import (DEPTHS, N_WORLDS, commas, emit, lg, lowest, per_world_r2,
                    r2_by_depth, seed_of, timing_line, within_share)


def t_star(ep, rec, law):
    i = detect_emergence(rec["retrieval"][law], rec["geometric"][law], S.hold, S.tau)
    return float(ep[int(i)]) if np.isfinite(i) else np.nan


def load():
    """{depth: seeds, laws, ep, observed and predicted t* (worlds x laws), and
    accuracy and geometric error (worlds x laws x evaluations) per source}."""
    data = {}
    for d in DEPTHS:
        files = {seed_of(f): f for f in
                 glob.glob(os.path.join(glob.escape(RESULTS), "f1", "w*_d%d.json" % d))}
        seeds = sorted(files)[:N_WORLDS]
        if len(seeds) < N_WORLDS:
            raise SystemExit("Figure 1 has %d worlds at N=%d, not %d"
                             % (len(seeds), d, N_WORLDS))
        laws, ep = None, None
        T = {"net": [], "pred": []}
        acc = {"net": [], "pred": []}
        geo = {"net": [], "pred": []}
        for s in seeds:
            r = json.load(open(files[s]))
            if laws is None:
                laws = sorted(r["net"]["retrieval"])
                ep = np.array(r["net"]["epochs"], float)
            for src in ("net", "pred"):
                rec = r[src]
                assert np.array_equal(np.array(rec["epochs"], float), ep)
                T[src].append([t_star(ep, rec, l) for l in laws])
                acc[src].append([rec["retrieval"][l] for l in laws])
                geo[src].append([rec["geometric"][l] for l in laws])
        data[d] = dict(seeds=np.array(seeds), laws=laws, ep=ep,
                       tn=np.array(T["net"]), tp=np.array(T["pred"]),
                       acc={k: np.array(v, float) for k, v in acc.items()},
                       geo={k: np.array(v, float) for k, v in geo.items()})
    return data


def emergence(data):
    return (["**Emergence, network against prediction** (the same t* criterion "
             "for both)"]
            + ["- N=%d: %s" % (d, timing_line(data[d]["tn"].ravel(), data[d]["tp"].ravel(),
                                              data[d]["ep"][1] - data[d]["ep"][0],
                                              "the law emerges within %s epochs"
                                              % commas(data[d]["ep"][-1]), "law-worlds"))
               for d in DEPTHS])


def law_spread(data):
    """Descriptive only: the ladder was built so the laws differ, so the
    question is by how much, not whether."""
    ranges = []
    for d in DEPTHS:
        budget = data[d]["ep"][-1]
        tn = data[d]["tn"]
        med = np.median(np.where(np.isfinite(tn), tn, np.inf), axis=0)
        fin = med[np.isfinite(med)]
        late = int((~np.isfinite(med)).sum())
        ranges.append("N=%d %d laws, %s–%s epochs (%.1f-fold)%s" % (
            d, len(fin), commas(fin.min()), commas(fin.max()), fin.max() / fin.min(),
            "" if not late else "; the other %d had no t* within %s epochs in at "
                                "least half the worlds" % (late, commas(budget))))
    return ["**Differences between laws** (network)",
            "- Median t*, among laws with an estimable median (a t* in more than "
            "half the worlds): %s" % "; ".join(ranges)]


def trajectories(data):
    """Report lines, and the lowest law-world R² as diagnostic lines."""
    out = ["**Supporting: whole trajectories** (each law in each world separately; "
           "%s)" % "; ".join("N=%d: 0 to %s epochs" % (d, commas(data[d]["ep"][-1]))
                              for d in DEPTHS)]
    diag = []
    for key, label, tf in (("acc", "Held-out accuracy", np.asarray),
                           ("geo", "Geometric error (log10)", lg)):
        per = {}
        for d in DEPTHS:
            net, pred = tf(data[d][key]["net"]), tf(data[d][key]["pred"])
            per[d] = per_world_r2(net.reshape(-1, net.shape[-1]),
                                  pred.reshape(-1, pred.shape[-1]))
        out.append("- %s: %s" % (label, r2_by_depth(per, unit="law-world")))
        diag.append("- %s: %s" % (label, lowest(per)))
    return out, diag


def law_share(data):
    n_laws = len(data[1]["laws"])
    share, left = [], []
    for d in DEPTHS:
        tn = data[d]["tn"]
        ok = np.isfinite(tn)
        world = np.broadcast_to(data[d]["seeds"][:, None], tn.shape)[ok]
        law = np.broadcast_to(np.arange(n_laws)[None, :], tn.shape)[ok]
        L = (law[:, None] == np.arange(1, n_laws)[None, :]).astype(float)
        share.append(100.0 * within_share(np.log10(tn[ok]), L, world))
        left.append(int((~ok).sum()))
    return ["**Law identity and emergence time** (network; conditional on an "
            "observed t*)",
            "- Law identity accounts for %.1f%%, %.1f%% and %.1f%% of the within-world "
            "variance in log10 t* at N = 1, 2 and 3, among law-worlds with an observed "
            "t* (without one: %s)"
            % (*share, ", ".join("%s at N=%d" % (commas(n), d)
                                 for n, d in zip(left, DEPTHS)))]


def descriptions(data):
    """Sharpness: of the time until accuracy first reaches 100%, the share
    spent climbing from the last evaluation at or below its starting level.
    Ordering: within a world, the rank correlation across laws between that
    epoch and the epoch geometric error falls to half its peak."""
    rel, rise, rho = {}, {}, {}
    for d in DEPTHS:
        ep, A, G = data[d]["ep"], data[d]["acc"]["net"], data[d]["geo"]["net"]
        rel[d], rise[d], rho[d] = [], [], []
        for i in range(A.shape[0]):
            ta, tg = [], []
            for j in range(A.shape[1]):
                a = A[i, j]
                full = np.flatnonzero(a >= 100.0 - 1e-9)
                if len(full) and full[0] > 0:
                    k = full[0]
                    k0 = np.flatnonzero(a[:k] <= a[0] + 1e-9)[-1]
                    rise[d].append(ep[k] - ep[k0])
                    rel[d].append((ep[k] - ep[k0]) / ep[k])
                    ta.append(ep[k])
                else:
                    ta.append(np.nan)
                g = G[i, j]
                p = int(np.argmax(g))
                half = np.flatnonzero(g[p:] <= 0.5 * g[p])
                tg.append(ep[p + half[0]] if len(half) else np.nan)
            ta, tg = np.array(ta), np.array(tg)
            ok = np.isfinite(ta) & np.isfinite(tg)
            if ok.sum() >= 5 and np.ptp(ta[ok]) > 0 and np.ptp(tg[ok]) > 0:
                rho[d].append(ss.spearmanr(ta[ok], tg[ok])[0])
    return ["**Descriptions, only if the Results keeps these sentences** (network)",
            "- Sharpness of the accuracy transition, share of the time to first "
            "100%% spent climbing from the starting level (smaller is sharper), "
            "median: %s" % "; ".join("N=%d %.2f (%s epochs)"
                                     % (d, np.median(rel[d]), commas(np.median(rise[d])))
                                     for d in DEPTHS),
            "- Geometric ordering, within-world Spearman ρ across laws between the "
            "epoch accuracy first reaches 100%% and the epoch geometric error falls "
            "to half its peak, median: %s" % "; ".join(
                "N=%d %.3f (%d worlds)" % (d, np.median(rho[d]), len(rho[d]))
                for d in DEPTHS)]


def report():
    data = load()
    traj, diag = trajectories(data)
    out = ["## Spectral structure predicts law-specific compositional emergence "
           "(Figure 1)", ""]
    for block in (emergence(data), law_spread(data), traj):
        out += block + [""]
    out += ["### Supplementary (Figure 1)", ""]
    for block in (law_share(data),
                  ["**Diagnostics: lowest R² of any single law-world** (R² depends "
                   "on how much variance a trajectory has)"] + diag,
                  descriptions(data)):
        out += block + [""]
    return out


if __name__ == "__main__":
    emit(report())
