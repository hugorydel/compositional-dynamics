"""Statistics reported alongside Figures 1 to 3.

Everything here is read from the stored records the figures draw, on the same
world sets, so a number in the text and a curve in a panel always come from
the same runs.  Nothing is trained.

For each figure three kinds of number are produced:

  structure    what the training data does and does not determine, from the
               constraint matrix alone: rank, null-space dimension, and the
               identifiability residual of the law or comparison in question.
  effect       the behavioural and geometric outcome, compared between
               conditions on the same worlds with a paired two-sided Wilcoxon
               signed-rank test, and the number of worlds in which the effect
               goes the stated way.
  prediction   agreement between each network and its parameter-free
               prediction: event times (emergence, or reaching a fixed level),
               whole-curve error, and, for Figure 3, the fitted time-rescaling
               between the two.

  usage:  python stats.py            all three
          python stats.py f1         one figure
"""

import glob
import json
import os
import sys

import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS
from scipy import stats as ss
from relspec import System, worlds
from relspec.config import DEFAULT as S
from relspec.measure import detect_emergence, unlocked

N_WORLDS = 200
DEPTHS = (1, 2, 3)


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #

def seed_of(path):
    return int(os.path.basename(path).split("_")[0][1:])


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "P < 1e-300"
    return "P = %.1e" % p if p < 1e-3 else "P = %.3f" % p


def msem(x):
    x = np.asarray(x, float)
    return "%.2f +/- %.2f" % (x.mean(), x.std(ddof=1) / np.sqrt(len(x)))


def paired(label, a, b, unit=""):
    """a and b paired by world; two-sided Wilcoxon signed-rank test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        test = "all differences zero"
    else:
        w = ss.wilcoxon(a, b)
        z = ss.norm.isf(w.pvalue / 2.0)
        test = "W = %.0f, z = %.1f, %s" % (w.statistic, z, fmt_p(w.pvalue))
    print("    %-34s %s%s vs %s%s | higher in %d, lower in %d, equal in %d of %d | %s"
          % (label, msem(a), unit, msem(b), unit, int((d > 0).sum()), int((d < 0).sum()),
             int((d == 0).sum()), len(d), test))


def change(label, before, after, unit=""):
    """Within-condition change over the window; Wilcoxon signed-rank against zero."""
    d = np.asarray(after, float) - np.asarray(before, float)
    if np.allclose(d, 0):
        test = "all changes zero"
    else:
        w = ss.wilcoxon(d)
        test = "W = %.0f, %s" % (w.statistic, fmt_p(w.pvalue))
    print("    %-34s change %s%s, median %.2f | up in %d, down in %d of %d | %s"
          % (label, msem(d), unit, np.median(d), int((d > 0).sum()), int((d < 0).sum()),
             len(d), test))


def agreement(label, net, pred, log=False):
    """Per-world curve agreement, pooled R^2, and the median per-world error."""
    net, pred = np.asarray(net, float), np.asarray(pred, float)
    if log:
        x, y = np.log10(np.maximum(net, 1e-12)), np.log10(np.maximum(pred, 1e-12))
        per = np.median(np.abs(x - y), axis=1)
        what = "median |log10 net/pred|"
    else:
        x, y = net, pred
        per = np.sqrt(np.mean((x - y) ** 2, axis=1))
        what = "RMSE"
    r2 = 1.0 - np.sum((x - y) ** 2) / np.sum((x - x.mean()) ** 2)
    print("    %-34s pooled R^2 = %.4f | per-world %s: median %.4f, 95th pct %.4f"
          % (label, r2, what, np.median(per), np.percentile(per, 95)))


def times_agree(label, tn, tp, resolution):
    """Observed against predicted event times, where both occurred."""
    tn, tp = np.asarray(tn, float), np.asarray(tp, float)
    both = np.isfinite(tn) & np.isfinite(tp) & (tn > 0) & (tp > 0)
    only_n = int((np.isfinite(tn) & ~np.isfinite(tp)).sum())
    only_p = int((~np.isfinite(tn) & np.isfinite(tp)).sum())
    neither = int((~np.isfinite(tn) & ~np.isfinite(tp)).sum())
    a, b = tn[both], tp[both]
    lr = np.log10(a / b)
    r = ss.pearsonr(np.log10(a), np.log10(b))[0] if len(a) > 2 and np.ptp(b) > 0 else np.nan
    print("    %-34s n = %d | Pearson r(log t) = %.4f | median |log10 obs/pred| = %.4f"
          " | within one evaluation (%d epochs) %.1f%% | within 10%% %.1f%%"
          % (label, both.sum(), r, np.median(np.abs(lr)), resolution,
             100.0 * np.mean(np.abs(a - b) <= resolution),
             100.0 * np.mean(np.abs(a / b - 1.0) <= 0.10)))
    if only_n or only_p or neither:
        print("    %-34s occurred in network only %d, prediction only %d, neither %d"
              % ("", only_n, only_p, neither))


def first_reach(ep, v, level):
    k = np.where(np.asarray(v) >= level - 1e-9)[0]
    return float(ep[k[0]]) if len(k) else np.inf


# --------------------------------------------------------------------------- #
#  Figure 1                                                                    #
# --------------------------------------------------------------------------- #

def figure1():
    print("=" * 78)
    print("FIGURE 1  emergence of 16 laws from initialisation, eta_target 0.03")
    w = worlds.emergence_world(0)
    s = System.build(w, settings=S)
    rep = s.rank_report(S)
    held = worlds.held_composites(w)
    rho = max(s.identifiability(l, S)["rho"] for l in w.laws)
    print("  structure (world 0): %d laws, %d facts, %d anchors, %d tokens, rank %d,"
          " null dim %d, max identifiability residual %.1e, held-out per law %s"
          % (len(w.laws), len(w.facts), len(w.anchors), w.P, rep["rank"],
             rep["null_dim"], rho, sorted({len(v) for v in held.values()})))
    law_names = [l.name for l in w.laws]
    drawn = list(worlds.F1_LAWS)
    for nm, lab in zip(drawn, "ABC"):
        c = worlds.LADDER[int(nm[1:].split("_")[0])]
        print("    law %s = %s, lattice %dx%d, premise repetitions x %d, y %d"
              % (lab, nm, c["m"], c["n"], c["rep_x"], c["rep_y"]))

    T = {}          # depth -> (worlds x laws) observed and predicted t*
    for d in DEPTHS:
        fs = sorted(glob.glob(os.path.join(glob.escape(RESULTS), "f1", "w*_d%d.json" % d)),
                    key=seed_of)[:N_WORLDS]
        tn = np.full((len(fs), len(law_names)), np.inf)
        tp = np.full_like(tn, np.inf)
        acc_n, acc_p, geo_n, geo_p = [], [], [], []
        for i, f in enumerate(fs):
            r = json.load(open(f))
            ep = np.array(r["net"]["epochs"], float)
            epp = np.array(r["pred"]["epochs"], float)
            assert np.array_equal(ep, epp)
            for j, law in enumerate(law_names):
                a = detect_emergence(r["net"]["retrieval"][law],
                                     r["net"]["geometric"][law], S.hold, S.tau)
                b = detect_emergence(r["pred"]["retrieval"][law],
                                     r["pred"]["geometric"][law], S.hold, S.tau)
                tn[i, j] = ep[int(a)] if np.isfinite(a) else np.inf
                tp[i, j] = epp[int(b)] if np.isfinite(b) else np.inf
                acc_n.append(r["net"]["retrieval"][law])
                acc_p.append(r["pred"]["retrieval"][law])
                geo_n.append(r["net"]["geometric"][law])
                geo_p.append(r["pred"]["geometric"][law])
        T[d] = (tn, tp, ep[-1])
        every = int(ep[1] - ep[0])
        print("  N=%d  (%d worlds, %d law-worlds, run to %d epochs, evaluated every %d)"
              % (d, len(fs), tn.size, ep[-1], every))
        print("   prediction")
        times_agree("emergence time, all 16 laws", tn.ravel(), tp.ravel(), every)
        rs = []
        for i in range(len(fs)):
            a = np.where(np.isfinite(tn[i]), tn[i], 1e9)
            b = np.where(np.isfinite(tp[i]), tp[i], 1e9)
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                rs.append(ss.spearmanr(a, b)[0])
        print("    %-34s median Spearman rho = %.3f (IQR %.3f-%.3f) over %d worlds"
              % ("order of the 16 laws, per world", np.median(rs),
                 np.percentile(rs, 25), np.percentile(rs, 75), len(rs)))
        agreement("accuracy curves, all laws", acc_n, acc_p)
        agreement("geometric-error curves, all laws", geo_n, geo_p, log=True)
        print("   effect")
        idx = [law_names.index(n) for n in drawn]
        abc = tn[:, idx]
        order = np.mean((abc[:, 0] < abc[:, 1]) & (abc[:, 1] < abc[:, 2]))
        weak = np.mean((abc[:, 0] <= abc[:, 1]) & (abc[:, 1] <= abc[:, 2]))
        fr = ss.friedmanchisquare(*[abc[:, k] for k in range(3)])
        print("    %-34s strictly A < B < C in %.1f%% of worlds (A <= B <= C %.1f%%);"
              " Friedman chi2(2) = %.1f, %s"
              % ("drawn laws, emergence order", 100 * order, 100 * weak,
                 fr.statistic, fmt_p(fr.pvalue)))
        for k, lab in enumerate("ABC"):
            col = abc[:, k]
            fin = np.isfinite(col)
            print("    %-34s emerged in %d of %d worlds, median t* = %s"
                  % ("law %s" % lab, fin.sum(), len(col),
                     ("%.0f" % np.median(col[fin])) if fin.any() else "none"))
        med = np.array([np.median(tn[:, j]) for j in range(len(law_names))])
        fin = np.isfinite(med)
        print("    %-34s %d of 16 laws emerge in most worlds; median t* spans %.0f-%.0f"
              " epochs (%.1f-fold)"
              % ("spread across laws", fin.sum(), med[fin].min(), med[fin].max(),
                 med[fin].max() / med[fin].min()))
        if np.all(np.isfinite(tn)):
            kw = ss.kruskal(*[np.log10(tn[:, j]) for j in range(len(law_names))])
            print("    %-34s Kruskal-Wallis H(15) = %.0f, %s"
                  % ("law identity vs emergence time", kw.statistic, fmt_p(kw.pvalue)))

    print("  one-number spectral summary")
    # the slowest eigenvalue among the modes a law's contrast loads on, as a
    # ratio to the largest; a single-mode stand-in for the full prediction
    slow = np.full((N_WORLDS, len(law_names)), np.nan)
    for seed in range(N_WORLDS):
        ws = worlds.emergence_world(seed)
        sy = System.build(ws, settings=S)
        for j, l in enumerate(ws.laws):
            assert l.name == law_names[j]
            slow[seed, j] = sy.evals_M.max() / sy.identifiability(l, S)["sigma_slow_loaded"]
    for d in DEPTHS:
        tn = T[d][0]
        ok = np.isfinite(tn)
        rho = ss.spearmanr(slow[ok], tn[ok])
        print("    N=%d  lambda_max / slowest loaded eigenvalue vs t*: Spearman rho = %.3f,"
              " %s, n = %d" % (d, rho[0], fmt_p(rho[1]), ok.sum()))

    print("  depth")
    for (d1, d2) in ((1, 2), (1, 3), (2, 3)):
        a, b = T[d1][0], T[d2][0]
        both = np.isfinite(a) & np.isfinite(b)
        ratio = a[both] / b[both]
        w_ = ss.wilcoxon(np.log10(a[both]), np.log10(b[both]))
        print("    N=%d vs N=%d: %d law-worlds emerging at both; median t*(N=%d)/t*(N=%d)"
              " = %.2f (IQR %.2f-%.2f); faster at N=%d in %.1f%%; Wilcoxon %s"
              % (d1, d2, both.sum(), d1, d2, np.median(ratio),
                 np.percentile(ratio, 25), np.percentile(ratio, 75), d2,
                 100 * np.mean(ratio > 1), fmt_p(w_.pvalue)))


# --------------------------------------------------------------------------- #
#  Figure 2                                                                    #
# --------------------------------------------------------------------------- #

def f2_curve(rec, arm, src, key):
    d = rec["arms"][arm][src]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    m = ep >= 0
    law = rec["open"]
    if key == "acc":
        v = 100.0 * np.array(d["hits"][law], bool).mean(axis=1)
    else:
        v = np.array(d["geometric"][law], float)
    return ep[m], v[m]


def figure2():
    print("=" * 78)
    print("FIGURE 2  one linking fact on an undetermined composite, eta_target 0.03")
    rho_b, rho_a, nulls = [], [], set()
    for seed in range(N_WORLDS):
        for closed in ("A", "B"):
            op = "B" if closed == "A" else "A"
            for nb, store in ((0, rho_b), (1, rho_a)):
                w = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed,
                                                 n_bridge=nb)
                s = System.build(w, settings=S)
                law = [l for l in w.laws if l.name == op][0]
                store.append(s.identifiability(law, S)["rho"])
                nulls.add((nb, s.rank_report(S)["null_dim"]))
    w = worlds.identifiability_world(0, K=16, k=4, closed_block="A")
    print("  structure: %d facts, %d anchors, %d tokens; null dim before/after %s;"
          " residual of the undetermined law before %.4f-%.4f, after max %.1e (%d arms)"
          % (len(w.facts), len(w.anchors), w.P, sorted(nulls), min(rho_b), max(rho_b),
             max(rho_a), len(rho_a)))

    s0 = System.build(worlds.identifiability_world(0, K=16, k=4, closed_block="A"), settings=S)
    s1 = System.build(worlds.identifiability_world(0, K=16, k=4, closed_block="A", n_bridge=1),
                      settings=S)
    e0, e1 = np.sort(s0.evals_M)[::-1], np.sort(s1.evals_M)[::-1]
    n0, n1 = e0[e0 > S.rank_rtol * e0[0]], e1[e1 > S.rank_rtol * e1[0]]
    print("  spectrum: largest eigenvalue %.3f -> %.3f; smallest nonzero %.5f -> %.5f"
          " (%.1e of largest), %.1f times below the next smallest; identical in every world"
          % (e0[0], e1[0], n0[-1], n1[-1], n1[-1] / e1[0], n1[-2] / n1[-1]))

    import fig2
    seeds, _, arms = fig2.world_set(keep_all=True)
    print("  %d worlds, 2 counterbalanced arms each, arms averaged within world" % len(seeds))
    for d in DEPTHS:
        win = fig2.XLIM[1]
        acc = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        geo = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        for seed in seeds:
            recs = [json.load(open(f)) for f in sorted(arms[d][seed])]
            for key, store in (("acc", acc), ("geo", geo)):
                for (a, s) in store:
                    cur = [f2_curve(r, a, s, key) for r in recs]
                    ep = cur[0][0]
                    V = np.array([v for _, v in cur])
                    store[a, s].append(V.mean(0) if key == "acc"
                                       else np.exp(np.log(np.maximum(V, 1e-12)).mean(0)))
        ep = f2_curve(json.load(open(sorted(arms[d][seeds[0]])[0])), "hold", "net", "acc")[0]
        k_win = int(np.argmin(np.abs(ep - win)))
        in_win = ep <= win
        every = int(ep[1] - ep[0])
        A = {k: np.array(v) for k, v in acc.items()}
        G = {k: np.array(v) for k, v in geo.items()}
        print("  N=%d  (switch at %d, %d epochs after, evaluated every %d; plotted window %d)"
              % (d, worlds.F2_SWITCH[d], ep[-1], every, win))
        print("   effect")
        print("    %-34s both arms %s%%" % ("accuracy at the linking fact", msem(A["hold", "net"][:, 0])))
        paired("accuracy at %d epochs, link vs none" % win,
               A["insert", "net"][:, k_win], A["hold", "net"][:, k_win], "%")
        change("accuracy without the link, 0-%d" % win,
               A["hold", "net"][:, 0], A["hold", "net"][:, k_win], "%")
        change("accuracy without the link, run",
               A["hold", "net"][:, 0], A["hold", "net"][:, -1], "%")
        paired("accuracy at run end, link vs none",
               A["insert", "net"][:, -1], A["hold", "net"][:, -1], "%")
        m0 = A["insert", "net"][:, 0].mean()
        rel = lambda m: 100 * (m - m0) / (100 - m0)
        print("    %-34s link %.1f, none %.1f (0 = mean at the fact, 100 = all correct)"
              % ("plotted value at %d epochs" % win, rel(A["insert", "net"][:, k_win].mean()),
                 rel(A["hold", "net"][:, k_win].mean())))
        ratio = G["hold", "net"][:, k_win] / G["insert", "net"][:, k_win]
        paired("log10 geometric error, none vs link", np.log10(G["hold", "net"][:, k_win]),
               np.log10(G["insert", "net"][:, k_win]))
        print("    %-34s median %.1f-fold lower with the link at %d epochs (IQR %.1f-%.1f)"
              % ("geometric error", np.median(ratio), win, np.percentile(ratio, 25),
                 np.percentile(ratio, 75)))
        print("   prediction")
        for a, lab in (("insert", "link"), ("hold", "none")):
            agreement("accuracy curve, %s, 0-%d" % (lab, win),
                      A[a, "net"][:, in_win], A[a, "pred"][:, in_win])
            agreement("geometric-error curve, %s, 0-%d" % (lab, win),
                      G[a, "net"][:, in_win], G[a, "pred"][:, in_win], log=True)
        for level in (50, 90):
            tn = [first_reach(ep, v, level) for v in A["insert", "net"]]
            tp = [first_reach(ep, v, level) for v in A["insert", "pred"]]
            times_agree("epochs to %d%% accuracy, link" % level, tn, tp, every)


# --------------------------------------------------------------------------- #
#  Figure 3                                                                    #
# --------------------------------------------------------------------------- #

def f3_curve(rec, arm, src, key):
    d = rec["arms"][arm][src]
    ep = np.array(d["epochs"], float) - rec["t_switch"]
    m = ep >= 0
    if key == "hits":
        return ep[m], np.array(d["hits"]["cross"], bool)[m]
    return ep[m], np.array(d["geometric"]["cross"], float)[m]


def figure3():
    print("=" * 78)
    print("FIGURE 3  one linking fact between two lattices, eta_target 0.003")
    for link in (False, True):
        w = worlds.integration_world(0, link=link)
        s = System.build(w, settings=S)
        rep = s.rank_report(S)
        ev = np.sort(s.evals_M)[::-1]
        nz = ev[ev > S.rank_rtol * ev[0]]
        print("  structure, link %-5s: %d facts, %d anchors, %d tokens, rank %d, null dim %d,"
              " largest eigenvalue %.3f, smallest nonzero %.5f (%.1e of largest)"
              % (link, len(w.facts), len(w.anchors), w.P, rep["rank"], rep["null_dim"],
                 ev[0], nz[-1], nz[-1] / ev[0]))
    print("  the eigenvalues depend only on the facts, so they are identical in every world;"
          " the link adds one mode %.1f times below the next smallest"
          % (np.sort(nz)[1] / np.sort(nz)[0]))

    import fig3
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
    from check_clock import fit

    seeds, dropped, have = fig3.world_set("f3_lr0p003")
    print("  %d worlds%s" % (len(seeds), "" if not dropped else ", dropped %s" % dropped))
    for d in DEPTHS:
        recs = [json.load(open(have[d][s])) for s in seeds]
        n_items = recs[0]["n_pairs"]
        sc = fig3.scale_of(recs[0])
        win = fig3.WINDOW * sc
        cnt = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        geo = {(a, s): [] for a in ("hold", "insert") for s in ("net", "pred")}
        spans, alphas = {"net": [], "pred": []}, []
        for r in recs:
            for (a, s) in cnt:
                ep, h = f3_curve(r, a, s, "hits")
                cnt[a, s].append(h.sum(axis=1).astype(float))
                geo[a, s].append(f3_curve(r, a, s, "geo")[1])
            for s in ("net", "pred"):
                ep, h = f3_curve(r, "insert", s, "hits")
                todo = ~h[0]
                t = unlocked(ep, h[:, todo])
                t = t[np.isfinite(t)]
                if len(t) >= 5:
                    spans[s].append(np.percentile(t, [10, 50, 90]))
            e, v = f3_curve(r, "insert", "net", "geo")
            ep_p, q = f3_curve(r, "insert", "pred", "geo")
            e, ep_p = e / sc, ep_p / sc
            m = (e >= 100) & (e <= 3000)
            alphas.append(fit(e[m], v[m], ep_p, q)[0])
        ep = f3_curve(recs[0], "hold", "net", "hits")[0]
        in_win = ep <= win
        k_win = int(np.argmin(np.abs(ep - win)))
        every = int(ep[1] - ep[0])
        C = {k: np.array(v) for k, v in cnt.items()}
        G = {k: np.array(v) for k, v in geo.items()}
        print("  N=%d  (switch at %d, %d epochs after, evaluated every %d; plotted window %d)"
              % (d, recs[0]["t_switch"], ep[-1], every, win))
        print("   effect")
        print("    %-34s both arms %s of %d" % ("correct at the linking fact",
                                                  msem(C["hold", "net"][:, 0]), n_items))
        paired("correct at %d epochs, link vs none" % win,
               C["insert", "net"][:, k_win], C["hold", "net"][:, k_win])
        gain = C["insert", "net"][:, k_win] - C["insert", "net"][:, 0]
        print("    %-34s %s (min %d, max %d); all %d correct in %d of %d worlds"
              % ("new compositions with the link", msem(gain), gain.min(), gain.max(),
                 n_items, int((C["insert", "net"][:, k_win] == n_items).sum()), len(gain)))
        change("correct without the link, 0-%d" % win,
               C["hold", "net"][:, 0], C["hold", "net"][:, k_win])
        gctl = C["hold", "net"][:, k_win] - C["hold", "net"][:, 0]
        print("    %-34s %s (min %d, max %d)" % ("new compositions without it", msem(gctl),
                                               gctl.min(), gctl.max()))
        ratio = G["hold", "net"][:, k_win] / G["insert", "net"][:, k_win]
        print("    %-34s median %.0f-fold lower with the link at %d epochs (IQR %.0f-%.0f)"
              % ("geometric error", np.median(ratio), win, np.percentile(ratio, 25),
                 np.percentile(ratio, 75)))
        for s in ("net", "pred"):
            q = np.array(spans[s])
            print("    %-34s median over worlds: 10%% at %.0f, 50%% at %.0f, 90%% at %.0f epochs"
                  % ("item resolution times, %s" % s, np.median(q[:, 0]), np.median(q[:, 1]),
                     np.median(q[:, 2])))
        print("   prediction")
        for a, lab in (("insert", "link"), ("hold", "none")):
            agreement("count curve, %s, 0-%d" % (lab, win),
                      C[a, "net"][:, in_win], C[a, "pred"][:, in_win])
            agreement("geometric-error curve, %s, 0-%d" % (lab, win),
                      G[a, "net"][:, in_win], G[a, "pred"][:, in_win], log=True)
        for frac in (0.5, 0.9):
            level = C["insert", "net"][:, 0].mean() + frac * (
                n_items - C["insert", "net"][:, 0].mean())
            tn = [first_reach(ep, v, level) for v in C["insert", "net"]]
            tp = [first_reach(ep, v, level) for v in C["insert", "pred"]]
            times_agree("epochs to %d%% of the gain, link" % int(100 * frac), tn, tp, every)
        al = np.array(alphas)
        print("    %-34s median %.3f (IQR %.3f-%.3f, min %.3f); 1 = same clock"
              % ("time rescaling alpha, link", np.median(al), np.percentile(al, 25),
                 np.percentile(al, 75), al.min()))


if __name__ == "__main__":
    which = sys.argv[1:] or ["f1", "f2", "f3"]
    for name in which:
        {"f1": figure1, "f2": figure2, "f3": figure3}[name]()
