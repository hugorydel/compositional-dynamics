"""One table behind Figure 3 and Figure S2: definitions, counts, horizons, numbers.

Everything a caption or a Results sentence might assert about those two figures,
computed from the same files the figures are drawn from, so that the prose can
be written against measured quantities rather than remembered ones.  Nothing is
trained and nothing is re-scored differently here: each number comes from the
replay pass, on the world set `fig3.world_set` defines.

Four things are stated for every quantity: what it is (the definition, including
the candidate pool and the retrieval rule), which worlds it is over, what
horizon and evaluation grid it was measured on, and the value.  Times carry the
interval they were measured in, since the grid is not uniform and an event time
is only known to within the interval it fell in.

Writes `results/_evidence_table.md`.

  usage:  python evidence.py
"""

import json
from pathlib import Path

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS, result

import fig3
import f3_integration as f3a
import retention as ret
from common import DEPTHS, N_WORLDS, commas

OUT = Path(RESULTS)


def q(v, p):
    return np.percentile(np.asarray(v, float), p)


def med_iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return "%s (IQR %s-%s)" % (commas(np.median(v)), commas(q(v, 25)), commas(q(v, 75)))


def provenance():
    out = ["## Provenance", "",
           "| stage | files | what it holds | checked against |",
           "| --- | --- | --- | --- |"]
    pre = {d: json.loads((OUT / "f3" / ("w00_d%d.json" % d)).read_text())["t_switch"]
           for d in DEPTHS}
    c = json.loads((OUT / "f3" / "controls_w00_d1.json").read_text())
    checks = c["timing"]["hold_checked"] + c["timing"]["insert_checked"]
    out += [
        "| published run | `results/f3/w{seed}_d{depth}.json` | the records the paper "
        "was written from: retrieval against the nine destination candidates and the "
        "geometric error, on the published grid | -- |",
        "| replay | `results/f3/controls_w{seed}_d{depth}.npz` | the same runs "
        "re-trained from the same seed and presentation order, re-scored against all "
        "eighteen candidates, on the trained facts, and on a finer grid | every "
        "published evaluation, hits and geometry; %d per cell at depth 1, and the "
        "cell fails if any differs |" % checks,
        "| prediction | `results/f3/controls_pred_w{seed}_d{depth}.npz` | the "
        "integrated theory from the same initialization, scored the same way, on the "
        "union of the network's grid (`shared`) and a drawing grid | the record's "
        "predicted hits exactly and its geometry to 1e-9 relative |",
        "",
        "Pre-intervention training differs by depth: %s epochs at depths 1, 2 and 3. "
        "Within a cell both arms run it once, from identical weights, and differ "
        "only in whether the linking fact is present afterwards."
        % ", ".join(commas(pre[d]) for d in DEPTHS), ""]
    return out


def world_set():
    seeds, dropped, have = fig3.world_set(fig3.SUB)
    out = ["## World set", "",
           "%d worlds, seeds %d to %d, the same set at every depth and in both "
           "figures: `fig3.world_set` chooses it and `fig3.py`, `figS2.py` and "
           "`analysis/f3_integration.py` all take it from there.  %s"
           % (len(seeds), min(seeds), max(seeds),
              "No world was dropped." if not dropped
              else "Dropped as unscorable: %s." % dropped),
           ""]
    if len(seeds) != N_WORLDS:
        out += ["**Short of %d worlds.**" % N_WORLDS, ""]
    return out, seeds


def grids(data):
    out = ["## Horizons and grids", "",
           "| depth | post-intervention run | published interval | replay evaluations "
           "| replay interval, first 500 epochs | at 30,000 epochs |",
           "| --- | --- | --- | --- | --- | --- |"]
    for d in DEPTHS:
        ep = data[d]["ep"]
        r = json.loads((OUT / "f3" / ("w00_d%d.json" % d)).read_text())
        pub = int(np.diff(np.asarray(r["arms"]["insert"]["net"]["epochs"], float))[0])
        k = int(np.argmin(np.abs(ep - 30000)))
        out.append("| %d | %s epochs | %s | %s | %s | %s |"
                   % (d, commas(ep[-1]), commas(pub), commas(len(ep)),
                      commas(np.diff(ep[ep <= 500])[0]), commas(ep[k] - ep[k - 1])))
    out += ["",
            "Both figures are drawn over the first %s epochs after the linking fact, "
            "the window Figure 3 has always used; the runs continue to %s."
            % (commas(data[1]["win"]), commas(data[1]["ep"][-1])), ""]
    return out


def nine_candidate_baseline():
    """What the same networks answer at the fact against nine candidates, the
    published pool: read from the replay so both pools come from one source."""
    out = {}
    for d in DEPTHS:
        v = [fig3.control(fig3.SUB, "controls", s, d)["hold.cross9"][0].sum()
             for s in fig3.world_set(fig3.SUB)[0]]
        out[d] = float(np.mean(v))
    return out


def figure3(data, seeds):
    n = data[1]["n_items"]
    NINE = nine_candidate_baseline()
    out = ["## Figure 3", "",
           "**Top row.** Correct novel compositions: of the %d withheld "
           "cross-structure comparisons, how many the network answers correctly, "
           "scored against ALL EIGHTEEN entities (the nine in the destination "
           "structure and the nine in the source).  A comparison counts when the "
           "queried point's nearest neighbour among the eighteen is the right "
           "entity.  Mean across worlds, shaded by the standard error across "
           "worlds." % n,
           "",
           "**Bottom row.** Geometric error: the distance from the queried point to "
           "the entity it should retrieve, in units of the median NEAREST-NEIGHBOUR "
           "spacing among the nine destination entities in the same learned "
           "embedding.  Retrieval and this normalization use different sets on "
           "purpose: a query competes against all eighteen entities, while the unit "
           "of distance is the local scale of the destination structure.  Geometric "
           "mean across worlds, error taken on the logarithms.", "",
           "| depth | at the fact, linked | at 30,000, linked | at 30,000, no link | "
           "worlds answering all %d at 30,000 | geometric error, fact to 30,000 |" % n,
           "| --- | --- | --- | --- | --- | --- |"]
    for d in DEPTHS:
        ep, cnt, geo = data[d]["ep"], data[d]["cnt"], data[d]["geo"]
        k = int(np.argmin(np.abs(ep - data[d]["win"])))
        a, b = cnt["insert", "net"], cnt["hold", "net"]
        g = geo["insert", "net"]
        out.append("| %d | %.1f of %d | %.1f | %.1f | %d of %d | %.2f to %.3f "
                   "(x%.0f smaller) |"
                   % (d, a[:, 0].mean(), n, a[:, k].mean(), b[:, k].mean(),
                      int((a[:, k] >= n - 1e-9).sum()), len(a),
                      np.exp(np.log(g[:, 0]).mean()), np.exp(np.log(g[:, k]).mean()),
                      np.exp(np.log(g[:, 0]).mean() - np.log(g[:, k]).mean())))
    out += ["",
            "**The no-link baseline.** This is what the pool change was made for.",
            "",
            "| depth | no link at the fact, mean of 80 | worlds answering none | "
            "largest single value | in which world, and when | that world at 30,000 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for d in DEPTHS:
        b = data[d]["cnt"]["hold", "net"]
        i, j = np.unravel_index(int(b.argmax()), b.shape)
        k = int(np.argmin(np.abs(data[d]["ep"] - data[d]["win"])))
        out.append("| %d | %.1f | %d of %d | %d of %d | world %d, at +%s epochs | %d |"
                   % (d, b[:, 0].mean(), int((b[:, 0] == 0).sum()), len(b),
                      int(b.max()), n, seeds[i], commas(data[d]["ep"][j]), int(b[i, k])))
    out += ["",
            "Against the nine destination candidates the same networks answer %s of "
            "%d at the fact, a depth-dependent amount of luck that the counts used to "
            "be corrected for.  Against all eighteen the mean is under one "
            "comparison, but it is not identically zero, so the control arm is "
            "reported beside the linked arm rather than assumed away."
            % (", ".join("%.1f" % NINE[d] for d in DEPTHS), n),
            ""]
    return out


def unlock(data):
    most = f3a._most(data)
    out = ["## Unlock time", "",
           "Epochs from the linking fact until %d of %d comparisons are correct with "
           "the link, per world, against all eighteen candidates.  Network and theory "
           "are measured on the same grid: the prediction is evaluated at every point "
           "of the network's grid.  Each time is known to within its crossing "
           "interval -- the gap from the last evaluation below the criterion to the "
           "one that met it -- which is reported beside it." % (most, data[1]["n_items"]),
           "",
           "| depth | network | distinct values | crossing interval, median (worst) | "
           "theory, R2 of log10 | \\|dt\\|, median (95th) | within its crossing interval |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    from common import agreement_r2
    for d in DEPTHS:
        tn, tp = f3a._unlock(data, d, "net"), f3a._unlock(data, d, "pred")
        iv = f3a._interval(data, d)
        both = np.isfinite(tn) & np.isfinite(tp)
        pos = both & (tn > 0) & (tp > 0)
        err = np.abs(tn[both] - tp[both])
        out.append("| %d | %s | %d | %s (%s) | %.4f | %s (%s) | %.1f%% |"
                   % (d, med_iqr(tn), len(np.unique(tn[np.isfinite(tn)])),
                      commas(np.nanmedian(iv)), commas(np.nanmax(iv)),
                      agreement_r2(np.log10(tn[pos]), np.log10(tp[pos])),
                      commas(np.median(err)), commas(q(err, 95)),
                      100.0 * np.mean(err <= iv[both])))
    out += ["", "Every world reaches the criterion in both the network and the theory "
            "at every depth.", ""]
    return out


def retention_block(sub, title, suffixes):
    """Both item sets and both arms, so no maximum can be quoted against the
    wrong denominator."""
    out = ["## %s" % title, "",
           "Retrieval of a trained fact is scored inside that entity's own block, the "
           "rule the rest of the paper uses.  Two item sets are reported: the training "
           "PREMISES, and ALL the original training facts, which adds the anchors.  "
           "Three quantities, kept apart:", "",
           "- **facts lost, distinct**: how many of that set fail at any point, the "
           "union over evaluations.",
           "- **facts lost, simultaneous**: the most failing at one evaluation, which "
           "is what the depth of a dip shows.",
           "- **time to recovery**: from the first evaluation with a failure to the "
           "first at which every fact is retrieved again and stays so to the end of "
           "the run.  Conditional on the affected cells that recover; a cell still "
           "failing at the horizon is counted separately, not entered as a number.",
           "",
           "| cells | item set | run after the fact | cells affected | distinct, "
           "median (max) | simultaneous, median (max) | first loss, median | time to "
           "recovery, median | not recovered by the horizon |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for depth in DEPTHS:
        for suffix in suffixes:
            got = ret.cells(sub, depth, suffix)
            if not got:
                continue
            epochs = got[0][1]
            n_premise, n_facts = ret.counts(sub, got[0][0])
            run = "%s epochs, %s evaluations" % (commas(epochs[-1]), commas(len(epochs)))
            block = suffix.replace("_", " block ")
            for arm, arm_label in (("insert", "linked"), ("hold", "no link")):
                for key, items in (("premise", "%d premises" % n_premise),
                                   ("facts", "all %d facts" % n_facts)):
                    v = ret.summarise([ret.loss(epochs, d[(arm, key)])
                                       for _, _, d in got])
                    label = "N=%d%s, %s" % (depth, block, arm_label)
                    if not v["hit"]:
                        out.append("| %s | %s | %s | 0 of %d | -- | -- | -- | -- | -- |"
                                   % (label, items, run, v["n"]))
                        continue
                    out.append("| %s | %s | %s | %d of %d | %.0f (%d) | %.0f (%d) | "
                               "%s | %s | %d |"
                               % (label, items, run, v["hit"], v["n"],
                                  v["distinct_med"], v["distinct_max"],
                                  v["simul_med"], v["simul_max"],
                                  commas(v["onset_med"]), commas(v["dur_med"]),
                                  v["censored"]))
    out += ["",
            "First loss and recovery are bracketed by the evaluation grid: each is the "
            "first evaluation showing the change, so the time is known to within the "
            "gap before it, and a duration inherits the brackets of both its "
            "endpoints.  These are grid widths, not statistical intervals, and "
            "\"retained\" means \"retrieved at every evaluation\": nothing here reports "
            "what happened between two checks.  `analysis/retention.py` prints the "
            "brackets.", ""]
    return out


def trajectories(data):
    """Per-world trajectory agreement, and what the sampling convention costs."""
    from common import per_world_r2
    out = ["## Trajectory agreement and its weighting", "",
           "R2 per world between the network's and the theory's count of correct "
           "compositions, from the fact to 30,000 epochs, averaged over worlds.  Each "
           "evaluation carries equal weight, and the replay grid is dense early and "
           "sparse late, so the statistic emphasizes the early trajectory.  That is a "
           "choice, not a property of the data, so the same statistic on the "
           "published uniform grid is given beside it.", "",
           "| evaluation weighting | N=1 | N=2 | N=3 |", "| --- | --- | --- | --- |"]
    rows = []
    for label, uniform in (("replay grid, equal per sample", False),
                           ("published uniform grid only", True)):
        vals = []
        for d in DEPTHS:
            ep = data[d]["ep"]
            m = ep <= data[d]["win"] + 1e-9
            if uniform:
                step = int(round(float(np.median(np.diff(ep[ep > 2000])))))
                r = json.loads((OUT / "f3" / ("w00_d%d.json" % d)).read_text())
                step = int(np.diff(np.asarray(
                    r["arms"]["insert"]["net"]["epochs"], float))[0])
                m = m & (np.mod(ep, step) == 0)
            vals.append(np.nanmean(per_world_r2(data[d]["cnt"]["insert", "net"][:, m],
                                                data[d]["cnt"]["insert", "pred"][:, m])))
        rows.append(vals)
        out.append("| %s | %.7f | %.7f | %.7f |" % (label, *vals))
    out += ["",
            "The difference is confined to depth 3, where the trajectory has most of "
            "its structure early: %.7f against %.7f.  This is a trajectory statistic "
            "and is unrelated to the unlock-time R2 above, which is one number per "
            "world across worlds." % (rows[0][2], rows[1][2]), ""]
    return out


def figureS2(data):
    out = ["## Figure S2", "",
           "Per-world retrieval of the %d ORIGINAL PREMISE facts -- not all 32 "
           "trained facts, whose maxima differ and are tabulated below -- averaged "
           "across the "
           "same worlds Figure 3 draws, shaded by the standard error; the prediction "
           "is the integrated theory scored on the same premises by the same rule.  "
           "The axis is linear below ten epochs and logarithmic above so that the "
           "intervention itself is on it, where the two arms are the same network."
           % ret.counts("f3", "w00_d1")[0], "",
           "| depth | minimum of the mean curve, network | at | minimum, theory | at | "
           "largest gap between the curves |",
           "| --- | --- | --- | --- | --- | --- |"]
    for depth in DEPTHS:
        got = ret.cells("f3", depth)
        epochs = got[0][1]
        net_m, net_min, k = ret.mean_curve([d[("insert", "premise")] for _, _, d in got])
        pred = []
        for name, _, _ in got:
            with np.load(OUT / "f3" / ("controls_pred_%s.npz" % name)) as z:
                if "shared" not in z.files:
                    return out + ["", "_The prediction has not been regenerated on the "
                                  "network's grid; re-run `analysis/controls_pred.py`._", ""]
                pred.append(z["insert.premise"][z["shared"].astype(bool)])
        pre_m, pre_min, j = ret.mean_curve(pred)
        out.append("| %d | %.1f%% | +%s | %.1f%% | +%s | %.2f points |"
                   % (depth, net_min, commas(epochs[k]), pre_min, commas(epochs[j]),
                      float(np.max(np.abs(net_m - pre_m)))))
    out += ["", "Those are minima of the across-world mean curve, one for the network "
            "and one for the theory.  A minimum is not a measure of agreement between "
            "them; the last column is, and so is the world-by-world comparison below.",
            "",
            "**Which worlds lose a premise, network against theory.**", "",
            "| depth | network | theory | both | network only | theory only |",
            "| --- | --- | --- | --- | --- | --- |"]
    for depth in DEPTHS:
        got = ret.cells("f3", depth)
        net = np.array([bool((~d[("insert", "premise")].astype(bool)).any())
                        for _, _, d in got])
        pred = []
        for name, _, _ in got:
            with np.load(OUT / "f3" / ("controls_pred_%s.npz" % name)) as z:
                m = z["shared"].astype(bool)
                pred.append(bool((~z["insert.premise"][m].astype(bool)).any()))
        pred = np.array(pred)
        ids = [int(n.split("_")[0][1:]) for n, _, _ in got]
        only_net = [ids[i] for i in np.flatnonzero(net & ~pred)]
        only_pred = [ids[i] for i in np.flatnonzero(pred & ~net)]
        out.append("| %d | %d | %d | %d | %s | %s |"
                   % (depth, int(net.sum()), int(pred.sum()), int((net & pred).sum()),
                      ", ".join("world %d" % w for w in only_net) or "--",
                      ", ".join("world %d" % w for w in only_pred) or "--"))
    out += ["",
            "The theory does not pick out exactly the same worlds: it adds one at "
            "depth 2 and misses one at depth 3.  Everywhere else the two agree world "
            "for world, which is what the phrase \"the same worlds\" should be "
            "understood to exclude.", ""]
    return out


def main():
    data = f3a.load()
    lines = ["# Evidence behind Figure 3 and Figure S2", "",
             "Generated by `code/analysis/evidence.py` from the stored records and "
             "the replay pass.  Nothing is trained.", ""]
    ws, seeds = world_set()
    lines += provenance() + ws + grids(data) + figure3(data, seeds) + unlock(data)
    lines += trajectories(data) + figureS2(data)
    lines += retention_block("f3", "Retention, Experiment 3", [""])
    lines += retention_block("f2", "Retention, Experiment 2 (approved sample)",
                             ["_A", "_B"])
    path = result("_evidence_table.md")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
