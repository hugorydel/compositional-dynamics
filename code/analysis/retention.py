"""Retention of the trained facts, defined exactly, for Experiments 2 and 3.

Three quantities were being conflated in earlier summaries, so each is named
and computed separately here.

  facts lost          two different numbers.  DISTINCT is how many of the
                      world's trained facts fail at any point in the run, the
                      union over evaluations.  SIMULTANEOUS is the largest
                      number failing at one evaluation, which is what a dip in
                      a retention curve shows.  A world that loses one fact,
                      recovers it and then loses another has distinct 2 and
                      simultaneous 1.
  time to recovery    from the first evaluation with a failure to the first
                      evaluation at which every trained fact is retrieved again
                      AND stays retrieved to the end of the run.  Not the span
                      between the first and last failing evaluation, which
                      ignores how long the last failure took to clear.  It is
                      summarized over the affected worlds that recover; a world
                      still failing at the end of the run is censored, counted
                      separately as "not recovered by the horizon", and left out
                      of the median rather than entered as a large number.
  minimum retention   the lowest value of the across-world mean curve.  It is a
                      minimum of a curve, not an agreement with anything; the
                      network and the prediction each have one, and the
                      comparison between them is reported separately.

Both are grid-limited in the same way event times are: a loss is seen at the
first evaluation that shows it, and recovery at the first that does not, so
each time is bracketed by the two evaluations it fell between.  Those sampling
brackets are reported beside the numbers, and a duration inherits both of them,
its own and its onset's.  They are widths of the evaluation grid, not
statistical uncertainty: nothing here says what happened between two checks, and
"retained" everywhere means "retrieved at every evaluation".

Reads the replay pass only (`analysis/controls.py`, and for Experiment 3 also
`analysis/controls_pred.py`).  Nothing is trained here.

  usage:  python retention.py                both experiments
          python retention.py f3             one of them
"""

import json
import sys
from pathlib import Path

import _boot  # noqa: F401
import _paths  # noqa: F401
import numpy as np
from _paths import RESULTS

DEPTHS = (1, 2, 3)
KEYS = ("premise", "facts")


def counts(sub, name):
    """How many premises, and how many trained facts, the cell scored."""
    d = json.loads((Path(RESULTS) / sub / ("controls_%s.json" % name)).read_text())
    return d["n_premise"], d["n_facts"]


def cells(sub, depth, arm_suffix="", expect=None):
    """Every replayed cell at one depth: (name, epochs, {key: hits}).

    `expect` is the seed set the caller believes is there; a missing or extra
    seed is an error rather than a quietly different sample.  Every cell must
    also be on one grid, for the same reason.
    """
    out = []
    for path in sorted((Path(RESULTS) / sub).glob("controls_w*_d%d%s.npz"
                                                  % (depth, arm_suffix))):
        with np.load(path) as z:
            data = {(arm, key): z["%s.%s" % (arm, key)]
                    for arm in ("hold", "insert") for key in KEYS}
            out.append((path.stem.replace("controls_", ""), z["epochs"], data))
    # the glob comes back in string order, where w100 precedes w99, so compare
    # the sets rather than the sequences
    seeds = sorted(int(n.split("_")[0][1:]) for n, _, _ in out)
    if expect is not None and seeds != sorted(expect):
        missing = sorted(set(expect) - set(seeds))
        extra = sorted(set(seeds) - set(expect))
        raise SystemExit("%s depth %d%s: expected %d seeds, found %d%s%s"
                         % (sub, depth, arm_suffix, len(list(expect)), len(seeds),
                            "; missing %s" % missing if missing else "",
                            "; unexpected %s" % extra if extra else ""))
    for name, epochs, _ in out[1:]:
        if not np.array_equal(epochs, out[0][1]):
            raise SystemExit("%s is on a different grid from %s" % (name, out[0][0]))
    return out


def loss(epochs, hits):
    """Everything about one arm's retention in one world.

    `hits` is (evaluations x items) and True where the fact was retrieved.
    """
    bad = ~hits.astype(bool)
    any_bad = bad.any(axis=1)
    out = dict(distinct=int(bad.any(axis=0).sum()),
               simultaneous=int(bad.sum(axis=1).max()),
               onset=np.nan, recovery=np.nan, duration=np.nan,
               onset_interval=np.nan, recovery_interval=np.nan, recovered=True)
    if not any_bad.any():
        return out
    k = int(np.flatnonzero(any_bad)[0])
    last = int(np.flatnonzero(any_bad)[-1])
    out["onset"] = float(epochs[k])
    out["onset_interval"] = float(epochs[k] - epochs[k - 1]) if k else 0.0
    if last == len(any_bad) - 1:                      # still failing at the end
        out["recovered"] = False
        return out
    out["recovery"] = float(epochs[last + 1])
    out["recovery_interval"] = float(epochs[last + 1] - epochs[last])
    out["duration"] = out["recovery"] - out["onset"]
    return out


def summarise(rows):
    """Across-world summary of `loss` dictionaries, worlds as the unit.

    Loss magnitudes are conditional on the affected worlds; the recovery
    median is conditional on the affected worlds that recover, with the
    censored ones counted separately.
    """
    hit = [r for r in rows if r["distinct"]]
    if not hit:
        return dict(n=len(rows), hit=0)
    back = [r for r in hit if r["recovered"]]
    return dict(
        n=len(rows), hit=len(hit), recovered=len(back),
        distinct_med=np.median([r["distinct"] for r in hit]),
        distinct_max=max(r["distinct"] for r in hit),
        simul_med=np.median([r["simultaneous"] for r in hit]),
        simul_max=max(r["simultaneous"] for r in hit),
        onset_med=np.median([r["onset"] for r in hit]),
        onset_bracket=np.median([r["onset_interval"] for r in hit]),
        dur_med=np.median([r["duration"] for r in back]) if back else np.nan,
        dur_onset_bracket=np.median([r["onset_interval"] for r in back])
        if back else np.nan,
        dur_recovery_bracket=np.median([r["recovery_interval"] for r in back])
        if back else np.nan,
        censored=len(hit) - len(back))


def line(tag, s):
    """Three lines: how many worlds, how much they lost, and when."""
    if not s["hit"]:
        return "    %-22s no world loses a trained fact (%d cells)" % (tag, s["n"])
    out = ["    %-22s %3d of %3d cells affected" % (tag, s["hit"], s["n"]),
           "    %-22s affected cells: distinct facts lost median %.0f (max %d), "
           "most at once median %.0f (max %d)"
           % ("", s["distinct_med"], s["distinct_max"], s["simul_med"], s["simul_max"]),
           "    %-22s first loss seen at median %s epochs (sampling bracket %s)"
           % ("", "{:,.0f}".format(s["onset_med"]),
              "{:,.0f}".format(s["onset_bracket"]))]
    if s["recovered"]:
        out.append("    %-22s recovered %d of %d: median %s epochs from first loss to "
                   "sustained retrieval (brackets %s at onset, %s at recovery)"
                   % ("", s["recovered"], s["hit"], "{:,.0f}".format(s["dur_med"]),
                      "{:,.0f}".format(s["dur_onset_bracket"]),
                      "{:,.0f}".format(s["dur_recovery_bracket"])))
    if s["censored"]:
        out.append("    %-22s not recovered by the horizon: %d" % ("", s["censored"]))
    return "\n".join(out)


def mean_curve(rows_hits):
    """Across-world mean retention in per cent, and its minimum."""
    v = np.array([100.0 * h.mean(axis=1) for h in rows_hits])
    m = v.mean(axis=0)
    return m, float(m.min()), int(m.argmin())


def experiment_3():
    print("Experiment 3 (results/f3): 200 worlds per depth, the linked arm against "
          "its control")

    for depth in DEPTHS:
        got = cells("f3", depth, expect=range(200))
        if not got:
            print("  N=%d: no replay pass" % depth)
            continue
        epochs = got[0][1]
        n_premise, n_facts = counts("f3", got[0][0])
        print("  N=%d  %d worlds, %s evaluations to %s epochs after the fact, "
              "%d premises of %d trained facts"
              % (depth, len(got), "{:,}".format(len(epochs)),
                 "{:,.0f}".format(epochs[-1]), n_premise, n_facts))
        for key in KEYS:
            for arm in ("insert", "hold"):
                rows = [loss(epochs, d[(arm, key)]) for _, _, d in got]
                print(line("%s, %s" % (key, arm), summarise(rows)))
        # the prediction, on the same grid, scored the same way
        pred, stale = [], False
        for name, _, _ in got:
            with np.load(Path(RESULTS) / "f3" / ("controls_pred_%s.npz" % name)) as z:
                if "shared" not in z.files:
                    stale = True
                    break
                pred.append(z["insert.premise"][z["shared"].astype(bool)])
        if stale:
            print("    %-22s not on the network grid yet; re-run "
                  "analysis/controls_pred.py" % "prediction")
        else:
            rows = [loss(epochs, h) for h in pred]
            print(line("premise, prediction", summarise(rows)))
            net_m, net_min, k = mean_curve([d[("insert", "premise")] for _, _, d in got])
            pre_m, pre_min, j = mean_curve(pred)
            net_rows = [loss(epochs, d[("insert", "premise")]) for _, _, d in got]
            both = sum(1 for a, b in zip(net_rows, rows) if a["distinct"] and b["distinct"])
            print("    %-22s minimum of the mean curve: network %.1f%% at +%s, "
                  "prediction %.1f%% at +%s"
                  % ("network vs theory", net_min, "{:,.0f}".format(epochs[k]),
                     pre_min, "{:,.0f}".format(epochs[j])))
            print("    %-22s both lose at least one fact in %d of %d worlds; "
                  "largest gap between the two mean curves %.2f points"
                  % ("", both, len(rows), float(np.max(np.abs(net_m - pre_m)))))
        print()


def experiment_2():
    print("Experiment 2 (results/f2): the approved sample, both structural "
          "assignments, both arms")

    for depth in DEPTHS:
        for block in ("A", "B"):
            got = cells("f2", depth, "_" + block, expect=range(30))
            if not got:
                print("  N=%d block %s: no replay pass" % (depth, block))
                continue
            epochs = got[0][1]
            seeds = sorted(int(n.split("_")[0][1:]) for n, _, _ in got)
            n_premise, n_facts = counts("f2", got[0][0])
            print("  N=%d block %s  %d worlds (seeds %d-%d), %s evaluations to "
                  "%s epochs after the fact, %d premises of %d trained facts"
                  % (depth, block, len(got), seeds[0], seeds[-1],
                     "{:,}".format(len(epochs)), "{:,.0f}".format(epochs[-1]),
                     n_premise, n_facts))
            for key in KEYS:
                for arm in ("insert", "hold"):
                    rows = [loss(epochs, d[(arm, key)]) for _, _, d in got]
                    print(line("%s, %s" % (key, arm), summarise(rows)))
        print()


def main():
    which = sys.argv[1:] or ["f3", "f2"]
    if "f3" in which:
        experiment_3()
    if "f2" in which:
        experiment_2()


if __name__ == "__main__":
    main()
