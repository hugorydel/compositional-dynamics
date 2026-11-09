"""Redraw every figure and regenerate every reported number, from the records.

Nothing here trains a network.  The repository ships the record each run wrote,
so the whole paper rebuilds from them in a few minutes; producing the records
themselves is a long parallel run, and README.md says how.

Each step is a script in `code/`, run exactly as it would be run by hand, with
its own output left on screen.  The figures print the numbers they draw, so a
disagreement with the paper is visible while it happens rather than afterwards.

  usage:  python reproduce.py             check, then figures, then analyses
          python reproduce.py check       report what is on disk, draw nothing
          python reproduce.py figures     the schematics, the theory figure, the results figures
          python reproduce.py analyses    the statistics and the evidence table
"""

import glob
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE, RESULTS = ROOT / "code", ROOT / "results"

DEPTHS = (1, 2, 3)

# What each experiment needs on disk and what writes it: the file pattern, how
# many worlds it covers, how many files one world contributes (two for the
# counterbalanced arms of Experiment 2), and the command that fills the gaps.
# Spare worlds beyond these counts change nothing: every figure takes its world
# set from `fig3.world_set` rather than from whatever files it happens to find.
RECORDS = (
    ("Experiment 1  training runs", "f1/w*_d%d.json", 200, 1,
     "run_experiments.py run {seeds} --only f1"),
    ("Experiment 2  training runs", "f2/w*_d%d_?.json", 200, 2,
     "run_experiments.py run {seeds} --only f2"),
    ("Experiment 3  training runs", "f3/w*_d%d.json", 200, 1,
     "run_experiments.py run {seeds} --only f3"),
    ("Experiment 3  network replay", "f3/controls_w*_d%d.npz", 200, 1,
     "analysis/controls.py --cells f3:0-199:{d} --nproc 12"),
    ("Experiment 3  theory, same grid", "f3/controls_pred_w*_d%d.npz", 200, 1,
     "analysis/controls_pred.py --cells f3:0-199:{d} --nproc 12"),
    ("Experiment 2  retention replay", "f2/controls_w*_d%d_?.npz", 30, 2,
     "analysis/controls.py --cells f2:0-29:{d}:A f2:0-29:{d}:B --nproc 12"),
)

FIGURES = (
    ("schematic_world.py", "the relational world of Section 3.1"),
    ("schematic_experiments.py", "the three experimental designs"),
    ("spectral_modes.py", "a fast and a slow law's spectral support, Section 4.3"),
    ("fig1.py", "emergence times differ between laws"),
    ("fig2.py", "identifiability decides whether a law can emerge"),
    ("fig3.py", "one linking fact unlocks many inferences"),
    ("figS1.py", "predicted against observed, all sixteen laws"),
    ("figS2.py", "the trained facts while the structure is remodelled"),
    ("figS3.py", "the geometric error behind Figures 1 to 3"),
)

ANALYSES = (
    ("analysis/run_all.py", "every statistic the Results section quotes"),
    ("analysis/retention.py", "what was lost and recovered, both experiments"),
    ("analysis/evidence.py", "the table behind Figures 3 and S2"),
)


def seeds_present(pattern, per_world):
    """Which world seeds this pattern covers completely.

    The directory is escaped, not the pattern: a checkout can sit under a path
    containing brackets, which glob would otherwise read as a character class
    and match nothing.
    """
    seen = {}
    for path in glob.glob(glob.escape(str(RESULTS)) + "/" + pattern):
        seed = int(re.search(r"w(\d+)_d", Path(path).name).group(1))
        seen[seed] = seen.get(seed, 0) + 1
    return {s for s, n in seen.items() if n >= per_world}


def check():
    """Print the record inventory; return whether everything needed is there."""
    print("Records in %s\n" % RESULTS)
    print("  %-34s %6s %6s %6s    %s" % ("", "N=1", "N=2", "N=3", "worlds needed"))
    gaps = []
    for name, pattern, worlds, per_world, command in RECORDS:
        have = [seeds_present(pattern % d, per_world) for d in DEPTHS]
        short = [(d, sorted(set(range(worlds)) - h)) for d, h in zip(DEPTHS, have)
                 if len(h & set(range(worlds))) < worlds]
        print("  %-34s %6d %6d %6d    %d%s"
              % ((name,) + tuple(len(h) for h in have)
                 + (worlds, "" if not short else "   INCOMPLETE")))
        gaps += [(name, command, d, miss) for d, miss in short]
    if not gaps:
        print("\nComplete: every record the figures and the analyses read.")
        return True
    print("\nMissing records.  A replay command may be run in full, since it "
          "skips\nthe cells already on disk; a training command lists only the "
          "worlds it needs.")
    for name, command, depth, miss in gaps:
        shown = " ".join(str(s) for s in miss[:8]) + (" ..." if len(miss) > 8 else "")
        print("\n  %s, N=%d: %d worlds missing" % (name, depth, len(miss)))
        print("  python " + command.format(seeds=shown, d=depth))
    return False


def run(script):
    """One step, with its own output, timed."""
    start = time.time()
    code = subprocess.run([sys.executable] + script.split(), cwd=CODE).returncode
    if code:
        raise SystemExit("\n%s exited %d" % (script, code))
    print("  [%s: %.0f s]\n" % (script, time.time() - start), flush=True)


def steps(title, items):
    print("\n=== %s ===\n" % title)
    for script, what in items:
        print("--- %-22s %s" % (script, what), flush=True)
        run(script)


def main(argv):
    what = argv[0] if argv else "all"
    if what not in ("all", "check", "figures", "analyses"):
        raise SystemExit(__doc__)
    if not check() and what != "check":
        raise SystemExit("\nNothing drawn: the records above are incomplete.")
    started = time.time()
    if what in ("all", "figures"):
        steps("Figures, into figures/", FIGURES)
    if what in ("all", "analyses"):
        steps("Analyses, into results/", ANALYSES)
    if what != "check":
        print("Done in %.1f min." % ((time.time() - started) / 60))


if __name__ == "__main__":
    main(sys.argv[1:])
