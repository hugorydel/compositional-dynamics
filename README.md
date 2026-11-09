# Compositional dynamics

Code, records and figures for **A Spectral Theory of Compositional Learning**.

Deep linear relational embeddings (`E = W_N ... W_1`, at depths N = 1, 2, 3) are
trained on synthetic relational worlds whose compositional laws are known
exactly, and each run is compared against a parameter-free theory integrated
from the same initialisation. Nothing is fitted to an observed trajectory.

| experiment | the question | figures |
| --- | --- | --- |
| 1 emergence | does the spectral structure supporting a law predict when that law becomes available? | 1, S1 |
| 2 identifiability | can a law emerge while the evidence leaves it undetermined, and what does one bridging fact change? | 2 |
| 3 integration | what does a single fact linking two separately learned structures unlock? | 3, S2 |

Every experiment is 200 worlds per depth, seeds 0 to 199, and every world is
measured against its own prediction.

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py
```

Nine figures into `figures/` (two schematics, one theory figure and six results), every reported
statistic into `results/`, four and a half minutes on one core here. Nothing is
trained: the repository carries the record each run wrote, and these scripts
read them.

```bash
python reproduce.py check       # what is on disk, draw nothing
python reproduce.py figures     # schematics and results      (~3 min)
python reproduce.py analyses    # reports and evidence table  (~1.5 min)
```

Python 3.12 with numpy, scipy and matplotlib; no GPU and nothing else. Last run
here on 3.12.8 with numpy 2.5, scipy 1.18 and matplotlib 3.11. Redrawing
the figures is deterministic: every PNG comes back byte-identical, and
`figS1_timing.pdf` identical apart from the creation date matplotlib embeds in
a PDF. On 26 September 2026 Figures 1 to 3 became one row each (the behavioural
measure), Figure 2 switched to plain accuracy, and their geometric rows moved to
the new appendix Figure S3 (`figS3.py`); until `manuscript/figures/` is synced,
those four differ from the copies the manuscript includes. The two schematics are drawn from the
world constructors alone and write their PDFs without a creation date, so every
file they produce comes back byte-identical.

## What is where

| path | what it holds |
| --- | --- |
| `code/fig1.py` … `figS3.py` | one script per result figure; each prints the numbers it draws. `figS3.py` stacks the geometric rows of Figures 1 to 3, drawn by those figures' own code |
| `code/schematic_world.py`, `schematic_experiments.py` | the two explanatory schematics, drawn from `relspec`'s world constructors and checked against them; no records needed |
| `code/spectral_modes.py` | the Section 4.3 figure: two laws' weights on the spectral modes of world 0, and the stored prediction of their geometric error |
| `code/run_experiments.py` | trains the cells, one file per (experiment, world, depth), resumably |
| `code/relspec/` | the worlds, the model, the training loop and the theory |
| `code/analysis/` | the reported statistics (`run_all.py`), the replay passes (`controls.py`, `controls_pred.py`), retention (`retention.py`), the evidence table (`evidence.py`) |
| `code/stats.py` | a per-figure console summary, including each world's structural description |
| `code/tests/` | the checks the design decisions rest on, one script per decision |
| `results/f1`, `f2`, `f3` | one JSON record per cell, and the replay `.npz` beside it |
| `results/_analysis_*.md`, `_evidence_table.md` | generated: every number the paper quotes, and where it comes from |
| `figures/` | what the figure scripts write, and what the manuscript includes |

A record holds the evaluation grid and, at every point on it, held-out retrieval,
per-item hits, rank and geometric error, for the network and for the prediction
alike. `results/` is 3.8 GB, so a clone is large.

The paper's LaTeX sources are not in this repository; they sit beside it in
`manuscript/`, with their own README.

## What the records show

- **Emergence times are predicted law by law.** Over 3,200 law-worlds per depth,
  network and prediction agree on whether a law emerges in every case, and the R²
  of log₁₀ emergence epoch is 1.0000, 0.9979 and 0.9793 at N = 1, 2, 3, with a
  median absolute error of 0 epochs.
- **Identifiability, not exposure, decides whether composition appears.** Two
  thousand epochs after one bridging fact, held-out accuracy is 77.4% against
  10.9% without it at N = 1, and 100.0% against 3.2% and 2.6% at N = 2 and 3.
- **One linking fact unlocks many inferences at once.** Of 80 withheld
  cross-structure comparisons, networks at N = 3 answer 0.4 when the fact
  arrives and 80.0 of 80 thirty thousand epochs later, against 0.0 without it;
  the epoch at which they unlock is predicted with R² above 0.999 at every depth.

Every one of these comes from `results/_analysis_results.md`, which
`analysis/run_all.py` writes from the records; `results/_evidence_table.md`
adds the definitions, world sets and horizons behind Figures 3 and S2.

## Producing the records

`run_experiments.py` writes one file per cell and skips cells already on disk, so
a run can be stopped and restarted at any point.

```bash
python run_experiments.py plan 0 1 2              # what would run
python run_experiments.py run 0 1 2               # three worlds, all experiments
python run_experiments.py run 0 --only f1 --nproc 8
```

The published set is seeds 0 to 199: `run 0 1 2 ... 199`, which bash writes as
`run $(seq 0 199)` and PowerShell as `run @(0..199)`. That is 2,400 cells, each
running its epoch budget twice over — once as descent and once as the integrated
prediction — so it is a parallel run of many core-hours rather than an
afternoon, which is why the records ship with the repository. Two things are
worth knowing before starting one.

**Experiment 3 is not at the settings now in `relspec/worlds.py`.** Its stored
records were produced at `lr_target = 0.003` with every budget ten times longer
(the intervention at 200,000 epochs at N = 1) — the same amount of learning on a
ten times finer step, which is what `tests/f3_at_rate.py 0.003` runs, into
`results/f3_lr0p003/`. `run_experiments.py` carries the faster equivalent, so
`plan` reports every stored Experiment 3 cell as stale and a run would replace
it. Experiments 1 and 2 re-run at the settings their records were made under.

**The replay passes come after the training runs.** `analysis/controls.py`
re-trains each stored cell from its own seed and presentation order, checks every
published hit and geometric score against the record before adding anything to
it, and scores the same states against the full eighteen-entity candidate pool;
`analysis/controls_pred.py` does the same for the theory, on the same grid.
Figures 3 and S2 and the retention tables read those files, and
`python reproduce.py check` prints the command for anything missing.

## Conventions

- A cell is deterministic given its seed: the world, the initialisation
  (`init_seed = 1000 + seed`) and the presentation order all follow from it, so a
  recomputed cell is identical to the one it replaces.
- Figures average one world set for every depth, chosen by `fig3.world_set`,
  rather than whatever files happen to be on disk, so a difference between
  columns cannot be a difference in sample.
- `analysis/` reports estimates and intervals and no P values; `analysis/common.py`
  states that convention and the reason for it.
