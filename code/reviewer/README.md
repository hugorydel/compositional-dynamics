# Reviewer analyses

Analyses written in response to the ICLR 2027 reviews. They read the stored
records and the repository's own code; none of them changes `relspec/`,
the experiment runners, or any stored result.

Each analysis is one script, `<reviewer>_<question>_<topic>.py`, and writes
only to `results/reviewer_outputs/<same name>/`:

- `report.md`: the numbers, under the reviewer's wording
- `summary.json`: the same numbers, machine-readable
- `cells.csv`: one row per world (and depth, arm, ...)

Run everything with `python code/reviewer/run_reviewer.py`, or one script on its own.

| Script | Question | Reads | Time |
|---|---|---|---|
| `r2_q4_construction.py` | Reviewer 2, Q4: disjoint tokens, null space, rho = 1/sqrt(15), two consistent solutions, effect of the link | rebuilt Exp. 2 worlds | seconds |
| `r2_q4_selection.py` | Reviewer 2, Q4: non-identifiability vs optimizer selection | `results/f2/controls_w*_d*_[AB].npz` | about 1 min |
| `r2_q2_variance.py` | Reviewer 2, Q2: between-law vs within-law agreement | `results/_timing_summary.npz` | under 1 s |

To add an analysis:
1. Write `<reviewer>_<question>_<topic>.py` with a `main()` that ends in `common.write_outputs(...)`.
2. Register it in `ANALYSES` in `run_reviewer.py`.
