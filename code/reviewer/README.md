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
| `r2_q3_rehearsal.py` | Reviewer 2, Q3/W5: inference and premise retention after the link under full, 25%, anchors + link, and link-only rehearsal (plus link-only at a higher rate) | `results/f3/controls_w*_d*.npz` (pretrained weights), `controls_pred_*.npz` | about 1 core-minute per world and depth |
| `r2_q5_partial.py` | Reviewer 2, Q5/W4: three lattices, two free translations, one link; resolved vs unresolved queries, theory vs network | new worlds built in the script | about 1 core-minute per world and depth |

The last two run new training. They store one file per world and depth in
`results/reviewer_outputs/<name>/cells/`, skip cells that already exist, and
then summarize all cells. Typical use:

    python code/reviewer/r2_q3_rehearsal.py --worlds 0-49 --depths 1-3 --nproc 4
    python code/reviewer/r2_q5_partial.py --worlds 0-49 --depths 1-3 --nproc 4
    python code/reviewer/r2_q5_partial.py --summarize      # tables from existing cells only

`r2_q3_rehearsal.py` also checks, on `--verify-worlds` (default 0-4), that its
full-rehearsal condition reproduces the stored network and prediction exactly.
`r2_q3_rehearsal.py` takes its learning rate and schedule from the stored
records (the paper's settings). `r2_q5_partial.py` uses the current
`relspec.worlds` Exp. 3 defaults (`lr_target` 0.03 and the shorter
F3_SWITCH / F3_AFTER schedule), not the paper's 0.003.

To add an analysis:
1. Write `<reviewer>_<question>_<topic>.py` with a `main()` that ends in `common.write_outputs(...)`.
2. Register it in `ANALYSES` in `run_reviewer.py`.
