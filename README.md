# Toy unified-lattice model

Self-contained. Nothing here is imported by the paper's own experiments, and
nothing here writes outside this folder. The only outside dependency is the
paper's `relspec` package (`../code/relspec`) and the shared figure style
(`../code/analysis/style.py`), both located relative to this folder by
`code/_paths.py`, so `toy_model/` can be moved as long as it stays one level
below the paper root.

## Why it exists

To test a proposed replacement measure set before committing to it, rather than
assuming it works. One world carries all three experimental roles at once.

## The world

68 tokens, 54 entities, 14 relations, 83 facts. Three roles:

| role | construction | identifiability |
|---|---|---|
| E1 | two anchored 3x3 lattices sharing base relation `a`, composite shown on 1 of 4 cells | rho = 0 |
| E2 | one anchored lattice with the composite shown once, one with it never shown | rho = 0 and rho = 0.577 |
| E3 | an anchored lattice plus an unanchored copy reusing its relations `p, q, s` | rho = 0, but no cross-lattice comparison is determined |

`build(link=True)` adds the single fact that joins E3's two lattices. That fact
takes the null-space dimension from 2 to 1 and unlocks all 81 cross-lattice
comparisons.

## Layout

```
code/       everything needed to reproduce the results and figures
figures/    the four rendered outputs
results/    every JSON the code reads or writes
```

### code/

| file | role |
|---|---|
| `_paths.py` | locates `relspec`, `results/` and `figures/` relative to this folder |
| `toy.py` | the world. Run it directly for the structural report |
| `toy_score.py` | scores every measure on a sequence of embeddings. Used for BOTH the trained and the predicted trajectory, so the overlay cannot drift from the curve it overlays |
| `toy_run.py` | trains and collects. Calls `toy_theory.ensure` for each cell as it finishes |
| `toy_theory.py` | the prospective prediction: integrates the mean dynamics from the same initialisation, on the same epoch grid, with no fitted parameters |
| `toy_common.py` | shared loading and styling for the figures |
| `toy_fig1.py` `toy_fig2.py` `toy_fig3.py` | the three figures |
| `toy_diag.py` | the diagnostic that located the theory-to-network gap |

## Reproducing

From `code/`:

```bash
python toy_run.py
```

That retrains all six cells and regenerates every prediction, roughly 15
minutes. Depth 1 dominates: it needs 120k epochs to reach the cross-lattice
error depth 2 reaches in 4k, because depth preconditions the slowest mode and
the unanchored copy is that mode.

Then, for the figures:

```bash
python toy_fig1.py
python toy_fig2.py
python toy_fig3.py
```

The figure scripts read `results/` only. They never retrain, and if a
prediction file is missing they compute and cache it on first use.

The diagnostic is separate and slow, roughly 12 minutes:

```bash
python toy_diag.py
```

## What the runs established

**The held-out compositional accuracy measure is broken as specified.** It
reaches 100% for a law whose composite is never shown and is provably
non-identifiable. Withholding the composite *fact* does not withhold the
composite: the pair's endpoints stay joined by an observed x-step then y-step,
so the query point is pinned by training facts either way. Figure 1.

**The geometric error must not be normalised by the composite's own length.**
When the composite is never shown, `z` lies in the null space and its length is
free, so `||z-(x+y)|| / ||z||` runs to 8 through 120 and cannot share an axis
with the identifiable law's 1e-4. Dividing by `||x+y||`, which the observed
facts do determine, bounds it, and 1.0 reads as "entirely wrong". Figure 2.

**The cross-lattice measures work.** Without the linking fact, 0 of 81
inferences ever resolve and the error sits at 4.45 forever. With it, all 81
resolve and the error falls to 0.009 at depth 1. Figure 3.

**The distance ordering does not.** Spearman correlation between true
cross-lattice distance and resolution time flips sign across depth, -0.391 at
depth 1 and +0.494 at depth 3, so it is not measuring a property of the world.
Replaced by the cumulative unlock curve.

**The theory-to-network gap is a step size problem, not a modelling one.** At
the frozen `lr_target = 0.3`, full-batch descent diverges outright at depth 2
and 3, overflowing by epoch 40. The network is training past the stability edge
of the very process the prediction integrates, and per-fact stochastic descent
survives only because each update is a fraction of an epoch-averaged step. At
0.03 the median absolute log ratio between network and prediction drops from
0.074 to 0.003 at depth 2 and from 0.123 to 0.001 at depth 3, using the same
integrator settings. Figure `toy_diag.png`.

## Known limitations

- Three worlds only, and 3 to 6 held-out composites per law, so the accuracy
  curves in Figures 1 and 2 render as steps rather than sigmoids. That is a
  resolution limit, not a finding.
- Every cell runs at `lr_target = 0.3`, which the diagnostic shows is past the
  stability edge at depth. The figures are honest about what the network does;
  the prediction overlay is the part that suffers.
- The rows of Figures 2 and 3, and the columns of Figure 3, use different epoch
  ranges. Each panel is labelled, but they are not comparable epoch for epoch.
