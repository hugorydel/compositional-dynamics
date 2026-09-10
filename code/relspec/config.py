"""
Experimental settings.

Everything that could have been tuned lives here, in one object, so that a
reader can see the whole configuration of any result at a glance and so that
"what was configured" is a diff-able artefact rather than a set of constants
scattered across scripts.

Each experiment declares the `Settings` it uses (usually `DEFAULT`, sometimes
`dataclasses.replace(DEFAULT, ...)`) and records them alongside its results.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    # ---- representation -------------------------------------------------- #
    d: int = 16                       # embedding dimension
    init_scale_shallow: float = 1e-2  # sd of E(0) entries, N = 1
    init_scale_deep: float = 0.05     # sd of each W_l entry, N > 1
    init_seed: int = 1000             # RNG seed for the initialisation

    # ---- optimisation ---------------------------------------------------- #
    lr_target: float = 0.3            # learning rate is set by lr * sigma_max = this
    anchor_weight: float = 1.0        # weight on anchor rows of A

    # ---- measurement ----------------------------------------------------- #
    eval_every: int = 2               # epochs between recorded evaluations
    hold: int = 10                    # consecutive evaluations the criterion must hold
    tau: float = 0.5                  # composition-error threshold in the criterion

    # ---- linear algebra -------------------------------------------------- #
    rank_rtol: float = 1e-9           # sigma > rank_rtol * sigma_max counts as non-zero
    identifiable_rtol: float = 1e-7   # rho below this counts as zero (see note)
    mode_significance: float = 0.05   # a mode is "loaded" at >= this x the max weight

    # `rank_rtol` and `identifiable_rtol` are deliberately loose: the measured
    # separation is ~13 orders of magnitude (rho is either ~1e-15 or ~0.26), so
    # no result depends on where in that gap the line is drawn.

    # ---- deep mean-dynamics integration ---------------------------------- #
    ode_method: str = "rk4"
    ode_substeps: Mapping[int, int] = field(
        default_factory=lambda: {1: 1, 2: 8, 3: 32, 4: 32, 5: 32})

    def init_scale(self, depth: int) -> float:
        return self.init_scale_shallow if depth == 1 else self.init_scale_deep

    def substeps(self, depth: int) -> int:
        return self.ode_substeps.get(depth, 32)

    def to_dict(self):
        d = asdict(self)
        d["ode_substeps"] = {str(k): v for k, v in d["ode_substeps"].items()}
        return d


DEFAULT = Settings()


def override(**kw) -> Settings:
    """`override(eval_every=1, hold=15)` -> a copy of DEFAULT with those changes."""
    return replace(DEFAULT, **kw)
