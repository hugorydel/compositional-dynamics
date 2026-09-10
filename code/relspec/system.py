"""
The relational world as a linear system, and its spectral structure.

Every fact `(a, r, b)` contributes a row `e_a + e_r - e_b` with target 0; every
anchor contributes a row `e_e` with target `c_e`.  Stacking them gives

    A E ~ C ,      L(E) = 1/(2|D|) ||A E - C||_F^2

whose gradient flow is  `tau dE/dt = B - H E`  with

    H = A^T A / |D| = Q Lambda Q^T ,     B = A^T C / |D|

`System` bundles the world, (A, C) and the eigendecomposition, and answers the
questions the theory asks of them: how fast does each mode decay, is a given
law's contrast inside the row space, what is the minimum-norm solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, Settings
from .worlds import World


# --------------------------------------------------------------------------- #
#  Assembly                                                                    #
# --------------------------------------------------------------------------- #

def build_matrices(world: World, facts=None, anchors=None, anchor_weight=1.0):
    """(A, C) for a chosen subset of facts/anchors (defaults: the whole world).

    Selecting a subset is how curricula are expressed: phase 1 and phase 2 are
    two (A, C) pairs over the *same* token space, so a model can be carried
    from one to the other unchanged.
    """
    facts = world.facts if facts is None else facts
    anchors = world.anchors if anchors is None else anchors
    tok = world.tok_index
    P = world.P

    rows = np.zeros((len(facts) + len(anchors), P))
    targets = np.zeros((len(facts) + len(anchors), world.d))
    for i, (h, r, t) in enumerate(facts):
        rows[i, tok[h]] += 1.0
        rows[i, tok[r]] += 1.0
        rows[i, tok[t]] -= 1.0
    aw = np.sqrt(anchor_weight)
    for j, (e, val) in enumerate(anchors.items()):
        rows[len(facts) + j, tok[e]] = aw
        targets[len(facts) + j] = np.asarray(val) * aw
    return rows, targets


def sparse_rows(A):
    """Per-row (nonzero indices, values), so per-fact SGD never touches a dense
    P-vector.  Each row has at most three nonzeros."""
    return [(np.nonzero(A[i])[0], A[i][np.nonzero(A[i])[0]]) for i in range(A.shape[0])]


# --------------------------------------------------------------------------- #
#  System                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class System:
    world: World
    A: np.ndarray
    C: np.ndarray
    evals_M: np.ndarray      # eigenvalues of M = A^T A, descending (sigma_k)
    Q: np.ndarray            # eigenvectors, columns aligned with evals_M
    Estar: np.ndarray        # minimum-norm least-squares solution (P x d)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def build(cls, world: World, facts=None, anchors=None,
              settings: Settings = DEFAULT):
        A, C = build_matrices(world, facts, anchors, settings.anchor_weight)
        evals, Q = np.linalg.eigh(A.T @ A)
        order = np.argsort(evals)[::-1]
        Estar, *_ = np.linalg.lstsq(A, C, rcond=None)
        return cls(world=world, A=A, C=C,
                   evals_M=evals[order], Q=Q[:, order], Estar=Estar)

    # -- basic quantities --------------------------------------------------- #

    @property
    def n_rows(self):
        return self.A.shape[0]

    @property
    def evals_H(self):
        return self.evals_M / self.n_rows

    @property
    def H(self):
        return self.A.T @ self.A / self.n_rows

    def sparse(self):
        cached = getattr(self, "_sparse", None)
        if cached is None:
            cached = sparse_rows(self.A)
            self._sparse = cached
        return cached

    def lr(self, depth=1, target=None, settings: Settings = DEFAULT):
        """Learning rate set by `lr * sigma_max = target`, so step sizes are
        comparable across worlds and depths.  `depth` is accepted (and ignored)
        so call sites read explicitly; the rule is depth-independent.
        """
        t = settings.lr_target if target is None else target
        return t / self.evals_M[0]

    def loss(self, E):
        R = self.A @ E - self.C
        return float(0.5 * np.mean(np.sum(R * R, axis=1)))

    # -- law contrasts and identifiability ---------------------------------- #

    def law_contrast(self, law) -> np.ndarray:
        """`ell_j` in token space: +1 on x, +1 on y, -1 on z."""
        law = self.world.law(law) if isinstance(law, str) else law
        ell = np.zeros(self.world.P)
        tok = self.world.tok_index
        ell[tok[law.x_rel]] += 1.0
        ell[tok[law.y_rel]] += 1.0
        ell[tok[law.z_rel]] -= 1.0
        return ell

    def row_basis(self, settings: Settings = DEFAULT):
        """`Q_+` (eigenvectors with non-zero eigenvalue) and the mask."""
        mask = self.evals_M > settings.rank_rtol * self.evals_M[0]
        return self.Q[:, mask], mask

    def project_row(self, X, settings: Settings = DEFAULT):
        Qr, _ = self.row_basis(settings)
        return Qr @ (Qr.T @ X)

    def project_null(self, X, settings: Settings = DEFAULT):
        return X - self.project_row(X, settings)

    def identifiability(self, law, settings: Settings = DEFAULT) -> dict:
        """Is `ell_j^T E` determined by the data?

        Returns `rho` = ||(I - Q_+ Q_+^T) ell_hat||, zero iff identifiable, plus
        the unnormalised direction `ell_perp` and the unit vector along
        it (the direction in which the initialisation survives training).
        """
        ell = self.law_contrast(law)
        unit = ell / np.linalg.norm(ell)
        perp_unit = self.project_null(unit, settings)
        rho = float(np.linalg.norm(perp_unit))
        ell_perp = self.project_null(ell, settings)
        _, mask = self.row_basis(settings)
        load = np.abs(self.Q.T @ unit) * mask
        signif = load >= settings.mode_significance * (load.max() + 1e-30)
        return dict(
            rho=rho,
            identifiable=bool(rho < settings.identifiable_rtol),
            ell_perp=ell_perp,
            ell_perp_norm=float(np.linalg.norm(ell_perp)),
            n_hat=(perp_unit / rho) if rho > 1e-12 else np.zeros_like(perp_unit),
            sigma_slow_loaded=float(self.evals_M[signif].min()) if signif.any()
            else float("nan"))

    def rank_report(self, settings: Settings = DEFAULT) -> dict:
        """Rank, null dimension, and the size of the gap the rank call sits in
        -- so a reader can see it is not a tolerance choice."""
        sig = self.evals_M
        mask = sig > settings.rank_rtol * sig[0]
        nz, z = sig[mask], sig[~mask]
        return dict(rank=int(mask.sum()), null_dim=int((~mask).sum()),
                    sigma_max=float(sig[0]),
                    sigma_min_nonzero=float(nz.min()) if nz.size else float("nan"),
                    sigma_max_zero=float(np.abs(z).max()) if z.size else 0.0)

    def endpoint(self, E0, settings: Settings = DEFAULT) -> np.ndarray:
        """`E(inf) = E* + Pi_null E(0)`.

        Exact for the shallow model under gradient flow *and* under per-fact
        SGD, because every SGD update lies in span(a_i) < row(A) and therefore
        never changes the null-space component.
        """
        return self.Estar + self.project_null(E0, settings)

    # -- spectral profile of a law ------------------------------------------ #

    def gating_weights(self, law, E0=None, settings: Settings = DEFAULT):
        """Per-mode weight `w_k = |a_jk| * ||z_k(0)||` -- each mode's
        contribution to the law's composition-error trajectory.  `z_k(0)` is
        the initial residual along mode k; with `E0=None` it is measured from
        the minimum-norm solution (equivalent to a zero initialisation)."""
        ell = self.law_contrast(law)
        unit = ell / np.linalg.norm(ell)
        a = np.abs(self.Q.T @ unit)
        ref = -self.Estar if E0 is None else (E0 - self.Estar)
        z0 = np.linalg.norm(self.Q.T @ ref, axis=1)
        _, mask = self.row_basis(settings)
        return a * z0 * mask
