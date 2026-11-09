"""Reviewer 2, Q4 (first part): Experiment 2's construction and its ambiguity.

  "Do the two separate blocks use disjoint relation tokens (yielding
   rho = 1/sqrt(15))?"

Rebuilds each Experiment 2 world with the repository's own constructor
(`worlds.identifiability_world`, K = 16, k = 4, as in `run_experiments.run_f2`)
and checks, per world and block assignment:

  1. the two laws share no relation token;
  2. ker A is one-dimensional and spanned by v = e_z(open) + the open block's
     4 unanchored pair tails;
  3. rho (as `System.identifiability` computes it) for the open and closed laws;
  4. E* and E* + v u^T fit every training row equally well, and their open-law
     errors differ by exactly -u^T;
  5. after the linking fact (`n_bridge=1`), ker A is empty and rho ~ 0.

Nothing is trained; a run takes seconds.

  usage:  python r2_q4_construction.py            worlds 0-29
          python r2_q4_construction.py 0 199      a seed range
"""

import sys

import numpy as np

from common import write_outputs
from relspec import System, worlds

NAME = "r2_q4_construction"
QUESTION = ("Q4, 'Do the two separate blocks use disjoint relation tokens "
            "(yielding rho = 1/sqrt(15))?'")


def null_dim(A):
    s = np.linalg.svd(A, compute_uv=False)
    return A.shape[1] - int(np.sum(s > 1e-9 * s[0]))


def open_null_vector(w, open_block):
    tok = w.tok_index
    v = np.zeros(w.P)
    v[tok["z" + open_block]] = 1.0
    tails = [t for t in tok if t.startswith("%sF" % open_block) and t.endswith("_1")]
    for t in tails:
        v[tok[t]] = 1.0
    return v, len(tails)


def cell(seed, closed):
    op = "B" if closed == "A" else "A"
    w = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed)
    s = System.build(w)
    la, lb = w.law("A"), w.law("B")
    disjoint = not ({la.x_rel, la.y_rel, la.z_rel} & {lb.x_rel, lb.y_rel, lb.z_rel})
    v, n_tails = open_null_vector(w, op)
    Estar = np.linalg.lstsq(s.A, s.C, rcond=None)[0]
    u = np.random.default_rng(seed).standard_normal(w.d)
    E2 = Estar + np.outer(v, u)
    ell = s.law_contrast(op)
    w1 = worlds.identifiability_world(seed, K=16, k=4, closed_block=closed, n_bridge=1)
    s1 = System.build(w1)
    return dict(
        seed=seed, closed=closed, disjoint_tokens=disjoint, n_tokens=w.P,
        null_dim=null_dim(s.A), A_v_norm=float(np.linalg.norm(s.A @ v)), n_free_tails=n_tails,
        rho_open=s.identifiability(op)["rho"], rho_closed=s.identifiability(closed)["rho"],
        loss_Estar=s.loss(Estar), loss_Estar_plus_vu=s.loss(E2),
        law_error_shift_equals_minus_u=bool(np.allclose(ell @ E2 - ell @ Estar, -u)),
        null_dim_after_link=null_dim(s1.A), rho_open_after_link=s1.identifiability(op)["rho"])


def main(lo=0, hi=29):
    cells = [cell(s, c) for s in range(lo, hi + 1) for c in ("A", "B")]
    a = lambda k: np.array([c[k] for c in cells])
    summ = dict(
        n_cells=len(cells), seeds=[lo, hi],
        all_disjoint=bool(a("disjoint_tokens").all()),
        null_dim=sorted(set(a("null_dim").tolist())),
        max_A_v_norm=float(a("A_v_norm").max()),
        rho_open_min=float(a("rho_open").min()), rho_open_max=float(a("rho_open").max()),
        one_over_sqrt15=float(1 / np.sqrt(15)),
        rho_closed_max=float(a("rho_closed").max()),
        max_loss_Estar=float(a("loss_Estar").max()),
        max_loss_Estar_plus_vu=float(a("loss_Estar_plus_vu").max()),
        all_shift_equals_minus_u=bool(a("law_error_shift_equals_minus_u").all()),
        null_dim_after_link=sorted(set(a("null_dim_after_link").tolist())),
        rho_open_after_link_max=float(a("rho_open_after_link").max()))
    rep = [
        "Worlds %d-%d, both block assignments (%d cells)." % (lo, hi, len(cells)), "",
        "| Check | Result |", "|---|---|",
        "| Laws share no relation token | %s |" % summ["all_disjoint"],
        "| dim ker A | %s (of %d tokens) |" % (summ["null_dim"], cells[0]["n_tokens"]),
        "| ‖A v‖, v = e_z + %d open tails | max %.1e |" % (cells[0]["n_free_tails"], summ["max_A_v_norm"]),
        "| ρ, open law | %.10f – %.10f (1/√15 = %.10f) |" % (summ["rho_open_min"], summ["rho_open_max"], summ["one_over_sqrt15"]),
        "| ρ, closed law | max %.1e |" % summ["rho_closed_max"],
        "| Training loss of E* / of E* + v uᵀ | max %.1e / %.1e |" % (summ["max_loss_Estar"], summ["max_loss_Estar_plus_vu"]),
        "| Open-law error of E* + v uᵀ minus that of E* equals −uᵀ | %s |" % summ["all_shift_equals_minus_u"],
        "| dim ker A after the linking fact | %s |" % summ["null_dim_after_link"],
        "| ρ, open law, after the linking fact | max %.1e |" % summ["rho_open_after_link_max"],
        "",
        "Derivation: v = e_z + Σ_tails e_t has ‖v‖² = 5 and ℓᵀv = −1 for ℓ = e_x + e_y − e_z, "
        "so P₀ℓ = −v/5, ‖P₀ℓ‖ = 1/√5, ‖ℓ‖ = √3 and ρ = ‖P₀ℓ‖/‖ℓ‖ = 1/√15."]
    write_outputs(NAME, "Reviewer 2, Q4: Experiment 2 construction", QUESTION, rep, summ, cells,
                  "rebuilt worlds (`relspec.worlds.identifiability_world`)")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:3]]
    main(*args) if args else main()
