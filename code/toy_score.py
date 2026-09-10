"""Score every proposed measure on a sequence of embeddings.

One function, used for BOTH the trained trajectory and the prospective
theory trajectory, so the prediction is scored by exactly the test applied to
the network and the overlay cannot drift from the curve it overlays.
"""
import numpy as np

M = 3
ROLES = ("E1", "E2_id", "E2_no")
LAW_OF = {"E1": "E1_c", "E2_id": "E2_id", "E2_no": "E2_no"}
CROSS = [(i, j, k, l) for i in range(M) for j in range(M)
         for k in range(M) for l in range(M)]


def _nearest(E, idx, ents, pt):
    return ents[int(np.argmin(np.linalg.norm(E[idx] - pt[None, :], axis=1)))]


def score(world, Es):
    """{measure: array} for a list of (P x d) embeddings, in order."""
    ti, ents = world.tok_index, world.entities
    idx = [ti[e] for e in ents]
    held, gt_e, gt_r = (world.meta["held"], world.meta["gt_ent"],
                        world.meta["gt_rel"])
    rn = np.linalg.norm(gt_r["p"])
    T = len(Es)

    out = dict(acc_xy={r: np.zeros(T) for r in ROLES},
               acc_z={r: np.zeros(T) for r in ROLES},
               geo_z={r: np.zeros(T) for r in ROLES},
               geo_xy={r: np.zeros(T) for r in ROLES},
               e3_acc=np.zeros(T), e3_err=np.zeros((T, len(CROSS))),
               e3_hit=np.zeros((T, len(CROSS)), bool))

    for t, E in enumerate(Es):
        for role in ROLES:
            L = world.law(LAW_OF[role])
            hx, hz = [], []
            for a, b, xr, yr in held[role]:
                hx.append(_nearest(E, idx, ents,
                                   E[ti[a]] + E[ti[xr]] + E[ti[yr]]) == b)
                hz.append(_nearest(E, idx, ents,
                                   E[ti[a]] + E[ti[L.z_rel]]) == b)
            out["acc_xy"][role][t] = 100.0 * np.mean(hx)
            out["acc_z"][role][t] = 100.0 * np.mean(hz)
            z, xy = E[ti[L.z_rel]], E[ti[L.x_rel]] + E[ti[L.y_rel]]
            gap = float(np.linalg.norm(z - xy))
            out["geo_z"][role][t] = gap / (np.linalg.norm(z) + 1e-12)
            out["geo_xy"][role][t] = gap / (np.linalg.norm(xy) + 1e-12)
        hs = []
        for c, (i, j, k, l) in enumerate(CROSS):
            a, b = "A_%d_%d" % (i, j), "Acopy_%d_%d" % (k, l)
            pt = E[ti[a]] + (M + k - i) * E[ti["p"]] + (l - j) * E[ti["q"]]
            hs.append(_nearest(E, idx, ents, pt) == b)
            out["e3_err"][t, c] = np.linalg.norm(
                (E[ti[b]] - E[ti[a]]) - (gt_e[b] - gt_e[a])) / rn
        out["e3_hit"][t] = hs
        out["e3_acc"][t] = 100.0 * np.mean(hs)
    return out


def as_json(o):
    return dict(acc_xy={k: v.tolist() for k, v in o["acc_xy"].items()},
                acc_z={k: v.tolist() for k, v in o["acc_z"].items()},
                geo_z={k: v.tolist() for k, v in o["geo_z"].items()},
                geo_xy={k: v.tolist() for k, v in o["geo_xy"].items()},
                e3_acc=o["e3_acc"].tolist(), e3_err=o["e3_err"].tolist(),
                e3_hit=o["e3_hit"].astype(int).tolist())
