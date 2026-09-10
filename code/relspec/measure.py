"""
Measurement: one behavioural scorer, one geometric scorer, one criterion.

A `Trajectory` is produced identically by per-fact SGD, by the shallow closed
form and by integrating the deep mean dynamics, and the same two measures are
applied to all three.  A predicted result and an observed result are therefore
the same kind of object and can be plotted on one axis.

The two measures, for a composition law:

  retrieval   held-out composite retrieval.  For a pair `a -> b` whose
              composite fact was withheld, is `b` ranked first among all
              entities for the query `a + z`?  Reported as the percentage of
              held-out pairs correct.

              The query uses the composite RELATION.  It is deliberately not
              `a + x + y`, which walks two trained premise edges and lands on
              `b` by algebra once they are fit, and not `x + y` ranked among
              relation candidates, which is a thresholded view of the geometric
              measure rather than an independent test of generalisation.

  geometric   composition error `||z - (x+y)|| / ||x+y||`.  The denominator is
              the premise sum, which the observed facts determine, not the
              composite's own length, which is free whenever the law is
              undetermined and leaves the ratio with no bounded scale.

For a cross-structure comparison the same two roles are filled by whether the
intended entity is ranked first for a query stepped across the seam, and by the
offset error against ground truth in units of one relation step.

Which items are held out is a property of the experiment, not of the world
alone, so the caller builds a plan and passes it in.  For a staged run the plan
must be built from the POST-intervention world, so the evaluation set does not
move when a fact is inserted mid-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT, Settings
from .worlds import World

# --------------------------------------------------------------------------- #
#  Plans: which items an experiment scores                                     #
# --------------------------------------------------------------------------- #


def _sqdist(pts, Cand):
    """(n_pts x n_cand) squared distances."""
    return ((pts ** 2).sum(1)[:, None] - 2.0 * pts @ Cand.T
            + (Cand ** 2).sum(1)[None, :])


def _rank_of(pts, Cand, target):
    """Zero-based rank of each query's true target among the candidates."""
    d2 = _sqdist(pts, Cand)
    true = d2[np.arange(len(target)), target]
    return (d2 < true[:, None]).sum(1)


def law_plan(world: World, held, candidates=None):
    """Score the held-out composites of every law.  `held` is
    `{law name: [(head, tail)]}`, as `worlds.held_composites` returns.

    Each law is ranked against ITS OWN structure, not the whole world.  A query
    from one lattice competing against every other lattice makes the measure
    depend on how many unrelated laws the world contains, and a rank flip
    against a foreign entity says nothing about the law.  `candidates` overrides
    the pool per law; the default comes from `worlds.law_entities`.
    """
    from .worlds import law_entities

    ti = world.tok_index
    plan = dict(kind="law", names=[])
    for law in world.laws:
        pairs = held.get(law.name, [])
        if not pairs:
            continue
        pool = sorted((candidates or {}).get(law.name) or law_entities(world, law))
        pos = {e: k for k, e in enumerate(pool)}
        if any(b not in pos for _, b in pairs):
            raise ValueError("held-out target outside %s's candidate pool" % law.name)
        plan["names"].append(law.name)
        plan[law.name] = dict(
            cand=np.array([ti[e] for e in pool]),
            n_cand=len(pool),
            a=np.array([ti[a] for a, _ in pairs]),
            b=np.array([pos[b] for _, b in pairs]),
            x=ti[law.x_rel], y=ti[law.y_rel], z=ti[law.z_rel])
    return plan


def cross_plan(world: World, pairs, name="cross", step="x", baseline=None,
               candidates=None):
    """Score held-out across-block comparisons.  `pairs` is
    `[(head, tail, n_x, n_y)]`, as `worlds.held_cross` returns.

    Candidates default to the destination block only, for the same reason law
    queries are ranked within their own lattice.

    `baseline` is the per-pair offset error of the minimum-norm solution.  The
    geometric measure is divided by it, so an arm in which the comparison stays
    undetermined sits at 1.0 by construction instead of wandering.  Without
    that normalisation the control's level and slope are pure gauge: they track
    where ground truth happens to sit relative to the minimum-norm
    representative, which is a property of the world's construction and not of
    learning.
    """
    ti, gt = world.tok_index, world.meta["gt_ent"]
    dest = sorted(candidates or {e for _, b, _, _ in pairs for e in (b,)})
    pos = {e: k for k, e in enumerate(dest)}
    base = (np.ones(len(pairs)) if baseline is None
            else np.maximum(np.asarray(baseline, float), 1e-12))
    return dict(
        kind="cross", name=name, names=[name],
        cand=np.array([ti[e] for e in dest]), n_cand=len(dest),
        a=np.array([ti[a] for a, _, _, _ in pairs]),
        b=np.array([ti[b] for _, b, _, _ in pairs]),
        bpos=np.array([pos[b] for _, b, _, _ in pairs]),
        nx=np.array([nx for _, _, nx, _ in pairs], float),
        ny=np.array([ny for _, _, _, ny in pairs], float),
        gt=np.array([gt[b] - gt[a] for a, b, _, _ in pairs]),
        x=ti["x"], y=ti["y"],
        scale=float(np.linalg.norm(world.meta["gt_rel"][step])),
        baseline=base, normalised=baseline is not None)


def apply_plan(plan, E):
    """One evaluation.  Returns `(retrieval, geometric, hits, errs, rank)`.

    `retrieval` is the percentage of held-out items whose true target is ranked
    first.  `rank` is the mean normalised rank of that target, `1` when it is
    first and `0` when it is last, which moves smoothly as the query migrates
    and does not jump by a whole item at a time.  The rank test drives the
    emergence criterion; the smooth one is what a trajectory panel should plot.
    """
    ret, geo, hits, errs, rank = {}, {}, {}, {}, {}
    if plan["kind"] == "law":
        for n in plan["names"]:
            p = plan[n]
            z, xy = E[p["z"]], E[p["x"]] + E[p["y"]]
            r = _rank_of(E[p["a"]] + z[None, :], E[p["cand"]], p["b"])
            hit = r == 0
            ret[n], hits[n], errs[n] = 100.0 * hit.mean(), hit, None
            rank[n] = 100.0 * float(np.mean(1.0 - r / max(p["n_cand"] - 1, 1)))
            geo[n] = float(np.linalg.norm(z - xy)
                           / (np.linalg.norm(xy) + 1e-12))
    else:
        n = plan["name"]
        pts = (E[plan["a"]] + plan["nx"][:, None] * E[plan["x"]][None, :]
               + plan["ny"][:, None] * E[plan["y"]][None, :])
        r = _rank_of(pts, E[plan["cand"]], plan["bpos"])
        hit = r == 0
        err = np.linalg.norm((E[plan["b"]] - E[plan["a"]]) - plan["gt"],
                             axis=1) / plan["scale"] / plan["baseline"]
        ret[n], hits[n], errs[n] = 100.0 * hit.mean(), hit, err
        rank[n] = 100.0 * float(np.mean(1.0 - r / max(plan["n_cand"] - 1, 1)))
        geo[n] = float(np.exp(np.log(np.maximum(err, 1e-16)).mean()))
    return ret, geo, hits, errs, rank


def unlocked(epochs, hits):
    """Per item, the first epoch after its LAST failure, so a curve built from
    this is monotone.  Instantaneous accuracy is not: an item that flips back
    makes it fall, which is why a panel built on the raw fraction cannot be
    labelled as items resolved."""
    hits = np.asarray(hits, bool)
    out = np.full(hits.shape[1], np.inf)
    for c in range(hits.shape[1]):
        bad = np.where(~hits[:, c])[0]
        k = 0 if not len(bad) else bad[-1] + 1
        if k < len(epochs):
            out[c] = epochs[k]
    return out


# --------------------------------------------------------------------------- #
#  Emergence                                                                   #
# --------------------------------------------------------------------------- #


def detect_emergence(retrieval, geometric, hold, tau, level=100.0):
    """Index of the first recorded evaluation at which retrieval reaches
    `level` percent *and* the geometric error is below `tau`, with both holding
    for `hold` consecutive evaluations.  NaN if it never happens.

    The geometric guard is not cosmetic.  A rank test can pass transiently on a
    quantity the training data does not determine, because ranks are invariant
    to a global scale the geometry is not, so retrieval alone reports false
    early emergence.  This was observed in every world tested.
    """
    s = (np.asarray(retrieval, float) >= level - 1e-9) & (
        np.asarray(geometric, float) < tau
    )
    for t in range(len(s)):
        if s[t] and np.all(s[t : min(len(s), t + hold)]):
            return t
    return np.nan


def threshold_time(epochs, series, thresh):
    """First (linearly interpolated) epoch at which a decreasing series crosses
    `thresh`.  A continuous-valued alternative to `t*`, used where the discrete
    evaluation grid would otherwise dominate the measurement."""
    s, e = np.asarray(series, float), np.asarray(epochs, float)
    if s[0] <= thresh:
        return e[0]
    for i in range(1, len(s)):
        if s[i] <= thresh < s[i - 1]:
            frac = (s[i - 1] - thresh) / (s[i - 1] - s[i] + 1e-12)
            return e[i - 1] + frac * (e[i] - e[i - 1])
    return np.nan


# --------------------------------------------------------------------------- #
#  Trajectory                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class Trajectory:
    """What a run produces, empirical or predicted."""

    epochs: np.ndarray
    retrieval: dict = field(default_factory=dict)  # name -> (T,) percent
    rank: dict = field(default_factory=dict)  # name -> (T,) mean normalised rank
    geometric: dict = field(default_factory=dict)  # name -> (T,) error
    hits: dict = field(default_factory=dict)  # name -> (T, n) bool
    errs: dict = field(default_factory=dict)  # name -> (T, n) or None
    loss: np.ndarray = None
    probes: dict = field(default_factory=dict)  # name -> (T, ...) array

    def emergence(self, settings: Settings = DEFAULT, level=100.0):
        """`{name: t* in epochs}`, NaN where it never emerged."""
        out = {}
        for n in self.retrieval:
            i = detect_emergence(
                self.retrieval[n], self.geometric[n], settings.hold, settings.tau, level
            )
            out[n] = float(self.epochs[int(i)]) if np.isfinite(i) else np.nan
        return out

    def emergence_of(self, name, settings: Settings = DEFAULT, level=100.0):
        return self.emergence(settings, level)[name]


def trajectory_from_embeddings(
    world: World, epochs, embeddings, system=None, probes=None, plan=None
):
    """Build a `Trajectory` by applying the measures to a sequence of E(t).

    This is the single point at which embeddings become measurements, shared by
    SGD, the shallow closed form and the deep ODE.  `embeddings` may be a
    generator, so a long predicted trajectory never has to be held in memory.
    With `plan=None` nothing is measured and only probes and loss are recorded.
    """
    names = plan["names"] if plan else []
    ret = {n: [] for n in names}
    geo = {n: [] for n in names}
    hit = {n: [] for n in names}
    err = {n: [] for n in names}
    rnk = {n: [] for n in names}
    losses = [] if system is not None else None
    probe_rec = {k: [] for k in (probes or {})}
    for E in embeddings:
        if plan:
            r, g, h, e, q = apply_plan(plan, E)
            for n in names:
                ret[n].append(r[n])
                geo[n].append(g[n])
                hit[n].append(h[n])
                err[n].append(e[n])
                rnk[n].append(q[n])
        for k, v in (probes or {}).items():
            probe_rec[k].append(np.asarray(v @ E).copy())
        if system is not None:
            losses.append(system.loss(E))
    return Trajectory(
        epochs=np.asarray(epochs, float),
        retrieval={n: np.asarray(v, float) for n, v in ret.items()},
        rank={n: np.asarray(v, float) for n, v in rnk.items()},
        geometric={n: np.asarray(v, float) for n, v in geo.items()},
        hits={n: np.asarray(v, bool) for n, v in hit.items()},
        errs={
            n: (None if err[n][0] is None else np.asarray(err[n], float)) for n in names
        },
        loss=None if losses is None else np.asarray(losses, float),
        probes={k: np.asarray(v, float) for k, v in probe_rec.items()},
    )
