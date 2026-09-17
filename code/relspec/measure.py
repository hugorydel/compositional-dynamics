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


def _spacing(Cand):
    """Typical distance between neighbouring candidates, in the LEARNED
    embedding.  Used as the unit for the geometric measure so that shrinking
    the whole representation cannot move it, which is what let the behavioural
    and geometric rows disagree: a rank test is blind to overall scale and a
    raw distance is not."""
    d2 = _sqdist(Cand, Cand)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1)))) + 1e-12


def _miss(q, Cand, target):
    """Mean distance from each query to the entity it should have retrieved,
    in units of candidate spacing.  This is the direct companion to the
    behavioural test: behaviour asks whether the right entity was picked, this
    asks how far the predicted point landed from it."""
    return float(np.mean(np.linalg.norm(q - Cand[target], axis=1)
                         / _spacing(Cand)))


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
        cnt = {}
        for _, b in pairs:
            cnt[b] = cnt.get(b, 0) + 1
        plan["names"].append(law.name)
        plan[law.name] = dict(
            chance=max(cnt.values()) / float(len(pairs)),
            cand=np.array([ti[e] for e in pool]),
            n_cand=len(pool),
            a=np.array([ti[a] for a, _ in pairs]),
            b=np.array([pos[b] for _, b in pairs]),
            x=ti[law.x_rel], y=ti[law.y_rel], z=ti[law.z_rel])
    return plan


def cross_plan(world: World, pairs, name="cross", candidates=None):
    """Score held-out across-block comparisons.  `pairs` is
    `[(head, tail, n_x, n_y)]`, as `worlds.held_cross` returns.

    Candidates default to the destination block only, for the same reason law
    queries are ranked within their own lattice.

    Candidates default to the destination block only, for the same reason law
    queries are ranked within their own lattice.  The geometric companion is
    the distance from the predicted point to the entity it should have
    retrieved, in units of candidate spacing, so nothing here needs a
    ground-truth alignment or a reference offset.
    """
    ti = world.tok_index
    dest = sorted(candidates or {b for _, b, _, _ in pairs})
    pos = {e: k for k, e in enumerate(dest)}
    cnt = {}
    for _, b, _, _ in pairs:
        cnt[b] = cnt.get(b, 0) + 1
    return dict(
        kind="cross", name=name, names=[name],
        chance=max(cnt.values()) / float(len(pairs)),
        cand=np.array([ti[e] for e in dest]), n_cand=len(dest),
        a=np.array([ti[a] for a, _, _, _ in pairs]),
        bpos=np.array([pos[b] for _, b, _, _ in pairs]),
        nx=np.array([nx for _, _, nx, _ in pairs], float),
        ny=np.array([ny for _, _, _, ny in pairs], float),
        x=ti["x"], y=ti["y"])


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
            z = E[p["z"]]
            Cand, q = E[p["cand"]], E[p["a"]] + z[None, :]
            r = _rank_of(q, Cand, p["b"])
            hit = r == 0
            ret[n], hits[n], errs[n] = 100.0 * hit.mean(), hit, None
            rank[n] = 100.0 * float(np.mean(1.0 - r / max(p["n_cand"] - 1, 1)))
            geo[n] = _miss(q, Cand, p["b"])
    else:
        n = plan["name"]
        pts = (E[plan["a"]] + plan["nx"][:, None] * E[plan["x"]][None, :]
               + plan["ny"][:, None] * E[plan["y"]][None, :])
        r = _rank_of(pts, E[plan["cand"]], plan["bpos"])
        hit = r == 0
        Cand = E[plan["cand"]]
        sp = _spacing(Cand)
        err = np.linalg.norm(pts - Cand[plan["bpos"]], axis=1) / sp
        ret[n], hits[n], errs[n] = 100.0 * hit.mean(), hit, err
        rank[n] = 100.0 * float(np.mean(1.0 - r / max(plan["n_cand"] - 1, 1)))
        geo[n] = float(err.mean())
    return ret, geo, hits, errs, rank


def resolved(epochs, hits, chance=0.0):
    """Percentage of items STABLY resolved by each epoch, which is monotone.

    An item counts from the first evaluation after its last failure, the same
    persistence idea the emergence criterion uses.  Instantaneous top-1
    accuracy is not the right object for a panel about emergence: single items
    cross and recross a nearest-neighbour boundary, so the curve steps and
    reverses even while learning is monotone underneath.

    Retrospective by construction, since an item's status depends on whether it
    fails later, so the curve is defined relative to the run's own budget.

    `chance` is the accuracy of the best constant answer, and the result is
    rescaled so that scoring it reads as zero.  Without that, a model whose
    global offset is still undetermined returns essentially the same entity to
    every query and is credited with the fraction of items that happen to have
    that entity as their target.  In the cross-structure world that is nine of
    eighty, and the control sits at exactly 11.25% for precisely this reason.
    """
    t = unlocked(epochs, hits)
    ep = np.asarray(epochs, float)
    raw = np.array([(t <= e).mean() for e in ep])
    if chance <= 0:
        return 100.0 * raw
    return 100.0 * np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0)


def instantaneous(hits, chance=0.0):
    """Percentage of items correct AT each evaluation, chance-corrected.

    The causal counterpart of `resolved`.  Its value at an epoch depends only
    on the state at that epoch, so two runs holding identical weights score
    identically -- which `resolved` does not guarantee, because it credits an
    item from after its last failure and so reads the future.  Two branches
    leaving a switch with the same weights scored 4.55 and 10.89 per cent under
    `resolved` at depth 1 in Figure 2, and 10.89 and 10.89 under this.

    The price is that it can fall when an item crosses back over the retrieval
    boundary.  `tests/check_reversals.py` measures how much, in every figure,
    and whether the prediction falls in the same worlds.
    """
    raw = np.asarray(hits, bool).mean(axis=1)
    if chance <= 0:
        return 100.0 * raw
    return 100.0 * np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0)


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
    rng_state: dict = None  # order generator at the end; training only

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
