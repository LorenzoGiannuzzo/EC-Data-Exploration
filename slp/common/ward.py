"""Ward linkage over weighted points.

The dictionary of Section 2.3 is built by summarising every daily shape into a
fixed number of micro-clusters and running Ward on those. A micro-cluster is not
one observation: it stands for the shapes that fell in it, and that cardinality
has to enter the linkage, or a micro-cluster holding three hundred thousand days
weighs exactly as much as one holding three and the tree spends its first merges
shaving isolated points off the population.

scipy.cluster.hierarchy.linkage tracks cluster sizes itself and offers no way to
declare them, so the weighted tree is built here. The criterion is unchanged:
Ward merges the pair whose union increases the within-cluster inertia the least,

    delta(A, B) = w_A w_B / (w_A + w_B) * ||c_A - c_B||^2

with c the weighted centroid and w the total weight, which is exactly what Ward
minimises when every point carries a multiplicity. The returned matrix follows
the scipy convention, distances stored as sqrt(2 delta) so that unit weights
reproduce scipy's own output, and can be passed to fcluster unchanged.
"""
from __future__ import annotations

import numpy as np


def ward_weighted(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return a scipy-compatible linkage matrix for weighted observations.

    points  (n, d) the coordinates, one row per weighted observation
    weights (n,)   the multiplicity of each row; None means unit weights, in
                   which case the result matches scipy's ward exactly
    """
    x = np.asarray(points, dtype="float64")
    n = len(x)
    if n < 2:
        raise ValueError("at least two observations are needed")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype="float64").copy()
    if len(w) != n or np.any(w <= 0):
        raise ValueError("weights must be positive and as many as the points")

    cent = x.copy()
    wt = w.copy()
    size = np.ones(n)                       # observations behind each cluster
    cid = np.arange(n)                      # scipy identifier of each active slot
    active = np.ones(n, dtype=bool)

    #Lorenzo Giannuzzo: the full matrix of merge costs is held once and only the
    # row of the cluster just formed is recomputed, which keeps the whole tree
    # within one pass over the matrix per merge. On four thousand micro-clusters
    # the matrix is sixty-four megabytes and the tree takes a few seconds.
    delta = _cost_matrix(cent, wt)
    np.fill_diagonal(delta, np.inf)

    nn = delta.argmin(axis=1)
    nn_cost = delta[np.arange(n), nn]

    Z = np.empty((n - 1, 4))
    next_id = n

    for step in range(n - 1):
        idx = np.where(active)[0]
        a = idx[nn_cost[idx].argmin()]
        b = nn[a]
        cost = delta[a, b]

        #Lorenzo Giannuzzo: scipy stores sqrt(2 * delta) for ward, so that the
        # merge of two singletons is their Euclidean distance. Keeping the same
        # scale means fcluster and every plot downstream need no adjustment.
        Z[step] = (min(cid[a], cid[b]), max(cid[a], cid[b]),
                   np.sqrt(2.0 * max(cost, 0.0)), size[a] + size[b])

        total = wt[a] + wt[b]
        cent[a] = (cent[a] * wt[a] + cent[b] * wt[b]) / total
        wt[a] = total
        size[a] = size[a] + size[b]
        cid[a] = next_id
        next_id += 1

        active[b] = False
        delta[b, :] = np.inf
        delta[:, b] = np.inf
        nn_cost[b] = np.inf

        idx = np.where(active)[0]
        if len(idx) == 1:
            break

        others = idx[idx != a]
        d2 = ((cent[others] - cent[a]) ** 2).sum(axis=1)
        row = wt[others] * wt[a] / (wt[others] + wt[a]) * d2
        delta[a, others] = row
        delta[others, a] = row

        nn[a] = others[row.argmin()]
        nn_cost[a] = row.min()

        #Lorenzo Giannuzzo: only the clusters whose nearest neighbour was one of
        # the two just merged need their own row scanned again. Everything else
        # keeps a valid cached neighbour, since Ward is reducible and no merge
        # can bring a pair closer than it already was.
        stale = others[(nn[others] == a) | (nn[others] == b) | (row < nn_cost[others])]
        for k in stale:
            nn[k] = delta[k].argmin()
            nn_cost[k] = delta[k, nn[k]]

    return Z


def _cost_matrix(cent: np.ndarray, wt: np.ndarray) -> np.ndarray:
    sq = (cent ** 2).sum(axis=1)
    d2 = sq[:, None] - 2.0 * cent @ cent.T + sq[None, :]
    np.maximum(d2, 0.0, out=d2)
    return wt[:, None] * wt[None, :] / (wt[:, None] + wt[None, :]) * d2


def weighted_centroids(points: np.ndarray, weights: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
    """Weighted mean of each cluster, in the order of the sorted labels."""
    x = np.asarray(points, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    return np.vstack([np.average(x[labels == k], axis=0, weights=w[labels == k])
                      for k in np.unique(labels)])
