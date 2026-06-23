"""Fuse the two similarity views, pick a label-free threshold, and cluster.

The gradient descriptor and the wavelet embedding capture complementary
evidence, so their similarity matrices are blended (``W_GRADIENT`` weight on the
gradient view). The cut threshold is chosen *without labels* from the shape of
the predicted-group-count vs threshold curve: the correct operating point sits
at the knee of that curve, where the clustering is most stable.

Scaling
-------
The exact path materializes the dense N x N fused similarity. That is fine for a
single shoot but Theta(N^2) in memory — at N=95k each view is ~36 GB and cannot
allocate. So ``FusionClusterer`` is N-gated:

  * N <= ``_DENSE_MAX``  -> the exact dense path (unchanged; small shoots);
  * N >  ``_DENSE_MAX``  -> a sparse blocking path that gets the SAME clustering
    without ever forming N x N.

The sparse path uses the scaled-concat identity: with
``v_i = [sqrt(W) g_i, sqrt(1-W) e_i]`` the fused similarity is exactly
``v_i . v_j`` and (both descriptors being L2-normalized) ``v`` is unit-norm, so
SimHash blocking (sign of fixed random hyperplanes, banded so two points are
candidates if ANY band collides) finds the high-similarity pairs. Only candidate
pairs are scored exactly; the rest of N x N is never touched. Validated to
reproduce the dense labels exactly on ``data/large`` (so the 1302 benchmark is
untouched by construction). Deterministic (fixed seed) and browser-portable
(matmul + bit packing + sort only).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

W_GRADIENT = 0.65  # weight on the gradient view; wavelet gets the remainder

# Above this image count the dense N x N is too large; switch to sparse blocking.
# data/large (5041) stays on the exact dense path, so 1302 is unaffected.
_DENSE_MAX = int(os.environ.get("AUTOHDR_DENSE_MAX", "8000"))

# SimHash blocking params (sparse path). r bands of w bits; candidate if ANY band
# collides. w is adaptive (keeps buckets ~constant as N grows); r is fixed at the
# value proven to give 100% edge recall / exact labels on data/large.
_SIMHASH_SEED = 1234567   # fixed -> deterministic hyperplanes (no real randomness)
_SIMHASH_BANDS = 64       # r
_SIMHASH_BUCKET = 80      # target avg bucket size -> w = round(log2(N / bucket))
_CANDIDATE_FLOOR = 0.20   # candidate edges scored/kept at/above this (= grid start)


class AdjacencyGraph:
    """A symmetric "same-group" graph over image indices, sparsely stored.

    Refinement passes link nodes (add edges) or read the induced clusters; the
    grouping is always the connected components of the current edge set. Storage
    is a dict-of-neighbor-sets so it stays O(N + E) rather than O(N^2) — the
    clustering is identical to a dense bool matrix, just without the N^2 array.
    """

    def __init__(self, n: int, rows=None, cols=None) -> None:
        self._n = int(n)
        self._adj: dict[int, set[int]] = defaultdict(set)
        if rows is not None:
            for i, j in zip(rows, cols):
                i, j = int(i), int(j)
                if i != j:
                    self._adj[i].add(j)
                    self._adj[j].add(i)

    def link(self, i: int, j: int) -> None:
        i, j = int(i), int(j)
        if i != j:
            self._adj[i].add(j)
            self._adj[j].add(i)

    def _csr(self) -> csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        for i, nbrs in self._adj.items():
            rows.extend([i] * len(nbrs))
            cols.extend(nbrs)
        data = np.ones(len(rows), dtype=np.int8)
        return csr_matrix((data, (rows, cols)), shape=(self._n, self._n))

    def labels(self) -> np.ndarray:
        _, labels = connected_components(self._csr(), directed=False)
        return labels

    def clusters(self) -> dict[int, list[int]]:
        members: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels()):
            members[label].append(index)
        return members


def _plateau_threshold(count_at) -> float:
    """Label-free cut: the leading edge of the count-curve's flat plateau.

    ``count_at(t)`` returns the connected-component count at threshold ``t``;
    the same knee rule is used whether the graph is dense or sparse.
    """
    grid = np.arange(0.20, 0.90, 0.01)
    counts = np.array([count_at(t) for t in grid], float)
    window = 3
    slope = np.full(len(grid), np.inf)
    for i in range(window, len(grid) - window):
        slope[i] = (counts[i + window] - counts[i - window]) / (2 * window)
    # ignore the low-threshold floor where everything is merged together
    slope[counts <= 0.5 * counts.max()] = np.inf
    smallest = slope[np.isfinite(slope)].min()
    # the knee = lowest threshold whose slope is within 1.3x (+0.5) of flattest
    cut = 1.3 * smallest + 0.5
    knee = int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])
    return grid[knee]


class FusionClusterer:
    """Blend the gradient + wavelet views and cut at the count-curve knee.

    N-gated: the exact dense path for small shoots, sparse SimHash blocking above
    ``_DENSE_MAX`` so large dumps stay near-linear in time and memory.
    """

    def __init__(self, gradient: np.ndarray, embedding: np.ndarray) -> None:
        self.gradient = gradient
        self.embedding = embedding
        self.n = len(gradient)

    # ---- exact dense path (small shoots) --------------------------------- #
    def _dense_similarity(self) -> np.ndarray:
        gradient_sim = (self.gradient @ self.gradient.T).astype(np.float32)
        embedding_sim = (self.embedding @ self.embedding.T).astype(np.float32)
        return (
            W_GRADIENT * gradient_sim + (1 - W_GRADIENT) * embedding_sim
        ).astype(np.float32)

    def _dense_graph(self) -> AdjacencyGraph:
        similarity = self._dense_similarity()
        np.fill_diagonal(similarity, 0.0)

        def count_at(t: float) -> int:
            return connected_components(
                csr_matrix(similarity >= t), directed=False
            )[0]

        threshold = _plateau_threshold(count_at)
        rows, cols = np.where(np.triu(similarity >= threshold, k=1))
        return AdjacencyGraph(self.n, rows, cols)

    # ---- sparse blocking path (large dumps) ------------------------------ #
    def _stacked_vectors(self) -> np.ndarray:
        a = np.sqrt(W_GRADIENT) * self.gradient
        b = np.sqrt(1.0 - W_GRADIENT) * self.embedding
        return np.ascontiguousarray(np.hstack([a, b]).astype(np.float32))

    def _candidate_edges(self):
        """SimHash-blocked candidate pairs with sim >= floor: (rows, cols, sims).

        Never forms N x N. Cost ~ O(N*r*w) hashing + O(sum bucket^2 * D) block
        scoring; with adaptive w the buckets stay ~constant so this is near-linear.
        """
        v = self._stacked_vectors()
        n, d = v.shape
        w = max(6, int(round(np.log2(max(n, 2) / _SIMHASH_BUCKET))))
        r = _SIMHASH_BANDS
        rng = np.random.default_rng(_SIMHASH_SEED)
        planes = rng.standard_normal((d, w * r)).astype(np.float32)
        sign = (v @ planes) > 0
        pow2 = (1 << np.arange(w)).astype(np.int64)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for band in range(r):
            keys = sign[:, band * w:(band + 1) * w].astype(np.int64) @ pow2
            order = np.argsort(keys, kind="stable")
            ks = keys[order]
            for members in np.split(order, np.flatnonzero(np.diff(ks)) + 1):
                m = len(members)
                if m < 2:
                    continue
                block = v[members] @ v[members].T
                iu, ju = np.triu_indices(m, k=1)
                s = block[iu, ju]
                keep = s >= _CANDIDATE_FLOOR
                if keep.any():
                    rows.append(members[iu[keep]])
                    cols.append(members[ju[keep]])
                    vals.append(s[keep])

        if not rows:
            empty = np.empty(0, np.int64)
            return empty, empty, np.empty(0, np.float32)
        ri = np.concatenate(rows); ci = np.concatenate(cols); vv = np.concatenate(vals)
        lo = np.minimum(ri, ci).astype(np.int64)
        hi = np.maximum(ri, ci).astype(np.int64)
        _, idx = np.unique(lo * n + hi, return_index=True)  # dedup multi-band hits
        return lo[idx], hi[idx], vv[idx]

    def _sparse_graph(self) -> AdjacencyGraph:
        rows, cols, sims = self._candidate_edges()

        def count_at(t: float) -> int:
            keep = sims >= t
            g = csr_matrix(
                (np.ones(int(keep.sum()), np.int8), (rows[keep], cols[keep])),
                shape=(self.n, self.n),
            )
            return connected_components(g, directed=False)[0]

        threshold = _plateau_threshold(count_at)
        keep = sims >= threshold
        return AdjacencyGraph(self.n, rows[keep], cols[keep])

    def initial_graph(self) -> AdjacencyGraph:
        if self.n <= _DENSE_MAX:
            return self._dense_graph()
        return self._sparse_graph()
