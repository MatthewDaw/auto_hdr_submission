"""FIX4 — merge same-scene clusters the fusion threshold split apart.

When one camera angle gets split into two clusters (often by an exposure step),
the pieces are still near each other in the wavelet embedding. For each cluster
we look at its embedding-nearest neighbours (plus brightness-nearest neighbours
for small or clipped clusters — the typical split pieces) and merge them when a
brightness-adjacent cross pair has a strong masked edge link. Different rooms
score ~0.30 masked, so the masked check is the precision guard.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ..clustering import AdjacencyGraph
from .context import RefinementContext

_SMALL_CLUSTER = 3
_CLIP_LOW, _CLIP_HIGH = 45, 210
_WELL_LOW, _WELL_HIGH = 50, 205   # "well-exposed" band for representative frames
_MIN_STEP = 25


def _seq_key(filename: str):
    """(prefix, number) from trailing digits — see clipped_reattachment._seq_key.
    Strips the synthetic gNNNN_ group tag so capture-order never keys on the label."""
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else None




def _capture_adjacency(clusters, filenames):
    """cid -> set of cids that hold a frame captured immediately before/after one
    of this cluster's frames. Consecutive shots are almost always the same bracket,
    so this links an all-clipped split piece to its other half even when brightness
    and embedding proximity both fail."""
    if filenames is None:
        return defaultdict(set)
    seq, cid_of = {}, {}
    for c, members in clusters.items():
        for m in members:
            cid_of[m] = c
            key = _seq_key(filenames[m])
            if key is not None:
                seq[key] = m
    adj = defaultdict(set)
    for c, members in clusters.items():
        for m in members:
            key = _seq_key(filenames[m])
            if key is None:
                continue
            prefix, n = key
            for nb in ((prefix, n - 1), (prefix, n + 1)):
                j = seq.get(nb)
                if j is not None and cid_of[j] != c:
                    adj[c].add(cid_of[j])
    return adj


class _UnionFind:
    def __init__(self, keys):
        self._parent = {k: k for k in keys}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        self._parent[self.find(a)] = self.find(b)


class ClusterMerging:
    def apply(self, graph: AdjacencyGraph, ctx: RefinementContext) -> None:
        B, E = ctx.brightness, ctx.embedding
        clusters = graph.clusters()
        cids = list(clusters.keys())

        centroids = np.stack([
            _normalize(E[clusters[c]].mean(0)) for c in cids
        ])
        centroid_sim = centroids @ centroids.T
        mean_brightness = np.array([B[clusters[c]].mean() for c in cids])
        lo_brightness = np.array([B[clusters[c]].min() for c in cids])
        hi_brightness = np.array([B[clusters[c]].max() for c in cids])

        cid_index = {c: i for i, c in enumerate(cids)}
        capture_adj = _capture_adjacency(clusters, ctx.filenames)

        uf = _UnionFind(cids)
        for ai, c in enumerate(cids):
            members = clusters[c]
            candidates = set(np.argsort(-centroid_sim[ai])[1:20].tolist())
            # consecutive-capture neighbours are strong same-bracket candidates
            candidates |= {cid_index[c2] for c2 in capture_adj[c]}
            is_split_piece = (
                len(members) <= _SMALL_CLUSTER
                or (B[members] < _CLIP_LOW).any()
                or (B[members] > _CLIP_HIGH).any()
            )
            if is_split_piece:
                candidates |= set(
                    np.argsort(np.abs(mean_brightness - mean_brightness[ai]))[1:15].tolist()
                )
                # brightness-RANGE adjacency: a partial bracket (e.g. an all-bright
                # split piece) continues the exposure ladder of the cluster whose
                # range ends just below its own — even when their MEANS are far
                # apart and their clipped wavelets put them far in embedding space.
                range_gap = np.maximum.reduce([
                    np.zeros(len(cids)),
                    lo_brightness - hi_brightness[ai],
                    lo_brightness[ai] - hi_brightness,
                ])
                candidates |= set(np.argsort(range_gap)[1:15].tolist())
            for bi in sorted(candidates):  # deterministic order
                c2 = cids[bi]
                if uf.find(c) == uf.find(c2):
                    continue
                if self._should_merge(clusters[c], clusters[c2], ctx):
                    uf.union(c, c2)

        # realize merges by linking a representative edge between unioned clusters
        for c in cids:
            root = uf.find(c)
            if root != c:
                graph.link(clusters[c][0], clusters[root][0])

    def _should_merge(self, members_a, members_b, ctx: RefinementContext) -> bool:
        B = ctx.brightness
        # best brightness-adjacent cross-frame masked link (needs a real exposure step)
        best = (-1.0, 0, 999.0)
        for x in members_a:
            y = min(members_b, key=lambda k: abs(B[k] - B[x]))
            gap = abs(B[x] - B[y])
            if gap < _MIN_STEP:
                continue
            zncc, overlap = ctx.masked.score(x, y)
            if zncc > best[0]:
                best = (zncc, overlap, gap)
        zncc, overlap, gap = best
        if overlap >= 1500 and (
            (zncc >= 0.62 and gap <= 120) or (zncc >= 0.50 and overlap >= 15000)
        ):
            return True

        # well-exposed-rep path: same scene split at OVERLAPPING exposures (no >=25
        # step, so the ladder rule above can't fire). Their mid-exposure reps still
        # masked-match strongly if it is one scene; different rooms score ~0.30.
        well_a = [k for k in members_a if _WELL_LOW <= B[k] <= _WELL_HIGH]
        well_b = [k for k in members_b if _WELL_LOW <= B[k] <= _WELL_HIGH]
        if well_a and well_b:
            r1 = min(well_a, key=lambda k: abs(B[k] - 128))
            r2 = min(well_b, key=lambda k: abs(B[k] - 128))
            rep_zncc, rep_overlap = ctx.masked.score(r1, r2)
            if rep_overlap >= 1500 and rep_zncc >= 0.55:
                return True
        return False


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)
