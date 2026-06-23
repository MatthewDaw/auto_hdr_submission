"""FIX2 — re-attach clipped orphans along an exposure ladder.

A near-black or near-white bracket often strands as a tiny cluster because its
edges are washed out. We try to slot each such orphan back next to the
brightness-adjacent frame of another cluster: if a real exposure step (>=25
gray levels) plus a strong masked edge match links them, and that match is
clearly the best option (unique by margin), we add the edge.
"""
from __future__ import annotations

import numpy as np

from ..clustering import AdjacencyGraph
from .context import RefinementContext

_CLIP_LOW, _CLIP_HIGH = 45, 210  # "orphan" brightness band to try re-attaching
_MIN_STEP = 25                   # required exposure gap to call it a ladder rung
_UNIQUE_MARGIN = 0.12            # best must beat 2nd best by this...
_MASK_THR = 0.58                 # ...or the 2nd best must itself be weak


def _accept(masked_zncc: float, overlap: int, gap: float, cluster_step: float) -> bool:
    """Calibrated rule for accepting a clipped frame onto a cluster's ladder."""
    if overlap < 1500:
        return False
    if masked_zncc >= 0.85 and overlap >= 15000:
        return True   # near-certain same scene — reattach regardless of how small
        #               the exposure step is (a 0.9 ZNCC at huge overlap is not a
        #               coincidence; the _MIN_STEP gate would otherwise strand it)
    if gap < _MIN_STEP:
        return False
    if masked_zncc >= 0.58 and gap <= 1.8 * max(cluster_step, 30.0):
        return True   # strong match that extends the exposure ladder by ~one rung
    if masked_zncc >= 0.50 and overlap >= 15000:
        return True   # weaker match but a huge well-exposed overlap
    return False


class OrphanReattachment:
    def apply(self, graph: AdjacencyGraph, ctx: RefinementContext) -> None:
        B = ctx.brightness
        labels = graph.labels()
        clusters = graph.clusters()
        # typical exposure spacing within each cluster, used to bound a new rung
        step = {
            c: (float(np.median(np.diff(np.sort(B[mem])))) if len(mem) >= 2 else 80.0)
            for c, mem in clusters.items()
        }
        orphans = [i for i in range(ctx.count) if B[i] < _CLIP_LOW or B[i] > _CLIP_HIGH]
        for i in orphans:
            if len(clusters[labels[i]]) > 2:
                continue  # only re-home true orphans; never bridge two real clusters
            scored = []
            for c, mem in clusters.items():
                if c == labels[i]:
                    continue
                j = min(mem, key=lambda k: abs(B[k] - B[i]))  # brightness-adjacent rung
                zncc, overlap = ctx.masked.score(i, j)
                if _accept(zncc, overlap, abs(B[i] - B[j]), step[c]):
                    scored.append((zncc, j))
            scored.sort(reverse=True)
            if not scored:
                continue
            second = scored[1][0] if len(scored) > 1 else -1
            if scored[0][0] - second >= _UNIQUE_MARGIN or second < _MASK_THR:
                graph.link(scored[0][1], i)
