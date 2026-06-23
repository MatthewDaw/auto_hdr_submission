"""FIX8 — re-merge same-scene clusters the splitters over-separated, by COLOUR.

The base clusterer + edge-ZNCC splitters sometimes leave ONE scene as two clusters
when its brackets correlate only weakly on edges (e.g. a flat sign/monument at a
clipped exposure: edge-ZNCC ~0.38, below every merge bar). But two frames of that
one scene at the SAME exposure level are near-identical in mid-tone chromaticity.

Colour is reliable as a MERGE confirmer (it failed only as a *separator*, where
bright bridge frames lose discrimination — see the reverted veto). This pass runs
LAST so nothing re-splits its merges. It merges two clusters when a same-exposure-
level mid pair is near-identical in colour AND still shows a moderate edge link —
the edge floor preventing a coincidental colour match between different rooms.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .context import RefinementContext

_MID_LOW, _MID_HIGH = 70, 185
_LEVEL = 30          # compare colour only within this brightness gap (same level)
_COLOR_SAME = 0.06   # same-level chroma divergence below this = same scene
_EDGE_FLOOR = 0.30   # require at least this much edge link (precision guard)
_MIN_OVERLAP = 1500


class ColorMerge:
    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        chroma = getattr(ctx, "chroma", None)
        if chroma is None:
            return labels
        B = ctx.brightness
        labels = labels.copy()

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)
        cids = list(clusters.keys())
        if len(cids) < 2:
            return labels

        # mid-band frames per cluster (the only ones colour can be trusted on)
        mids = {c: [k for k in clusters[c] if _MID_LOW <= B[k] <= _MID_HIGH and chroma.has(k)]
                for c in cids}

        idx = {c: i for i, c in enumerate(cids)}
        link = np.zeros((len(cids), len(cids)), bool)
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                if self._same_scene(mids[cids[a]], mids[cids[b]], B, ctx):
                    link[a, b] = link[b, a] = True
        n_comp, comp = connected_components(csr_matrix(link), directed=False)
        if n_comp == len(cids):
            return labels  # nothing merged

        # relabel each connected component to a single id
        for a, c in enumerate(cids):
            new = cids[int(np.where(comp == comp[a])[0][0])]
            if new != c:
                for k in clusters[c]:
                    labels[k] = new
        return labels

    def _same_scene(self, mids_a, mids_b, B, ctx) -> bool:
        for x in mids_a:
            for y in mids_b:
                if abs(B[x] - B[y]) > _LEVEL:
                    continue
                d = ctx.chroma.diverge(x, y)
                if d is None or d > _COLOR_SAME:
                    continue
                z, o = ctx.masked.score(x, y)
                if o >= _MIN_OVERLAP and z >= _EDGE_FLOOR:
                    return True
        return False
