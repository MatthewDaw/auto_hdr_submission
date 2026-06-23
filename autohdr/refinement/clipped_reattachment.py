"""FIX2c — re-attach pure-clipped singletons by uniqueness margin.

A frame that is almost entirely black (B<12) or white (B>245) keeps only a
handful of valid pixels, so a global threshold can't place it: real matches
land anywhere from 0.27 to 0.75. But the *best* cluster still beats the
runner-up by a clear gap, so we decide each orphan by its own uniqueness margin
rather than an absolute cutoff. Each candidate cluster is represented by its
mid-exposure (fully valid) frame, which maximizes co-valid overlap with the
orphan's few surviving pixels.
"""
from __future__ import annotations

import re

import numpy as np

from .context import RefinementContext

_WELL_LOW, _WELL_HIGH = 50, 205
_MATCH_THR = 0.38       # absolute floor for the best match
_UNIQUE_MARGIN = 0.12   # best must beat 2nd best by this
_MIN_OVERLAP = 400      # minimum co-valid pixels to trust a comparison


def _seq_key(filename: str):
    """Capture key = (prefix, number) from a filename's trailing digits, e.g.
    DSC00421.jpg -> ('DSC', 421). The prefix keeps different cameras/shoots that
    reuse the same numbers (DSC_2377 vs DSC02377) from being treated as adjacent.
    The synthetic benchmark group tag (gNNNN_) is stripped so capture-order never
    keys on the ground-truth label; no-op on real filenames."""
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return stem[: m.start()], int(m.group(1))


class ClippedReattachment:
    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        B = ctx.brightness
        labels = labels.copy()

        clusters: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            clusters.setdefault(label, []).append(index)

        # mid-exposure representative for each multi-frame cluster
        reps: dict[int, int] = {}
        for label, members in clusters.items():
            well = [k for k in members if _WELL_LOW <= B[k] <= _WELL_HIGH]
            if len(members) >= 2 and well:
                reps[label] = min(well, key=lambda k: abs(B[k] - 128))

        # (prefix, number) -> row index, for consecutive-capture adjacency
        seq = {}
        if ctx.filenames is not None:
            for k, name in enumerate(ctx.filenames):
                key = _seq_key(name)
                if key is not None:
                    seq[key] = k

        for i in range(ctx.count):
            is_singleton = len(clusters[labels[i]]) == 1
            is_pure_clipped = B[i] > 245 or B[i] < 12
            if not (is_singleton and is_pure_clipped):
                continue
            scored = []
            for label, rep in reps.items():
                zncc, overlap = ctx.masked.score(i, rep)
                if overlap >= _MIN_OVERLAP:
                    scored.append((zncc, label))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= _MATCH_THR and (
                len(scored) == 1 or scored[0][0] - scored[1][0] >= _UNIQUE_MARGIN
            ):
                labels[i] = scored[0][1]
                continue
            # Fallback: a pure-clipped frame carries too little edge signal for a
            # confident masked match (own group can tie a different scene). But an
            # HDR bracket is a consecutive burst, so the frame numbered immediately
            # before/after it is almost always its own bracket. Attach to that
            # neighbour's cluster when exactly one adjacent capture exists and it
            # belongs to a real (multi-frame) cluster.
            self._attach_by_capture_order(i, labels, clusters, seq, ctx)
        return labels

    def _attach_by_capture_order(self, i, labels, clusters, seq, ctx) -> None:
        if ctx.filenames is None:
            return
        key = _seq_key(ctx.filenames[i])
        if key is None:
            return
        prefix, n = key
        neighbour_clusters = {
            labels[seq[m]]
            for m in ((prefix, n - 1), (prefix, n + 1))
            if m in seq and len(clusters[labels[seq[m]]]) >= 2
        }
        if len(neighbour_clusters) == 1:
            labels[i] = next(iter(neighbour_clusters))
