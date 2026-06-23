"""SPLIT a foreign near-black frame out of a cluster by clipped-vs-clipped lights.

Root cause this pass targets
----------------------------
Two different dark scenes fuse when one contributes a heavily under-exposed
(near-black, mean < 30) frame: masked edge ZNCC is blind there, so the foreign
frame bridges into a cluster it does not belong to (e.g. 1038's B5 landing in
10280's bracket). The other passes cannot see it.

Signal (pixel-only, training-free, deterministic, browser-portable)
------------------------------------------------------------------
The few SATURATED point-lights that survive in a near-black frame are the scene's
light sources at fixed positions. Two near-black frames of the SAME scene share
those positions (clipped-vs-clipped light match > 0); a foreign scene's dark frame
shares none (match 0). Crucially this compares dark-to-dark (both clipped the same
way), so it is exposure-robust — unlike a dark-vs-well comparison whose lights are
exposure-shifted and information-limited.

Within a cluster we connect its near-black frames (each with >= _MIN_LIGHTS detected
lights) whenever their light match exceeds zero, take connected components, and if
two or more components exist the minority components are foreign scenes and split
off. A genuine single bracket's dark frames all share light positions => one
component => never splits (validated: 0 false-splits across 195 same-scene dark
frames in data/large).

No filenames, no labels, no ML, no randomness.
"""
from __future__ import annotations

import numpy as np

from ..features import extreme_anchor as _ea
from .context import RefinementContext

_DARK_MAX = 30.0     # near-black regime where bright spots are the only cue
_MIN_LIGHTS = 2      # need >= this many lights to judge a match (avoids 1-light flukes)
_TOL = 0.06          # spot-match tolerance (fraction of frame)
_MAX_T = 0.10        # max global translation searched (same camera)


def _match_points(A: np.ndarray, B: np.ndarray) -> float | None:
    if len(A) < _MIN_LIGHTS or len(B) < _MIN_LIGHTS:
        return None
    cands = [np.zeros(2, np.float32)]
    for a in A:
        for b in B:
            t = b - a
            if np.hypot(*t) <= _MAX_T:
                cands.append(t)
    best = 0
    for t in cands:
        At = A + t
        pr = sorted((float(np.hypot(*(At[i] - B[j]))), i, j)
                    for i in range(len(At)) for j in range(len(B)))
        ui: set = set(); uj: set = set(); n = 0
        for d, i, j in pr:
            if d > _TOL:
                break
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j); n += 1
        best = max(best, n)
    return best / min(len(A), len(B))


class ClippedForeignSplitter:
    """Split clusters that fused different dark scenes via an indistinguishable
    near-black bridge frame. ``apply(labels, ctx) -> labels``. Pixel-only."""

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None:
            return labels
        labels = labels.copy()
        B = ctx.brightness
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        # cache each near-black frame's saturated lights once (the expensive op)
        lights: dict[int, np.ndarray] = {}
        for i in range(len(labels)):
            if B[i] < _DARK_MAX:
                lights[i] = _ea.clip_lights(gray[i])

        for members in list(clusters.values()):
            darks = [k for k in members
                     if k in lights and len(lights[k]) >= _MIN_LIGHTS]
            if len(darks) < 2:
                continue
            # connect dark frames that share any light position (same scene)
            n = len(darks)
            link = np.zeros((n, n), bool)
            for a in range(n):
                for b in range(a + 1, n):
                    m = _match_points(lights[darks[a]], lights[darks[b]])
                    if m is not None and m > 0.0:
                        link[a, b] = link[b, a] = True
            comp = self._components(link)
            groups: dict[int, list[int]] = {}
            for li, k in enumerate(darks):
                groups.setdefault(comp[li], []).append(k)
            if len(groups) < 2:
                continue  # one coherent dark scene -> nothing foreign
            # keep the largest dark-light component on the cluster; split the rest
            ordered = sorted(groups.values(), key=len, reverse=True)
            for grp in ordered[1:]:
                for k in grp:
                    labels[k] = next_id
                next_id += 1
        return labels

    @staticmethod
    def _components(link: np.ndarray) -> list[int]:
        w = link.shape[0]
        comp = [-1] * w
        c = 0
        for s in range(w):
            if comp[s] >= 0:
                continue
            stack = [s]; comp[s] = c
            while stack:
                u = stack.pop()
                for v in range(w):
                    if link[u, v] and comp[v] < 0:
                        comp[v] = c; stack.append(v)
            c += 1
        return comp
