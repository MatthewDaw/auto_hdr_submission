"""FIX7 — split brackets that captured a SCENE CHANGE.

A single HDR bracket captures each exposure level once and the scene is perfectly
still. So if a cluster holds two frames at the SAME exposure level (near-equal
brightness) that DIFFER at all beyond sensor noise — an object moved (door,
cabinet, flag) OR the viewpoint shifted — the scene was not still between them,
so they are not HDR-mergeable and must be separated. Pure exposure differences
are ruled out by comparing frames at matched brightness on CLAHE-normalized
intensity; sensor noise is ruled out by requiring a real change.

Once a change is *flagged*, the cluster is partitioned into its brackets in
capture order (a new bracket starts whenever an exposure level repeats) and the
partition is then VERIFIED by full-resolution masked edge-ZNCC: a candidate
split only commits when its brackets are confidently DIFFERENT scenes (low ZNCC
at real pixel overlap). This guard is essential — the CLAHE-intensity change
signal is sensitive to exposure/clipping artifacts and on its own fires on
genuinely-still scenes (window glare shifting, a frame washing toward white).
Edge-ZNCC is exposure-invariant, so brackets that stay well-correlated (a still
scene merely re-exposed, or a small object that moved but the scene is otherwise
identical) are re-merged and the cluster is left as GT has it.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .context import RefinementContext

_LEVEL = 12          # frames within this brightness are the "same exposure level"
_PEAK = 80.0         # change above this (vs ~31 noise median) is real, not noise
_MIN_VALID = 12000   # co-valid pixels needed for a trustworthy comparison
_MIN_CHANGED = 300   # below this the change is just noise / a few stray pixels
_WELL_LOW, _WELL_HIGH = 45, 215   # only compare well-exposed frames (clipped
#                                   frames give unreliable CLAHE differences)
_SEPARATE_BELOW = 0.35   # brackets are confidently different scenes below this
_MIN_OVERLAP = 3000      # co-valid pixels needed to trust a masked comparison


class SceneChangeSplitter:
    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._norm: dict[int, np.ndarray] = {}

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None:
            return labels
        labels = labels.copy()
        next_id = int(labels.max()) + 1
        B = ctx.brightness

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(i)

        for members in list(clusters.values()):
            if len(members) < 4:
                continue
            if not self._has_change(members, B, gray):
                continue
            brackets = self._into_brackets(members, B, ctx.filenames)
            if len(brackets) < 2:
                continue
            # VERIFY: re-merge brackets that are not confidently different scenes
            # by exposure-invariant masked edge-ZNCC. Only genuine scene changes
            # (low ZNCC at real overlap) survive; high-ZNCC near-duplicates and
            # clipped-frame artifacts collapse back to one group.
            scenes = self._confirm(brackets, B, ctx.masked)
            if len(scenes) < 2:
                continue
            for grp in scenes[1:]:
                for idx in grp:
                    labels[idx] = next_id
                next_id += 1
        self._norm.clear()
        return labels

    def _confirm(self, brackets, B, masked):
        """Merge brackets whose well-exposed representatives are NOT confidently
        different (masked ZNCC >= _SEPARATE_BELOW, or too clipped to compare).
        Returns the surviving scene groups (>=2 only when a real change holds)."""
        reps = [self._rep(grp, B) for grp in brackets]
        m = len(brackets)
        link = np.ones((m, m), bool)   # link = "same scene (keep together)"
        for a in range(m):
            for b in range(a + 1, m):
                if reps[a] is None or reps[b] is None:
                    continue
                zncc, overlap = masked.score(reps[a], reps[b])
                if overlap >= _MIN_OVERLAP and zncc < _SEPARATE_BELOW:
                    link[a, b] = link[b, a] = False
        np.fill_diagonal(link, False)
        _, comp = connected_components(csr_matrix(link), directed=False)
        groups: dict[int, list[int]] = {}
        for bi, c in enumerate(comp):
            groups.setdefault(int(c), []).extend(brackets[bi])
        return list(groups.values())

    @staticmethod
    def _rep(grp, B):
        """Most well-exposed frame of a bracket (closest to mid-grey), or None
        if the whole bracket is clipped beyond the comparable range."""
        well = [k for k in grp if _WELL_LOW <= B[k] <= _WELL_HIGH]
        if not well:
            return None
        return min(well, key=lambda k: abs(B[k] - 128))

    def _n(self, i, gray):
        if i not in self._norm:
            self._norm[i] = self._clahe.apply(gray[i]).astype(np.float32)
        return self._norm[i]

    def _changed(self, i, j, gray):
        """(peak, #changed-px) of the masked CLAHE-normalized difference."""
        a, b = self._n(i, gray), self._n(j, gray)
        v = (gray[i] >= 8) & (gray[i] <= 247) & (gray[j] >= 8) & (gray[j] <= 247)
        if v.sum() < _MIN_VALID:
            return 0.0, 0
        d = cv2.GaussianBlur(np.abs(a - b), (0, 0), 3)
        d[~v] = 0
        return float(d.max()), int((d > 40).sum())

    def _has_change(self, members, B, gray):
        well = [k for k in members if _WELL_LOW <= B[k] <= _WELL_HIGH]
        for a in range(len(well)):
            for b in range(a + 1, len(well)):
                i, j = well[a], well[b]
                if abs(B[i] - B[j]) <= _LEVEL:
                    peak, changed = self._changed(i, j, gray)
                    if peak >= _PEAK and changed >= _MIN_CHANGED:
                        return True
        return False

    def _into_brackets(self, members, B, filenames):
        # capture order; start a new bracket when an exposure level repeats (a
        # single bracket visits each level once, so a repeat means the next bracket
        # has begun).
        cap = (lambda x: _capture(filenames[x])) if filenames else (lambda x: B[x])
        order = sorted(members, key=cap)
        brackets, cur = [], []
        for k in order:
            if any(abs(B[k] - B[p]) <= _LEVEL for p in cur):
                brackets.append(cur); cur = []
            cur.append(k)
        if cur:
            brackets.append(cur)
        return brackets


def _capture(name):
    import re
    m = re.search(r"(\d+)\D*$", name)
    return int(m.group(1)) if m else 0
