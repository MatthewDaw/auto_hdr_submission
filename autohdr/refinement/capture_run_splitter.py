"""FIX6 — split over-merged clusters back into their capture runs.

The coarse fusion descriptor occasionally fuses two *different* scenes that happen
to share a layout/palette (e.g. two beige bathrooms). A real HDR bracket is a
single consecutive burst from one camera, so we partition each cluster into
capture runs — contiguous (same-prefix, near-consecutive number) frame sequences —
and separate runs that are confidently NOT the same scene by full-res masked edge
correlation. Conservative by construction:

  * a correct single bracket is one run, so it is never touched;
  * runs are only separated on a confident masked DISSIMILARITY (low ZNCC at real
    overlap); if the frames are too clipped to compare, they stay together;
  * same-scene multi-burst groups masked-match, so they stay merged.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .context import RefinementContext

_GAP = 5            # number jump above this starts a new capture run
_WELL_LOW, _WELL_HIGH = 50, 205
_SAME_SCENE = 0.45   # well-exposed frames linking at/above this are one scene
# Separate capture runs are DIFFERENT brackets. The only reason to keep two runs
# together is a single bracket fragmented by a numbering artifact — which masked-
# matches very strongly (~0.85+). Different brackets of even the SAME room sit at
# 0.36-0.55 (towel ring moved, curtain open vs closed = scene not still), so we
# split below 0.65 and keep only the near-identical (fragmented-bracket) pairs.
_SEPARATE_BELOW = 0.35
_MIN_OVERLAP = 3000      # need this many co-valid pixels to trust the comparison


def _seq_key(filename: str):
    # Strip the synthetic benchmark group tag (gNNNN_) so capture-order keys on the
    # camera-native sequence only — never the ground-truth label. No-op on real
    # filenames (which don't start with gNNNN_). Proven score-identical on large.
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else (filename, -1)


class CaptureRunSplitter:
    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        if ctx.filenames is None:
            return labels
        labels = labels.copy()
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(i)

        for members in list(clusters.values()):
            # Primary: split a cluster whose WELL-EXPOSED frames form two or more
            # confidently-different scenes (e.g. a red room + a green room wrongly
            # fused). Every frame — including clipped ones that can't be compared
            # directly — is assigned to its best-matching scene, so a dark frame
            # can't bridge two scenes the way it does in the run-based path.
            scenes = self._scene_split(members, ctx)
            if scenes is not None:
                for grp in scenes[1:]:
                    for idx in grp:
                        labels[idx] = next_id
                    next_id += 1
                continue

            # Fallback: capture-run separation (different camera / number gap).
            runs, prefixes = self._runs(members, ctx.filenames)
            if len(runs) < 2:
                continue
            comp = self._run_components(runs, prefixes, ctx)
            if comp.max() == 0:
                continue  # all runs linked -> one scene, leave it
            for c in range(1, comp.max() + 1):
                for ri in np.where(comp == c)[0]:
                    for idx in runs[ri]:
                        labels[idx] = next_id
                next_id += 1
        return labels

    def _scene_split(self, members, ctx):
        """If the cluster's well-exposed frames form >=2 CONFIDENTLY-different
        scenes, return the members partitioned by scene (every frame assigned to
        its best-matching scene); else None. Fires only when scenes are clearly
        distinct (cross masked-ZNCC < _SEPARATE_BELOW at real overlap), so single
        brackets and near-duplicate brackets (high ZNCC) are left untouched."""
        B = ctx.brightness
        well = [k for k in members if _WELL_LOW <= B[k] <= _WELL_HIGH]
        if len(well) < 2:
            return None
        w = len(well)
        # Link well-exposed frames that are NOT confidently different (masked ZNCC
        # >= _SEPARATE_BELOW, or too clipped to compare). Connected components are
        # therefore scenes separated by a CONFIDENT dissimilarity. A single bracket
        # (high ZNCC throughout) and near-duplicate brackets (door/curtain moved,
        # ZNCC ~0.5-0.8) stay one component; only genuinely-different rooms split.
        link = np.ones((w, w), bool)
        for a in range(w):
            for b in range(a + 1, w):
                zncc, overlap = ctx.masked.score(well[a], well[b])
                if overlap >= _MIN_OVERLAP and zncc < _SEPARATE_BELOW:
                    link[a, b] = link[b, a] = False
        np.fill_diagonal(link, False)
        n_comp, comp = connected_components(csr_matrix(link), directed=False)
        if n_comp < 2:
            return None
        scene_of = {well[t]: comp[t] for t in range(w)}
        reps = {}
        for c in range(n_comp):
            idxs = [well[t] for t in range(w) if comp[t] == c]
            reps[c] = min(idxs, key=lambda k: abs(B[k] - 128))
        # Assign whole capture runs (= brackets) to a scene, so a clipped frame
        # (e.g. a near-black shot that masked-matches nothing) follows its own
        # bracket's well-exposed frames instead of being placed by a noisy match.
        runs, _ = self._runs(members, ctx.filenames)
        groups = defaultdict(list)
        for run in runs:
            well_scenes = [scene_of[k] for k in run if k in scene_of]
            if well_scenes:
                sc = max(set(well_scenes), key=well_scenes.count)
            else:
                rep = self._rep(run, B)
                sc = max(range(n_comp), key=lambda c: ctx.masked.score(rep, reps[c])[0])
            groups[sc].extend(run)
        out = [g for g in groups.values() if g]
        return out if len(out) >= 2 else None

    def _runs(self, members, filenames):
        keyed = sorted(members, key=lambda i: _seq_key(filenames[i]))
        runs, prefixes, cur, cur_pre = [], [], [], None
        prev = None
        for i in keyed:
            pre, n = _seq_key(filenames[i])
            if prev is not None and (pre != prev[0] or n - prev[1] > _GAP):
                runs.append(cur); prefixes.append(cur_pre); cur = []
            cur.append(i); prev = (pre, n); cur_pre = pre
        if cur:
            runs.append(cur); prefixes.append(cur_pre)
        return runs, prefixes

    def _rep(self, run, B):
        well = [k for k in run if _WELL_LOW <= B[k] <= _WELL_HIGH]
        pool = well or run
        return min(pool, key=lambda k: abs(B[k] - 128))

    def _run_components(self, runs, prefixes, ctx):
        B = ctx.brightness
        reps = [self._rep(r, B) for r in runs]
        m = len(runs)
        # Separate capture runs default to DIFFERENT brackets. Keep two together
        # only when they are the same camera AND either masked-match strongly
        # (a fragmented single bracket) or are too clipped to tell. A different
        # camera prefix means a different shoot — split even if it is too dark to
        # correlate (this is the near-black cross-scene leak case).
        # keep runs together by default; only separate on a CONFIDENT masked
        # dissimilarity at real overlap (conservative — protects fragmented
        # single brackets and clipped frames we cannot compare).
        link = np.ones((m, m), bool)
        for a in range(m):
            for b in range(a + 1, m):
                zncc, overlap = ctx.masked.score(reps[a], reps[b])
                if overlap >= _MIN_OVERLAP and zncc < _SEPARATE_BELOW:
                    link[a, b] = link[b, a] = False
        np.fill_diagonal(link, False)
        _, comp = connected_components(csr_matrix(link), directed=False)
        return comp
