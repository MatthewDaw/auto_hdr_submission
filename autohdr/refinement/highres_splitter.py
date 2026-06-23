"""FIX5 — split coarse-descriptor over-merges using 256px masked correlation.

The 64x64 fusion descriptor sometimes merges similar-layout-but-different rooms
(fusion ~0.70, yet 256px masked ~0.30). Within each predicted cluster we
re-examine the members with the sharp masked ZNCC. Three passes, in order, peel
apart the ways an over-merge shows up — each guarded so legitimately varied
single-angle groups are left intact:

  Pass 1  a well-exposed frame that masked-links to nothing -> a wrong singleton
  Pass 2  two or more internally-tight bracket-sets fused together
  Pass 3  two different rooms joined only by a weak bridge, separated via their
          well-exposed sub-scenes and each frame assigned to its best match

Exposure-ladder chaining is preserved: members link transitively, so genuine
brackets that share no single strong pair still stay together via intermediates.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .context import RefinementContext

_MIN_CLUSTER = 3
_LINK_THR = 0.38          # masked ZNCC above which two members are "same scene"
_WELL_LOW, _WELL_HIGH = 55, 200
_SUBSCENE_LINK = 0.50     # well-exposed frames linked above this share an angle.
#                           A genuinely same-angle pair sits well above this even
#                           across exposure; two different rooms can correlate up to
#                           ~0.46 by shared layout, so a single such bridge must not
#                           fuse two internally-tight cliques (the _SEPARATION_GAP
#                           check below still guards true brackets from over-splitting).
_SEPARATION_GAP = 0.15    # a sub-scene's internal cohesion must beat its best
                          # cross-link by at least this to count as distinct


class HighResSplitter:
    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        self._score = ctx.masked.score
        B = ctx.brightness
        new_labels = labels.copy()
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            clusters.setdefault(label, []).append(index)

        for members in clusters.values():
            k = len(members)
            if k < _MIN_CLUSTER:
                continue

            # masked ZNCC link graph within the cluster, plus each member's best
            # (max) score to any sibling
            link = np.zeros((k, k), bool)
            best = np.full(k, -2.0)
            for a in range(k):
                for b in range(a + 1, k):
                    zncc, _ = self._score(members[a], members[b])
                    best[a], best[b] = max(best[a], zncc), max(best[b], zncc)
                    if zncc >= _LINK_THR:
                        link[a, b] = link[b, a] = True
            n_comp, sub = connected_components(csr_matrix(link), directed=False)
            sizes = np.bincount(sub, minlength=n_comp)
            largest = int(np.argmax(sizes))
            changed = False

            # Pass 1: split off well-exposed frames that link to nothing (wrongly
            # merged singletons). Clipped orphans are protected — FIX2 legitimately
            # attaches them with low masked overlap.
            for i, m in enumerate(members):
                if sizes[sub[i]] == 1 and _WELL_LOW <= B[m] <= _WELL_HIGH and best[i] < 0.32:
                    new_labels[m] = next_id
                    next_id += 1
                    changed = True

            # Pass 2: multiple internally-tight bracket-sets wrongly merged. The
            # tightness guard distinguishes this from a legitimately varied group.
            multi = [comp for comp in range(n_comp) if sizes[comp] >= 2]
            if len(multi) >= 2 and all(
                self._component_min(members, sub, comp) >= 0.55 for comp in multi
            ):
                for i, m in enumerate(members):
                    if sizes[sub[i]] >= 2 and sub[i] != largest:
                        new_labels[m] = next_id + sub[i]
                next_id += n_comp
                changed = True

            # Pass 3: different rooms joined by a weak bridge — split by well-exposed
            # sub-scenes (where masked is reliable), then assign every frame to the
            # sub-scene it best matches.
            if not changed:
                next_id = self._split_by_rooms(members, k, B, new_labels, next_id)
        return new_labels

    def _component_min(self, members, sub, comp) -> float:
        idx = [members[i] for i in range(len(members)) if sub[i] == comp]
        worst = 2.0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                worst = min(worst, self._score(idx[a], idx[b])[0])
        return worst

    def _split_by_rooms(self, members, k, B, new_labels, next_id) -> int:
        well = [i for i in range(k) if _WELL_LOW <= B[members[i]] <= _WELL_HIGH]
        if len(well) < 4:
            return next_id
        # cluster the well-exposed frames into sub-scenes
        w = len(well)
        link = np.zeros((w, w), bool)
        wv = np.zeros((w, w))
        for a in range(w):
            for b in range(a + 1, w):
                v, _ = self._score(members[well[a]], members[well[b]])
                wv[a, b] = wv[b, a] = v
                if v >= _SUBSCENE_LINK:
                    link[a, b] = link[b, a] = True
        n_comp, wsub = connected_components(csr_matrix(link), directed=False)
        sizes = np.bincount(wsub, minlength=n_comp)
        multi = [comp for comp in range(n_comp) if sizes[comp] >= 2]

        def internal_min(comp) -> float:
            idx = [t for t in range(w) if wsub[t] == comp]
            worst = 2.0
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    worst = min(worst, wv[idx[a], idx[b]])
            return worst

        def cross_best(comp) -> float:
            inside = [t for t in range(w) if wsub[t] == comp]
            outside = [t for t in range(w) if wsub[t] != comp]
            return max((wv[a, b] for a in inside for b in outside), default=-1.0)

        # A sub-scene is a genuinely distinct camera angle (not just exposure
        # variation) when its members cohere far more tightly with each other
        # than with any frame outside it. This relative gap is robust where an
        # absolute tightness floor was not: two near-identical angles can sit at
        # masked ZNCC ~0.6 internally yet ~0.2 across the gap — clearly separate,
        # but below any single fixed "tight enough" line.
        if len(multi) < 2 or not all(
            internal_min(comp) >= cross_best(comp) + _SEPARATION_GAP
            for comp in multi
        ):
            return next_id
        # mid-exposure representative of each sub-scene
        reps = {
            comp: min(
                (well[t] for t in range(w) if wsub[t] == comp),
                key=lambda i: abs(B[members[i]] - 128),
            )
            for comp in multi
        }
        base = multi[0]
        for m in members:
            best_comp = max(
                multi, key=lambda comp: self._score(members[reps[comp]], m)[0]
            )
            if best_comp != base:
                new_labels[m] = next_id + best_comp
        return next_id + n_comp
