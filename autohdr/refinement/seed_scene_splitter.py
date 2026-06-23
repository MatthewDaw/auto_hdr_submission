"""FIX9 — split a wrongly-merged cluster into scenes by SEED ASSIGNMENT.

Two different scenes with a similar layout (a house vs a field, two beige rooms)
get fused because their edges correlate ~0.4. Connected-components can't separate
them: one bright bridge frame (which loses both edge and colour discrimination)
reconnects the two. Seed assignment is non-transitive and immune to that:

  1. SEEDS: among mid-exposed frames, find the most DIVERGENT pair (low edge-ZNCC
     or high same-exposure-level colour divergence). One seed -> one scene, never
     split. Two+ mutually-divergent seeds -> a genuine multi-scene cluster.
  2. ASSIGN: every frame joins its NEAREST seed by masked edge-ZNCC; frames too
     clipped to score fall back to capture-order nearest seed. A bridge frame can
     only join ONE seed, so it cannot re-merge the scenes.

Colour is used here only as a same-exposure-level SEED test (where it is reliable),
never to bridge across exposures — so it does not reintroduce the reverted veto's
bright-frame failure.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .context import RefinementContext

_MID_LOW, _MID_HIGH = 70, 185
_EDGE_DIFF = 0.35       # edge-ZNCC below this (at overlap) => confidently different
_COLOR_LEVEL = 25       # only trust colour within this brightness gap
_COLOR_DIFF = 0.20      # same-level chroma divergence above this => different scene.
#                         Within a single bracket, same-level mid divergence stays
#                         <=0.184 (measured); genuine different scenes sit at 0.23+.
_MIN_OVERLAP = 3000
_MIN_MEMBERS = 4        # only consider sizable clusters for a scene split


def _seqnum(name: str) -> int:
    stem = re.sub(r"^g\d+_", "", name.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1


class SeedSceneSplitter:
    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        if getattr(ctx, "chroma", None) is None:
            return labels
        labels = labels.copy()
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        for members in list(clusters.values()):
            if len(members) < _MIN_MEMBERS:
                continue
            scenes = self._seed_scenes(members, ctx)
            if len(scenes) < 2:
                continue
            for grp in scenes[1:]:
                for idx in grp:
                    labels[idx] = next_id
                next_id += 1
        return labels

    def _seed_scenes(self, members, ctx):
        B = ctx.brightness

        def diff_score(i, j):
            # SEEDING uses colour ONLY, at the same exposure level — the high-
            # precision signal. (Edge-ZNCC dips below 0.35 within a single bracket
            # on bright-window frames, which would spawn spurious seeds and
            # over-split genuine brackets.) Edge is used for ASSIGNMENT below.
            if _MID_LOW <= B[i] <= _MID_HIGH and _MID_LOW <= B[j] <= _MID_HIGH \
               and abs(B[i] - B[j]) <= _COLOR_LEVEL:
                d = ctx.chroma.diverge(i, j)
                if d is not None and d >= _COLOR_DIFF:
                    return 1.0
            return 0.0

        mids = [k for k in members if _MID_LOW <= B[k] <= _MID_HIGH]
        best, bp = 0.0, None
        for a in range(len(mids)):
            for b in range(a + 1, len(mids)):
                s = diff_score(mids[a], mids[b])
                if s > best:
                    best, bp = s, (mids[a], mids[b])
        if bp is None or best <= 0:
            return [members]                         # one scene — no split
        seeds = list(bp)
        for k in mids:
            if k not in seeds and all(diff_score(k, s) > 0 for s in seeds):
                seeds.append(k)

        groups = defaultdict(list)
        for i in members:
            scored = [(ctx.masked.score(i, s)[0], ctx.masked.score(i, s)[1], si)
                      for si, s in enumerate(seeds)]
            usable = [(z, si) for z, o, si in scored if o >= _MIN_OVERLAP]
            if usable:
                best_si = max(usable)[1]
            elif ctx.filenames is not None:          # too clipped — capture-order
                best_si = min(range(len(seeds)),
                              key=lambda si: abs(_seqnum(ctx.filenames[i]) - _seqnum(ctx.filenames[seeds[si]])))
            else:
                best_si = max(scored)[2]
            groups[best_si].append(i)
        return [g for g in groups.values() if g]
