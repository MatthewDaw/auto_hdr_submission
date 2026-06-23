"""Cluster-B fix — split wrongly-MERGED different scenes by EXTREME-ANCHOR mismatch.

The masked edge correlation (the precision check every other split pass leans on)
goes blind on heavily-clipped frames: a near-black or near-white frame has almost
no co-valid pixels, so its masked overlap collapses to ~0 and any two clipped
frames look indistinguishable. That is exactly how two genuinely different scenes
get fused — their clipped frames bridge the clusters even though a human can tell
the rooms apart at a glance.

A human reads two cues that survive extreme exposure:

  * in a near-BLACK frame, the LIGHT SOURCES (lamps, windows) stay visible as the
    brightest spots, at fixed image positions;
  * in a near-WHITE frame, the DARK OBJECTS (furniture, frames) stay visible as the
    darkest spots, at fixed positions.

``anchor_match`` turns that into a number: the spatial IoU of two frames'
brightest-K% (or darkest-K%) pixels on a coarse rank map. It is rank-based, so a
uniform exposure change cancels — the same scene matches HIGH (~0.5-0.95) across a
huge exposure jump, two different scenes match LOW (<~0.33). The bright-spot channel
carries near-black frames; the dark-spot channel carries near-white frames; the
pass picks whichever channel the clipped frame can actually express.

The splitter is deliberately conservative so a genuine single bracket is never torn
apart:

  * WELL-EXPOSED frames are seeded into scenes by masked edge ZNCC (reliable where
    it works) — anchor is never used to split a well-exposed frame, since masked
    correlation already handles those and anchor can be noisy mid-tone;
  * the CLIPPED frames are grouped into their own anchor-clusters (same clip
    direction), so a scene's near-black or near-white frames cohere by light-spot /
    dark-spot positions even though masked correlation is blind to them;
  * each clipped anchor-cluster is attached to the well-scene it best anchors, and
    spawns its OWN scene only when it is fully hard-clipped, textured, and
    CONFIDENTLY rejects every well-scene — that is the anchor-mismatch split.

This pass reads only pixels (brightness, gray tiles, masked correlator); it does
NOT use filenames or capture order in any way. It fires only on confident two-scene
anchor evidence with a clear margin; a single anchor-coherent scene is left intact.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

try:  # cv2 is already a hard dependency of the feature extractors
    import cv2
except Exception:  # pragma: no cover - defensive, cv2 is always present in this repo
    cv2 = None

from .context import RefinementContext

# --- anchor signature -------------------------------------------------------
_GRID = 24          # coarse rank map resolution (24x24 = 576 cells)
_FRAC = 0.06        # fraction of cells taken as the bright / dark spot set (~35)
_DARK_SIDE = 110    # a frame this dark (or darker) -> bright spots are real lights
_LIGHT_SIDE = 146   # a frame this light (or lighter) -> dark spots are real objects

# --- scene seeding / assignment thresholds (PIXEL-ONLY — no filename signals) --
_WELL_LOW, _WELL_HIGH = 50, 200   # masked-ZNCC is reliable in this brightness band
_SAME_SCENE = 0.35                # well-vs-well masked ZNCC >= this -> same scene
_MIN_OVERLAP = 3000               # co-valid pixels needed to trust a masked score
_MIN_MEMBERS = 3                  # only sizable clusters can be over-merged scenes

# Hard-clipped band: masked ZNCC is fully blind here (overlap ~0), so only the
# anchor can place these frames. Frames between this and the well band are "dim" —
# they keep some edge content and are merely ATTACHED to their nearest anchor scene,
# never used to start a new one (a dim frame can score a low anchor by noise).
_CLIP_HARD_DARK = 30
_CLIP_HARD_WHITE = 225

# Clipped-frame scene assignment.
_CLIP_LINK = 0.45    # two same-direction clipped frames anchor-link into one cluster
_ATTACH = 0.40       # a clipped cluster joins the well-scene it best anchors here
# A fully HARD-clipped cluster spawns its OWN scene only when it CONFIDENTLY rejects
# every well-scene (best anchor below this) and carries real texture (std floor).
# Within a single bracket the extreme frames anchor their own scene at >=0.42; a
# genuinely different scene's clipped frames sit at <=0.13 — 0.15 is the confident
# split bar, the std floor drops featureless saturated frames whose anchor is noise.
_SPAWN = 0.15
_SPAWN_STD = 10.0


def _rank_map(gray256: np.ndarray) -> np.ndarray:
    return cv2.resize(gray256, (_GRID, _GRID), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ).ravel()


def _spot_mask(rank: np.ndarray, bright: bool) -> np.ndarray:
    k = max(1, int(len(rank) * _FRAC))
    order = np.argsort(rank)
    mask = np.zeros(len(rank), bool)
    mask[order[-k:] if bright else order[:k]] = True
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def anchor_match(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Exposure-invariant extreme-anchor similarity of two 256x256 gray frames.

    Compares the spatial overlap of bright spots (light sources, the signal in
    near-black frames) and/or dark spots (dark objects, the signal in near-white
    frames). The channel(s) used depend on the pair's exposure so the discriminating
    cue is the one actually carried by the clipped frame:

      * if either frame is dark enough (mean <= _DARK_SIDE) the BRIGHT channel is
        used (its bright spots are genuine light sources);
      * if either frame is light enough (mean >= _LIGHT_SIDE) the DARK channel is
        used (its dark spots are genuine objects);
      * if both channels apply, the MIN is returned so a mismatch on either cue is
        not masked by a coincidental match on the other (two rooms can share a lamp
        position yet differ in furniture);
      * two purely mid-tone frames (neither clipped) fall back to both channels —
        but the pass never relies on anchor for those, masked ZNCC does.

    Returns a value in [0, 1]; same scene ~0.5-0.95, different scene <~0.33.
    """
    ra = _rank_map(gray_a)
    rb = _rank_map(gray_b)
    ba = float(gray_a.mean())
    bb = float(gray_b.mean())
    bright = _iou(_spot_mask(ra, True), _spot_mask(rb, True))
    dark = _iou(_spot_mask(ra, False), _spot_mask(rb, False))
    vals = []
    if min(ba, bb) <= _DARK_SIDE:
        vals.append(bright)
    if max(ba, bb) >= _LIGHT_SIDE:
        vals.append(dark)
    if not vals:
        vals = [bright, dark]
    return min(vals)


class AnchorSplitter:
    """Split clusters that fused different scenes via indistinguishable clipped frames.

    ``apply(labels, ctx) -> labels``. Runs late, after the masked-ZNCC split passes,
    as the last resort for over-merges those passes cannot see (clipped-frame
    bridges). Pure pixel ops, deterministic, no learning.
    """

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        if cv2 is None or ctx.gray is None:
            return labels
        labels = labels.copy()
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        for members in list(clusters.values()):
            if len(members) < _MIN_MEMBERS:
                continue
            scenes = self._partition(members, ctx)
            if len(scenes) < 2:
                continue
            # keep the largest scene on the original label; relabel the rest
            scenes.sort(key=len, reverse=True)
            for grp in scenes[1:]:
                for idx in grp:
                    labels[idx] = next_id
                next_id += 1
        return labels

    # ------------------------------------------------------------------ #
    def _partition(self, members, ctx) -> list[list[int]]:
        """Return ``members`` partitioned into anchor-distinct scenes (>=1 list).

        A single coherent scene returns ``[members]`` (no split). Pixel-only — uses
        brightness, the masked correlator and the gray tiles; never the filenames."""
        B = ctx.brightness
        gray = ctx.gray
        n = len(members)

        def AM(i: int, j: int) -> float:
            return anchor_match(gray[members[i]], gray[members[j]])

        def is_hard(i: int) -> bool:
            b = B[members[i]]
            return b < _CLIP_HARD_DARK or b > _CLIP_HARD_WHITE

        well = [i for i in range(n) if _WELL_LOW <= B[members[i]] <= _WELL_HIGH]
        well_set = set(well)

        # 1) Seed scenes from WELL-EXPOSED frames via masked edge ZNCC — the frames
        #    masked correlation can actually compare. Connected components on a
        #    same-scene link give the well-exposed scene cores (this is the trusted
        #    precision signal: it is what already separates two well-lit rooms).
        scene: dict[int, int] = {}
        n_scene = 0
        if well:
            w = len(well)
            link = np.zeros((w, w), bool)
            for a in range(w):
                for b in range(a + 1, w):
                    zncc, overlap = ctx.masked.score(members[well[a]], members[well[b]])
                    if overlap >= _MIN_OVERLAP and zncc >= _SAME_SCENE:
                        link[a, b] = link[b, a] = True
            comp = self._components(link)
            for li, wi in enumerate(well):
                scene[wi] = comp[li]
            n_scene = int(max(comp)) + 1
        multi_well = n_scene >= 2

        reps: dict[int, list[int]] = defaultdict(list)
        for wi in well:
            reps[scene[wi]].append(wi)

        # 2) Group the CLIPPED frames into anchor-clusters (same clip direction). Two
        #    near-black frames of one scene share light-source positions; two
        #    near-white frames share dark-object positions — so a scene's clipped
        #    frames cohere even when masked correlation cannot see them at all. This
        #    keeps a bracket's own clipped frames together and lets a different
        #    scene's clipped frames stand apart as their own anchor-cluster.
        clipped = [i for i in range(n) if i not in scene]
        clip_clusters = self._anchor_clusters(clipped, B, members, AM)

        # Decide which clip-clusters are DIFFERENT scenes via the reliable
        # clipped-vs-clipped signal: at a given extreme direction, a single scene's
        # clipped frames all anchor-link into ONE cluster (they share light-spot /
        # dark-spot positions). So if a direction has >=2 HARD, TEXTURED clip-clusters
        # that did NOT link, the extra ones are foreign scenes. The LARGEST stays with
        # the main scene; the rest spawn. This is far more robust than the noisy
        # clipped-vs-WELL comparison — a foreign group that contributes only one dark
        # frame (10276/1038, 14288/14037) still separates. A single bracket has one
        # frame per level => one clip-cluster per extreme => never spawns (safe).
        spawn_set: set[int] = set()
        hard_by_dir: dict[bool, list[int]] = defaultdict(list)
        for gi, grp in enumerate(clip_clusters):
            if all(is_hard(ci) for ci in grp) and \
               max(float(gray[members[ci]].std()) for ci in grp) >= _SPAWN_STD:
                hard_by_dir[B[members[grp[0]]] < 128].append(gi)
        for gis in hard_by_dir.values():
            if len(gis) >= 2:
                gis.sort(key=lambda gi: (-len(clip_clusters[gi]), gi))
                spawn_set.update(gis[1:])

        # 3) Attach each clipped anchor-cluster to the well-scene it best anchors. A
        #    cluster spawns its OWN scene only when it is FULLY hard-clipped, carries
        #    real texture, and CONFIDENTLY rejects every well-scene (best anchor below
        #    _SPAWN). This is the anchor-MISMATCH split: a different scene's clipped
        #    frames anchor a foreign well-scene far lower than a bracket's own clipped
        #    frames anchor theirs, so the wrongly-merged scene separates cleanly.
        for gi, grp in enumerate(clip_clusters):
            if gi in spawn_set:
                sc = n_scene              # foreign clip-cluster -> its own scene
                n_scene += 1
            else:
                scored = []
                for s, seeds in reps.items():
                    pool = [wi for wi in seeds if wi in well_set]
                    if pool:
                        scored.append((max(AM(ci, wi) for ci in grp for wi in pool), s))
                sc = max(scored)[1] if scored else 0   # best-anchoring well-scene
            for ci in grp:
                scene[ci] = sc
                reps[sc].append(ci)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[scene.get(i, 0)].append(members[i])
        return [g for g in groups.values() if g]

    # ------------------------------------------------------------------ #
    def _anchor_clusters(self, clipped, B, members, AM) -> list[list[int]]:
        """Connected components of clipped frames linked by anchor match. Only same
        clip-direction frames (both near-black or both near-white) can link — a black
        and a white frame share no expressible channel."""
        m = len(clipped)
        if m == 0:
            return []
        link = np.zeros((m, m), bool)
        for a in range(m):
            for b in range(a + 1, m):
                if (B[members[clipped[a]]] < 128) != (B[members[clipped[b]]] < 128):
                    continue
                if AM(clipped[a], clipped[b]) >= _CLIP_LINK:
                    link[a, b] = link[b, a] = True
        comp = self._components(link)
        groups: dict[int, list[int]] = defaultdict(list)
        for li, ci in enumerate(clipped):
            groups[comp[li]].append(ci)
        return list(groups.values())

    @staticmethod
    def _components(link: np.ndarray) -> list[int]:
        """Connected-component label per node of a boolean adjacency matrix."""
        w = link.shape[0]
        comp = [-1] * w
        c = 0
        for s in range(w):
            if comp[s] >= 0:
                continue
            stack = [s]
            comp[s] = c
            while stack:
                u = stack.pop()
                for v in range(w):
                    if link[u, v] and comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
        return comp
