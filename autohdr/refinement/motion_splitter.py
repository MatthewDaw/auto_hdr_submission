"""FIX10 — split brackets that captured LOCALIZED IN-SCENE MOTION.

Same camera viewpoint, but something physically MOVED between two brackets — a
shower curtain opened, a door swung, a pool-table cover came off, a window
changed state, a finger crossed the lens. Globally the two brackets stay almost
identical, so edge-ZNCC remains high and the coarse pipeline MERGES them. The
giveaway is a single LOCALIZED, COHERENT region that differs, while the rest of
the frame matches; a genuinely-still scene re-exposed differs only by diffuse
sensor/CLAHE shimmer (no compact changed object).

SAFETY PROPERTY (why this can't run wild): a single correct HDR bracket visits
each exposure level exactly ONCE, so it has ZERO same-exposure-level frame pairs
and this detector — which only ever compares two frames at the SAME level — can
never fire on it. Same-level pairs exist only when a cluster already holds >=2
brackets (a multi-bracket merge). For a genuinely re-shot still scene the
same-level pair shows no localized change; for a moved object it shows a compact,
coherent changed region.

Detection is a deliberate OR of two complementary, exposure-robust tracks run on
same-level WELL-EXPOSED frame pairs:

  * Track A (different framing / scene): masked edge-ZNCC at real overlap is LOW
    (< _A_ZNCC). Catches re-framed scenes whose layout actually changed (curtain,
    pool) even though the descriptor merged them.
  * Track B (same framing, a compact object moved): the largest connected blob of
    the masked CLAHE intensity difference is a real OBJECT — meaningful in BOTH
    dimensions (not a thin exposure-highlight strip), well-FILLED (a solid region,
    not scattered noise), and LOCALIZED (strong inside the blob, quiet outside).
    Catches door / finger / window where edge-ZNCC stays high.

Both gates are tuned so the user's named single-bracket safety groups never fire
and so genuine still-scene same-level pairs (validated against a baseline of real
same-scene pairs and self+noise) stay below threshold. Once a cluster is flagged,
it is partitioned into its two contiguous capture-order brackets: the most
dissimilar same-level anchor pair straddles the bracket boundary, and the cluster
is cut at the most balanced contiguous capture-order point between the anchors so
each bracket stays whole and clipped frames follow their own bracket.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .context import RefinementContext

# --- comparison range -------------------------------------------------------
_WELL_LOW, _WELL_HIGH = 45, 215   # only compare well-exposed frames; clipped
#                                   highlights/shadows give unreliable diffs
_LEVEL = 35           # |Δbrightness| <= this == "same exposure level" (a bit
#                       loose: a localized change shifts a bracket's mean a little)

# --- Track A: different framing / scene -------------------------------------
_A_ZNCC = 0.45        # masked edge-ZNCC below this == confidently different scene
_A_MIN_OVERLAP = 8000 # co-valid pixels needed to trust the ZNCC

# --- Track B: a compact object moved ----------------------------------------
_B_HI = 60.0          # CLAHE-diff above this (blurred) is a real change, not noise
_B_OPEN = 5           # morphological-open radius: erase thin shimmer, keep objects
_B_MIN_AREA = 600     # blob smaller than this is noise / a sliver
_B_MIN_FILL = 0.30    # blob area / bbox area — a solid region, not scattered px
_B_MIN_DIM = 25       # smaller bbox side; a real object is wide in BOTH dims
_B_STRONG_LOC = 9.0   # in-blob vs out-of-blob diff ratio for a STRONG compact move
_B_STRONG_DIM = 40    #   (door / finger): high contrast, moderate size
_B_WIDE_DIM = 60      # a LARGE moved object (window / curtain panel): big in both
_B_WIDE_LOC = 4.3     #   dims, so a more moderate localization ratio suffices
# A SMALL but EXTREMELY localized change (a curtain region opening/closing): the
# changed region is modest in size (below the general _B_MIN_AREA floor) yet a
# solid, square-ish island that stands out far above the still background (very
# high in-vs-out ratio). This rule fills exactly the sub-_B_MIN_AREA niche, so it
# carries both an AREA FLOOR (reject slivers) and an AREA CEILING (a larger
# high-loc blob is handled by the strong/wide tracks and, more importantly, a
# re-shot still scene with broad moving shimmer produces large high-loc blobs that
# must NOT split — e.g. group 5091 has loc~26 but area>1000; the curtain is
# area~420). Validated: no single-bracket / correctly-merged same-level pair lands
# in this tight (small-area, very-high-loc, well-filled, square) box.
_B_COMPACT_LOC = 17.0
_B_COMPACT_DIM = 25
_B_COMPACT_AREA = 300
_B_COMPACT_AREA_MAX = 700
_B_COMPACT_FILL = 0.40

# --- partition --------------------------------------------------------------
_LOC_NORM = 3000.0    # normaliser folding blob-area into the anchor dissimilarity


def _seq_key(filename: str):
    """Capture-order key on the camera-native sequence only.

    Strips the synthetic ``gNNNN_`` benchmark tag so ordering never peeks at the
    ground-truth label (a no-op on real filenames). Mirrors CaptureRunSplitter.
    """
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else (filename, -1)


class MotionSplitter:
    """Splits clusters whose brackets captured a localized in-scene change."""

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_B_OPEN, _B_OPEN))
        self._norm: dict[int, np.ndarray] = {}

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None or ctx.filenames is None:
            return labels
        labels = labels.copy()
        next_id = int(labels.max()) + 1
        B = ctx.brightness

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        for members in list(clusters.values()):
            if len(members) < 4:
                continue  # a single bracket has no same-level pair to compare
            anchors = self._fire(members, B, ctx)
            if anchors is None:
                continue
            parts = self._partition(members, anchors, B, ctx)
            if parts is None or len(parts) < 2:
                continue
            for grp in parts[1:]:
                for idx in grp:
                    labels[idx] = next_id
                next_id += 1
        self._norm.clear()
        return labels

    # --- detection ----------------------------------------------------------
    def _same_level_pairs(self, members, B):
        well = [k for k in members if _WELL_LOW <= B[k] <= _WELL_HIGH]
        for a in range(len(well)):
            for b in range(a + 1, len(well)):
                i, j = well[a], well[b]
                if abs(B[i] - B[j]) <= _LEVEL:
                    yield i, j

    def _fire(self, members, B, ctx):
        """Return the most-dissimilar same-level anchor pair if this cluster shows
        a confident localized change (Track A OR Track B), else None."""
        best = None
        fired = False
        for i, j in self._same_level_pairs(members, B):
            zncc, overlap = ctx.masked.score(i, j)
            track_a = overlap >= _A_MIN_OVERLAP and -1.0 < zncc < _A_ZNCC
            blob = self._changed_blob(i, j, ctx.gray)
            track_b = self._is_object(blob)
            if track_a or track_b:
                fired = True
                # dissimilarity: low ZNCC (different scene) + big localized change
                base = (1.0 - zncc) if overlap >= _A_MIN_OVERLAP else 0.0
                diss = base + (blob["area"] / _LOC_NORM if blob else 0.0)
                if best is None or diss > best[0]:
                    best = (diss, i, j)
        return (best[1], best[2]) if fired and best is not None else None

    @staticmethod
    def _is_object(blob) -> bool:
        if blob is None:
            return False
        compact = (
            blob["loc"] >= _B_COMPACT_LOC
            and blob["mindim"] >= _B_COMPACT_DIM
            and _B_COMPACT_AREA <= blob["area"] <= _B_COMPACT_AREA_MAX
            and blob["fill"] >= _B_COMPACT_FILL
        )
        if blob["area"] < _B_MIN_AREA or blob["fill"] < _B_MIN_FILL:
            return compact
        if blob["mindim"] < _B_MIN_DIM:
            return compact
        strong = blob["loc"] >= _B_STRONG_LOC and blob["mindim"] >= _B_STRONG_DIM
        wide = blob["mindim"] >= _B_WIDE_DIM and blob["loc"] >= _B_WIDE_LOC
        return strong or wide or compact

    def _norm_img(self, i, gray):
        if i not in self._norm:
            self._norm[i] = self._clahe.apply(gray[i]).astype(np.float32)
        return self._norm[i]

    def _changed_blob(self, i, j, gray):
        """Largest connected blob of the masked CLAHE intensity difference, with
        shape/localization stats. Exposure-robust: CLAHE flattens the brightness
        offset, morphological-open removes thin shimmer lines, and the blob is
        scored by how localized it is (strong inside, quiet outside)."""
        gi, gj = gray[i], gray[j]
        valid = (gi >= 8) & (gi <= 247) & (gj >= 8) & (gj <= 247)
        if valid.sum() < 12000:
            return None
        a = self._norm_img(i, gray)
        b = self._norm_img(j, gray)
        d = cv2.GaussianBlur(np.abs(a - b), (0, 0), 3)
        d[~valid] = 0
        changed = ((d > _B_HI) & valid).astype(np.uint8)
        opened = cv2.morphologyEx(changed, cv2.MORPH_OPEN, self._kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[k, cv2.CC_STAT_AREA])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        blob = lab == k
        outside = valid & ~blob
        out_med = float(np.median(d[outside])) if int(outside.sum()) > 500 else 0.0
        loc = float(d[blob].mean()) / (out_med + 1.0)
        return {
            "area": area,
            "fill": area / max(1, bw * bh),
            "mindim": min(bw, bh),
            "loc": loc,
        }

    # --- partition ----------------------------------------------------------
    def _partition(self, members, anchors, B, ctx):
        """Cut the cluster into its two contiguous capture-order brackets.

        The trigger anchors straddle the bracket boundary, so order the frames by
        capture sequence and cut at the most BALANCED contiguous point strictly
        between the two anchors. Each bracket stays whole; clipped frames follow
        their own bracket (they sit on their bracket's side of the cut)."""
        order = sorted(members, key=lambda i: _seq_key(ctx.filenames[i]))
        a1, a2 = anchors
        try:
            p1, p2 = sorted((order.index(a1), order.index(a2)))
        except ValueError:
            return None
        best_k = None
        best_bal = None
        for k in range(p1 + 1, p2 + 1):
            bal = abs((k) - (len(order) - k))
            if best_bal is None or bal < best_bal:
                best_bal, best_k = bal, k
        if best_k is None:
            return None
        return [order[:best_k], order[best_k:]]
