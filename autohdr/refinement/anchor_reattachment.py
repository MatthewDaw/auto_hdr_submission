"""ANCHOR — re-attach a clipped frame across a large exposure jump.

Root cause this pass targets
----------------------------
A near-black (mean < ~30) or near-white (mean > ~225) bracket loses almost all
its valid (un-clipped) pixels, so :class:`MaskedCorrelation` has ~0 co-valid
overlap with the rest of its scene and *cannot* compare them. The scene then
over-splits: the extreme frame strands as its own tiny cluster even though a
human reads it as obviously the same room.

Signal (exposure-invariant, training-free, browser-portable)
------------------------------------------------------------
A still scene's light sources are bright spots at FIXED image positions — they
stay visible even in a near-black frame. Symmetrically, dark objects are dark
spots at fixed positions, visible even in a near-white frame. We compare two
frames by the spatial overlap (IoU) of their brightest-K% and darkest-K% pixels
on a coarse rank map. Rank-based ⇒ invariant to the exposure level; only the
*positions* of the extremes matter.

Adaptive polarity is the key refinement over a naive both-halves average: in a
near-black frame the dark mask is meaningless (almost everything is dark / noise)
— only the BRIGHT spots (the lights) carry the scene's fingerprint. In a
near-white frame only the DARK spots do. So when an extreme frame is involved we
score on the polarity that survived the clipping.

Decision rule
-------------
Only fires for a clipped frame whose masked-ZNCC overlap with the candidate
cluster is too low to decide (``overlap < _MIN_OVERLAP``). The frame's cluster is
merged into the candidate cluster it best anchor-matches, but only when that best
match clears an absolute floor AND beats the runner-up by a clear uniqueness
margin — a clipped frame carries little signal, so we never reattach on a tie.
Each candidate cluster is scored by its single best-matching member (so a
saturated white frame can chain onto its near-white sibling, which in turn sits
in the well-exposed cluster).

Deterministic, no ML, no randomness. Operates on the label array in place of a
copy and never mutates ``ctx``.
"""
from __future__ import annotations

import cv2
import numpy as np

from .context import RefinementContext

# --- anchor signal -----------------------------------------------------------
_GRID = 32          # coarse rank map side; 32 separates same/diff scenes best
_FRAC = 0.06        # top/bottom 6% of cells are the bright/dark "anchor" spots
_DARK_B = 30.0      # mean below this ⇒ near-black: trust BRIGHT spots (lights)
_WHITE_B = 225.0    # mean above this ⇒ near-white: trust DARK spots

# --- reattachment guards -----------------------------------------------------
_CLIP_LOW, _CLIP_HIGH = 30.0, 225.0  # only clipped frames are candidates
_MIN_OVERLAP = 1500   # masked-ZNCC needs this many co-valid px to be trusted;
#                       below it the edge comparator is blind and anchor decides
_FLOOR = 0.34         # absolute floor for the best anchor match
_MARGIN = 0.10        # best must beat the runner-up cluster by this
_MAX_CLUSTER = 2      # only re-home true orphans (cluster of size <= 2); never
#                       bridge two already-substantial clusters on a coarse cue


def _anchor_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bright_mask, dark_mask) as flat boolean arrays on a _GRID×_GRID rank map.

    ``gray`` is a 2-D uint8 tile. INTER_AREA pools each cell, then the cells are
    ranked; the top/bottom _FRAC become the bright/dark anchor sets.
    """
    cells = cv2.resize(
        gray, (_GRID, _GRID), interpolation=cv2.INTER_AREA
    ).astype(np.float32).ravel()
    k = max(1, int(round(len(cells) * _FRAC)))
    order = np.argsort(cells, kind="stable")
    dark = np.zeros(len(cells), dtype=bool)
    bright = np.zeros(len(cells), dtype=bool)
    dark[order[:k]] = True
    bright[order[-k:]] = True
    return bright, dark


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def anchor_match(gray_a: np.ndarray, gray_b: np.ndarray,
                 mean_a: float | None = None,
                 mean_b: float | None = None) -> float:
    """Exposure-invariant anchor similarity of two grayscale tiles, in [0, 1].

    Compares the spatial overlap of the brightest-K% (light sources) and
    darkest-K% (dark objects) cells. When either frame is clipped the score uses
    only the polarity that survives the clipping (bright spots for a near-black
    frame, dark spots for a near-white frame); otherwise it averages both.
    """
    if mean_a is None:
        mean_a = float(gray_a.mean())
    if mean_b is None:
        mean_b = float(gray_b.mean())
    ba, da = _anchor_masks(gray_a)
    bb, db = _anchor_masks(gray_b)
    scores: list[float] = []
    if min(mean_a, mean_b) < _DARK_B:   # a near-black frame ⇒ lights are the cue
        scores.append(_iou(ba, bb))
    if max(mean_a, mean_b) > _WHITE_B:  # a near-white frame ⇒ dark objects are it
        scores.append(_iou(da, db))
    if not scores:                      # neither clipped: use both halves
        scores.append(0.5 * (_iou(ba, bb) + _iou(da, db)))
    return max(scores)


class AnchorReattachment:
    """Merge a clipped frame's cluster into the scene it best anchor-matches.

    Runs after the masked-ZNCC passes (ClippedReattachment / CaptureRunSplitter):
    it only acts on frames those passes could not place because the clipped
    overlap was too small to read. See module docstring for the signal.
    """

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        B = ctx.brightness
        gray = ctx.gray
        if gray is None:
            return labels
        labels = labels.copy()

        # cache anchor masks once per frame (deterministic, no per-pair recompute)
        masks = [_anchor_masks(gray[i]) for i in range(ctx.count)]

        def pair_anchor(i: int, j: int) -> float:
            bi, di = masks[i]
            bj, dj = masks[j]
            sc: list[float] = []
            if min(B[i], B[j]) < _DARK_B:
                sc.append(_iou(bi, bj))
            if max(B[i], B[j]) > _WHITE_B:
                sc.append(_iou(di, dj))
            if not sc:
                sc.append(0.5 * (_iou(bi, bj) + _iou(di, dj)))
            return max(sc)

        def cluster_map() -> dict[int, list[int]]:
            out: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                out.setdefault(int(label), []).append(index)
            return out

        clusters = cluster_map()

        def well_rep(members: list[int]):
            """The member masked-ZNCC would actually reference: the most fully
            exposed (nearest mid-grey) frame. ``None`` if the cluster is entirely
            clipped, in which case masked-ZNCC can never decide for it."""
            well = [k for k in members if _CLIP_LOW <= B[k] <= _CLIP_HIGH]
            if not well:
                return None
            return min(well, key=lambda k: abs(B[k] - 128.0))

        # clipped orphans, brightest/darkest first so the most-clipped (hardest,
        # but also most distinctive in polarity) frames are placed deterministically
        clipped = [
            i for i in range(ctx.count)
            if (B[i] < _CLIP_LOW or B[i] > _CLIP_HIGH)
            and len(clusters[int(labels[i])]) <= _MAX_CLUSTER
        ]
        clipped.sort(key=lambda i: (min(B[i], 255.0 - B[i]), i))

        for i in clipped:
            own = int(labels[i])
            if len(clusters[own]) > _MAX_CLUSTER:
                continue  # may have grown via an earlier merge this pass

            scored: list[tuple[float, int]] = []
            for c, members in clusters.items():
                if c == own or not members:
                    continue  # skip self and clusters emptied by an earlier merge
                # masked-ZNCC must be UNABLE to decide. It compares against the
                # cluster's well-exposed representative — the only frame with the
                # edge content the comparator needs. If THAT has enough co-valid
                # overlap with the clipped orphan, the edge comparator already had
                # its say and we don't override it with a coarse anchor cue.
                rep = well_rep(members)
                if rep is not None:
                    _, overlap = ctx.masked.score(i, rep)
                    if overlap >= _MIN_OVERLAP:
                        continue
                anchor = max(pair_anchor(i, j) for j in members)
                scored.append((anchor, c))

            if not scored:
                continue
            scored.sort(reverse=True)
            best, best_c = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best >= _FLOOR and best - runner >= _MARGIN:
                # absorb the orphan's whole (tiny) cluster into the winner
                for k in clusters[own]:
                    labels[k] = best_c
                clusters[best_c].extend(clusters[own])
                clusters[own] = []
        return labels
