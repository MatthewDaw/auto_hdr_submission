"""LINK — rejoin a frame stranded by a small CAMERA DRIFT (sub-frame shift/zoom).

Root cause this pass targets
----------------------------
Two captures of the same still scene can be offset by a few pixels (the camera
nudged between brackets, or a re-shot frame is slightly zoomed). The masked edge
ZNCC is translation-sensitive, so even a ~8 px shift collapses it from ~0.9 to
~0.45 at FULL pixel overlap — the comparator reads "different scene" and the frame
strands as its own cluster. Example: GT 19300's IMG_7607 (a B129 bathroom view)
splits from its bracket although a small vertical shift makes it identical.

Signal (training-free, deterministic, browser-portable)
-------------------------------------------------------
Phase correlation recovers the dominant global shift between two edge maps in one
FFT. After applying that shift, the edge ZNCC of the SAME scene jumps back high
(0.45 -> 0.83 for 19300); a genuinely different scene does not recover (no single
shift aligns its structure). So we reattach a small orphan to the cluster whose
DRIFT-ALIGNED edge ZNCC is high, only when the raw ZNCC was too low for the normal
passes to link it, the shift is small (a real drift, not a big translation), and
the best candidate beats the runner-up by a margin.

Conservative guards (merging is the dangerous direction)
--------------------------------------------------------
Only a small orphan (<= _MAX_ORPHAN) reattaches; only on WELL-exposed reps (edges
reliable); only when raw ZNCC < _RAW_MAX (the normal passes already failed) yet
aligned ZNCC >= _ALIGN_MIN with |shift| <= _MAX_SHIFT; and the best aligned score
must beat the runner-up cluster by _MARGIN. Holds the data/large 1302/1302 gate.

Reads only pixels. No labels, no ML, no randomness.
"""
from __future__ import annotations

import cv2
import numpy as np

from .context import RefinementContext

_WELL_LOW, _WELL_HIGH = 45.0, 215.0   # reps must have reliable edge content
_MAX_ORPHAN = 2          # only a stranded piece reattaches, never a whole scene
_RAW_MAX = 0.55          # raw edge ZNCC below this == the normal passes couldn't link
_ALIGN_MIN = 0.72        # drift-aligned edge ZNCC at/above this == same scene
_MAX_SHIFT = 14.0        # px; a real camera drift, not a different framing
_MARGIN = 0.10           # best aligned score must beat the runner-up by this


def _edge(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def _zncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def aligned_zncc(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float]:
    """(drift-aligned edge ZNCC, shift magnitude). Phase-correlates the two frames,
    rolls B onto A by the recovered shift, and scores edge ZNCC on the valid region."""
    fa = gray_a.astype(np.float32)
    fb = gray_b.astype(np.float32)
    (sx, sy), _ = cv2.phaseCorrelate(fa, fb)
    dy = int(round(-sy)); dx = int(round(-sx))
    mag = float(np.hypot(sx, sy))
    if abs(dy) > 64 or abs(dx) > 64:
        return 0.0, mag
    Ea = _edge(gray_a); Eb = _edge(gray_b)
    Bs = np.roll(np.roll(Eb, dy, 0), dx, 1)
    mask = np.zeros(Ea.shape, bool)
    ys = slice(max(0, dy), Ea.shape[0] + min(0, dy))
    xs = slice(max(0, dx), Ea.shape[1] + min(0, dx))
    mask[ys, xs] = True
    return _zncc(Ea, Bs, mask), mag


class DriftReattachment:
    """Merge a small orphan into the cluster it matches once a small camera drift is
    corrected. ``apply(labels, ctx) -> labels``."""

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None:
            return labels
        labels = labels.copy()
        B = ctx.brightness

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        def rep(members: list[int]):
            well = [k for k in members if _WELL_LOW <= B[k] <= _WELL_HIGH]
            if not well:
                return None
            return min(well, key=lambda k: abs(B[k] - 128.0))

        for c, members in list(clusters.items()):
            if not clusters.get(c) or len(members) > _MAX_ORPHAN:
                continue
            r = rep(members)
            if r is None:
                continue

            scored: list[tuple[float, int]] = []
            for oc, om in clusters.items():
                if oc == c or not om:
                    continue
                orep = rep(om)
                if orep is None:
                    continue
                raw, _ = ctx.masked.score(r, orep)
                if raw >= _RAW_MAX:
                    continue  # normal passes already had a reliable verdict
                az, mag = aligned_zncc(gray[r], gray[orep])
                if mag <= _MAX_SHIFT:
                    scored.append((az, oc))
            if not scored:
                continue
            scored.sort(reverse=True)
            best, host = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best >= _ALIGN_MIN and best - runner >= _MARGIN:
                for k in members:
                    labels[k] = host
                clusters[host].extend(members)
                clusters[c] = []
        return labels
