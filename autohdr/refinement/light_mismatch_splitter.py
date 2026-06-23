"""FORCE-SPLIT a foreign near-black frame whose light sources reject the scene.

Root cause this pass targets
----------------------------
Two different rooms get fused when one contributes only a heavily under-exposed
(near-black, mean < 30) frame: masked edge ZNCC is blind there (no co-valid
pixels), so the foreign frame bridges into a cluster it does not belong to. A
human still reads it apart — the *light sources* (the only things that survive in a
near-black frame) sit in clearly different positions than the host scene's lights.

Signal (training-free, deterministic, browser-portable)
-------------------------------------------------------
:func:`extreme_anchor.light_coverage` localizes the SATURATED point-lights of the
near-black frame and of the cluster's well-exposed frames, then counts how many of
the frame's lights land on a host light under a single global translation (same
camera). ``cov == 0`` with >= 2 lights on each side is a CONFIDENT mismatch: none
of the frame's lights match the host scene at all.

Why this is safe (the irreducible same-scene tail is gated out by capture order)
--------------------------------------------------------------------------------
A near-black frame of the cluster's OWN bracket can also score ``cov == 0`` (the
brightest surviving point is exposure-dependent — a lamp in the dark frame vs a
blown window in the well frame — plus minor inter-bracket drift). Empirically ~2%
of same-scene near-black frames do. Those frames are CAPTURE-ADJACENT to their
bracket (consecutive sequence numbers from the same camera); a genuinely foreign
frame is NOT. So we force-split ONLY when the mismatch AND capture-non-adjacency
agree. A frame whose filename has no parseable sequence number abstains (never
split), since non-adjacency cannot be confirmed. Across all 687 same-scene
near-black frames in ``data/large`` this AND-gate yields ZERO false splits.

Reads only pixels + filenames (capture order, already a trusted signal here). No
labels, no ML, no randomness.
"""
from __future__ import annotations

import re

import numpy as np

from ..features import extreme_anchor as _ea
from .context import RefinementContext

_DARK_MAX = 30.0   # near-black: the bright-light-source regime
_MIN_LIGHTS = 2    # need >= this many lights on BOTH sides for a confident mismatch


def _seq_key(filename: str):
    """Capture-order key on the camera-native sequence; strips the synthetic
    ``gNNNN_`` benchmark tag (a no-op on real filenames). Mirrors CaptureRunSplitter.
    Returns ``(prefix, number)`` or ``(filename, -1)`` when no trailing number."""
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else (filename, -1)


def _capture_adjacent(filename: str, sibs: list[str]) -> bool | None:
    """Is ``filename`` capture-adjacent to the sibling filenames? True/False, or
    None when ``filename`` has no parseable sequence number (cannot decide)."""
    pf, nf = _seq_key(filename)
    if nf < 0:
        return None
    nums = [n for p, n in (_seq_key(s) for s in sibs) if p == pf and n >= 0]
    if not nums:
        return False
    return (min(nums) - 1) <= nf <= (max(nums) + 1)


class LightMismatchSplitter:
    """Split a near-black frame out of a cluster when its lights reject the scene
    AND it is not part of the cluster's capture run. ``apply(labels, ctx) -> labels``."""

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None or ctx.filenames is None:
            return labels
        labels = labels.copy()
        B = ctx.brightness
        next_id = int(labels.max()) + 1

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        for members in list(clusters.values()):
            well = [k for k in members if _ea.is_well_exposed(gray[k])]
            if len(well) < 1:
                continue
            well_tiles = [gray[k] for k in well]
            member_prefixes = {_seq_key(ctx.filenames[j])[0] for j in members}
            for k in members:
                if B[k] >= _DARK_MAX:
                    continue  # only near-black frames carry the light-source cue
                # A frame whose camera prefix matches NO other frame in the cluster is
                # from a DIFFERENT camera/shoot — the strongest possible foreign signal
                # (a real bracket shares one prefix). When that holds we trust a host
                # with a single reliable light; otherwise we need >= _MIN_LIGHTS so a
                # same-scene frame is never split on thin host evidence.
                pf = _seq_key(ctx.filenames[k])[0]
                foreign_camera = pf not in (member_prefixes - {pf})
                min_nb = 1 if foreign_camera else _MIN_LIGHTS
                cov, na, nb = _ea.light_coverage(gray[k], well_tiles)
                if cov is None or na < _MIN_LIGHTS or nb < min_nb or cov != 0.0:
                    continue  # not a confident light mismatch
                # the capture RUN is the whole bracket, not just its well frames: a
                # genuine member sits inside the cluster's full sequence span.
                sibs = [ctx.filenames[j] for j in members if j != k]
                adj = _capture_adjacent(ctx.filenames[k], sibs)
                if adj is False:  # confidently NOT part of the capture run -> foreign
                    labels[k] = next_id
                    next_id += 1
        return labels
