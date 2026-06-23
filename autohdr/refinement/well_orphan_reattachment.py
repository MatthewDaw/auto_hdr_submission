"""LINK — rejoin a stranded WELL-exposed frame to its DARK bracket by light match.

Root cause this pass targets
----------------------------
A scene shot mostly under-exposed (a dim room: one lamp lit, everything else near
black) may include a single well-exposed frame. Across that large exposure gap the
masked edge correlator is blind (ZNCC ~0.15), so the well-exposed frame strands as
its own cluster while its dark siblings cohere — an over-split. Capture order does
not always help (the well frame can be numbered far from the dark run). Example:
GT 10125's well frame (B79) split from its three dark frames (B19/B40/B40).

Signal (training-free, deterministic, browser-portable)
-------------------------------------------------------
The scene's light fixtures (lamp, ceiling light, windows) are the few bright spots
that survive in the dark frames AND are the clean, dominant lights of the well
frame. :func:`extreme_anchor.lights_present_fraction` measures, in the well->dark
direction, how many of the well frame's CLEAN lights are present among the dark
frames' surviving bright spots (under one global translation). Same scene => most
of the well frame's lights reappear in the dark frames (>=~0.5); a different scene
=> they do not (<=~0.25). The well->dark direction matters: the well frame is the
clean side, so it is not diluted by the dark frames' dark-room noise blobs.

Conservative guards (merging is the dangerous direction)
--------------------------------------------------------
Fires only for a small, fully WELL-exposed orphan (<= 2 frames) reattaching to a
DARK bracket (>= 2 near-black frames), only when the well frame has enough clean
lights to judge (>= _MIN_WELL_LIGHTS), and only when the best dark bracket clears
an absolute floor AND beats the runner-up by a margin (so a well frame is never
merged on a tie between two candidate scenes). Holds the data/large 1302/1302 gate.

Reads only pixels. No labels, no ML, no randomness.
"""
from __future__ import annotations

import re

import numpy as np

from ..features import extreme_anchor as _ea
from .context import RefinementContext


def _prefix(filename: str) -> str:
    """Camera filename prefix (drops the synthetic gNNNN_ tag and trailing number)."""
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return stem[: m.start()] if m else stem

_WELL_LO, _WELL_HI = 70.0, 185.0   # the orphan must be genuinely well-exposed
_DARK_GAP = 15.0                   # a reference frame must be at least this much
#                                    darker than the orphan (its lights dominate)
_MIN_DARK = 1                      # at least one darker frame to read the lights from
_MAX_ORPHAN = 2                    # only a stranded piece, never a whole scene
_MIN_WELL_LIGHTS = 3               # enough clean lights to judge a match
_FLOOR = 0.45                      # best light-presence fraction to reattach
_MARGIN = 0.20                     # ...and beat the runner-up dark bracket by this


class WellOrphanReattachment:
    """Merge a stranded well-exposed orphan into the dark bracket whose lights it
    best matches. ``apply(labels, ctx) -> labels``."""

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        filenames = getattr(ctx, "filenames", None)
        if gray is None or filenames is None:
            return labels
        labels = labels.copy()
        B = ctx.brightness

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        for c, members in list(clusters.items()):
            if not clusters.get(c):
                continue
            if len(members) > _MAX_ORPHAN:
                continue
            if not all(_WELL_LO <= B[k] <= _WELL_HI for k in members):
                continue  # orphan must be fully well-exposed
            well_tile = min(members, key=lambda k: abs(B[k] - 128.0))
            ref_ceiling = min(B[k] for k in members) - _DARK_GAP
            orphan_prefixes = {_prefix(filenames[k]) for k in members}

            scored: list[tuple[float, int]] = []
            for dc, dmembers in clusters.items():
                if dc == c or not dmembers:
                    continue
                # a real bracket is one camera/shoot: the orphan must SHARE a filename
                # prefix with the target, else this is a cross-camera false merge
                if orphan_prefixes.isdisjoint(_prefix(filenames[k]) for k in dmembers):
                    continue
                # reference = the candidate's frames meaningfully darker than the
                # orphan, where the scene's lights dominate the surviving structure
                darks = [k for k in dmembers if B[k] < ref_ceiling]
                if len(darks) < _MIN_DARK:
                    continue
                frac, na, nb = _ea.lights_present_fraction(
                    gray[well_tile], [gray[k] for k in darks])
                if frac is None or na < _MIN_WELL_LIGHTS:
                    continue
                scored.append((frac, dc))
            if not scored:
                continue
            scored.sort(reverse=True)
            best, host = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best >= _FLOOR and best - runner >= _MARGIN:
                for k in members:
                    labels[k] = host
                clusters[host].extend(members)
                clusters[c] = []
        return labels
