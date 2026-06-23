"""Extreme-anchor matcher: align a heavily CLIPPED frame to a SCENE TEMPLATE.

A bracket may contain frames so over/under-exposed that almost all structure is
gone — a near-black frame keeps only the few brightest light sources (windows,
lamps, the sun) and a near-white frame keeps only the few darkest objects. Prior
attempts matched clipped-to-clipped (both sides noisy). The idea here is to match
the clipped frame against a TEMPLATE built from the cluster's WELL-EXPOSED frames
(one clean side), so the reference structure is reliable.

Pipeline
--------
1. ``build_template`` — from one or more well-exposed gray tiles, derive a stable
   bright-structure map (top-percentile pixels = lights/windows) and a
   dark-structure map (bottom-percentile = dark objects), intersected across frames.
2. ``extreme_spots`` — localize the dominant extreme blobs of a clipped frame with
   sub-pixel centroids (connected components on a per-frame adaptive threshold).
3. ``coverage_score`` — fraction of the clipped frame's extreme spots that land on
   the template's corresponding extreme structure within a small tolerance, under a
   single best global translation found by exhaustive 2-point hypotheses (same
   camera => one consistent shift explains many spots). Field-standard inlier count;
   immune to missing/spurious blobs. Returns a score in [0, 1].

Pure numpy + cv2, deterministic, training-free, browser-portable. No labels used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

# ---- exposure regime constants (mean gray of a tile) ----------------------------
BLACK_MAX = 30.0    # mean below this => near-black clipped frame (match bright spots)
WHITE_MIN = 225.0   # mean above this => near-white clipped frame (match dark spots)
WELL_LO = 70.0      # well-exposed band used to build the template
WELL_HI = 185.0


def brightness(tile: np.ndarray) -> float:
    return float(tile.mean())


def is_clipped(tile: np.ndarray) -> bool:
    m = brightness(tile)
    return m < BLACK_MAX or m > WHITE_MIN


def clip_polarity(tile: np.ndarray) -> Optional[str]:
    """'bright' if near-black (track bright spots), 'dark' if near-white, else None."""
    m = brightness(tile)
    if m < BLACK_MAX:
        return "bright"
    if m > WHITE_MIN:
        return "dark"
    return None


def is_well_exposed(tile: np.ndarray) -> bool:
    return WELL_LO <= brightness(tile) <= WELL_HI


# ---------------------------------------------------------------------------------
@dataclass
class SceneTemplate:
    """Per-cluster extreme-structure reference, built from well-exposed frames."""
    bright: np.ndarray  # float [0,1] map, high where lights/windows live
    dark: np.ndarray    # float [0,1] map, high where dark objects live
    size: int
    n_frames: int


def _norm_gray(tile: np.ndarray, size: int) -> np.ndarray:
    g = tile
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if g.shape[0] != size or g.shape[1] != size:
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def build_template(well_tiles: List[np.ndarray], size: int = 256,
                   bright_pct: float = 98.0, dark_pct: float = 2.0,
                   blur: float = 2.5) -> Optional[SceneTemplate]:
    """Build a SceneTemplate from one or more WELL-EXPOSED gray tiles.

    For each frame we mark the top ``bright_pct`` percentile pixels as bright
    structure and the bottom ``dark_pct`` as dark structure, then AVERAGE the
    (blurred) masks across frames so only consistently-extreme regions survive.
    """
    masks_b: List[np.ndarray] = []
    masks_d: List[np.ndarray] = []
    for t in well_tiles:
        g = _norm_gray(t, size)
        hi = np.percentile(g, bright_pct)
        lo = np.percentile(g, dark_pct)
        mb = (g >= hi).astype(np.float32)
        md = (g <= lo).astype(np.float32)
        if blur > 0:
            mb = cv2.GaussianBlur(mb, (0, 0), blur)
            md = cv2.GaussianBlur(md, (0, 0), blur)
        masks_b.append(mb)
        masks_d.append(md)
    if not masks_b:
        return None
    bright = np.mean(masks_b, axis=0)
    dark = np.mean(masks_d, axis=0)
    # normalize each map to [0,1] for a stable hit test
    if bright.max() > 0:
        bright = bright / bright.max()
    if dark.max() > 0:
        dark = dark / dark.max()
    return SceneTemplate(bright=bright, dark=dark, size=size, n_frames=len(masks_b))


# ---------------------------------------------------------------------------------
def extreme_spots(tile: np.ndarray, polarity: Optional[str] = None,
                  size: int = 256, k: int = 6, pct: float = 98.0,
                  min_area: int = 3, max_area_frac: float = 0.10,
                  min_sep: float = 0.04) -> Tuple[np.ndarray, str]:
    """Localize the dominant extreme blobs of a (clipped) frame.

    near-black / polarity 'bright' -> brightest blobs (light sources).
    near-white / polarity 'dark'   -> darkest blobs.

    Per-frame adaptive threshold (a high/low percentile of the frame's own
    intensity) keeps only the few surviving extreme regions; connected components
    give sub-pixel intensity-weighted centroids. Returns (spots[K,2] in [0,1] as
    (y,x), polarity). Spots are ordered by dominance (area*contrast) and de-duped.
    """
    g = _norm_gray(tile, size)
    if polarity is None:
        polarity = "bright" if g.mean() < 128 else "dark"

    if polarity == "bright":
        thr = np.percentile(g, pct)
        mask = (g >= thr).astype(np.uint8)
        weight = g
    else:
        thr = np.percentile(g, 100.0 - pct)
        mask = (g <= thr).astype(np.uint8)
        weight = (255.0 - g)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lbl = cv2.connectedComponents(mask, connectivity=8)[1]
    max_area = max_area_frac * size * size
    cand = []
    for c in range(1, n):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        ys, xs = np.where(lbl == c)
        w = weight[ys, xs]
        ws = w.sum()
        if ws <= 0:
            continue
        cy = float((ys * w).sum() / ws) / size
        cx = float((xs * w).sum() / ws) / size
        dominance = float(area) * float(w.mean())
        cand.append((dominance, cy, cx))
    cand.sort(key=lambda t: -t[0])
    out: List[Tuple[float, float]] = []
    for _, cy, cx in cand:
        if all(np.hypot(cy - oy, cx - ox) > min_sep for oy, ox in out):
            out.append((cy, cx))
        if len(out) >= k:
            break
    return np.array(out, dtype=np.float32).reshape(-1, 2), polarity


def _template_spots(tmpl: SceneTemplate, polarity: str, k: int = 6,
                    thr: float = 0.4, min_sep: float = 0.04) -> np.ndarray:
    """Peak locations of the template's extreme map for the given polarity."""
    m = tmpl.bright if polarity == "bright" else tmpl.dark
    size = tmpl.size
    mask = (m >= thr * m.max()).astype(np.uint8) if m.max() > 0 else np.zeros_like(m, np.uint8)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand = []
    for c in range(1, n):
        ys, xs = np.where(lbl == c)
        w = m[ys, xs]
        ws = w.sum()
        if ws <= 0:
            continue
        cy = float((ys * w).sum() / ws) / size
        cx = float((xs * w).sum() / ws) / size
        cand.append((stats[c, cv2.CC_STAT_AREA] * float(w.mean()), cy, cx))
    cand.sort(key=lambda t: -t[0])
    out: List[Tuple[float, float]] = []
    for _, cy, cx in cand:
        if all(np.hypot(cy - oy, cx - ox) > min_sep for oy, ox in out):
            out.append((cy, cx))
        if len(out) >= k:
            break
    return np.array(out, dtype=np.float32).reshape(-1, 2)


# ---------------------------------------------------------------------------------
# Saturated-light matcher (for the confident FORCE-SPLIT of foreign near-black frames).
# Unlike the percentile coverage above, this localizes only genuine SATURATED point
# lights on both sides, so it is precise enough to assert a CONFIDENT mismatch (zero
# lights align) rather than a soft score. It is paired with a capture-order
# non-adjacency gate by the caller, never used alone.
_SL_ABS_FLOOR = 45    # a surviving light in a near-black frame clears this abs level
_SL_REL = 0.55        # ...or this fraction of the frame's own max (handles dim frames)
_SL_SAT = 240         # near-saturation on a well frame = a true light / blown highlight
_SL_MAXA = 400        # compact: a point-light blob, not a whole wall (px @256)
_SL_MINA = 2
_SL_TOL = 0.05
_SL_MAXT = 0.08
_SL_K = 8


def _sat_blobs(tile: np.ndarray, thr: float, size: int = 256) -> np.ndarray:
    """Compact bright-blob centroids (intensity-weighted, y,x in [0,1]) above ``thr``."""
    g = _norm_gray(tile, size)
    mask = (g >= thr).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for c in range(1, n):
        a = stats[c, cv2.CC_STAT_AREA]
        if a < _SL_MINA or a > _SL_MAXA:
            continue
        ys, xs = np.where(lbl == c)
        w = g[ys, xs].astype(np.float32)
        ws = w.sum()
        if ws <= 0:
            continue
        out.append((float(a) * float(w.mean()),
                    float((ys * w).sum() / ws) / size, float((xs * w).sum() / ws) / size))
    out.sort(key=lambda t: -t[0])
    pts: List[Tuple[float, float]] = []
    for _, y, x in out:
        if all(np.hypot(y - py, x - px) > 0.04 for py, px in pts):
            pts.append((y, x))
        if len(pts) >= _SL_K:
            break
    return np.array(pts, np.float32).reshape(-1, 2)


def clip_lights(tile: np.ndarray) -> np.ndarray:
    """Surviving saturated point-lights of a near-black frame (y,x in [0,1])."""
    thr = max(_SL_ABS_FLOOR, float(_norm_gray(tile, 256).max()) * _SL_REL)
    return _sat_blobs(tile, thr)


def well_lights(well_tiles: List[np.ndarray]) -> np.ndarray:
    """Union (deduped) of the saturated light fixtures across a scene's well frames."""
    allp = [p for p in (_sat_blobs(t, _SL_SAT) for t in well_tiles) if len(p)]
    if not allp:
        return np.zeros((0, 2), np.float32)
    P = np.vstack(allp)
    keep: List[Tuple[float, float]] = []
    for y, x in P:
        if all(np.hypot(y - ky, x - kx) > 0.04 for ky, kx in keep):
            keep.append((y, x))
    return np.array(keep, np.float32)


def light_coverage(clip_tile: np.ndarray,
                   well_tiles: List[np.ndarray]) -> Tuple[Optional[float], int, int]:
    """(coverage, n_clip_lights, n_well_lights). Coverage = fraction of the clipped
    frame's saturated lights that land on a well-frame light under one global
    translation (same camera). ``cov == 0`` with enough lights on both sides is a
    CONFIDENT scene mismatch. Returns ``(None, na, nb)`` when either side has no
    lights (cannot decide)."""
    A = clip_lights(clip_tile)
    B = well_lights(well_tiles)
    if len(A) == 0 or len(B) == 0:
        return None, len(A), len(B)
    cands = [np.zeros(2, np.float32)]
    for a in A:
        for b in B:
            t = b - a
            if np.hypot(*t) <= _SL_MAXT:
                cands.append(t)
    best = 0
    for t in cands:
        At = A + t
        pr = sorted((float(np.hypot(*(At[i] - B[j]))), i, j)
                    for i in range(len(At)) for j in range(len(B)))
        ui: set = set(); uj: set = set(); n = 0
        for d, i, j in pr:
            if d > _SL_TOL:
                break
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j); n += 1
        best = max(best, n)
    return best / len(A), len(A), len(B)


def lights_present_fraction(well_tile: np.ndarray, dark_tiles: List[np.ndarray],
                            tol: float = 0.06, max_t: float = 0.10) -> Tuple[Optional[float], int, int]:
    """Fraction of a WELL-exposed frame's clean light fixtures that are also present
    among a set of DARK frames' surviving bright spots (well->dark direction), under
    one global translation. The well frame has few, clean lights; a dark frame of the
    SAME scene keeps those same lights (plus dark-room noise), so this fraction is
    high for a stranded well orphan and low for a different scene. Returns
    ``(fraction, n_well_lights, n_dark_lights)`` or ``(None, ., .)`` if undecidable.

    This is the reverse of :func:`light_coverage` (clip->well); it is the right
    direction when REATTACHING a well-exposed orphan to its dark bracket, because the
    well frame is the clean side and should not be diluted by the dark frame's noise.
    """
    A = well_lights([well_tile])
    parts = [p for p in (clip_lights(t) for t in dark_tiles) if len(p)]
    if len(A) == 0 or not parts:
        return None, len(A), 0
    P = np.vstack(parts)
    keep: List[Tuple[float, float]] = []
    for y, x in P:
        if all(np.hypot(y - ky, x - kx) > 0.04 for ky, kx in keep):
            keep.append((y, x))
    B = np.array(keep, np.float32)
    cands = [np.zeros(2, np.float32)]
    for a in A:
        for b in B:
            t = b - a
            if np.hypot(*t) <= max_t:
                cands.append(t)
    best = 0
    for t in cands:
        At = A + t
        pr = sorted((float(np.hypot(*(At[i] - B[j]))), i, j)
                    for i in range(len(At)) for j in range(len(B)))
        ui: set = set(); uj: set = set(); n = 0
        for d, i, j in pr:
            if d > tol:
                break
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j); n += 1
        best = max(best, n)
    return best / len(A), len(A), len(B)


def _one_to_one(A: np.ndarray, B: np.ndarray, tol: float) -> int:
    pairs = sorted(((float(np.hypot(*(A[i] - B[j]))), i, j)
                    for i in range(len(A)) for j in range(len(B))), key=lambda t: t[0])
    ua, ub, n = set(), set(), 0
    for d, i, j in pairs:
        if d > tol:
            break
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); n += 1
    return n


def coverage_score(clipped_tile: np.ndarray, tmpl: SceneTemplate,
                   polarity: Optional[str] = None,
                   tol: float = 0.045, max_t: float = 0.08,
                   k: int = 6) -> float:
    """Coverage/inlier score in [0,1]: fraction of the clipped frame's extreme spots
    that land on the template's corresponding extreme structure, allowing one global
    translation (exhaustive 2-point hypotheses; same camera). Immune to missing or
    spurious blobs (they are simply non-inliers)."""
    if polarity is None:
        polarity = clip_polarity(clipped_tile) or (
            "bright" if brightness(clipped_tile) < 128 else "dark")
    A, _ = extreme_spots(clipped_tile, polarity, size=tmpl.size, k=k)
    B = _template_spots(tmpl, polarity)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    cands = [np.zeros(2, np.float32)]
    for a in A:
        for b in B:
            t = b - a
            if np.hypot(*t) <= max_t:
                cands.append(t)
    best = 0
    for t in cands:
        best = max(best, _one_to_one(A + t, B, tol))
    return best / float(len(A))
