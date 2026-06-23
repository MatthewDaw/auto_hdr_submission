"""Sweep extreme-anchor scorer knobs to maximize positive/negative margin.

Reuses case assembly from spike_extreme_anchor; reimplements localization + scoring
locally with tunable params so we can pick the best config WITHOUT editing the module.
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autohdr.features import extreme_anchor as ea
import spike_extreme_anchor as S


def spots(g256, polarity, size, k, min_sep, pct, min_area, max_area_frac):
    g = g256.astype(np.float32)
    if g.shape[0] != size:
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    if polarity == "bright":
        thr = np.percentile(g, pct); mask = (g >= thr).astype(np.uint8); w = g
    else:
        thr = np.percentile(g, 100 - pct); mask = (g <= thr).astype(np.uint8); w = 255.0 - g
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    maxa = max_area_frac * size * size
    cand = []
    for c in range(1, n):
        a = stats[c, cv2.CC_STAT_AREA]
        if a < min_area or a > maxa:
            continue
        ys, xs = np.where(lbl == c); ww = w[ys, xs]; ws = ww.sum()
        if ws <= 0:
            continue
        cy = (ys * ww).sum() / ws / size; cx = (xs * ww).sum() / ws / size
        cand.append((a * ww.mean(), cy, cx))
    cand.sort(key=lambda t: -t[0])
    out = []
    for _, cy, cx in cand:
        if all(np.hypot(cy - oy, cx - ox) > min_sep for oy, ox in out):
            out.append((cy, cx))
        if len(out) >= k:
            break
    return np.array(out, np.float32).reshape(-1, 2)


def tmpl_spots(tmpl, polarity, k, min_sep, thr):
    m = tmpl.bright if polarity == "bright" else tmpl.dark
    size = tmpl.size
    if m.max() <= 0:
        return np.zeros((0, 2), np.float32)
    mask = (m >= thr * m.max()).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand = []
    for c in range(1, n):
        ys, xs = np.where(lbl == c); w = m[ys, xs]; ws = w.sum()
        if ws <= 0:
            continue
        cy = (ys * w).sum() / ws / size; cx = (xs * w).sum() / ws / size
        cand.append((stats[c, cv2.CC_STAT_AREA] * w.mean(), cy, cx))
    cand.sort(key=lambda t: -t[0])
    out = []
    for _, cy, cx in cand:
        if all(np.hypot(cy - oy, cx - ox) > min_sep for oy, ox in out):
            out.append((cy, cx))
        if len(out) >= k:
            break
    return np.array(out, np.float32).reshape(-1, 2)


def inlier(A, B, tol, max_t, norm):
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
        best = max(best, ea._one_to_one(A + t, B, tol))
    den = {"A": len(A), "min": min(len(A), len(B))}[norm]
    return best / float(den)


def evaluate(P):
    pos, neg = CASES
    pv = {"bright": [], "dark": []}; nv = {"bright": [], "dark": []}
    cache = {}
    for bucket, store in ((pos, pv), (neg, nv)):
        for label, f, t, pol, ds, well_g in bucket:
            key = (id(ds), well_g)
            if key not in cache:
                wt = S.well_tiles(ds, well_g)
                cache[key] = ea.build_template(wt, size=256) if wt else None
            tmpl = cache[key]
            if tmpl is None:
                continue
            A = spots(t, pol, 256, P["k"], P["min_sep"], P["pct"], P["min_area"], P["max_area_frac"])
            B = tmpl_spots(tmpl, pol, P["tk"], P["min_sep"], P["tthr"])
            s = inlier(A, B, P["tol"], P["max_t"], P["norm"])
            store[pol].append(s)
    res = {}
    for pol in ("bright", "dark"):
        Pv, Nv = pv[pol], nv[pol]
        if not Pv or not Nv:
            res[pol] = (None, None, None); continue
        res[pol] = (auc(Pv, Nv), np.median(Pv), np.median(Nv))
    return res, pv, nv


def auc(P, N):
    P = np.asarray(P); N = np.asarray(N)
    wins = sum((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()
               for _ in [0])
    return wins / (len(P) * len(N))


if __name__ == "__main__":
    CASES = S.build_cases()
    base = dict(k=6, tk=6, min_sep=0.06, pct=99.0, tthr=0.5,
                min_area=3, max_area_frac=0.10, tol=0.045, max_t=0.12, norm="min")
    grid = {
        "min_sep": [0.04, 0.06, 0.10, 0.14],
        "pct": [98.0, 99.0, 99.5],
        "tol": [0.03, 0.045, 0.06],
        "max_t": [0.08, 0.12, 0.18],
        "norm": ["A", "min"],
        "k": [4, 5, 6],
        "tthr": [0.4, 0.5, 0.6],
    }
    best = None
    # coordinate ascent
    cur = dict(base)
    for rounds in range(2):
        for key, vals in grid.items():
            scored = []
            for v in vals:
                p = dict(cur); p[key] = v
                res, _, _ = evaluate(p)
                # objective: minimize max(neg) overlap; use sum of margins (clip)
                aucs = [res[pol][0] for pol in ("bright", "dark") if res[pol][0] is not None]
                obj = sum(aucs)
                scored.append((obj, v, res))
            scored.sort(key=lambda x: -x[0])
            cur[key] = scored[0][1]
        print(f"round{rounds} cur={cur}")
    res, pv, nv = evaluate(cur)
    print("\nBEST CONFIG:", cur)
    for pol in ("bright", "dark"):
        Pv, Nv = pv[pol], nv[pol]
        if not Pv or not Nv:
            continue
        # best balanced-accuracy threshold
        cands = sorted(set([round(x, 4) for x in Pv + Nv]))
        bestthr, bestbal = 0.5, -1
        for c in cands:
            tp = sum(s >= c for s in Pv) / len(Pv)
            tn = sum(s < c for s in Nv) / len(Nv)
            if (tp + tn) / 2 > bestbal:
                bestbal, bestthr = (tp + tn) / 2, c
        fn = sum(s < bestthr for s in Pv); fp = sum(s >= bestthr for s in Nv)
        print(f"[{pol}] n_pos={len(Pv)} n_neg={len(Nv)} AUC={auc(Pv,Nv):.3f} "
              f"pos(min={min(Pv):.3f},med={np.median(Pv):.3f}) "
              f"neg(max={max(Nv):.3f},med={np.median(Nv):.3f}) "
              f"bestthr={bestthr:.3f} balacc={bestbal:.3f} FN={fn}/{len(Pv)} FP={fp}/{len(Nv)}")
