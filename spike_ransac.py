"""SPIKE #1: RANSAC inlier-COUNT discriminator for clipped-frame blob matching.

Replaces greedy chamfer (which averages in the misses) with the field-standard
inlier count: hypothesize a translation from one blob correspondence, apply it,
and count how many OTHER blobs land on a counterpart. Same scene (same camera) =>
a consistent translation (~0) explains many blobs => high inlier fraction.
Different scene => no translation explains many => low. Missing/spurious blobs are
simply non-inliers (no penalty) — the property chamfer lacked.

Goal: does inlier fraction cleanly separate FOREIGN (want LOW) from SAME-scene
(want HIGH) clipped-frame pairs?
"""
import csv, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import cv2

ddir = Path("data/full_subset")
raw = np.load(ddir / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
pred = json.load(open(ddir / "pred_labels.json"))
f2g = {}
for r in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
    f2g[r["filename"]] = r["group_id"]
gt = defaultdict(list)
for f, g in f2g.items():
    gt[g].append(f)
B = lambda f: float(imgs[idx[f]].mean())

_mser = cv2.MSER_create()
_mser.setDelta(5); _mser.setMinArea(4); _mser.setMaxArea(int(0.15 * 256 * 256))

def blobs(gray, k=6):
    """Up to k DOMINANT MSER blob centroids (y,x in [0,1]). MSER is exposure-invariant
    by construction (extremal regions = pixel ordering). Run on the INVERTED image for
    a dark frame (bright light sources become dark extremal regions) and on the raw
    image for a bright frame (dark objects)."""
    m = float(gray.mean())
    src = (255 - gray) if m < 128 else gray
    regions, _ = _mser.detectRegions(src)
    bl = []
    for reg in regions:
        if len(reg) >= 4:
            cx, cy = reg[:, 0].mean(), reg[:, 1].mean()
            bl.append((len(reg), cy / 256.0, cx / 256.0))
    bl.sort(key=lambda t: -t[0])
    # dedupe near-duplicate MSER regions (nested extremal regions share a centroid)
    out = []
    for _, y, x in bl:
        if all(np.hypot(y - oy, x - ox) > 0.04 for oy, ox in out):
            out.append((y, x))
        if len(out) >= k:
            break
    return np.array(out)

def _one_to_one(A, Bb, tol):
    """Greedy one-to-one inlier count: each B-blob matches at most one A-blob."""
    pairs = sorted(((np.hypot(*(A[i] - Bb[j])), i, j)
                    for i in range(len(A)) for j in range(len(Bb))), key=lambda t: t[0])
    ua, ub, n = set(), set(), 0
    for d, i, j in pairs:
        if d > tol: break
        if i in ua or j in ub: continue
        ua.add(i); ub.add(j); n += 1
    return n

def inlier_frac(A, Bb, tol=0.035, max_t=0.10):
    """RANSAC inlier fraction: best SMALL translation (|t|<=max_t, same camera) with
    ONE-TO-ONE matching at a TIGHT tolerance. Same scene => identical blob positions
    => a near-zero t matches all; different scene => no small t matches many."""
    if len(A) == 0 or len(Bb) == 0:
        return None
    best = 0
    # hypotheses: t=0 (exact same position) + every small blob-correspondence shift
    cands = [np.zeros(2)] + [b - a for a in A for b in Bb if np.hypot(*(b - a)) <= max_t]
    for t in cands:
        best = max(best, _one_to_one(A + t, Bb, tol))
    return best / min(len(A), len(Bb))

def darkest(g, lo=45, n=3):
    return sorted((f for f in gt[g] if f in idx and B(f) <= lo), key=B)[:n]

print("=== FOREIGN (different scene, want LOW inlier-frac) ===")
fvals = []
for a, b in [("1038","10280"),("10464","10613"),("10886","10593"),
             ("14288","14037"),("14279","14983")]:
    cs = [{pred[f] for f in gt[g] if f in pred} for g in (a, b)]
    sh = set.intersection(*cs) if all(cs) else set()
    if not sh: print(f"  {a}+{b}: not co-merged"); continue
    cl = next(iter(sh)); fr = [f for f in pred if pred[f] == cl]
    cnt = Counter(f2g[f] for f in fr if f in f2g)
    minor, major = min(cnt, key=cnt.get), max(cnt, key=cnt.get)
    fd = [f for f in fr if f2g.get(f)==minor and f in idx and B(f) <= 45]
    md = [f for f in fr if f2g.get(f)==major and f in idx and B(f) <= 45]
    best = max((inlier_frac(blobs(imgs[idx[fa]]), blobs(imgs[idx[ma]])) or 0
                for fa in fd for ma in md), default=None)
    fvals.append(best)
    print(f"  {a}+{b} (foreign={minor}): inlier-frac={best:.2f}")

print("\n=== SAME scene (want HIGH inlier-frac): two near-black frames, same scene ===")
svals = []
for g in ["10463","13675","10147","10160","13093","13609","13586","12583","10149","10152"]:
    df = darkest(g, lo=55, n=3)
    if len(df) < 2:
        print(f"  {g}: <2 dark frames"); continue
    # best inlier-frac among same-scene dark pairs
    best = max((inlier_frac(blobs(imgs[idx[df[i]]]), blobs(imgs[idx[df[j]]])) or 0
                for i in range(len(df)) for j in range(i+1, len(df))), default=None)
    svals.append(best)
    print(f"  {g}: inlier-frac={best:.2f}  (dark B={[round(B(f)) for f in df]})")

print(f"\nSEPARATION: foreign max={max([v for v in fvals if v is not None], default=0):.2f}  "
      f"same min={min([v for v in svals if v is not None], default=0):.2f}")
