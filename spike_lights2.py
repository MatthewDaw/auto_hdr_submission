"""SPIKE v2: robust dominant-light/dark-anchor matching, both polarities.

v1 failed because blob COUNT drifts with exposure. Fixes:
  * detect only the K DOMINANT anchors (strongest blobs) — stable across exposure;
  * polarity auto: near-black -> bright blobs (lights); near-white -> dark blobs;
  * one-directional cost: does every anchor of the LOWER-count frame land on an
    anchor of the other? (a foreign light with no counterpart = mismatch);
  * fair same-scene baseline: a scene's own extreme frame vs its well/other frame.
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

def anchors(gray, k=4):
    """Up to K dominant anchor centroids (y,x in [0,1]). Polarity by exposure:
    dark frame -> bright blobs (light sources); bright frame -> dark blobs."""
    g = gray.astype(np.float32)
    m = g.mean()
    bright = m < 128
    if bright:
        thr = max(float(np.percentile(g, 99.0)), m + 4 * g.std(), 35.0)
        mask = (g >= thr).astype(np.uint8)
    else:
        thr = min(float(np.percentile(g, 1.0)), m - 4 * g.std(), 220.0)
        mask = (g <= thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    blobs = sorted(((stats[c, cv2.CC_STAT_AREA], cent[c]) for c in range(1, n)
                    if stats[c, cv2.CC_STAT_AREA] >= 2), key=lambda t: -t[0])[:k]
    return [(cy / 256.0, cx / 256.0) for _, (cx, cy) in blobs]

def mismatch(la, lb):
    """Largest 'orphan' distance: for the frame with FEWER anchors, the max distance
    of its anchors to the nearest anchor of the other. A foreign light with no
    counterpart => large. 0 if positions all coincide. None if either empty."""
    if not la or not lb:
        return None
    if len(la) > len(lb):
        la, lb = lb, la
    pb = np.array(lb)
    return float(max(np.min(np.hypot(*(pb - np.array(p)).T)) for p in la))

def ext(g, dark_hi=45, white_lo=215):
    fs = [f for f in gt[g] if f in idx]
    d = [f for f in fs if imgs[idx[f]].mean() <= dark_hi]
    w = [f for f in fs if imgs[idx[f]].mean() >= white_lo]
    return d, w

print("=== FOREIGN (want HIGH mismatch) ===")
cases = [("1038","10280"),("10464","10613"),("10886","10593"),
         ("14288","14037"),("14279","14983"),("17040","17060")]
for a, b in cases:
    cs = [{pred[f] for f in gt[g] if f in pred} for g in (a, b)]
    sh = set.intersection(*cs) if all(cs) else set()
    if not sh: print(f"  {a}+{b}: not co-merged"); continue
    cl = next(iter(sh)); fr = [f for f in pred if pred[f] == cl]
    cnt = Counter(f2g[f] for f in fr if f in f2g)
    minor, major = min(cnt, key=cnt.get), max(cnt, key=cnt.get)
    fd = [f for f in fr if f2g.get(f)==minor and f in idx and (imgs[idx[f]].mean()<=45 or imgs[idx[f]].mean()>=215)]
    md = [f for f in fr if f2g.get(f)==major and f in idx]
    best = min(filter(lambda x: x is not None,
               (mismatch(anchors(imgs[idx[fa]]), anchors(imgs[idx[ma]]))
                for fa in fd for ma in md if abs(imgs[idx[fa]].mean()-imgs[idx[ma]].mean())<=60)),
               default=None)
    print(f"  {a}+{b} (foreign={minor}): mismatch={best}")

print("\n=== SAME scene (want LOW mismatch): extreme frame vs same-scene siblings ===")
for g in ["10463","13675","14258","14259","10147","10160","13093","13609"]:
    d, w = ext(g)
    extreme = (d[:1] or w[:1])
    if not extreme: print(f"  {g}: no extreme frame"); continue
    fs = [f for f in gt[g] if f in idx and f != extreme[0]]
    best = min(filter(lambda x: x is not None,
               (mismatch(anchors(imgs[idx[extreme[0]]]), anchors(imgs[idx[s]])) for s in fs)),
               default=None)
    print(f"  {g}: mismatch={best}  (extreme B={imgs[idx[extreme[0]]].mean():.0f})")
