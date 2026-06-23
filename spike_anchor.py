"""SPIKE: rank-based EXTREME-ANCHOR signal for clipped frames.

At extreme exposure, edge-ZNCC dies (no co-valid overlap), but the scene's light
sources (bright spots, visible even in near-black frames) and dark objects (visible
even in near-white frames) sit at FIXED image positions. Compare two frames by the
spatial overlap of their brightest-K% and darkest-K% pixels (rank-based => exposure
invariant). Same still scene => high overlap; different scene => low.

Goal: does it separate Cluster A (same scene, want HIGH) from Cluster B (different
scenes, want LOW)?
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

ddir = Path("data/full_subset")
raw = np.load(ddir / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
gt = defaultdict(list)
for r in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
    gt[r["group_id"]].append(r["filename"])

GRID = 24
def anchor(img, frac=0.06):
    """(bright_mask, dark_mask) on a GRIDxGRID rank map."""
    d = cv2.resize(img, (GRID, GRID), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    k = max(1, int(len(d) * frac))
    order = np.argsort(d)
    dark = np.zeros(len(d), bool); dark[order[:k]] = True
    bright = np.zeros(len(d), bool); bright[order[-k:]] = True
    return bright, dark

def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0

def sig(f):
    return anchor(imgs[idx[f]])

def best_cross(fa_list, fb_list):
    """max anchor overlap between any frame of A and any frame of B (bright+dark)."""
    best = -1.0
    for fa in fa_list:
        ba, da = sig(fa)
        for fb in fb_list:
            bb, db = sig(fb)
            s = 0.5 * (iou(ba, bb) + iou(da, db))
            best = max(best, s)
    return best

def brightness(f):
    return float(imgs[idx[f]].mean())

def frames(g, lo=None, hi=None):
    fs = [f for f in gt[g] if f in idx]
    if lo is not None: fs = [f for f in fs if brightness(f) >= lo]
    if hi is not None: fs = [f for f in fs if brightness(f) <= hi]
    return fs

print("=== Cluster A (SAME scene, large exposure jump) — want HIGH anchor match ===")
# extreme (very dark/white) frames vs the group's well-exposed frames
A = ["10463", "14055", "19300", "13675", "10370"]
for g in A:
    if not gt[g]: print(f"  {g}: not in subset"); continue
    ext = frames(g, hi=30) + frames(g, lo=225)       # extreme frames
    well = frames(g, lo=70, hi=185)                   # well-exposed frames
    if not ext or not well: print(f"  {g}: no extreme/well frames (ext={len(ext)},well={len(well)})"); continue
    print(f"  {g}: extreme<->well anchor = {best_cross(ext, well):.3f}")

print("\n=== Cluster B (DIFFERENT scenes wrongly merged) — want LOW anchor match ===")
B = [("10276","1038"),("10464","10613"),("10886","10593"),("11533","11604"),
     ("11534","11605"),("14288","14037"),("14279","14983"),("17040","17060")]
for a, b in B:
    if not gt[a] or not gt[b]: print(f"  {a} vs {b}: missing ({len(gt[a])},{len(gt[b])})"); continue
    # compare at matched extreme: both dark, or both white
    for label, fa, fb in [("dark", frames(a, hi=40), frames(b, hi=40)),
                          ("white", frames(a, lo=220), frames(b, lo=220))]:
        if fa and fb:
            print(f"  {a} vs {b} [{label}]: anchor = {best_cross(fa, fb):.3f}")
