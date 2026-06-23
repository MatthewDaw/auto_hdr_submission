"""SPIKE: localized in-scene MOTION detection for Cluster C.

Same viewpoint, but something moved (curtain, door, pool-cover, finger). Edge-ZNCC
stays high (mostly identical), so it can't tell. But a real change is a LOCALIZED,
COHERENT region that differs between two frames at the SAME exposure level, while
a genuine still scene differs only by diffuse sensor noise.

Signal: align two same-level well-exposed frames, CLAHE-normalize, take the abs
diff, blur, and measure the largest localized changed region (peak * area). High =
something moved -> split; low = still scene -> keep.

Goal: does the localized-change score separate Cluster C pairs (different, want
HIGH) from genuine same-scene same-level pairs (want LOW)?
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

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def brightness(f): return float(imgs[idx[f]].mean())

def change_score(fa, fb):
    """Largest localized changed region between two frames (masked to co-valid)."""
    a = _clahe.apply(imgs[idx[fa]]).astype(np.float32)
    b = _clahe.apply(imgs[idx[fb]]).astype(np.float32)
    ga, gb = imgs[idx[fa]], imgs[idx[fb]]
    valid = (ga >= 8) & (ga <= 247) & (gb >= 8) & (gb <= 247)
    d = cv2.GaussianBlur(np.abs(a - b), (0, 0), 3)
    d[~valid] = 0
    changed = (d > 40).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    area = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0   # largest blob
    return float(d.max()), int(area)

def well(g):
    return [f for f in gt[g] if f in idx and 60 <= brightness(f) <= 195]

def matched_pairs(fa_list, fb_list, tol=8):
    out = []
    for fa in fa_list:
        ba = brightness(fa)
        cand = [fb for fb in fb_list if abs(brightness(fb) - ba) <= tol]
        if cand:
            out.append((fa, min(cand, key=lambda fb: abs(brightness(fb) - ba))))
    return out

def report(a, b, label):
    pairs = matched_pairs(well(a), well(b))
    if not pairs:
        print(f"  {label}: no same-level well pairs"); return
    best = max((change_score(*p) for p in pairs), key=lambda t: t[1])
    print(f"  {label}: peak={best[0]:.0f} changed_area={best[1]}")

print("=== Cluster C (in-scene MOTION, different) — want HIGH localized change ===")
for a, b, what in [("10600","10601","curtain"),("1087","1093","pool-cover"),
                   ("11573","11574","window"),("13339","13340","door"),
                   ("17879","17880","finger")]:
    if not gt[a] or not gt[b]: print(f"  {a} vs {b}: missing"); continue
    report(a, b, f"{a} vs {b} ({what})")

print("\n=== Safety: single-bracket groups — want NO same-level pairs (can't fire) ===")
for g in ["10147","13601","12583","13093","13586","10160","13609","14011"]:
    fs = well(g)
    pairs = matched_pairs(fs, fs)
    pairs = [(a, b) for a, b in pairs if a != b]
    if not pairs:
        print(f"  {g}: 0 same-level pairs (safe)")
    else:
        best = max((change_score(*p) for p in pairs), key=lambda t: t[1])
        print(f"  {g}: {len(pairs)} same-level pairs! peak={best[0]:.0f} area={best[1]}")
