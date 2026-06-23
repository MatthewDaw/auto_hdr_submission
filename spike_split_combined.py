"""Process scene-A + scene-B frames TOGETHER and report cluster count.
2 == splitter correctly separates the two GT scenes; 1 == over-merge survives."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

DIRS = [Path("data/full_subset"), Path("data/large")]
loaded = []
for d in DIRS:
    if not (d / "raw256.npz").exists():
        continue
    raw = np.load(d / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    col = np.load(d / "img128c.npz", allow_pickle=True)
    cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
    gt = defaultdict(list)
    with open(d / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["group_id"]].append(row["filename"])
    loaded.append((imgs, idx, cimgs, cidx, gt))

def get(gid):
    for imgs, idx, cimgs, cidx, gt in loaded:
        fr = [f for f in gt.get(gid, []) if f in idx]
        if fr:
            g = np.stack([imgs[idx[f]] for f in fr])
            c = np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
            return fr, g, c
    return None, None, None

pairs = [("10280","1038"),("10464","10613"),("11533","11604"),
         ("14037","14288"),("14279","14983"),("12226","13169")]
for a, b in pairs:
    fa, ga, ca = get(a); fb, gb, cb = get(b)
    if fa is None or fb is None:
        print(f"{a}/{b}: missing"); continue
    fr = fa + fb
    gray = np.concatenate([ga, gb])
    color = np.concatenate([ca, cb]) if ca is not None and cb is not None else None
    out = ImageGrouper().group(Photoshoot(fr, gray, color))
    setA = set(fa)
    # how cleanly does the partition match GT?
    ok = {frozenset(fa), frozenset(fb)} == {frozenset(g) for g in out}
    print(f"{a}/{b}: {len(out)} cluster(s)  exact-2-split={ok}")
