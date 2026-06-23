"""Quick proxy: run the grouper on each named GT group's frames in isolation and
report how many clusters we produce. 1 == we keep GT's grouping (merged)."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

ddir = Path("data/full_subset")
raw = np.load(ddir / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]
idx = {f: i for i, f in enumerate(files)}
col = np.load(ddir / "img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]

gt = defaultdict(list)
with open(ddir / "public_manifest.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gt[row["group_id"]].append(row["filename"])

groups = sys.argv[1:] or ["3221","44897","5091","51090","58773","58916","62532",
    "10147","10149","10150","10152","10153","10159","10160","11574","12583",
    "13093","13360","13586","13590","13601","13609","14011","10125","10129",
    "40667","60452"]
for g in groups:
    fr = [f for f in gt.get(g, []) if f in idx]
    if not fr:
        print(f"{g}: (no frames in cache)"); continue
    gray = np.stack([imgs[idx[f]] for f in fr])
    color = np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    ps = Photoshoot(fr, gray, color)
    out = ImageGrouper().group(ps)
    print(f"{g}: {len(fr)} frames -> {len(out)} cluster(s)")
