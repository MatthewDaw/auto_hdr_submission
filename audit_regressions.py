"""For each named (newly-missed) GT group, show how our prediction differs from GT:
is it an over-SPLIT (we broke the GT group into pieces) or an over-MERGE (we pulled
in frames from other GT groups), and which other groups are involved. Helps decide
whether the disagreement is a true regression or a ground-truth correction.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

data = Path("data/large")
raw = np.load(data/"raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]
col = np.load(data/"img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
color = np.stack([cimgs[cidx[f]] for f in files]) if all(f in cidx for f in files) else None
B = {f: float(gray[i].mean()) for i, f in enumerate(files)}

f2g = {}; gt = defaultdict(set)
with open(data/"public_manifest.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        gt[row["group_id"]].add(row["filename"]); f2g[row["filename"]] = row["group_id"]

pred = ImageGrouper().group(Photoshoot(list(files), gray, color))
f2c = {}
for ci, grp in enumerate(pred):
    for f in grp: f2c[f] = ci
cluster = defaultdict(set)
for f, c in f2c.items(): cluster[c].add(f)

missed = sys.argv[1:] or ['14579','15545','15665','20040','22534','23690','34588',
 '40648','42093','43596','44888','49265','5125','5332','54656','55384','63350',
 '63560','65892','69279','77535','89342','93065','9635']

for g in missed:
    gframes = gt[g]
    cl_ids = {f2c[f] for f in gframes if f in f2c}
    # how the GT group is distributed across our clusters
    parts = []
    foreign = defaultdict(int)
    for c in cl_ids:
        members = cluster[c]
        comp = defaultdict(int)
        for f in members: comp[f2g.get(f, "?")] += 1
        parts.append((c, dict(comp)))
        for og, n in comp.items():
            if og != g: foreign[og] += n
    kind = []
    if len(cl_ids) > 1: kind.append(f"SPLIT into {len(cl_ids)}")
    if foreign: kind.append(f"MERGE with {dict(foreign)}")
    if not kind: kind = ["exact?"]
    brange = [round(B[f]) for f in sorted(gframes)]
    print(f"\n=== {g}  ({len(gframes)} frames, B={brange}) :: {' + '.join(kind)} ===")
    for c, comp in parts:
        print(f"   cluster{c}: {comp}")
