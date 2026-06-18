"""
Enumerate exact failing cases at a given threshold, using the cached descriptors.
Categorizes each failed reference group so we know what to iterate on:
  - over-split (pieces, no contamination) vs over-merge (contaminated)
  - drone? (DJI in any member filename)
  - size, intra-group ZNCC min/mean, brightness range
Run after eval_large.py has cached descriptors.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.44
IMG = DATA / "images"
d = np.load(DATA / "desc_cache.npz", allow_pickle=True)
M = d["M"]; files = list(d["files"])

groups = defaultdict(set); f2g = {}
for r in csv.DictReader(open(DATA / "public_manifest.csv")):
    groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]

idxof = {f: i for i, f in enumerate(files)}
sim = M @ M.T
A = sim >= THR; np.fill_diagonal(A, False)
_, lab = connected_components(csr_matrix(A), directed=False)
pred = defaultdict(set)
for i, f in enumerate(files): pred[lab[i]].add(f)
predlk = set(frozenset(v) for v in pred.values())
file2pred = {f: lab[i] for i, f in enumerate(files)}

def bright(f):
    buf = np.fromfile(str(IMG / f), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return float(im.mean()) if im is not None else float("nan")

rows = []
for g, members in groups.items():
    rs = frozenset(members)
    if rs in predlk: continue
    ix = [idxof[f] for f in members]
    drone = any("DJI" in f for f in members)
    if len(ix) >= 2:
        sub = sim[np.ix_(ix, ix)]; u = np.triu_indices(len(ix), 1)
        imin, imean = sub[u].min(), sub[u].mean()
    else:
        imin = imean = float("nan")
    clusters = set(file2pred[f] for f in members)
    contaminated = any((pred[c] - members) for c in clusters)
    kind = "OVER-MERGE" if contaminated else "OVER-SPLIT"
    rows.append((g, len(members), drone, kind, imin, imean, len(clusters)))

rows.sort(key=lambda r: (r[3], -r[1]))
nsplit = sum(1 for r in rows if r[3] == "OVER-SPLIT")
nmerge = sum(1 for r in rows if r[3] == "OVER-MERGE")
ndrone = sum(1 for r in rows if r[2])
print(f"thr={THR}: {len(rows)} failed groups | over-split={nsplit} over-merge={nmerge} | drone-involved={ndrone}")
print(f"{'group':>8} {'sz':>3} {'drone':>5} {'kind':>11} {'intra-min':>9} {'intra-mean':>10} {'pieces':>6}")
for g, sz, dr, kind, imin, imean, npc in rows[:60]:
    print(f"{g:>8} {sz:>3} {str(dr):>5} {kind:>11} {imin:>9.3f} {imean:>10.3f} {npc:>6}")
print(f"\n(showing up to 60 of {len(rows)} failures)")
