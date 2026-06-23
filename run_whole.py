"""Run the grouper on an ENTIRE cached dataset as one photoshoot (no chunking).

Unlike process_subset.py (which splits into group-preserving chunks so the dense
N x N stays small), this feeds every frame at once — so for N > AUTOHDR_DENSE_MAX
the sparse SimHash blocking path is exercised on real data at scale. Writes
pred_labels to the given path and reports timing.

Usage: python run_whole.py data/full_subset [out_pred.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

data = Path(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else str(data / "pred_labels.sparse.json")

raw = np.load(data / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]
col = np.load(data / "img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
color = (np.stack([cimgs[cidx[f]] for f in files])
         if all(f in cidx for f in files) else None)
print(f"loaded {len(files)} imgs (color={'yes' if color is not None else 'no'})",
      flush=True)

t = time.perf_counter()
groups = ImageGrouper().group(Photoshoot(list(files), gray, color))
dt = time.perf_counter() - t

labels = {}
for cid, g in enumerate(groups):
    for fn in g:
        labels[fn] = cid
json.dump(labels, open(out, "w"))
print(f"{len(files)} imgs -> {len(groups)} clusters in {dt:.1f}s -> {out}", flush=True)
