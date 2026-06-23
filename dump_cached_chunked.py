"""Cached, group-preserving CHUNKED regroup for a large dataset (e.g. full_subset).

Mirrors process_subset.py's chunking but loads raw256.npz + img128c.npz instead of
re-decoding images/ — so only the grouping compute runs. Writes pred_labels.json.
Usage: dump_cached_chunked.py <data_dir> [chunk_size]
"""
import json, sys
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

data = Path(sys.argv[1])
chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
raw = np.load(data/"raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]; gid = list(raw["gid"])
col = np.load(data/"img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]

# group-preserving chunks (never split a GT group across chunks)
by_group = {}
for i, g in enumerate(gid):
    by_group.setdefault(g, []).append(i)
chunks, cur = [], []
for members in by_group.values():
    if cur and len(cur) + len(members) > chunk_size:
        chunks.append(cur); cur = []
    cur += members
if cur:
    chunks.append(cur)

labels, next_id = {}, 0
for ci, idxs in enumerate(chunks):
    fr = [files[i] for i in idxs]
    g = gray[idxs]
    c = np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    for grp in ImageGrouper().group(Photoshoot(fr, g, c)):
        for fn in grp:
            labels[fn] = next_id
        next_id += 1
    print(f"  chunk {ci}: {len(idxs)} imgs -> done", flush=True)

json.dump(labels, open(data/"pred_labels.json", "w"))
print(f"wrote {data}/pred_labels.json ({len(labels)} frames, {next_id} clusters)")
