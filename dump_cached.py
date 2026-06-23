"""Fast cached pred_labels dump — runs the real ImageGrouper from raw256.npz +
img128c.npz instead of re-decoding images/. Usage: dump_cached.py <data_dir>"""
import json, sys
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

data = Path(sys.argv[1])
raw = np.load(data/"raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]
col = np.load(data/"img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
color = np.stack([cimgs[cidx[f]] for f in files]) if all(f in cidx for f in files) else None

groups = ImageGrouper().group(Photoshoot(list(files), gray, color))
labels = {fn: cid for cid, grp in enumerate(groups) for fn in grp}
json.dump(labels, open(data/"pred_labels.json", "w"))
print(f"wrote {data}/pred_labels.json ({len(labels)} frames, {len(groups)} clusters)")
