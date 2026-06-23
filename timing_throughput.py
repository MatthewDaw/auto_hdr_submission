"""Measure end-to-end compute throughput (ms/image) of the current pipeline on a
cached dataset: descriptor extract + fusion clustering + all refinement passes.
Mirrors the slideshow's extract / cluster+refine split. Decode is excluded (the
caches are pre-decoded), matching the existing slide's basis."""
import sys, time
from pathlib import Path
import numpy as np

from autohdr import ImageGrouper
from autohdr.features import GradientDescriptor, WaveletEmbedding
from autohdr.image_loader import Photoshoot

data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = whole dataset
raw = np.load(data / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]
if limit and limit < len(files):
    files = files[:limit]; gray = gray[:limit]
col = np.load(data / "img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
color = (np.stack([cimgs[cidx[f]] for f in files])
         if all(f in cidx for f in files) else None)
n = len(files)
ps = Photoshoot(list(files), gray, color)

# warm import/JIT of cv2/pywt, then measure extract alone
GradientDescriptor().encode(gray[:8]); WaveletEmbedding().encode(gray[:8])
t = time.perf_counter()
GradientDescriptor().encode(gray)
WaveletEmbedding().encode(gray)
t_extract = time.perf_counter() - t

# full end-to-end (re-extracts internally, then clusters + runs every pass)
t = time.perf_counter()
groups = ImageGrouper().group(ps)
t_total = time.perf_counter() - t
t_cr = t_total - t_extract

print(f"dataset {data}  N={n}  -> {len(groups)} groups")
print(f"  extract        {t_extract:7.2f} s   ({t_extract/n*1000:5.2f} ms/img)")
print(f"  cluster+refine {t_cr:7.2f} s   ({t_cr/n*1000:5.2f} ms/img)")
print(f"  TOTAL          {t_total:7.2f} s   ({t_total/n*1000:5.2f} ms/img)")
