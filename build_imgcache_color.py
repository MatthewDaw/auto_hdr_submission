"""Cache 128x128 BGR uint8 (color) for a dataset dir. Lab conversion + CLAHE done in trainer."""
import csv, sys
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
RES = 128
OUT = DATA / f"img{RES}c.npz"

def read_bgr(p):
    try:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is not None: return im
    except Exception: pass
    try:
        from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
        return cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception: return None

def proc(fname):
    im = read_bgr(DATA / "images" / fname)
    if im is None: return np.zeros((RES, RES, 3), np.uint8)
    return cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)

def main():
    f2g = {}
    for r in csv.DictReader(open(DATA / "public_manifest.csv")):
        f2g[r["filename"]] = r["group_id"]
    files = sorted(f2g.keys())
    cv2.setNumThreads(1)
    with Pool(max(1, cpu_count()-1)) as pool:
        imgs = np.array(list(pool.imap(proc, files, chunksize=16)), np.uint8)
    gid = np.array([f2g[f] for f in files])
    np.savez(OUT, imgs=imgs, files=np.array(files), gid=gid)
    print(f"cached {len(files)} -> {OUT} ({imgs.nbytes/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
