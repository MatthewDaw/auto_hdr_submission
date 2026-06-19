"""Cache raw 256px grayscale (pre-CLAHE) for valid-pixel masking + masked correlation."""
import csv, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np, cv2
DATA=Path(sys.argv[1] if len(sys.argv)>1 else "data/large"); RES=256; OUT=DATA/f"raw{RES}.npz"
def read_gray(p):
    try:
        im=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is not None: return im
    except Exception: pass
    try:
        from PIL import Image,ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
        return np.array(Image.open(p).convert("L"))
    except Exception: return None
def proc(f):
    im=read_gray(DATA/"images"/f)
    if im is None: return np.zeros((RES,RES),np.uint8)
    return cv2.resize(im,(RES,RES),interpolation=cv2.INTER_AREA)
def main():
    f2g={r["filename"]:r["group_id"] for r in csv.DictReader(open(DATA/"public_manifest.csv", encoding="utf-8"))}
    files=sorted(f2g); cv2.setNumThreads(1)
    with Pool(max(1,cpu_count()-1)) as pool: imgs=np.array(list(pool.imap(proc,files,chunksize=16)),np.uint8)
    np.savez(OUT,imgs=imgs,files=np.array(files),gid=np.array([f2g[f] for f in files]))
    print(f"cached {len(files)} -> {OUT} ({imgs.nbytes/1e6:.0f} MB)")
if __name__=="__main__":
    main()
