"""Build feat_cache.npz (gradient descriptor M) for a dataset dir, matching the
format fix2_eval.py expects: 256px CLAHE -> Sobel magnitude -> 64x64 -> z-normalized.
Files sorted (same order as build_raw256.py). Usage: build_feat.py <data_dir>"""
import csv, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np, cv2
DATA=Path(sys.argv[1] if len(sys.argv)>1 else "data/large"); SIZE=256; ZNCC=64
OUT=DATA/"feat_cache.npz"
def read_gray(p):
    try:
        im=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is not None: return im
    except Exception: pass
    try:
        from PIL import Image,ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
        return np.array(Image.open(p).convert("L"))
    except Exception: return None
def desc(f):
    im=read_gray(DATA/"images"/f)
    if im is None: return np.zeros(ZNCC*ZNCC,np.float32)
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    g=clahe.apply(cv2.resize(im,(SIZE,SIZE),interpolation=cv2.INTER_AREA)).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    z=cv2.resize(cv2.magnitude(gx,gy),(ZNCC,ZNCC),interpolation=cv2.INTER_AREA).ravel()
    z=z-z.mean(); nn=np.linalg.norm(z)
    return (z/nn if nn>0 else z).astype(np.float32)
def main():
    f2g={r["filename"]:r["group_id"] for r in csv.DictReader(open(DATA/"public_manifest.csv",encoding="utf-8"))}
    files=sorted(f2g); cv2.setNumThreads(1); t0=time.time()
    with Pool(max(1,cpu_count()-1)) as pool: M=np.array(list(pool.imap(desc,files,chunksize=16)),np.float32)
    bad=int((np.linalg.norm(M,axis=1)==0).sum())
    np.savez(OUT,M=M,files=np.array(files))
    print(f"cached {len(files)} -> {OUT} ({M.nbytes/1e6:.0f} MB, {bad} undecodable, {time.time()-t0:.0f}s)")
if __name__=="__main__":
    main()
