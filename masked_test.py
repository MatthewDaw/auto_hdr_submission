"""
FIX 1 diagnostic: does co-valid-pixel masked gradient correlation recover the
within-group adjacent-exposure links that plain ZNCC misses on clipping groups?
For each failing group, sort members by brightness and compare, for each
brightness-adjacent pair, plain gradient-ZNCC vs co-valid-masked gradient-ZNCC.
"""
import csv
from collections import defaultdict
import numpy as np, cv2

DATA="data/large"
d=np.load(f"{DATA}/raw256.npz",allow_pickle=True)
RAW=d["imgs"]; files=list(d["files"]); gid=d["gid"]
idxof={f:i for i,f in enumerate(files)}
groups=defaultdict(list)
for f,g in zip(files,gid): groups[g].append(f)

VLO,VHI=8,247   # valid (non-clipped) intensity band
clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))

def grad_and_mask(raw):
    g=clahe.apply(raw).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(gx,gy)
    valid=(raw>=VLO)&(raw<=VHI)
    return mag, valid

def plain_zncc(m1,m2):
    a=m1.ravel()-m1.mean(); b=m2.ravel()-m2.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def masked_zncc(m1,v1,m2,v2):
    v=(v1&v2).ravel()
    if v.sum()<200: return float('nan'), float(v.mean())
    a=m1.ravel()[v]; b=m2.ravel()[v]
    a=a-a.mean(); b=b-b.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)), float(v.mean())

for G in ["94278","33501","11393","40599","40615"]:
    mem=groups[G]
    feats=[(RAW[idxof[f]].mean(), f) for f in mem]
    feats.sort()
    print(f"\n=== group {G} ({len(mem)} frames), sorted by brightness ===")
    gm=[grad_and_mask(RAW[idxof[f]]) for _,f in feats]
    print(f"{'brightA':>8} {'brightB':>8} {'plainZNCC':>10} {'maskedZNCC':>11} {'validFrac':>9}")
    for k in range(len(feats)-1):
        bA=feats[k][0]; bB=feats[k+1][0]
        pz=plain_zncc(gm[k][0],gm[k+1][0])
        mz,vf=masked_zncc(gm[k][0],gm[k][1],gm[k+1][0],gm[k+1][1])
        print(f"{bA:>8.0f} {bB:>8.0f} {pz:>10.3f} {mz:>11.3f} {vf:>9.2f}")
