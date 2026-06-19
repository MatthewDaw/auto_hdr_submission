import csv
from collections import defaultdict
import numpy as np, cv2
DATA="data/large"; VLO,VHI=8,247
rd=np.load(f"{DATA}/raw256.npz",allow_pickle=True); raw=rd["imgs"]; files=list(rd["files"]); gid=rd["gid"]
idx={f:i for i,f in enumerate(files)}
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=fc["M"]; mfiles=list(fc["files"])
mi={f:k for k,f in enumerate(mfiles)}
clahe=cv2.createCLAHE(3.0,(8,8))
def gm(i):
    g=clahe.apply(raw[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((raw[i]>=VLO)&(raw[i]<=VHI))
def mz(i,j):
    a,va=gm(i); b,vb=gm(j); v=(va&vb).ravel(); c=int(v.sum())
    if c<50: return -1.0,c
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9)),c
g2i=defaultdict(list)
for i,f in enumerate(files): g2i[gid[i]].append(i)
B=np.array([raw[i].mean() for i in range(len(raw))])
sim=(M@M.T)
for g in ["10370","35098","48484","56453","56460","70877","82871","86016"]:
    mem=g2i[g]; orph=[i for i in mem if B[i]>245 or B[i]<12]
    well=[i for i in mem if 50<=B[i]<=205]
    if not well: print(f"g{g}: NO well-exposed member!"); continue
    rep=min(well,key=lambda i:abs(B[i]-128))
    for o in orph:
        own,ownc=mz(o,rep); vf=gm(o)[1].mean()
        # best WRONG: top gradient-sim frames in other groups, well-exposed
        oi=mi[files[o]]; order=np.argsort(-sim[oi])
        bw,bwg,bwc=-1,"",0; seen=0
        for k in order:
            f2=mfiles[k]; j=idx[f2]
            if gid[j]==g or not(50<=B[j]<=205): continue
            v,c=mz(o,j)
            if v>bw: bw,bwg,bwc=v,gid[j],c
            seen+=1
            if seen>=40: break
        print(f"g{g} orphan B{B[o]:.0f} validfrac={vf:.3f}  OWN mz={own:.2f}(n{ownc})  bestWRONG mz={bw:.2f} g{bwg}(n{bwc})  sep={own-bw:+.2f}")
