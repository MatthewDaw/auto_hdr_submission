from collections import defaultdict
import numpy as np, cv2
DATA="data/large"; VLO,VHI=8,247
rd=np.load(f"{DATA}/raw256.npz",allow_pickle=True); raw=rd["imgs"]; files=list(rd["files"]); gid=rd["gid"]
g2i=defaultdict(list)
for i,f in enumerate(files): g2i[gid[i]].append(i)
B=np.array([raw[i].mean() for i in range(len(raw))])
clahe=cv2.createCLAHE(3.0,(8,8))
def gm(i):
    g=clahe.apply(raw[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((raw[i]>=VLO)&(raw[i]<=VHI))
def mz(i,j):
    a,va=gm(i); b,vb=gm(j); v=(va&vb).ravel(); c=int(v.sum())
    if c<300: return None
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))
def vf(i): return float(((raw[i]>=VLO)&(raw[i]<=VHI)).mean())
for g in ["10370","35098","82871","56453","56460","70877","22901"]:
    mem=g2i[g]; dead=[i for i in mem if vf(i)<0.03]
    well=[i for i in mem if 55<=B[i]<=200]
    # coherence among well-exposed (one scene?) ; if <2 well, use mid-exposed
    pool=well if len(well)>=2 else [i for i in mem if vf(i)>=0.15]
    cohs=[mz(pool[a],pool[b]) for a in range(len(pool)) for b in range(a+1,len(pool))]
    cohs=[c for c in cohs if c is not None]
    coh=min(cohs) if cohs else None
    print(f"g{g}: n={len(mem)} dead(vf<3%)={len(dead)} (B={[int(B[i]) for i in dead]}) wellN={len(well)} "
          f"well-coherence(min masked)={coh if coh is None else round(coh,2)}  "
          f"-> {'ONE SCENE (info-limited)' if (coh is not None and coh>=0.45) else ('cohLOW?' if coh is not None else 'n/a')}")
