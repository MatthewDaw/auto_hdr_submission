"""
FIX 3: drone-drift recovery via ORB + RANSAC homography.
For drone frames (oracle: 'DJI' in filename; real deploy needs a pixel classifier),
add an edge between two drone frames in different clusters if ORB feature matching
+ RANSAC finds a near-identity homography with enough inliers. Tolerates small
hover/rotation drift that pixel-aligned ZNCC misses, without merging different scenes.
"""
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA="data/large"; W_GRAD=0.65
MIN_INLIERS=22; RATIO=0.75; NFEAT=1500

class Model(nn.Module):
    def __init__(s,dim=128):
        super().__init__(); m=mobilenet_v3_small(weights=None)
        s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,dim))
    def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)
def load_lab(p):
    d=np.load(p,allow_pickle=True); bgr=d["imgs"]; cl=cv2.createCLAHE(3.0,(8,8)); lab=np.empty_like(bgr)
    for i in range(len(bgr)):
        L=cv2.cvtColor(bgr[i],cv2.COLOR_BGR2Lab); L[:,:,0]=cl.apply(L[:,:,0]); lab[i]=L
    return lab,d["gid"],list(d["files"])
@torch.no_grad()
def emb(model,lab,bs=256):
    model.eval(); M=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); S=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(lab),bs):
        x=torch.from_numpy(lab[i:i+bs]).float().to(DEV)/255.0; x=x.permute(0,3,1,2); out.append(model((x-M)/S).cpu().numpy())
    return np.concatenate(out)
def best_thr(sim,files,groups):
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)
        if sc>best[0]: best=(sc,A.copy())
    return best
def score(A,files,groups):
    A=A.copy(); np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    refsets=set(frozenset(v) for v in groups.values()); predlk=set(frozenset(v) for v in pred.values())
    okset={g for g,v in groups.items() if frozenset(v) in predlk}
    return len(okset)/len(refsets), okset, lab

orb=cv2.ORB_create(nfeatures=NFEAT)
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
def orb_inliers(raw_i, raw_j):
    k1,d1=orb.detectAndCompute(raw_i,None); k2,d2=orb.detectAndCompute(raw_j,None)
    if d1 is None or d2 is None or len(k1)<8 or len(k2)<8: return 0,None
    matches=bf.knnMatch(d1,d2,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<RATIO*n.distance]
    if len(good)<8: return 0,None
    pts1=np.float32([k1[m.queryIdx].pt for m in good]); pts2=np.float32([k2[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(pts1,pts2,cv2.USAC_MAGSAC,5.0)
    if H is None: return 0,None
    return int(mask.sum()), H

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.where(np.array([g in val_g for g in gid]))[0]; vf=[files[i] for i in va]; n=len(vf)
    G=(d["M"][va]@d["M"][va].T).astype(np.float32); E=emb(mc,li[va]); Es=(E@E.T).astype(np.float32)
    Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    B0=np.array([raw[k].mean() for k in va])
    groups=defaultdict(set)
    for f,g in zip(vf,gid[va]): groups[g].add(f)
    sc0,A0=best_thr(Fz,vf,groups); base,ok0,lab0=score(A0,vf,groups)
    print(f"BASE fusion: {base:.4f} ({len(ok0)}/{len(groups)})")
    drone=[i for i in range(n) if "DJI" in vf[i]]
    print(f"drone frames in val: {len(drone)}")
    # DEBUG: inlier counts for SAME-ref split drone pairs (the targets) vs DIFF-ref
    print("  -- same-ref split drone pairs (targets) --")
    for a in range(len(drone)):
        for b in range(a+1,len(drone)):
            i,j=drone[a],drone[b]
            if gid[va][i]==gid[va][j] and lab0[i]!=lab0[j]:
                inl,_=orb_inliers(raw[va[i]],raw[va[j]])
                print(f"    g{gid[va][i]} B{B0[i]:.0f}/{B0[j]:.0f} emb={Es[i,j]:.2f} inliers={inl}")
    A=A0.copy(); added=0
    for a in range(len(drone)):
        i=drone[a]
        for b in range(a+1,len(drone)):
            j=drone[b]
            if lab0[i]==lab0[j] or A[i,j]: continue
            if Es[i,j]<0.4: continue                       # cheap embedding pre-filter (drone pairs share aerial look)
            inl,H=orb_inliers(raw[va[i]],raw[va[j]])
            if inl>=MIN_INLIERS:
                A[i,j]=A[j,i]=True; added+=1
                gi,gj=gid[va][i],gid[va][j]
                print(f"  edge: g{gi}--g{gj} inliers={inl} emb={Es[i,j]:.2f} {'SAME' if gi==gj else 'DIFF-REF'}")
    sc1,ok1,_=score(A,vf,groups)
    print(f"FIX3 drone ORB+RANSAC (added {added}): {sc1:.4f} ({len(ok1)}/{len(groups)})")
    print(f"  recovered: {sorted(ok1-ok0)}")
    print(f"  BROKEN: {sorted(ok0-ok1)}")

if __name__=="__main__":
    main()
