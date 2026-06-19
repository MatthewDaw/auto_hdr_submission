"""
FIX 1 integration + measurement on large-val.
Base: fusion (gradient + color embedding) -> oracle threshold -> CC.
Recall pass for clipping: for each CLIPPED frame (extreme brightness), among its
top-K embedding-nearest frames, add an edge to any that is brightness-adjacent
(adjacent exposure) AND has high co-valid-MASKED gradient ZNCC. Targeted +
candidate-restricted so it can't link unrelated rooms.
Measures: score before/after, and which target groups recovered. Precision guard:
report any NEW over-merges introduced.
"""
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA="data/large"; VLO,VHI=8,247
W_GRAD=0.65; KEMB=20; MASK_THR=0.45; MIN_VALID=0.03; NEAREST_EXP=3

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

clahe=cv2.createCLAHE(3.0,(8,8))
def grad_mask(raw):
    g=clahe.apply(raw).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy),((raw>=VLO)&(raw<=VHI))
def masked_zncc(m1,v1,m2,v2):
    v=(v1&v2).ravel()
    if v.mean()<MIN_VALID: return -1.0
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def best_thr_lab(sim,files,groups):
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)
        if sc>best[0]: best=(sc,A.copy())
    return best

def score(adj,files,groups):
    A=adj.copy(); np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    refsets=set(frozenset(v) for v in groups.values()); predlk=set(frozenset(v) for v in pred.values())
    okset={g for g,v in groups.items() if frozenset(v) in predlk}
    return len(okset)/len(refsets), okset

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid]); vidx=np.where(va)[0]
    vf=[files[i] for i in vidx]; n=len(vf)
    G=(d["M"][vidx]@d["M"][vidx].T).astype(np.float32); E=emb(mc,li[vidx])
    Es=(E@E.T).astype(np.float32); B=np.array([raw[i].mean() for i in vidx])
    Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(vf,gid[vidx]): groups[g].add(f)

    sc0,A0=best_thr_lab(Fz,vf,groups); base,ok0=score(A0,vf,groups)
    print(f"BASE fusion: {base:.4f} ({len(ok0)}/{len(groups)})")

    # precompute grad+mask for val frames
    gm=[grad_mask(raw[i]) for i in vidx]
    clipped=[i for i in range(n) if B[i]<45 or B[i]>210]
    print(f"clipped frames in val: {len(clipped)}")
    A=A0.copy(); added=0
    for i in clipped:
        cand=np.argsort(-Es[i])[1:KEMB+1]                 # embedding-nearest candidates
        # among candidates, try the exposure-nearest few (rank by brightness gap)
        cand=sorted(cand, key=lambda j: abs(B[i]-B[j]))[:NEAREST_EXP]
        for j in cand:
            if A[i,j]: continue
            mz=masked_zncc(gm[i][0],gm[i][1],gm[j][0],gm[j][1])
            if mz>=MASK_THR:
                A[i,j]=A[j,i]=True; added+=1
    sc1,ok1=score(A,vf,groups)
    print(f"FIX1 masked-adjacent (added {added} edges): {sc1:.4f} ({len(ok1)}/{len(groups)})")
    print(f"  recovered groups: {sorted(ok1-ok0)}")
    print(f"  BROKEN groups (precision loss): {sorted(ok0-ok1)}")

if __name__=="__main__":
    main()
