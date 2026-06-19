"""
FIX 2: exposure-ladder orphan re-attachment via MASKED similarity.
After base fusion clustering, for each CLIPPED frame, generate candidates by
BRIGHTNESS-ADJACENCY (its exposure-neighbors across the val set, NOT embedding —
the embedding is also dead on clipped frames), compute co-valid MASKED gradient
ZNCC, and add an edge to candidates with high masked ZNCC. Re-cluster, measure.
Precision guard: masked ZNCC on the (small) co-valid region only matches true
same-scene frames; optional uniqueness margin.
"""
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA="data/large"; VLO,VHI=8,247
W_GRAD=0.65; KBR=25; MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12

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

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.where(np.array([g in val_g for g in gid]))[0]; vf=[files[i] for i in va]; n=len(vf)
    G=(d["M"][va]@d["M"][va].T).astype(np.float32); Es=(emb(mc,li[va])@emb(mc,li[va]).T).astype(np.float32)
    B=np.array([raw[i].mean() for i in va]); Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(vf,gid[va]): groups[g].add(f)
    sc0,A0=best_thr(Fz,vf,groups); base,ok0,lab0=score(A0,vf,groups)
    print(f"BASE fusion: {base:.4f} ({len(ok0)}/{len(groups)})")
    gm=[grad_mask(raw[i]) for i in va]
    clipped=[i for i in range(n) if B[i]<45 or B[i]>210]
    # cluster -> member indices + per-cluster internal brightness step
    clmembers=defaultdict(list)
    for j in range(n): clmembers[lab0[j]].append(j)
    clstep={}
    for c,mem in clmembers.items():
        bs=np.sort(B[mem])
        clstep[c]=np.median(np.diff(bs)) if len(bs)>=2 else 80.0   # default step for singletons
    A=A0.copy(); added=0
    for i in clipped:
        # candidate = each OTHER cluster's brightness-nearest member to i (ladder endpoint)
        scored=[]
        for c,mem in clmembers.items():
            if c==lab0[i]: continue
            j=min(mem, key=lambda k: abs(B[k]-B[i]))
            # LADDER-CONTINUITY guard: orphan must extend ladder by ~<=1 EV step
            if abs(B[i]-B[j]) > 1.8*max(clstep[c], 30.0): continue
            mz=masked_zncc(gm[i][0],gm[i][1],gm[j][0],gm[j][1])
            scored.append((mz,j))
        scored.sort(reverse=True)
        if scored and scored[0][0]>=MASK_THR:
            bj=scored[0][1]
            second=scored[1][0] if len(scored)>1 else -1   # 2nd-best cluster (already distinct clusters)
            if scored[0][0]-second>=MARGIN or second<MASK_THR:
                A[i,bj]=A[bj,i]=True; added+=1
                gi=gid[va][i]; gj=gid[va][bj]
                print(f"  edge: i(g{gi},B{B[i]:.0f}) -- j(g{gj},B{B[bj]:.0f}) masked={scored[0][0]:.3f} 2nd={second:.3f} {'SAME' if gi==gj else 'DIFF-REF'}")
    sc1,ok1,_=score(A,vf,groups)
    print(f"FIX2 ladder re-attach (added {added}): {sc1:.4f} ({len(ok1)}/{len(groups)})")
    print(f"  recovered: {sorted(ok1-ok0)}")
    print(f"  BROKEN (precision loss): {sorted(ok0-ok1)}")

if __name__=="__main__":
    main()
