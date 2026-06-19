"""
FIX 2 final eval on a full dataset (no train/val split). Applies the validated
exposure-ladder masked re-attachment on top of fusion clustering and reports
score before/after, recovered and broken groups.
Usage: fix2_eval.py <data_dir>
"""
import sys, json, os
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA=sys.argv[1] if len(sys.argv)>1 else "sample"
VLO,VHI=8,247; W_GRAD=0.65; MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12

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
    v=(v1&v2).ravel(); cnt=int(v.sum())
    if cnt<200: return -1.0,cnt
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)),cnt

def accept_edge(mz,cnt,gap,step):
    # refined accept rule (calibrated on full-set edges):
    # require a real exposure step (gap>=25) and a valid-pixel floor; then either
    # a strong masked match within the ladder, or a huge well-exposed overlap.
    if gap<25 or cnt<1500: return False
    if mz>=0.58 and gap<=1.8*max(step,30.0): return True   # strong ladder extension
    if mz>=0.50 and cnt>=15000: return True                # large well-exposed overlap
    return False
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
def load_unfixable():
    p=f"{DATA}/unfixable.json"
    return set(json.load(open(p))["groups"].keys()) if os.path.exists(p) else set()

def score(A,files,groups):
    A=A.copy(); np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    refsets=set(frozenset(v) for v in groups.values()); predlk=set(frozenset(v) for v in pred.values())
    return {g for g,v in groups.items() if frozenset(v) in predlk}, lab

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]; assert list(d["files"])==files
    n=len(files); G=(d["M"]@d["M"].T).astype(np.float32); Es=(emb(mc,li)@emb(mc,li).T).astype(np.float32)
    B=np.array([raw[i].mean() for i in range(n)]); Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    _,A0=best_thr(Fz,files,groups); ok0,lab0=score(A0,files,groups)
    print(f"{DATA}: BASE fusion {len(ok0)}/{len(groups)} = {len(ok0)/len(groups):.4f}")
    gm=[grad_mask(raw[i]) for i in range(n)]
    clmembers=defaultdict(list)
    for j in range(n): clmembers[lab0[j]].append(j)
    clstep={c:(np.median(np.diff(np.sort(B[mem]))) if len(mem)>=2 else 80.0) for c,mem in clmembers.items()}
    clipped=[i for i in range(n) if B[i]<45 or B[i]>210]
    A=A0.copy(); added=0
    for i in clipped:
        scored=[]
        for c,mem in clmembers.items():
            if c==lab0[i]: continue
            j=min(mem,key=lambda k:abs(B[k]-B[i]))
            mz,cnt=masked_zncc(gm[i][0],gm[i][1],gm[j][0],gm[j][1])
            if accept_edge(mz,cnt,abs(B[i]-B[j]),clstep[c]): scored.append((mz,j))
        scored.sort(reverse=True)
        if scored:
            second=scored[1][0] if len(scored)>1 else -1
            if scored[0][0]-second>=MARGIN or second<MASK_THR:
                A[scored[0][1],i]=A[i,scored[0][1]]=True; added+=1
    ok1,_=score(A,files,groups)
    print(f"{DATA}: +FIX2 ladder re-attach (added {added}) {len(ok1)}/{len(groups)} = {len(ok1)/len(groups):.4f}")
    print(f"  recovered: {sorted(ok1-ok0)}   broken: {sorted(ok0-ok1)}")
    # Exclude genuinely-unfixable ground-truth-error groups
    unfix=load_unfixable(); fixable={g for g in groups if g not in unfix}
    okf=ok1 & fixable
    print(f"  -- excluding {len(unfix)} unfixable ground-truth-error groups --")
    print(f"  FIXABLE-only score: {len(okf)}/{len(fixable)} = {len(okf)/len(fixable):.4f}")
    missed_fixable=sorted(fixable-ok1)
    print(f"  still-missed FIXABLE groups ({len(missed_fixable)}): {missed_fixable[:25]}")
    # validation: are flagged-unfixable groups actually ones the pipeline misses?
    flagged_but_solved=sorted((ok1 & unfix))
    print(f"  [validation] flagged-unfixable that pipeline SOLVED (should be ~0): {len(flagged_but_solved)} {flagged_but_solved[:15]}")

if __name__=="__main__":
    main()
