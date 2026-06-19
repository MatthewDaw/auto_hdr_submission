"""
Characterize the still-missed FIXABLE groups on a dataset (full pipeline =
fusion + FIX2 ladder re-attach, excluding ground-truth-error groups).
For each missed fixable group: over-split vs over-merge, size, drone, brightness
range, intra-group min fusion/masked sims, and (for merges) the contaminating
groups + bridge sim. Reveals the dominant remaining failure mode.
Usage: analyze_missed.py <data_dir>
"""
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA=Path(sys.argv[1] if len(sys.argv)>1 else "data/large")
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
def grad_mask(r):
    g=clahe.apply(r).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((r>=VLO)&(r<=VHI))
def mzncc(m1,v1,m2,v2):
    v=(v1&v2).ravel()
    if v.mean()<MIN_VALID: return -1.0
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def main():
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]; n=len(files)
    G=(d["M"]@d["M"].T).astype(np.float32); mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    E=emb(mc,li); Es=(E@E.T).astype(np.float32); B=np.array([raw[i].mean() for i in range(n)])
    Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    idxof={f:i for i,f in enumerate(files)}
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=Fz>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))
        if sc>best[0]: best=(sc,A.copy(),t)
    A=best[1]; thr=best[2]; _,lab0=connected_components(csr_matrix(A),directed=False)
    # FIX2
    clm=defaultdict(list)
    for j in range(n): clm[lab0[j]].append(j)
    step={c:(np.median(np.diff(np.sort(B[m]))) if len(m)>=2 else 80.0) for c,m in clm.items()}
    gm=[grad_mask(raw[i]) for i in range(n)]
    def mzc(m1,v1,m2,v2):
        v=(v1&v2).ravel(); cnt=int(v.sum())
        if cnt<200: return -1.0,cnt
        a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)),cnt
    def accept(mz,cnt,gap,st):
        if gap<25 or cnt<1500: return False
        if mz>=0.58 and gap<=1.8*max(st,30.0): return True
        if mz>=0.50 and cnt>=15000: return True
        return False
    for i in [k for k in range(n) if B[k]<45 or B[k]>210]:
        scc=[]
        for c,m in clm.items():
            if c==lab0[i]: continue
            j=min(m,key=lambda k:abs(B[k]-B[i]))
            mz,cnt=mzc(*gm[i],*gm[j])
            if accept(mz,cnt,abs(B[i]-B[j]),step[c]): scc.append((mz,j))
        scc.sort(reverse=True)
        if scc and (scc[0][0]-(scc[1][0] if len(scc)>1 else -1)>=MARGIN or (scc[1][0] if len(scc)>1 else -1)<MASK_THR):
            A[i,scc[0][1]]=A[scc[0][1],i]=True
    np.fill_diagonal(A,False); _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    predlk=set(frozenset(v) for v in pred.values()); f2p={f:lab[idxof[f]] for f in files}
    okset={g for g,v in groups.items() if frozenset(v) in predlk}
    unfix=set(json.load(open(f"{DATA}/unfixable.json"))["groups"].keys())
    missed=[g for g in groups if g not in okset and g not in unfix]
    print(f"{DATA}: thr={thr:.2f} | solved {len(okset)}/{len(groups)} | {len(unfix)} ground-truth errors | {len(missed)} missed FIXABLE\n")
    nmerge=nsplit=0; rows=[]
    for g in missed:
        mem=groups[g]; ix=[idxof[f] for f in mem]; drone=any("DJI" in f for f in mem)
        clset=set(f2p[f] for f in mem); contam=set()
        for c in clset: contam|=({files[i] for i in range(n) if lab[i]==c}-mem)
        sub=Fz[np.ix_(ix,ix)]; iu=np.triu_indices(len(ix),1)
        fmin=sub[iu].min() if len(ix)>1 else float('nan')
        if contam:
            nmerge+=1; cix=[idxof[f] for f in contam]
            bridge=Fz[np.ix_(ix,cix)].max()
            cg=sorted(set(gid[idxof[f]] for f in contam))[:4]
            rows.append(("MERGE",g,len(mem),drone,f"{B[ix].min():.0f}-{B[ix].max():.0f}",f"bridge={bridge:.2f}",f"with {cg}"))
        else:
            nsplit+=1; mmin=min((mzncc(*gm[a],*gm[b]) for a in ix for b in ix if a<b),default=float('nan'))
            rows.append(("SPLIT",g,len(mem),drone,f"{B[ix].min():.0f}-{B[ix].max():.0f}",f"fusMin={fmin:.2f}",f"maskMin={mmin:.2f} pieces={len(clset)}"))
    print(f"OVER-SPLIT={nsplit}  OVER-MERGE={nmerge}\n")
    print(f"{'kind':>6} {'group':>8} {'sz':>3} {'drn':>4} {'bright':>9} {'sim':>12} {'detail':>26}")
    for r in sorted(rows,key=lambda x:(x[0],-x[2])):
        print(f"{r[0]:>6} {r[1]:>8} {r[2]:>3} {str(r[3])[0]:>4} {r[4]:>9} {r[5]:>12} {r[6]:>26}")

if __name__=="__main__":
    main()
