"""
Realistic deployment pipeline (no oracle): fused similarity (gradient + color
embedding, FIXED weight) -> plateau-selected threshold -> connected components
-> orphan re-attach. Reports on large-val and 500-set for a few fixed weights to
pick one that generalizes.
"""
from collections import defaultdict
import numpy as np
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576,256), nn.ReLU(), nn.Linear(256,dim))
    def forward(self,x): return F.normalize(self.proj(self.backbone(x)),dim=1)

def load_lab(p):
    d=np.load(p,allow_pickle=True); bgr=d["imgs"]; clahe=cv2.createCLAHE(3.0,(8,8)); lab=np.empty_like(bgr)
    for i in range(len(bgr)):
        L=cv2.cvtColor(bgr[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0]); lab[i]=L
    return lab,d["gid"],list(d["files"])

@torch.no_grad()
def emb_color(model,lab,bs=256):
    model.eval(); M=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); S=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(lab),bs):
        x=torch.from_numpy(lab[i:i+bs]).float().to(DEV)/255.0; x=x.permute(0,3,1,2)
        out.append(model((x-M)/S).cpu().numpy())
    return np.concatenate(out)

def plateau_thr(sim, lo=0.35, hi=0.80):
    # proven selector.py logic: leading edge of the min-slope plateau of the
    # group-count curve, over a bounded range where count stays near-optimal.
    grid=np.arange(lo,hi,0.01); counts=[]
    for t in grid:
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False); counts.append(len(set(lab)))
    counts=np.array(counts,float)
    W=3; slope=np.full(len(grid),np.inf)
    for i in range(W,len(grid)-W): slope[i]=(counts[i+W]-counts[i-W])/(2*W)
    slope[counts <= 0.5*counts.max()]=np.inf
    fin=slope[np.isfinite(slope)]
    if not len(fin): return (lo+hi)/2
    cut=1.3*fin.min()+0.5
    return grid[int(np.where(np.isfinite(slope)&(slope<=cut))[0][0])]

def cluster(sim,thr):
    A=sim>=thr; np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False); return lab

def score(lab,files,groups):
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    refsets=set(frozenset(v) for v in groups.values())
    return len(refsets & set(frozenset(v) for v in pred.values()))/len(refsets)

def orphan(lab,C,B,n):
    def crows(i,allidx): return 1.0-np.unpackbits(np.bitwise_xor(C[i],C[allidx]).reshape(len(allidx),-1),axis=1).sum(1)/(C.shape[1]*8.0)
    cl=defaultdict(list)
    for i,l in enumerate(lab): cl[l].append(i)
    EX=(B<45)|(B>210); lab=lab.copy()
    for l,mem in list(cl.items()):
        if len(mem)>2 or not any(EX[i] for i in mem): continue
        for i in mem:
            others=[j for j in range(n) if lab[j]!=lab[i]]
            cs=crows(i,np.array(others)); o=np.argsort(cs)[::-1]; bj=others[o[0]]
            sj=next((others[k] for k in o[1:] if lab[others[k]]!=lab[bj]),None)
            sec=crows(i,np.array([sj]))[0] if sj is not None else 0
            if cs[o[0]]>=0.62 and cs[o[0]]-sec>=0.03 and abs(B[i]-B[bj])>=25: lab[i]=lab[bj]
    return lab

def run(label, G, E, C, B, gid, files):
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    n=len(files); SE=(E@E.T).astype(np.float32)
    print(f"\n[{label}] n={n}, groups={len(groups)}")
    for w in [1.0, 0.5, 0.4, 0.3, 0.2]:
        fused=(w*G+(1-w)*SE).astype(np.float32)
        thr=plateau_thr(fused); lab=cluster(fused,thr)
        s0=score(lab,files,groups); lab2=orphan(lab,C,B,n); s1=score(lab2,files,groups)
        print(f"  w_grad={w:.1f}: selector_thr={thr:.2f}  score={s0:.4f}  +orphan={s1:.4f}")

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    d=np.load("data/large/feat_cache.npz",allow_pickle=True); li,gid,files=load_lab("data/large/img128c.npz")
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid])
    G=(d["M"][va]@d["M"][va].T).astype(np.float32)
    run("large-val", G, emb_color(mc,li[va]), d["C"][va], d["B"][va], gid[va], [f for f,m in zip(files,va) if m])
    ds=np.load("sample/feat_cache.npz",allow_pickle=True); sli,sgid,sfiles=load_lab("sample/img128c.npz")
    Gs=(ds["M"]@ds["M"].T).astype(np.float32)
    run("500-set", Gs, emb_color(mc,sli), ds["C"], ds["B"], sgid, sfiles)

if __name__=="__main__":
    main()
