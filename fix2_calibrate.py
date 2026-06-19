"""
Instrument every edge FIX 2 adds on the full large set: masked value, valid-pixel
fraction & absolute count, brightness gap, ladder-step ratio, and SAME/DIFF ref
group. Reveals how to separate good re-attachments from spurious near-clip matches.
"""
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA="data/large"; VLO,VHI=8,247; W_GRAD=0.65

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
def masked(m1,v1,m2,v2):
    v=(v1&v2).ravel(); cnt=int(v.sum())
    if cnt<200: return -1.0,cnt
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)),cnt

def main():
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]; n=len(files)
    G=(d["M"]@d["M"].T).astype(np.float32); mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    Es=(emb(mc,li)@emb(mc,li).T).astype(np.float32); B=np.array([raw[i].mean() for i in range(n)])
    Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.55,0.80,0.01):
        A=Fz>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))
        if sc>best[0]: best=(sc,A.copy())
    A=best[1]; _,lab0=connected_components(csr_matrix(A),directed=False)
    clm=defaultdict(list)
    for j in range(n): clm[lab0[j]].append(j)
    step={c:(np.median(np.diff(np.sort(B[m]))) if len(m)>=2 else 80.0) for c,m in clm.items()}
    gm=[grad_mask(raw[i]) for i in range(n)]
    print(f"{'mz':>5} {'cnt':>6} {'frac':>5} {'gap':>4} {'step':>5} {'ratio':>5} {'same':>5}")
    good=bad=0
    for i in [k for k in range(n) if B[k]<45 or B[k]>210]:
        cand=[]
        for c,m in clm.items():
            if c==lab0[i]: continue
            j=min(m,key=lambda k:abs(B[k]-B[i]))
            mz,cnt=masked(*gm[i],*gm[j]); cand.append((mz,j,cnt))
        cand.sort(reverse=True)
        if not cand: continue
        mz,j,cnt=cand[0]; second=cand[1][0] if len(cand)>1 else -1
        # current accept rule (pre-guard) to list candidates worth inspecting
        if mz>=0.50:
            gap=abs(B[i]-B[j]); st=max(step[lab0[j]],30.0); same=gid[i]==gid[j]
            print(f"{mz:>5.2f} {cnt:>6} {cnt/65536:>5.2f} {gap:>4.0f} {st:>5.0f} {gap/st:>5.1f} {('YES' if same else 'no'):>5}")
            if same: good+=1
            else: bad+=1
    print(f"\ncandidate edges with masked>=0.50: {good} SAME-ref (good), {bad} DIFF-ref (bad)")

if __name__=="__main__":
    main()
