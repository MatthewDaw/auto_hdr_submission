"""
Test smarter fusion EDGE RULES vs linear blend. Hypothesis: linear averaging
lets a dead gradient veto a confident embedding on extreme-exposure pairs.
An OR-style rule (merge if blend high OR embedding confidently high) should
recover those without breaking precision.
"""
from collections import defaultdict
import numpy as np
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(s,dim=128):
        super().__init__(); m=mobilenet_v3_small(weights=None)
        s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,dim))
    def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)

def load_lab(p):
    d=np.load(p,allow_pickle=True); bgr=d["imgs"]; clahe=cv2.createCLAHE(3.0,(8,8)); lab=np.empty_like(bgr)
    for i in range(len(bgr)):
        L=cv2.cvtColor(bgr[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0]); lab[i]=L
    return lab,d["gid"],list(d["files"])

@torch.no_grad()
def emb(model,lab,bs=256):
    model.eval(); M=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); S=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(lab),bs):
        x=torch.from_numpy(lab[i:i+bs]).float().to(DEV)/255.0; x=x.permute(0,3,1,2); out.append(model((x-M)/S).cpu().numpy())
    return np.concatenate(out)

def score_edges(edge, files, refsets):
    A=edge.copy(); np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    return len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)

def evalset(G,E,gid,files,label):
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values())
    # R1 linear
    r1=(0,None)
    for w in np.arange(0,1.01,0.1):
        for t in np.arange(0.2,0.95,0.01):
            s=score_edges(w*G+(1-w)*E>=t,files,refsets)
            if s>r1[0]: r1=(s,f"w={w:.1f},t={t:.2f}")
    # R2 OR: blend>=t  OR  E>=te (embedding-confident escape hatch)
    r2=(0,None)
    for w in [0.4,0.5,0.65]:
        for t in np.arange(0.45,0.85,0.02):
            for te in np.arange(0.70,0.95,0.02):
                s=score_edges(((w*G+(1-w)*E)>=t)|(E>=te),files,refsets)
                if s>r2[0]: r2=(s,f"w={w:.2f},t={t:.2f},te={te:.2f}")
    # R3 OR both single-signal: blend>=t OR E>=te OR G>=tg
    r3=(0,None)
    for t in np.arange(0.5,0.8,0.02):
        for te in np.arange(0.74,0.92,0.02):
            for tg in np.arange(0.6,0.85,0.02):
                s=score_edges(((0.5*G+0.5*E)>=t)|(E>=te)|(G>=tg),files,refsets)
                if s>r3[0]: r3=(s,f"t={t:.2f},te={te:.2f},tg={tg:.2f}")
    print(f"[{label}] R1 linear={r1[0]:.4f} ({r1[1]}) | R2 blend-OR-embConf={r2[0]:.4f} ({r2[1]}) | R3 blend-OR-G-OR-E={r3[0]:.4f} ({r3[1]})")

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab("data/large/img128c.npz"); d=np.load("data/large/feat_cache.npz",allow_pickle=True)
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid])
    evalset((d["M"][va]@d["M"][va].T).astype(np.float32),(emb(mc,li[va])@emb(mc,li[va]).T).astype(np.float32),
            gid[va],[f for f,m in zip(files,va) if m],"large-val")
    sli,sgid,sfiles=load_lab("sample/img128c.npz"); ds=np.load("sample/feat_cache.npz",allow_pickle=True)
    evalset((ds["M"]@ds["M"].T).astype(np.float32),(emb(mc,sli)@emb(mc,sli).T).astype(np.float32),sgid,sfiles,"500-set")

if __name__=="__main__":
    main()
