"""
Embedding v3 — color (Lab) + HARD-NEGATIVE MINING for the fusion objective.
Each batch = a seed group + its nearest groups in gradient-descriptor centroid
space (lookalike rooms), K images each. Forces SupCon to separate exactly the
confusable rooms where gradient-ZNCC over-merges. Cosine LR, temp 0.07.
Usage: train_embed3.py [epochs]
"""
import sys, time
from collections import defaultdict
import numpy as np
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 70

def load_lab(path):
    d = np.load(path, allow_pickle=True); bgr = d["imgs"]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); lab = np.empty_like(bgr)
    for i in range(len(bgr)):
        L = cv2.cvtColor(bgr[i], cv2.COLOR_BGR2Lab); L[:, :, 0] = clahe.apply(L[:, :, 0]); lab[i] = L
    return lab, d["gid"], list(d["files"])

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x): return F.normalize(self.proj(self.backbone(x)), dim=1)

MEAN = torch.tensor([0.5]*3).view(1,3,1,1); STD = torch.tensor([0.25]*3).view(1,3,1,1)
def to_input(batch, train, dev):
    x = torch.from_numpy(batch).float().to(dev)/255.0; x = x.permute(0,3,1,2)
    if train:
        B=x.shape[0]; L=x[:,0:1]
        L=L.clamp(1e-4,1).pow(torch.empty(B,1,1,1,device=dev).uniform_(0.45,2.2))
        L=(L*torch.empty(B,1,1,1,device=dev).uniform_(0.7,1.3)).clamp(0,1)
        x=torch.cat([L,x[:,1:]],1)
        x[:,1:]=(x[:,1:]+torch.empty(B,2,1,1,device=dev).uniform_(-0.03,0.03)).clamp(0,1)
        if torch.rand(1).item()<0.5: x=torch.flip(x,dims=[3])
    return (x-MEAN.to(dev))/STD.to(dev)

def supcon(emb, labels, temp=0.07):
    sim=emb@emb.t()/temp; B=emb.shape[0]; eye=torch.eye(B,device=emb.device).bool()
    sim.masked_fill_(eye,-1e9); pos=(labels[:,None]==labels[None,:])&~eye
    logp=sim-torch.logsumexp(sim,1,keepdim=True); pc=pos.sum(1); v=pc>0
    return (-(logp*pos).sum(1)[v]/pc[v]).mean()

def hard_batches(gid, Gdesc, P=24, K=4, steps=60, seed=0):
    rng=np.random.default_rng(seed)
    by=defaultdict(list)
    for i,g in enumerate(gid): by[g].append(i)
    glist=[g for g,v in by.items() if len(v)>=2]
    # group centroids in gradient-descriptor space
    cent=np.stack([Gdesc[by[g]].mean(0) for g in glist])
    cent/= (np.linalg.norm(cent,axis=1,keepdims=True)+1e-9)
    gsim=cent@cent.T; np.fill_diagonal(gsim,-1)
    order=np.argsort(-gsim,axis=1)  # nearest groups first
    for _ in range(steps):
        seed_i=rng.integers(len(glist))
        chosen=[seed_i]+list(order[seed_i,:P-1])   # seed + nearest confusable groups
        idx=[]
        for gi in chosen:
            v=by[glist[gi]]; idx+=list(rng.choice(v,size=min(K,len(v)),replace=len(v)<K))
        yield np.array(idx)

@torch.no_grad()
def embed_all(model, lab, bs=256):
    model.eval(); out=[]
    for i in range(0,len(lab),bs): out.append(model(to_input(lab[i:i+bs],False,DEV)).cpu().numpy())
    return np.concatenate(out)

def fusion_best(G,E,gid,files):
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values())
    def sc(sim):
        sim=sim.copy(); np.fill_diagonal(sim,-1); best=0
        for t in np.arange(0.15,0.97,0.01):
            A=sim>=t; np.fill_diagonal(A,False)
            _,lab=connected_components(csr_matrix(A),directed=False)
            pred=defaultdict(set)
            for i,f in enumerate(files): pred[lab[i]].add(f)
            best=max(best,len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets))
        return best
    fb=(0,0)
    for w in np.arange(0,1.001,0.05):
        s=sc(w*G+(1-w)*E)
        if s>fb[0]: fb=(s,round(w,2))
    return sc(E), sc(G), fb

def main():
    lab,gid,files=load_lab("data/large/img128c.npz")
    d=np.load("data/large/feat_cache.npz",allow_pickle=True); assert list(d["files"])==files
    M=d["M"]
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    tr=np.array([g not in val_g for g in gid]); va=~tr
    tr_lab,tr_gid,tr_M=lab[tr],gid[tr],M[tr]
    va_lab,va_gid,va_files=lab[va],gid[va],[f for f,m in zip(files,va) if m]
    Gva=(M[va]@M[va].T).astype(np.float32)
    print(f"device={DEV} train={tr.sum()} val={va.sum()} epochs={EPOCHS} (hard-neg mining)")
    model=Model().to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); losses=[]
        for idx in hard_batches(tr_gid,tr_M,steps=60,seed=ep):
            x=to_input(tr_lab[idx],True,DEV)
            _,lb=torch.unique(torch.from_numpy(tr_gid[idx].astype(np.int64)).to(DEV),return_inverse=True)
            loss=supcon(model(x),lb); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        if ep%5==0 or ep==1:
            E=embed_all(model,va_lab); se,sg,fb=fusion_best(Gva,(E@E.T).astype(np.float32),va_gid,va_files)
            tag=""
            if fb[0]>best: best=fb[0]; torch.save(model.state_dict(),"embed3_best.pt"); tag=" *"
            print(f"ep{ep:2d} loss={np.mean(losses):.3f} emb={se:.4f} fusion={fb[0]:.4f}@w={fb[1]} ({time.time()-t0:.0f}s){tag}")
    print(f"BEST val fusion = {best:.4f}")
    s_lab,s_gid,s_files=load_lab("sample/img128c.npz"); ds=np.load("sample/feat_cache.npz",allow_pickle=True)
    Gs=(ds["M"]@ds["M"].T).astype(np.float32); model.load_state_dict(torch.load("embed3_best.pt"))
    Es=embed_all(model,s_lab); se,sg,fb=fusion_best(Gs,(Es@Es.T).astype(np.float32),s_gid,s_files)
    print(f"CROSS-SIZE 500: emb={se:.4f} grad={sg:.4f} fusion={fb[0]:.4f}@w={fb[1]}")

if __name__=="__main__":
    main()
