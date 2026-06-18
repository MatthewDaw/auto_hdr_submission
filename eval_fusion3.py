"""
3-way fusion: gradient-ZNCC + gray-embedding + color-embedding.
Tests whether the two embeddings are complementary on top of gradient.
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
        self.proj = nn.Sequential(nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x): return F.normalize(self.proj(self.backbone(x)), dim=1)

def load_gray(p):
    d = np.load(p, allow_pickle=True); return d["imgs"], d["gid"], list(d["files"])
def load_lab(p):
    d = np.load(p, allow_pickle=True); bgr = d["imgs"]; clahe = cv2.createCLAHE(3.0, (8,8)); lab=np.empty_like(bgr)
    for i in range(len(bgr)):
        L=cv2.cvtColor(bgr[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0]); lab[i]=L
    return lab, d["gid"], list(d["files"])

@torch.no_grad()
def emb_gray(model, imgs, bs=256):
    model.eval(); im_m=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEV); im_s=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(imgs),bs):
        x=torch.from_numpy(imgs[i:i+bs]).float().to(DEV)/255.0; x=x.unsqueeze(1).repeat(1,3,1,1)
        out.append(model((x-im_m)/im_s).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def emb_color(model, lab, bs=256):
    model.eval(); M=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); S=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(lab),bs):
        x=torch.from_numpy(lab[i:i+bs]).float().to(DEV)/255.0; x=x.permute(0,3,1,2)
        out.append(model((x-M)/S).cpu().numpy())
    return np.concatenate(out)

def best_score(sim, refsets, files):
    sim=sim.copy(); np.fill_diagonal(sim,-1); best=0
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        best=max(best,len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets))
    return best

def evalset(G, Eg, Ec, gid, files, label):
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values())
    SG=(Eg@Eg.T).astype(np.float32); SC=(Ec@Ec.T).astype(np.float32)
    base=max(best_score(G,refsets,files), 0)
    fb=(0,None)
    for wg in np.arange(0,1.01,0.1):
        for we in np.arange(0,1.01-wg+1e-9,0.1):
            wc=1-wg-we
            if wc<-1e-9: continue
            s=best_score(wg*G+we*SG+wc*SC, refsets, files)
            if s>fb[0]: fb=(s, f"grad={wg:.1f},gray={we:.1f},color={wc:.1f}")
    print(f"  [{label}] grad-only={base:.4f}  3-way FUSION={fb[0]:.4f} ({fb[1]})")

def main():
    mg=Model().to(DEV); mg.load_state_dict(torch.load("embed_best.pt"))
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    # large-val
    gi,gid,files=load_gray("data/large/img128.npz"); li,_,_=load_lab("data/large/img128c.npz")
    d=np.load("data/large/feat_cache.npz",allow_pickle=True)
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid])
    G=(d["M"][va]@d["M"][va].T).astype(np.float32)
    evalset(G, emb_gray(mg,gi[va]), emb_color(mc,li[va]), gid[va], [f for f,m in zip(files,va) if m], "large-val")
    # 500
    sgi,sgid,sfiles=load_gray("sample/img128.npz"); sli,_,_=load_lab("sample/img128c.npz")
    ds=np.load("sample/feat_cache.npz",allow_pickle=True); Gs=(ds["M"]@ds["M"].T).astype(np.float32)
    evalset(Gs, emb_gray(mg,sgi), emb_color(mc,sli), sgid, sfiles, "500-set")

if __name__=="__main__":
    main()
