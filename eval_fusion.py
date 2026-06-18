"""
Canonical fusion eval (fine grids) for any embedding checkpoint.
Usage: eval_fusion.py <ckpt> <gray|color>
Reports emb-only / grad-only / fusion-best on large-val (held-out groups) and 500-set.
"""
import sys
from collections import defaultdict
import numpy as np
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT, MODE = sys.argv[1], sys.argv[2]

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=1)

def load_gray(path):
    d = np.load(path, allow_pickle=True); return d["imgs"], d["gid"], list(d["files"])
def load_color_lab(path):
    d = np.load(path, allow_pickle=True); bgr = d["imgs"]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)); lab = np.empty_like(bgr)
    for i in range(len(bgr)):
        L = cv2.cvtColor(bgr[i], cv2.COLOR_BGR2Lab); L[:, :, 0] = clahe.apply(L[:, :, 0]); lab[i] = L
    return lab, d["gid"], list(d["files"])

MEAN = torch.tensor([0.5]*3).view(1,3,1,1); STD = torch.tensor([0.25]*3).view(1,3,1,1)
def to_input(batch, dev):
    if MODE == "gray":
        x = torch.from_numpy(batch).float().to(dev)/255.0
        x = x.unsqueeze(1).repeat(1,3,1,1)
        im_m = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(dev); im_s = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(dev)
        return (x-im_m)/im_s
    x = torch.from_numpy(batch).float().to(dev)/255.0; x = x.permute(0,3,1,2)
    return (x-MEAN.to(dev))/STD.to(dev)

@torch.no_grad()
def embed_all(model, arr, bs=256):
    model.eval(); out=[]
    for i in range(0,len(arr),bs): out.append(model(to_input(arr[i:i+bs],DEV)).cpu().numpy())
    return np.concatenate(out)

def fusion_eval(G, E, gid, files, label):
    groups = defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    def sc(sim):
        sim=sim.copy(); np.fill_diagonal(sim,-1); best=(0,0)
        for t in np.arange(0.15,0.97,0.01):
            A=sim>=t; np.fill_diagonal(A,False)
            _,lab=connected_components(csr_matrix(A),directed=False)
            pred=defaultdict(set)
            for i,f in enumerate(files): pred[lab[i]].add(f)
            s=len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)
            if s>best[0]: best=(s,round(t,2))
        return best
    sg=sc(G); se=sc(E); fb=(0,None)
    for w in np.arange(0,1.001,0.05):
        s=sc(w*G+(1-w)*E)
        if s[0]>fb[0]: fb=(s[0], f"w_grad={w:.2f}@thr={s[1]}")
    print(f"  [{label}] grad={sg[0]:.4f} emb={se[0]:.4f} FUSION={fb[0]:.4f} ({fb[1]})")

def main():
    model = Model().to(DEV); model.load_state_dict(torch.load(CKPT)); print(f"ckpt={CKPT} mode={MODE}")
    loader = load_gray if MODE=="gray" else load_color_lab
    cpath = "data/large/img128.npz" if MODE=="gray" else "data/large/img128c.npz"
    arr, gid, files = loader(cpath)
    d = np.load("data/large/feat_cache.npz", allow_pickle=True); assert list(d["files"])==files
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid])
    E=embed_all(model,arr[va]); G=(d["M"][va]@d["M"][va].T).astype(np.float32)
    fusion_eval(G,(E@E.T).astype(np.float32),gid[va],[f for f,m in zip(files,va) if m],"large-val")
    # 500-set
    spath = "sample/img128.npz" if MODE=="gray" else "sample/img128c.npz"
    sarr,sgid,sfiles=loader(spath); ds=np.load("sample/feat_cache.npz",allow_pickle=True)
    Es=embed_all(model,sarr); Gs=(ds["M"]@ds["M"].T).astype(np.float32)
    fusion_eval(Gs,(Es@Es.T).astype(np.float32),sgid,sfiles,"500-set")

if __name__=="__main__":
    main()
