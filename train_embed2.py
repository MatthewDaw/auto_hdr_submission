"""
Embedding v2 — optimized for the FUSION objective.
- Input: Lab (CLAHE'd L + a + b chroma). Chroma discriminates rooms & is
  exposure-stable; CLAHE'd L carries structure. Exposure aug jitters only L.
- Cosine LR, 60 epochs, temp 0.07.
- Eval tracks FUSION score (w*grad + (1-w)*embed) on held-out groups; saves best
  by fusion. Reports cross-size fusion on the 500-set.
Usage: train_embed2.py [res] [epochs]
"""
import sys, time
from collections import defaultdict
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60

def load_lab(path):
    d = np.load(path, allow_pickle=True)
    bgr = d["imgs"]; gid = d["gid"]; files = list(d["files"])
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = np.empty_like(bgr)
    for i in range(len(bgr)):
        L = cv2.cvtColor(bgr[i], cv2.COLOR_BGR2Lab)
        L[:, :, 0] = clahe.apply(L[:, :, 0])
        lab[i] = L
    return lab, gid, files  # (N,H,W,3) uint8 Lab, L is CLAHE'd

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=1)

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)

def to_input(batch_lab, train, dev):
    x = torch.from_numpy(batch_lab).float().to(dev) / 255.0   # (B,H,W,3) Lab in [0,1]
    x = x.permute(0, 3, 1, 2)                                   # (B,3,H,W)
    if train:
        B = x.shape[0]
        L = x[:, 0:1]
        gamma = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.45, 2.2)
        L = L.clamp(1e-4, 1).pow(gamma)
        bright = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
        L = (L * bright).clamp(0, 1)
        x = torch.cat([L, x[:, 1:]], dim=1)
        # small chroma jitter
        x[:, 1:] = (x[:, 1:] + torch.empty(B, 2, 1, 1, device=dev).uniform_(-0.03, 0.03)).clamp(0, 1)
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
    return (x - MEAN.to(dev)) / STD.to(dev)

def supcon(emb, labels, temp=0.07):
    sim = emb @ emb.t() / temp
    B = emb.shape[0]; eye = torch.eye(B, device=emb.device).bool()
    sim.masked_fill_(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pc = pos.sum(1); v = pc > 0
    return (-(logp * pos).sum(1)[v] / pc[v]).mean()

def pk_batches(gid, P=24, K=4, steps=60, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, g in enumerate(gid): by[g].append(i)
    multi = [g for g, v in by.items() if len(v) >= 2]
    for _ in range(steps):
        gs = rng.choice(multi, size=min(P, len(multi)), replace=False); idx = []
        for g in gs:
            v = by[g]; idx += list(rng.choice(v, size=min(K, len(v)), replace=len(v) < K))
        yield np.array(idx)

@torch.no_grad()
def embed_all(model, lab, bs=256):
    model.eval(); out = []
    for i in range(0, len(lab), bs):
        out.append(model(to_input(lab[i:i+bs], False, DEV)).cpu().numpy())
    return np.concatenate(out)

def oracle_fusion(G, E, gid, files, ws=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0)):
    groups = defaultdict(set)
    for f, g in zip(files, gid): groups[g].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    def sc_of(sim):
        sim = sim.copy(); np.fill_diagonal(sim, -1); best = 0
        for thr in np.arange(0.2, 0.97, 0.02):
            A = sim >= thr; np.fill_diagonal(A, False)
            _, lab = connected_components(csr_matrix(A), directed=False)
            pred = defaultdict(set)
            for i, f in enumerate(files): pred[lab[i]].add(f)
            best = max(best, len(refsets & set(frozenset(v) for v in pred.values()))/len(refsets))
        return best
    res = {}
    for w in ws: res[w] = sc_of(w*G + (1-w)*E)
    return res

def main():
    lab, gid, files = load_lab("data/large/img128c.npz")
    d = np.load("data/large/feat_cache.npz", allow_pickle=True)
    assert list(d["files"]) == files
    Gfull = d["M"]
    uniq = sorted(set(gid)); rng = np.random.default_rng(0); rng.shuffle(uniq)
    val_g = set(uniq[:int(0.18*len(uniq))])
    tr = np.array([g not in val_g for g in gid]); va = ~tr
    tr_lab, tr_gid = lab[tr], gid[tr]
    va_lab, va_gid, va_files = lab[va], gid[va], [f for f, m in zip(files, va) if m]
    Gva = (Gfull[va] @ Gfull[va].T).astype(np.float32)
    print(f"device={DEV} train={tr.sum()} val={va.sum()} ({len(val_g)} groups) epochs={EPOCHS}")

    model = Model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = 0; t0 = time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); losses = []
        for idx in pk_batches(tr_gid, steps=60, seed=ep):
            x = to_input(tr_lab[idx], True, DEV)
            _, lb = torch.unique(torch.from_numpy(tr_gid[idx].astype(np.int64)).to(DEV), return_inverse=True)
            emb = model(x); loss = supcon(emb, lb)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        if ep % 5 == 0 or ep == 1:
            E = embed_all(model, va_lab); Esim = (E @ E.T).astype(np.float32)
            res = oracle_fusion(Gva, Esim, va_gid, va_files)
            fbest = max(res.values()); wbest = max(res, key=res.get)
            tag = ""
            if fbest > best: best = fbest; torch.save(model.state_dict(), "embed2_best.pt"); tag = " *"
            print(f"ep{ep:2d} loss={np.mean(losses):.3f} emb={res[0.0]:.4f} fusion={fbest:.4f}@w={wbest} ({time.time()-t0:.0f}s){tag}")
    print(f"BEST val fusion = {best:.4f}")

    # cross-size on 500-set
    s_lab, s_gid, s_files = load_lab("sample/img128c.npz")
    ds = np.load("sample/feat_cache.npz", allow_pickle=True)
    Gs = (ds["M"] @ ds["M"].T).astype(np.float32)
    model.load_state_dict(torch.load("embed2_best.pt"))
    Es = embed_all(model, s_lab); Essim = (Es @ Es.T).astype(np.float32)
    res = oracle_fusion(Gs, Essim, s_gid, s_files)
    print(f"CROSS-SIZE 500: emb={res[0.0]:.4f} grad={res[1.0]:.4f} fusion={max(res.values()):.4f}@w={max(res,key=res.get)}")

if __name__ == "__main__":
    main()
