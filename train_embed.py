"""
Contrastive per-image embedding for exposure-invariant scene grouping.

- Input: 128x128 CLAHE grayscale (exposure pre-normalized), replicated to 3ch.
- Backbone: MobileNetV3-small (pretrained) -> 128-d L2-normalized projection.
- Loss: supervised contrastive (SupCon), P-groups x K-images batches. Real
  bracket members (different exposures) are positives -> learns exposure
  invariance from real data; different groups (incl. lookalike rooms) are
  negatives -> learns sensitivity to small scene changes.
- Augmentation: strong photometric (gamma/brightness/contrast) for exposure
  robustness; only MILD geometric (we must stay sensitive to scene changes).
- Eval: embed -> cosine -> plateau-selected threshold -> connected components
  -> exact-set score, on held-out groups AND cross-size, vs gradient baseline.
"""
import sys, time
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 128

# ---------------- data ----------------
def load(path):
    d = np.load(path, allow_pickle=True)
    return d["imgs"], d["gid"], list(d["files"])

class Model(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=1)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_input(batch_u8, train, dev):
    # batch_u8: (B,H,W) uint8 -> (B,3,H,W) normalized float, optional augmentation
    x = torch.from_numpy(batch_u8).float().to(dev) / 255.0          # (B,H,W)
    if train:
        B = x.shape[0]
        gamma = torch.empty(B,1,1, device=dev).uniform_(0.45, 2.2)  # exposure/gamma jitter
        x = x.clamp(1e-4, 1).pow(gamma)
        bright = torch.empty(B,1,1, device=dev).uniform_(0.7, 1.3)
        x = (x * bright).clamp(0,1)
        mean = x.mean(dim=(1,2), keepdim=True)
        contrast = torch.empty(B,1,1, device=dev).uniform_(0.8, 1.2)
        x = ((x - mean) * contrast + mean).clamp(0,1)
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])   # h-flip ok (mirror angle still 1 scene state)
    x = x.unsqueeze(1).repeat(1,3,1,1)                              # 3ch
    x = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
    return x

def supcon(emb, labels, temp=0.1):
    # emb: (B,d) normalized ; labels: (B,)
    sim = emb @ emb.t() / temp
    B = emb.shape[0]
    self_mask = torch.eye(B, device=emb.device).bool()
    sim.masked_fill_(self_mask, -1e9)
    pos = (labels[:,None] == labels[None,:]) & ~self_mask
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    loss = -(logp * pos).sum(1)[valid] / pos_cnt[valid]
    return loss.mean()

# ---------------- PK sampler ----------------
def pk_batches(gid, P=24, K=4, steps=40, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, g in enumerate(gid): by[g].append(i)
    multi = [g for g, v in by.items() if len(v) >= 2]
    for _ in range(steps):
        gs = rng.choice(multi, size=min(P, len(multi)), replace=False)
        idx = []
        for g in gs:
            v = by[g]
            idx += list(rng.choice(v, size=min(K, len(v)), replace=len(v) < K))
        yield np.array(idx)

# ---------------- eval ----------------
def plateau_thr(sim, lo=0.30, hi=0.95):
    n = sim.shape[0]
    grid = np.arange(lo, hi, 0.01)
    counts = []
    for t in grid:
        A = sim >= t; np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        counts.append(len(set(lab)))
    counts = np.array(counts, float); W = 3; slope = np.full(len(grid), np.inf)
    for i in range(W, len(grid)-W): slope[i] = (counts[i+W]-counts[i-W])/(2*W)
    slope[counts <= 0.5*counts.max()] = np.inf
    cut = 1.3*slope[np.isfinite(slope)].min() + 0.5
    return grid[int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])]

def exact_score(emb, gid, files):
    sim = (emb @ emb.T).astype(np.float32); np.fill_diagonal(sim, -1)
    thr = plateau_thr(sim)
    A = sim >= thr; np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    groups = defaultdict(set)
    for f, g in zip(files, gid): groups[g].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), thr

@torch.no_grad()
def embed_all(model, imgs, bs=256):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = to_input(imgs[i:i+bs], False, DEV)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)

def main():
    imgs, gid, files = load("data/large/img128.npz")
    # disjoint group split
    uniq = sorted(set(gid)); rng = np.random.default_rng(0)
    rng.shuffle(uniq); val_g = set(uniq[:int(0.18*len(uniq))])
    tr = np.array([g not in val_g for g in gid]); va = ~tr
    tr_imgs, tr_gid = imgs[tr], gid[tr]
    va_imgs, va_gid, va_files = imgs[va], gid[va], [f for f, m in zip(files, va) if m]
    print(f"device={DEV} train={tr.sum()} val={va.sum()} ({len(val_g)} val groups)")

    # baseline (gradient-ZNCC) on the same val split, for reference
    sci_imgs = va_imgs  # gradient baseline computed elsewhere; here embedding only
    model = Model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = 0; t0 = time.time()
    for epoch in range(1, 41):
        model.train(); losses = []
        for idx in pk_batches(tr_gid, P=24, K=4, steps=60, seed=epoch):
            x = to_input(tr_imgs[idx], True, DEV)
            lab = torch.from_numpy(tr_gid[idx].astype(np.int64)).to(DEV)
            # remap labels to contiguous ints
            _, lab = torch.unique(lab, return_inverse=True)
            emb = model(x); loss = supcon(emb, lab)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if epoch % 4 == 0 or epoch == 1:
            emb = embed_all(model, va_imgs)
            sc, thr = exact_score(emb, va_gid, va_files)
            tag = ""
            if sc > best: best = sc; torch.save(model.state_dict(), "embed_best.pt"); tag = " *saved"
            print(f"ep{epoch:2d} loss={np.mean(losses):.3f} val_exact={sc:.4f} @thr={thr:.2f} ({time.time()-t0:.0f}s){tag}")
    print(f"BEST val exact-set = {best:.4f}")

    # cross-size check on the 500-set (fully held out)
    s_imgs, s_gid, s_files = load("sample/img128.npz")
    model.load_state_dict(torch.load("embed_best.pt"))
    emb = embed_all(model, s_imgs); sc, thr = exact_score(emb, s_gid, s_files)
    print(f"CROSS-SIZE (500-set) exact-set = {sc:.4f} @thr={thr:.2f}")

if __name__ == "__main__":
    main()
