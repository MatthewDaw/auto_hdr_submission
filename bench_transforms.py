"""
Limited benchmark: is a training-free wavelet / eigenface descriptor actually
faster than the learned CNN embedding (the heavy learned piece)? And does it
stay accurate? Times per-image descriptor extraction (CPU — the browser target),
similarity, and exact-set grouping accuracy on the 500-sample.
"""
import json, time
from collections import defaultdict
import numpy as np, cv2, pywt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA

DATA="sample"
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)            # grayscale 256
RAW=raw["imgs"]; gid=raw["gid"]; files=list(raw["files"]); n=len(files)
colc=np.load(f"{DATA}/img128c.npz",allow_pickle=True)["imgs"]  # BGR 128
groups=defaultdict(set)
for f,g in zip(files,gid): groups[g].add(f)
refsets=set(frozenset(v) for v in groups.values())
unfix=set(json.load(open(f"{DATA}/unfixable.json")).get("groups",{}).keys()) if __import__("os").path.exists(f"{DATA}/unfixable.json") else set()
fixable={g for g in groups if g not in unfix}
clahe=cv2.createCLAHE(3.0,(8,8))

def score(sim):
    best=0
    for t in np.arange(0.1,0.98,0.02):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        pl=set(frozenset(v) for v in pred.values())
        ok={g for g,v in groups.items() if frozenset(v) in pl}
        best=max(best,len(ok&fixable)/len(fixable))
    return best
def l2(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

# ---------- descriptor extractors (per image, from cached grayscale/color) ----------
def f_gradient(i):           # current core
    g=clahe.apply(cv2.resize(RAW[i],(256,256))).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    z=cv2.resize(cv2.magnitude(gx,gy),(64,64)).ravel(); z-=z.mean(); return z
def f_clahe64(i):            # raw input for eigenface/PCA
    return clahe.apply(cv2.resize(RAW[i],(64,64))).astype(np.float32).ravel()
def f_wavelet(i):            # multi-scale detail bands (exposure-robust, no LL)
    g=clahe.apply(cv2.resize(RAW[i],(128,128))).astype(np.float32)
    co=pywt.wavedec2(g,'db2',level=3)
    parts=[]
    for lvl in (1,2,3):      # detail bands at levels 1-3 (skip approximation cA)
        cH,cV,cD=co[lvl]
        for b in (cH,cV,cD): parts.append(cv2.resize(np.abs(b),(16,16)).ravel())
    return np.concatenate(parts).astype(np.float32)

def time_extract(fn,reps=1):
    t0=time.time()
    out=np.stack([fn(i) for i in range(n)])
    return out, (time.time()-t0)/n*1000   # ms/img

print(f"{DATA}: {n} images, {len(fixable)} fixable groups  (CPU timing)\n")
print(f"{'method':<26}{'extract ms/img':>15}{'sim ms':>9}{'score':>9}")

# 1) gradient-ZNCC (current core)
M,te=time_extract(f_gradient); t0=time.time(); S=l2(M)@l2(M).T; ts=(time.time()-t0)*1000
print(f"{'gradient-ZNCC (core)':<26}{te:>15.2f}{ts:>9.1f}{score(S):>9.4f}")

# 2) CNN color embedding (current learned piece) — forced to CPU (browser target)
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
class Model(nn.Module):
    def __init__(s,d=128):
        super().__init__(); m=mobilenet_v3_small(weights=None)
        s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,d))
    def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)
torch.set_num_threads(max(1,__import__("os").cpu_count()))
mc=Model(); mc.load_state_dict(torch.load("embed2_best.pt",map_location="cpu")); mc.eval()
MEAN=torch.tensor([0.5]*3).view(1,3,1,1); STD=torch.tensor([0.25]*3).view(1,3,1,1)
def lab_in(i):
    L=cv2.cvtColor(colc[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0])
    x=torch.from_numpy(L).float()/255.0; return ((x.permute(2,0,1)-MEAN[0])/STD[0])
t0=time.time()
with torch.no_grad():
    E=mc(torch.stack([lab_in(i) for i in range(n)])).numpy()
te_cnn=(time.time()-t0)/n*1000
t0=time.time(); Se=E@E.T; ts=(time.time()-t0)*1000
print(f"{'CNN embedding (learned)':<26}{te_cnn:>15.2f}{ts:>9.1f}{score(Se):>9.4f}")

# 3) eigenface / PCA on CLAHE pixels
X,te_x=time_extract(f_clahe64)
t0=time.time(); pca=PCA(n_components=128,whiten=True).fit(X); tfit=time.time()-t0
t0=time.time(); P=pca.transform(X); te_p=(time.time()-t0)/n*1000
P=P[:,1:]  # drop 1st component (mostly residual lighting)
ts=time.time(); Sp=l2(P)@l2(P).T; ts=(time.time()-ts)*1000
print(f"{'eigenface/PCA':<26}{te_x+te_p:>15.2f}{ts:>9.1f}{score(Sp):>9.4f}   (PCA fit {tfit*1000:.0f}ms one-time)")

# 4) wavelet (detail bands), direct
W,te_w=time_extract(f_wavelet)
t0=time.time(); Sw=l2(W)@l2(W).T; ts=(time.time()-t0)*1000
print(f"{'wavelet (multi-scale)':<26}{te_w:>15.2f}{ts:>9.1f}{score(Sw):>9.4f}")

# 5) wavelet + PCA (eigen-wavelets)
t0=time.time(); pcaw=PCA(n_components=128,whiten=True).fit(W); tfitw=time.time()-t0
t0=time.time(); PW=pcaw.transform(W); te_pw=(time.time()-t0)/n*1000
PW=PW[:,1:]
t0=time.time(); Spw=l2(PW)@l2(PW).T; ts=(time.time()-t0)*1000
print(f"{'wavelet + PCA':<26}{te_w+te_pw:>15.2f}{ts:>9.1f}{score(Spw):>9.4f}   (PCA fit {tfitw*1000:.0f}ms one-time)")

# fusions with the core gradient signal
print("\nfused with gradient-ZNCC core (best weight):")
def best_fuse(Sother):
    return max(score(w*S+(1-w)*Sother) for w in np.arange(0,1.01,0.1))
print(f"  gradient + CNN embedding : {best_fuse(Se):.4f}")
print(f"  gradient + eigenface/PCA : {best_fuse(Sp):.4f}")
print(f"  gradient + wavelet       : {best_fuse(Sw):.4f}")
print(f"  gradient + wavelet+PCA   : {best_fuse(Spw):.4f}")
