"""
Large-set validation: does a training-free wavelet+eigenface descriptor match
the CNN embedding through the FULL pipeline (fusion + FIX 2/4/5)?
Reports base-fusion fixable score for every descriptor + the full pipeline for
the combined wavelet+PCA. CPU descriptor timing included.
"""
import csv, json, time
from collections import defaultdict
import numpy as np, cv2, pywt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA

DATA="data/large"; VLO,VHI=8,247; W_GRAD=0.65; MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=fc["M"]; files=list(fc["files"]); n=len(files)
RAW=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
col=np.load(f"{DATA}/img128c.npz",allow_pickle=True)["imgs"]
gid={}; g2=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8")): gid[r["filename"]]=r["group_id"]; g2[r["group_id"]].append(r["filename"])
groups=defaultdict(set)
for f in files: groups[gid[f]].add(f)
refsets=set(frozenset(v) for v in groups.values())
unfix=set(json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"].keys())
fixable={g for g in groups if g not in unfix}
B=RAW.reshape(n,-1).mean(1)
clahe=cv2.createCLAHE(3.0,(8,8))
def l2(X): return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)

# ---- descriptors ----
def d_eigen():
    X=np.stack([clahe.apply(cv2.resize(RAW[i],(64,64))).astype(np.float32).ravel() for i in range(n)])
    P=PCA(128,whiten=True).fit_transform(X)[:,1:]; return l2(P)
def d_wavelet():
    out=np.zeros((n,9*256),np.float32)
    for i in range(n):
        g=clahe.apply(cv2.resize(RAW[i],(128,128))).astype(np.float32); co=pywt.wavedec2(g,'db2',level=3)
        parts=[cv2.resize(np.abs(b),(16,16)).ravel() for lvl in (1,2,3) for b in co[lvl]]
        out[i]=np.concatenate(parts)
    return l2(out)
def d_wavpca(W): return l2(PCA(128,whiten=True).fit_transform(W)[:,1:])
def d_cnn():
    import torch,torch.nn as nn,torch.nn.functional as F
    from torchvision.models import mobilenet_v3_small
    class Mdl(nn.Module):
        def __init__(s,d=128):
            super().__init__(); m=mobilenet_v3_small(weights=None)
            s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,d))
        def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    mc=Mdl().to(dev); mc.load_state_dict(torch.load("embed2_best.pt",map_location=dev)); mc.eval()
    MN=torch.tensor([0.5]*3).view(1,3,1,1).to(dev); SD=torch.tensor([0.25]*3).view(1,3,1,1).to(dev); out=[]
    with torch.no_grad():
        for s in range(0,n,256):
            xb=[]
            for i in range(s,min(s+256,n)):
                L=cv2.cvtColor(col[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0]); xb.append(torch.from_numpy(L).float().permute(2,0,1)/255.0)
            x=(torch.stack(xb).to(dev)-MN)/SD; out.append(mc(x).cpu().numpy())
    return np.concatenate(out)

G=(M@M.T).astype(np.float32)
def fixscore(sim,oracle=True):
    best=0
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        pl=set(frozenset(v) for v in pred.values()); ok={g for g,v in groups.items() if frozenset(v) in pl}
        best=max(best,len(ok&fixable)/len(fixable))
    return best

print(f"{DATA}: {n} imgs, {len(fixable)} fixable groups\n")
print("BASE-FUSION fixable score (gradient + descriptor, oracle threshold):")
t=time.time(); Ecnn=d_cnn(); print(f"  gradient + CNN embedding : {fixscore((W_GRAD*G+(1-W_GRAD)*(Ecnn@Ecnn.T)).astype(np.float32)):.4f}   (extract {time.time()-t:.0f}s)")
t=time.time(); Eei=d_eigen();  print(f"  gradient + eigenface/PCA : {fixscore((W_GRAD*G+(1-W_GRAD)*(Eei@Eei.T)).astype(np.float32)):.4f}   (extract {time.time()-t:.0f}s)")
t=time.time(); W=d_wavelet(); twav=time.time()-t
print(f"  gradient + wavelet       : {fixscore((W_GRAD*G+(1-W_GRAD)*(W@W.T)).astype(np.float32)):.4f}   (extract {twav:.0f}s)")
t=time.time(); Ewp=d_wavpca(W); print(f"  gradient + wavelet+PCA   : {fixscore((W_GRAD*G+(1-W_GRAD)*(Ewp@Ewp.T)).astype(np.float32)):.4f}   (extract {twav+time.time()-t:.0f}s)")

# ---- FULL pipeline with the combined wavelet+eigenface (wavelet+PCA) as E ----
def masked_zncc(a,b):
    g1=clahe.apply(RAW[a]).astype(np.float32); g2a=clahe.apply(RAW[b]).astype(np.float32)
    return None  # placeholder (real gm precomputed below)
print("\nFULL pipeline (fusion + FIX2/4/5) with wavelet+PCA in place of CNN:")
E=Ewp; Es=(E@E.T).astype(np.float32); Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
def gm(i):
    g=clahe.apply(RAW[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((RAW[i]>=VLO)&(RAW[i]<=VHI))
GM=[None]*n
def gmc(i):
    if GM[i] is None: GM[i]=gm(i)
    return GM[i]
def mz(i,j):
    a,va=gmc(i); b,vb=gmc(j); v=(va&vb).ravel(); cnt=int(v.sum())
    if cnt<200: return -1.0,cnt
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9)),cnt
def accept(mzv,cnt,gap,step):
    if gap<25 or cnt<1500: return False
    if mzv>=0.58 and gap<=1.8*max(step,30): return True
    if mzv>=0.50 and cnt>=15000: return True
    return False
def cc(A): A=A.copy(); np.fill_diagonal(A,False); _,l=connected_components(csr_matrix(A),directed=False); return l
def sc_of(lab):
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    pl=set(frozenset(v) for v in pred.values()); ok={g for g,v in groups.items() if frozenset(v) in pl}
    return ok
# oracle threshold base
best=(-1,None)
for t in np.arange(0.15,0.97,0.01):
    lab=cc(Fz>=t); ok=sc_of(lab)
    if len(ok&fixable)>best[0]: best=(len(ok&fixable),(Fz>=t).copy())
A=best[1]; lab0=cc(A)
clm=defaultdict(list)
for j in range(n): clm[lab0[j]].append(j)
clstep={c:(np.median(np.diff(np.sort(B[m]))) if len(m)>=2 else 80.0) for c,m in clm.items()}
# FIX2
for i in [k for k in range(n) if B[k]<45 or B[k]>210]:
    sc=[]
    for c,m in clm.items():
        if c==lab0[i]: continue
        j=min(m,key=lambda k:abs(B[k]-B[i])); mzv,cnt=mz(i,j)
        if accept(mzv,cnt,abs(B[i]-B[j]),clstep[c]): sc.append((mzv,j))
    sc.sort(reverse=True)
    if sc and (sc[0][0]-(sc[1][0] if len(sc)>1 else -1)>=MARGIN or (sc[1][0] if len(sc)>1 else -1)<MASK_THR): A[sc[0][1],i]=A[i,sc[0][1]]=True
lab=cc(A)
# FIX4 (embedding-near uses E=wavelet+PCA)
cl=defaultdict(list)
for j in range(n): cl[lab[j]].append(j)
cids=list(cl); cent=l2(np.stack([E[cl[c]].mean(0) for c in cids])); csim=cent@cent.T
par={c:c for c in cids}
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for ai,c in enumerate(cids):
    for bi in np.argsort(-csim[ai])[1:13]:
        c2=cids[bi]
        if find(c)==find(c2) or csim[ai,bi]<0.45: continue
        bst=(-1,0,999)
        for x in cl[c]:
            y=min(cl[c2],key=lambda k:abs(B[k]-B[x])); g=abs(B[x]-B[y])
            if g<25: continue
            mzv,cnt=mz(x,y)
            if mzv>bst[0]: bst=(mzv,cnt,g)
        if bst[1]>=1500 and ((bst[0]>=0.62 and bst[2]<=120) or (bst[0]>=0.50 and bst[1]>=15000)): par[find(c)]=find(c2)
for c in cids:
    if find(c)!=c: A[cl[c][0],cl[find(c)][0]]=A[cl[find(c)][0],cl[c][0]]=True
lab=cc(A)
# FIX5
cl5=defaultdict(list)
for j in range(n): cl5[lab[j]].append(j)
newlab=lab.copy(); nx=int(lab.max())+1
for c,mem in cl5.items():
    if len(mem)<3: continue
    k=len(mem); IM=np.zeros((k,k),bool); MX=np.full(k,-2.0)
    for a in range(k):
        for b in range(a+1,k):
            mzv,_=mz(mem[a],mem[b]); MX[a]=max(MX[a],mzv); MX[b]=max(MX[b],mzv)
            if mzv>=0.38: IM[a,b]=IM[b,a]=True
    nc,sub=connected_components(csr_matrix(IM),directed=False); sizes=np.bincount(sub,minlength=nc); big=int(np.argmax(sizes))
    for i,m in enumerate(mem):
        if sizes[sub[i]]==1 and 55<=B[m]<=200 and MX[i]<0.32: newlab[m]=nx; nx+=1
    multi=[c2 for c2 in range(nc) if sizes[c2]>=2]
    if len(multi)>=2:
        def cmin(comp):
            idx=[mem[i] for i in range(k) if sub[i]==comp]; mn=2.0
            for a in range(len(idx)):
                for b in range(a+1,len(idx)): mn=min(mn,mz(idx[a],idx[b])[0])
            return mn
        if all(cmin(c2)>=0.55 for c2 in multi):
            for i,m in enumerate(mem):
                if sizes[sub[i]]>=2 and sub[i]!=big: newlab[m]=nx+sub[i]
            nx+=nc
pred=defaultdict(set)
for i,f in enumerate(files): pred[newlab[i]].add(f)
pl=set(frozenset(v) for v in pred.values()); ok={g for g,v in groups.items() if frozenset(v) in pl}
print(f"  wavelet+PCA FULL pipeline fixable-only: {len(ok&fixable)}/{len(fixable)} = {len(ok&fixable)/len(fixable):.4f}")
print(f"  (CNN full-pipeline reference: 1303/1313 = 0.9924)")
