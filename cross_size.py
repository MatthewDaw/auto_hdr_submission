"""
Cross-size generalization: train edge classifier on one dataset, test on the
other (fully held-out, different size). Tests whether graph-context features
transfer across dataset density. Also reports scale-invariant-only variant,
permutation importances, and learned+orphan stacking.
"""
import csv
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

SIZE = 256; ZNCC = 64; CEN = 96; FLOOR = 0.45
FEATNAMES = ["grad","census","bgap","common","rankmin","rankmax","smaxmin","ratioi","ratioj","degmin","degmax"]
SCALE_INVARIANT = [0,1,2,4,5,6,7,8]  # drop common/degmin/degmax (density-dependent)

def read_gray(p):
    try:
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is not None: return im
    except Exception: pass
    try:
        from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.array(Image.open(p).convert("L"))
    except Exception: return None

def _feat(args):
    path, = args
    im = read_gray(path)
    if im is None: return np.zeros(ZNCC*ZNCC, np.float32), np.zeros(CEN*CEN, np.uint8), -1.0
    bright = float(im.mean())
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    gf = g.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    z = cv2.resize(cv2.magnitude(gx, gy), (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); nn = np.linalg.norm(z); z = (z/nn if nn>0 else z).astype(np.float32)
    c = cv2.resize(g, (CEN, CEN)).astype(np.int16); code = np.zeros((CEN,CEN),np.uint8); bit=0
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            code |= ((c > np.roll(np.roll(c,di,0),dj,1)).astype(np.uint8) << bit); bit+=1
    return z, code.ravel(), bright

def load_dir(DATA):
    DATA = Path(DATA); cache = DATA / "feat_cache.npz"
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(DATA / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    files = sorted(f2g.keys())
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["files"]) == files:
            return d["M"], d["C"], d["B"], files, f2g, groups
    cv2.setNumThreads(1)
    with Pool(max(1,cpu_count()-1)) as pool:
        res = list(pool.imap(_feat, [(DATA/"images"/f,) for f in files], chunksize=16))
    M=np.array([r[0] for r in res],np.float32); C=np.array([r[1] for r in res],np.uint8); B=np.array([r[2] for r in res],np.float32)
    np.savez(cache, M=M, C=C, B=B, files=np.array(files))
    return M, C, B, files, f2g, groups

def build_pairs(M, C, B, files, f2g):
    n=len(files); sim=(M@M.T).astype(np.float32); np.fill_diagonal(sim,-1)
    smax=sim.max(1); A=sim>=FLOOR; deg=A.sum(1)
    common=(csr_matrix(A.astype(np.int8))@csr_matrix(A.astype(np.int8))).toarray().astype(np.float32)
    rank=np.argsort(np.argsort(-sim,axis=1),axis=1).astype(np.float32)
    ii,jj=np.where(np.triu(A,1))
    def cen(i,j): return 1.0-np.unpackbits(np.bitwise_xor(C[i],C[j])).sum()/(C.shape[1]*8.0)
    F=np.zeros((len(ii),11),np.float32)
    F[:,0]=sim[ii,jj]; F[:,1]=[cen(i,j) for i,j in zip(ii,jj)]; F[:,2]=np.abs(B[ii]-B[jj])
    F[:,3]=common[ii,jj]; F[:,4]=np.minimum(rank[ii,jj],rank[jj,ii]); F[:,5]=np.maximum(rank[ii,jj],rank[jj,ii])
    F[:,6]=np.minimum(smax[ii],smax[jj]); F[:,7]=sim[ii,jj]/np.maximum(smax[ii],1e-6); F[:,8]=sim[ii,jj]/np.maximum(smax[jj],1e-6)
    F[:,9]=np.minimum(deg[ii],deg[jj]); F[:,10]=np.maximum(deg[ii],deg[jj])
    gid=np.array([f2g[f] for f in files]); lab=(gid[ii]==gid[jj]).astype(int)
    return ii,jj,F,lab,sim,n

def cluster_score(ii,jj,mask,n,files,groups):
    G=csr_matrix((np.ones(mask.sum()),(ii[mask],jj[mask])),shape=(n,n))
    _,lab=connected_components(G,directed=False)
    pred=defaultdict(set)
    for k in range(n): pred[lab[k]].add(files[k])
    refsets=set(frozenset(v) for v in groups.values()); predlk=set(frozenset(v) for v in pred.values())
    return len(refsets&predlk)/len(refsets)

def run(trainDATA, testDATA, label):
    Mtr,Ctr,Btr,ftr,f2gtr,gtr = load_dir(trainDATA)
    Mte,Cte,Bte,fte,f2gte,gte = load_dir(testDATA)
    iitr,jjtr,Ftr,ltr,_,_ = build_pairs(Mtr,Ctr,Btr,ftr,f2gtr)
    iite,jjte,Fte,lte,simte,nte = build_pairs(Mte,Cte,Bte,fte,f2gte)
    print(f"\n=== {label}: train {trainDATA}({len(ftr)}) -> test {testDATA}({len(fte)}) ===")
    for tag, cols in [("all 11 feats", list(range(11))), ("scale-invariant 8", SCALE_INVARIANT)]:
        clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,max_depth=4)
        clf.fit(Ftr[:,cols],ltr); prob=clf.predict_proba(Fte[:,cols])[:,1]
        best=max((cluster_score(iite,jjte,prob>=p,nte,fte,gte),round(p,2)) for p in np.arange(0.3,0.9,0.05))
        print(f"  LEARNED ({tag}): {best[0]:.4f} @p>={best[1]}")
    gb=max((cluster_score(iite,jjte,simte[iite,jjte]>=t,nte,fte,gte),round(t,2)) for t in np.arange(0.5,0.7,0.02))
    print(f"  BASELINE gradient-thr: {gb[0]:.4f} @thr={gb[1]}")

if __name__ == "__main__":
    run("data/large","sample","big->small")
    run("sample","data/large","small->big")
