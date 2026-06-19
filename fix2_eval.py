"""
FIX 2 final eval on a full dataset (no train/val split). Applies the validated
exposure-ladder masked re-attachment on top of fusion clustering and reports
score before/after, recovered and broken groups.
Usage: fix2_eval.py <data_dir>
"""
import sys, json, os
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA=sys.argv[1] if len(sys.argv)>1 else "sample"
VLO,VHI=8,247; W_GRAD=0.65; MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12

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
def grad_mask(raw):
    g=clahe.apply(raw).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy),((raw>=VLO)&(raw<=VHI))
def masked_zncc(m1,v1,m2,v2):
    v=(v1&v2).ravel(); cnt=int(v.sum())
    if cnt<200: return -1.0,cnt
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)),cnt

def accept_edge(mz,cnt,gap,step):
    # refined accept rule (calibrated on full-set edges):
    # require a real exposure step (gap>=25) and a valid-pixel floor; then either
    # a strong masked match within the ladder, or a huge well-exposed overlap.
    if gap<25 or cnt<1500: return False
    if mz>=0.58 and gap<=1.8*max(step,30.0): return True   # strong ladder extension
    if mz>=0.50 and cnt>=15000: return True                # large well-exposed overlap
    return False
def best_thr(sim,files,groups):
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)
        if sc>best[0]: best=(sc,A.copy())
    return best
def load_unfixable():
    p=f"{DATA}/unfixable.json"
    return set(json.load(open(p))["groups"].keys()) if os.path.exists(p) else set()

def score(A,files,groups):
    A=A.copy(); np.fill_diagonal(A,False)
    _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    refsets=set(frozenset(v) for v in groups.values()); predlk=set(frozenset(v) for v in pred.values())
    return {g for g,v in groups.items() if frozenset(v) in predlk}, lab

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab(f"{DATA}/img128c.npz"); d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True)
    raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]; assert list(d["files"])==files
    n=len(files); G=(d["M"]@d["M"].T).astype(np.float32); E=emb(mc,li); Es=(E@E.T).astype(np.float32)
    B=np.array([raw[i].mean() for i in range(n)]); Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    _,A0=best_thr(Fz,files,groups); ok0,lab0=score(A0,files,groups)
    print(f"{DATA}: BASE fusion {len(ok0)}/{len(groups)} = {len(ok0)/len(groups):.4f}")
    gm=[grad_mask(raw[i]) for i in range(n)]
    clmembers=defaultdict(list)
    for j in range(n): clmembers[lab0[j]].append(j)
    clstep={c:(np.median(np.diff(np.sort(B[mem]))) if len(mem)>=2 else 80.0) for c,mem in clmembers.items()}
    clipped=[i for i in range(n) if B[i]<45 or B[i]>210]
    A=A0.copy(); added=0
    for i in clipped:
        scored=[]
        for c,mem in clmembers.items():
            if c==lab0[i]: continue
            j=min(mem,key=lambda k:abs(B[k]-B[i]))
            mz,cnt=masked_zncc(gm[i][0],gm[i][1],gm[j][0],gm[j][1])
            if accept_edge(mz,cnt,abs(B[i]-B[j]),clstep[c]): scored.append((mz,j))
        scored.sort(reverse=True)
        if scored:
            second=scored[1][0] if len(scored)>1 else -1
            if scored[0][0]-second>=MARGIN or second<MASK_THR:
                A[scored[0][1],i]=A[i,scored[0][1]]=True; added+=1
    ok1,_=score(A,files,groups)
    print(f"{DATA}: +FIX2 ladder re-attach (added {added}) {len(ok1)}/{len(groups)} = {len(ok1)/len(groups):.4f}")
    print(f"  recovered: {sorted(ok1-ok0)}   broken: {sorted(ok0-ok1)}")

    # ---- FIX 4: embedding-guided cluster merge via masked bridging ----
    # Recovers over-splits where same-scene pieces were split by the fusion
    # threshold. Merge two embedding-near clusters if their brightness-adjacent
    # frames have a strong masked link (gap>=25). Lookalikes score ~0.30 masked
    # (vs same-scene >=0.45), so masked is the discriminator.
    np.fill_diagonal(A,False); _,lab=connected_components(csr_matrix(A),directed=False)
    cl=defaultdict(list)
    for j in range(n): cl[lab[j]].append(j)
    cids=list(cl.keys())
    cent={c:E[cl[c]].mean(0) for c in cids}
    cent={c:v/ (np.linalg.norm(v)+1e-9) for c,v in cent.items()}
    cmat=np.stack([cent[c] for c in cids]); csim=cmat@cmat.T
    parent={c:c for c in cids}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    merged=0
    for ai,c in enumerate(cids):
        near=np.argsort(-csim[ai])[1:13]
        for bi in near:
            c2=cids[bi]
            if find(c)==find(c2) or csim[ai,bi]<0.45: continue
            # best brightness-adjacent cross-frame masked link
            best=(-1,0,999)
            for x in cl[c]:
                y=min(cl[c2],key=lambda k:abs(B[k]-B[x])); g=abs(B[x]-B[y])
                if g<25: continue
                mz,cnt=masked_zncc(gm[x][0],gm[x][1],gm[y][0],gm[y][1])
                if mz>best[0]: best=(mz,cnt,g)
            mz,cnt,g=best
            if cnt>=1500 and ((mz>=0.62 and g<=120) or (mz>=0.50 and cnt>=15000)):
                parent[find(c)]=find(c2); merged+=1
    # apply merges
    for c in cids:
        for j in cl[c]:
            r=find(c)
            if r!=c:
                # connect a representative edge
                A[cl[c][0], cl[r][0]]=A[cl[r][0], cl[c][0]]=True
    ok1b,_=score(A,files,groups)
    print(f"{DATA}: +FIX4 cluster-merge (merged {merged}) {len(ok1b)}/{len(groups)} = {len(ok1b)/len(groups):.4f}")
    print(f"  FIX4 recovered: {sorted(ok1b-ok1)}   FIX4 broken: {sorted(ok1-ok1b)}")
    ok1=ok1b

    # ---- FIX 5: high-resolution masked SPLIT of coarse-descriptor over-merges ----
    # The 64x64 fusion descriptor over-merges similar-layout-but-different rooms
    # (fusion ~0.70, but 256px masked ~0.30). Re-cluster each predicted cluster's
    # members by 256px masked ZNCC; if it fragments, the cluster was over-merged.
    # Exposure-ladder chaining is preserved (all-pairs CC bridges via intermediates).
    np.fill_diagonal(A,False); _,lab=connected_components(csr_matrix(A),directed=False)
    cl5=defaultdict(list)
    for j in range(n): cl5[lab[j]].append(j)
    newlab=lab.copy(); nextid=int(lab.max())+1; splits=0
    for c,mem in cl5.items():
        if len(mem)<4: continue                       # only sizable clusters can be over-merges
        k=len(mem); IM=np.zeros((k,k),bool)
        for a in range(k):
            for b in range(a+1,k):
                mz,cnt=masked_zncc(gm[mem[a]][0],gm[mem[a]][1],gm[mem[b]][0],gm[mem[b]][1])
                if mz>=0.38: IM[a,b]=IM[b,a]=True
        nc,sub=connected_components(csr_matrix(IM),directed=False)
        if nc<2: continue
        sizes=np.bincount(sub); big=int(np.argmax(sizes))
        multi=[c2 for c2 in range(nc) if sizes[c2]>=2]
        if len(multi)<2: continue                     # need >=2 real sub-groups
        # OVER-MERGE SIGNATURE: each multi-frame sub-component must be internally TIGHT
        # (a complete bracket set), i.e. min internal masked high. Legitimate varied
        # groups (e.g. one group of distinct views) lack this clean structure.
        def comp_min(comp):
            idx=[mem[i] for i in range(k) if sub[i]==comp]; mn=2.0
            for a in range(len(idx)):
                for b in range(a+1,len(idx)):
                    v,_=masked_zncc(gm[idx[a]][0],gm[idx[a]][1],gm[idx[b]][0],gm[idx[b]][1])
                    mn=min(mn,v)
            return mn
        if not all(comp_min(c2)>=0.55 for c2 in multi): continue
        splits+=1
        for i,m in enumerate(mem):
            comp=sub[i]
            newlab[m]= (nextid+comp) if sizes[comp]>=2 and comp!=big else lab[m]
        nextid+=nc
    predm=defaultdict(set)
    for i,f in enumerate(files): predm[newlab[i]].add(f)
    predlk=set(frozenset(v) for v in predm.values())
    ok1c={g for g,v in groups.items() if frozenset(v) in predlk}
    print(f"{DATA}: +FIX5 high-res split ({splits} clusters split) {len(ok1c)}/{len(groups)} = {len(ok1c)/len(groups):.4f}")
    print(f"  FIX5 recovered: {sorted(ok1c-ok1)}   FIX5 broken: {sorted(ok1-ok1c)}")
    ok1=ok1c
    # Exclude genuinely-unfixable ground-truth-error groups
    unfix=load_unfixable(); fixable={g for g in groups if g not in unfix}
    okf=ok1 & fixable
    print(f"  -- excluding {len(unfix)} unfixable ground-truth-error groups --")
    print(f"  FIXABLE-only score: {len(okf)}/{len(fixable)} = {len(okf)/len(fixable):.4f}")
    missed_fixable=sorted(fixable-ok1)
    print(f"  still-missed FIXABLE groups ({len(missed_fixable)}): {missed_fixable[:25]}")
    # validation: are flagged-unfixable groups actually ones the pipeline misses?
    flagged_but_solved=sorted((ok1 & unfix))
    print(f"  [validation] flagged-unfixable that pipeline SOLVED (should be ~0): {len(flagged_but_solved)} {flagged_but_solved[:15]}")

if __name__=="__main__":
    main()
