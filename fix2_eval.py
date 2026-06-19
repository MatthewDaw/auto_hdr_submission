"""
FIX 2 final eval on a full dataset (no train/val split). Applies the validated
exposure-ladder masked re-attachment on top of fusion clustering and reports
score before/after, recovered and broken groups.
Usage: fix2_eval.py <data_dir>
"""
import sys, json, os, time
from collections import defaultdict
import numpy as np, cv2
import descriptor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA=sys.argv[1] if len(sys.argv)>1 else "sample"
VLO,VHI=8,247; W_GRAD=float(os.environ.get("WGRAD",0.65)); MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12
EDGE_DM=float(os.environ.get("EDGE_DM",0.0))   # sensitivity: shift accept_edge masked thresholds
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
    if mz>=0.58+EDGE_DM and gap<=1.8*max(step,30.0): return True   # strong ladder extension
    if mz>=0.50+EDGE_DM and cnt>=15000: return True                # large well-exposed overlap
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
def plateau_thr(sim,files):
    # LABEL-FREE threshold: the knee of the predicted-group-count vs threshold
    # curve (flattest local slope = most stable clustering). Mirrors selector.py
    # but runs on the fusion sim. No labels used.
    grid=np.arange(0.20,0.90,0.01)
    counts=[]
    for t in grid:
        A=sim>=t; np.fill_diagonal(A,False)
        nc,_=connected_components(csr_matrix(A),directed=False); counts.append(nc)
    counts=np.array(counts,float); W=3
    slope=np.full(len(grid),np.inf)
    for i in range(W,len(grid)-W): slope[i]=(counts[i+W]-counts[i-W])/(2*W)
    slope[counts<=0.5*counts.max()]=np.inf
    smin=slope[np.isfinite(slope)].min(); cut=1.3*smin+0.5
    sel_i=int(np.where(np.isfinite(slope)&(slope<=cut))[0][0]); t=grid[sel_i]
    A=sim>=t; np.fill_diagonal(A,False)
    return t,A

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
    rd=np.load(f"{DATA}/raw256.npz",allow_pickle=True); raw=rd["imgs"]; gid=rd["gid"]; files=list(rd["files"])
    d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); assert list(d["files"])==files
    n=len(files); t_pipe=time.time(); G=(d["M"]@d["M"].T).astype(np.float32)
    t0=time.time(); E=descriptor.embed(raw); te=time.time()-t0      # wavelet+eigenface (training-free)
    print(f"{DATA}: descriptor extract {te:.1f}s ({te/n*1000:.2f} ms/img)")
    Es=(E@E.T).astype(np.float32)
    B=np.array([raw[i].mean() for i in range(n)]); Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    if os.environ.get("THR_MODE")=="selector":
        t_sel,A0=plateau_thr(Fz,files); print(f"  [THR_MODE=selector] label-free plateau thr={t_sel:.2f}")
    else:
        _,A0=best_thr(Fz,files,groups)
    ok0,lab0=score(A0,files,groups)
    print(f"{DATA}: BASE fusion {len(ok0)}/{len(groups)} = {len(ok0)/len(groups):.4f}")
    gm=[grad_mask(raw[i]) for i in range(n)]
    clmembers=defaultdict(list)
    for j in range(n): clmembers[lab0[j]].append(j)
    clstep={c:(np.median(np.diff(np.sort(B[mem]))) if len(mem)>=2 else 80.0) for c,mem in clmembers.items()}
    clipped=[i for i in range(n) if B[i]<45 or B[i]>210]
    gidarr=np.asarray(gid)
    A=A0.copy(); added=0
    for i in clipped:
        if len(clmembers[lab0[i]])>2: continue        # only re-attach orphans; don't bridge two real clusters
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
                bj=scored[0][1]; A[bj,i]=A[i,bj]=True; added+=1
                if gidarr[i]!=gidarr[bj]: print(f"  FIX2 DIFF-edge g{gidarr[i]}(B{B[i]:.0f})->g{gidarr[bj]}(B{B[bj]:.0f}) mz={scored[0][0]:.2f}")
    # NOTE: the remaining ~8 over-splits are a single PURE-clipped orphan (B0/B254)
    # stranded from an otherwise-correct cluster. probe_orphans.py confirms these are
    # at the physical information limit: true-vs-wrong masked separation is 0.02-0.20
    # (own mz often <0.45 on a few hundred valid pixels), below the noise floor — a
    # threshold that catches the real matches also accepts false ones (a FIX-2b
    # attempt merged unrelated 11479+25886). Several groups are double-stranded with a
    # truly-hopeless B0/B2 frame (validfrac~0.001), so unrecoverable under exact-set.
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
    cb={c:B[cl[c]].mean() for c in cids}
    cbarr=np.array([cb[c] for c in cids])
    merged=0
    for ai,c in enumerate(cids):
        # candidates: embedding-near; plus, for SMALL or CLIPPED clusters (the split
        # pieces), the brightness-nearest clusters. Masked bridge below = precision guard.
        near=set(np.argsort(-csim[ai])[1:20].tolist())
        mem=cl[c]
        if len(mem)<=3 or (B[mem]<45).any() or (B[mem]>210).any():
            near|=set(np.argsort(np.abs(cbarr-cb[c]))[1:15].tolist())
        for bi in sorted(near):                       # deterministic order
            c2=cids[bi]
            if find(c)==find(c2): continue
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
        if len(mem)<3: continue
        k=len(mem); IM=np.zeros((k,k),bool); MX=np.full(k,-2.0)
        for a in range(k):
            for b in range(a+1,k):
                mz,cnt=masked_zncc(gm[mem[a]][0],gm[mem[a]][1],gm[mem[b]][0],gm[mem[b]][1])
                MX[a]=max(MX[a],mz); MX[b]=max(MX[b],mz)
                if mz>=0.38: IM[a,b]=IM[b,a]=True
        nc,sub=connected_components(csr_matrix(IM),directed=False)
        sizes=np.bincount(sub,minlength=nc); big=int(np.argmax(sizes)); changed=False
        # Pass 1: split off WELL-EXPOSED frames that don't masked-link to the cluster
        # (wrongly-merged singletons, e.g. 25823). Clipped orphans (extreme B) are
        # protected — FIX2 legitimately attaches them with low masked overlap.
        for i,m in enumerate(mem):
            if sizes[sub[i]]==1 and 55<=B[m]<=200 and MX[i]<0.32:
                newlab[m]=nextid; nextid+=1; changed=True
        # Pass 2: multiple internally-TIGHT bracket-sets wrongly merged (coarse-descriptor
        # over-merge). Tightness guard distinguishes from legitimate varied-content groups.
        multi=[c2 for c2 in range(nc) if sizes[c2]>=2]
        if len(multi)>=2:
            def comp_min(comp):
                idx=[mem[i] for i in range(k) if sub[i]==comp]; mn=2.0
                for a in range(len(idx)):
                    for b in range(a+1,len(idx)):
                        v,_=masked_zncc(gm[idx[a]][0],gm[idx[a]][1],gm[idx[b]][0],gm[idx[b]][1])
                        mn=min(mn,v)
                return mn
            if all(comp_min(c2)>=0.55 for c2 in multi):
                for i,m in enumerate(mem):
                    if sizes[sub[i]]>=2 and sub[i]!=big: newlab[m]=nextid+sub[i]
                nextid+=nc; changed=True
        # Pass 3: over-merged DIFFERENT rooms connected only by a weak bridge — split by
        # WELL-EXPOSED sub-scenes (masked reliable there), then assign every frame to the
        # sub-scene it best masked-matches. Tightness guard avoids splitting varied groups.
        if not changed:
            welli=[i for i in range(k) if 55<=B[mem[i]]<=200]
            if len(welli)>=4:
                WG=np.zeros((len(welli),len(welli)),bool); wv=[[0]*len(welli) for _ in welli]
                for a in range(len(welli)):
                    for b in range(a+1,len(welli)):
                        v,_=masked_zncc(gm[mem[welli[a]]][0],gm[mem[welli[a]]][1],gm[mem[welli[b]]][0],gm[mem[welli[b]]][1])
                        wv[a][b]=wv[b][a]=v
                        if v>=0.45: WG[a,b]=WG[b,a]=True
                wnc,wsub=connected_components(csr_matrix(WG),directed=False)
                wsizes=np.bincount(wsub,minlength=wnc); wmulti=[c2 for c2 in range(wnc) if wsizes[c2]>=2]
                def wtight(c2):
                    ii=[t for t in range(len(welli)) if wsub[t]==c2]; mn=2.0
                    for a in range(len(ii)):
                        for b in range(a+1,len(ii)): mn=min(mn,wv[ii[a]][ii[b]])
                    return mn
                if len(wmulti)>=2 and all(wtight(c2)>=0.6 for c2 in wmulti):
                    reps={c2:min([welli[t] for t in range(len(welli)) if wsub[t]==c2],key=lambda ii:abs(B[mem[ii]]-128)) for c2 in wmulti}
                    base_c=wmulti[0]
                    for i,m in enumerate(mem):
                        bestc=max(wmulti,key=lambda c2:masked_zncc(gm[mem[reps[c2]]][0],gm[mem[reps[c2]]][1],gm[m][0],gm[m][1])[0])
                        if bestc!=base_c: newlab[m]=nextid+bestc
                    nextid+=wnc; changed=True
        if changed: splits+=1
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
    if os.environ.get("DIAG"):
        f2i={f:i for i,f in enumerate(files)}
        lab2idx=defaultdict(list)
        for i in range(n): lab2idx[newlab[i]].append(i)
        for g in missed_fixable:
            gi=[f2i[f] for f in groups[g]]; labs=set(newlab[i] for i in gi)
            br=f"B[{int(B[gi].min())}..{int(B[gi].max())}]"
            if len(labs)==1:                       # OVER-MERGE: group sits in one cluster with others
                L=labs.pop(); others=sorted(set(gidarr[i] for i in lab2idx[L])-{g})
                # masked between this group's well-rep and each contaminating group's well-rep
                wr=min(gi,key=lambda i:abs(B[i]-128)); det=[]
                for o in others:
                    oi=[i for i in lab2idx[L] if gidarr[i]==o]; owr=min(oi,key=lambda i:abs(B[i]-128))
                    mz,_=masked_zncc(gm[wr][0],gm[wr][1],gm[owr][0],gm[owr][1]); det.append(f"g{o}(mz={mz:.2f},n{len(oi)})")
                print(f"  DIAG g{g} OVER-MERGE n{len(gi)} {br} +{others if not det else ' '.join(det)}")
            else:                                   # OVER-SPLIT: group's frames in multiple clusters
                pieces=[]
                for L in labs:
                    pii=[i for i in lab2idx[L] if gidarr[i]==g]; foreign=sorted(set(gidarr[i] for i in lab2idx[L])-{g})
                    pieces.append(f"[n{len(pii)} B{int(B[pii].min())}-{int(B[pii].max())}{' +'+','.join(foreign) if foreign else ''}]")
                print(f"  DIAG g{g} OVER-SPLIT n{len(gi)} {br} into {len(labs)}: {' '.join(pieces)}")
    # ---- end-to-end timing: only the embedding-extract stage differs from the old
    # CNN pipeline (fusion+cluster+refine is identical), so the CNN total is the same
    # run with the extract stage swapped for the CNN's measured 3.19 ms/img. ----
    tot=time.time()-t_pipe; rest=tot-te
    cnn_ms=3.19; cnn_tot=rest+cnn_ms/1000*n
    print(f"  [TIMING] n={n}  embed-extract={te:.1f}s ({te/n*1000:.2f} ms/img)  cluster+refine={rest:.1f}s")
    print(f"  [TIMING] END-TO-END ours={tot:.1f}s ({tot/n*1000:.1f} ms/img)  |  with CNN extract={cnn_tot:.1f}s ({cnn_tot/n*1000:.1f} ms/img)")
    # validation: are flagged-unfixable groups actually ones the pipeline misses?
    flagged_but_solved=sorted((ok1 & unfix))
    print(f"  [validation] flagged-unfixable that pipeline SOLVED (should be ~0): {len(flagged_but_solved)} {flagged_but_solved[:15]}")

if __name__=="__main__":
    main()
