"""
Dissect fusion failures on large-val (oracle threshold, best weight).
For each failing reference group, report kind (split/merge), size, drone, and
crucially whether gradient-ONLY or embedding-ONLY would have gotten it right
(at their own oracle thresholds) -- isolates which signal is the weak link.
"""
from collections import defaultdict
import numpy as np
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W_GRAD = 0.65

class Model(nn.Module):
    def __init__(s,dim=128):
        super().__init__(); m=mobilenet_v3_small(weights=None)
        s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,dim))
    def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)

def load_lab(p):
    d=np.load(p,allow_pickle=True); bgr=d["imgs"]; clahe=cv2.createCLAHE(3.0,(8,8)); lab=np.empty_like(bgr)
    for i in range(len(bgr)):
        L=cv2.cvtColor(bgr[i],cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0]); lab[i]=L
    return lab,d["gid"],list(d["files"])

@torch.no_grad()
def emb(model,lab,bs=256):
    model.eval(); M=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); S=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV); out=[]
    for i in range(0,len(lab),bs):
        x=torch.from_numpy(lab[i:i+bs]).float().to(DEV)/255.0; x=x.permute(0,3,1,2); out.append(model((x-M)/S).cpu().numpy())
    return np.concatenate(out)

def clusters_at_best(sim, files, groups):
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=sim>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))/len(refsets)
        if sc>best[0]: best=(sc,lab.copy())
    return best[1]

def correct_groups(lab, files, groups):
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    predlk=set(frozenset(v) for v in pred.values())
    return {g for g,v in groups.items() if frozenset(v) in predlk}, pred, {f:lab[i] for i,f in enumerate(files)}

def main():
    mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt"))
    li,gid,files=load_lab("data/large/img128c.npz"); d=np.load("data/large/feat_cache.npz",allow_pickle=True)
    assert list(d["files"])==files
    uniq=sorted(set(gid)); rng=np.random.default_rng(0); rng.shuffle(uniq); val_g=set(uniq[:int(0.18*len(uniq))])
    va=np.array([g in val_g for g in gid]); vf=[f for f,m in zip(files,va) if m]
    G=(d["M"][va]@d["M"][va].T).astype(np.float32); E=(emb(mc,li[va])@emb(mc,li[va]).T).astype(np.float32)
    B=d["B"][va]; vg=gid[va]; idxof={f:i for i,f in enumerate(vf)}
    groups=defaultdict(set)
    for f,g in zip(vf,vg): groups[g].add(f)
    Fz=(W_GRAD*G+(1-W_GRAD)*E).astype(np.float32)

    okF,predF,f2pF = correct_groups(clusters_at_best(Fz,vf,groups),vf,groups)
    okG,_,_ = correct_groups(clusters_at_best(G,vf,groups),vf,groups)
    okE,_,_ = correct_groups(clusters_at_best(E,vf,groups),vf,groups)
    fails=[g for g in groups if g not in okF]
    print(f"large-val: {len(groups)} groups, fusion correct={len(okF)} ({len(okF)/len(groups):.4f}), {len(fails)} fail")
    print(f"  grad-only correct={len(okG)}  emb-only correct={len(okE)}")
    print(f"  fails fixable by grad-only:{len(set(fails)&okG)}  by emb-only:{len(set(fails)&okE)}  by NEITHER:{len(set(fails)-okG-okE)}\n")
    print(f"{'group':>8} {'sz':>3} {'drn':>4} {'kind':>6} {'gOK':>4} {'eOK':>4} {'brightrange':>12} {'gMin':>6} {'eMin':>6}")
    for g in sorted(fails, key=lambda g:-len(groups[g])):
        mem=groups[g]; ix=[idxof[f] for f in mem]; drone=any("DJI" in f for f in mem)
        clset=set(f2pF[f] for f in mem); contam=any((predF[c]-mem) for c in clset)
        kind="MERGE" if contam else "SPLIT"
        gmin=G[np.ix_(ix,ix)][np.triu_indices(len(ix),1)].min() if len(ix)>1 else float('nan')
        emin=E[np.ix_(ix,ix)][np.triu_indices(len(ix),1)].min() if len(ix)>1 else float('nan')
        br=f"{B[ix].min():.0f}-{B[ix].max():.0f}"
        print(f"{g:>8} {len(mem):>3} {str(drone)[0]:>4} {kind:>6} {('Y' if g in okG else '-'):>4} {('Y' if g in okE else '-'):>4} {br:>12} {gmin:>6.2f} {emin:>6.2f}")

if __name__=="__main__":
    main()
