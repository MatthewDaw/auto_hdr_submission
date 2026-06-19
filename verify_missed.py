"""
Re-inspect the remaining missed-FIXABLE groups on large. For each, decode images
FRESH (utf-8 safe) and determine whether it's genuinely an algorithm failure or
actually a ground-truth error the coherence detector missed.

- OVER-SPLIT groups: are the members ONE coherent scene (algorithm under-connects
  -> fixable) or multiple visually-different scenes (reference error)?
- OVER-MERGE clusters (groups the pipeline lumped together): are the merged groups
  genuinely DIFFERENT scenes (algorithm over-merge -> fixable by discrimination)
  or the SAME scene the reference over-split (reference error)?

Writes verify_missed.html gallery + a classification summary.
"""
import csv, base64
from collections import defaultdict
import numpy as np, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA="data/large"; VLO,VHI=8,247
# the still-missed fixable groups, grouped by how the pipeline merged them
OVER_SPLIT=["11533","22901","35098","43807","48484","56453","56460","70877","86016","30093"]
OVER_MERGE_CLUSTERS=[["31580","31579"],["4986","33301","56922","63126","90309"],
                     ["86290","25823"],["17105","40599"]]

g2f=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv", encoding="utf-8")):
    g2f[r["group_id"]].append(r["filename"])

def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: buf=fh.read()
    return cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR)
clahe=cv2.createCLAHE(3.0,(8,8))
def gm(bgr):
    g=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(256,256),interpolation=cv2.INTER_AREA)
    gc=clahe.apply(g).astype(np.float32)
    gx=cv2.Sobel(gc,cv2.CV_32F,1,0,3); gy=cv2.Sobel(gc,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((g>=VLO)&(g<=VHI)),float(g.mean())
def mz(a,b):
    m1,v1,_=a; m2,v2,_=b; v=(v1&v2).ravel()
    if v.sum()<300: return -1.0
    x=m1.ravel()[v]-m1.ravel()[v].mean(); y=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))

cache={}
def feats(f):
    if f not in cache: cache[f]=gm(readbgr(f))
    return cache[f]
def well_rep(fs):
    cand=[(abs(feats(f)[2]-128),f) for f in fs if 55<=feats(f)[2]<=200]
    return min(cand)[1] if cand else None
def subscenes(fs):
    k=len(fs); A=np.zeros((k,k),bool)
    for a in range(k):
        for b in range(a+1,k):
            if mz(feats(fs[a]),feats(fs[b]))>=0.45: A[a,b]=A[b,a]=True
    nc,lab=connected_components(csr_matrix(A),directed=False); return nc,lab
def thumb(f):
    bgr=cv2.resize(readbgr(f),(80,80)); ok,b=cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,72])
    return base64.b64encode(b).decode()

rows=[]; cls=defaultdict(list)
print("=== OVER-SPLIT groups: is the group ONE coherent scene? ===")
for g in OVER_SPLIT:
    fs=g2f[g]; nc,lab=subscenes(fs)
    # are the sub-scenes the same scene (well-exposed reps match) -> truly 1 scene?
    if nc>=2:
        reps=[well_rep([fs[i] for i in range(len(fs)) if lab[i]==c]) for c in range(nc)]
        cross=[mz(feats(reps[a]),feats(reps[b])) for a in range(nc) for b in range(a+1,nc) if reps[a] and reps[b]]
        coherent = (cross and max(cross)>=0.45) or all(r is None for r in reps[1:])
    else:
        coherent=True; cross=[]
    verdict="FIXABLE (1 scene, algo under-connects)" if coherent else "GROUND-TRUTH ERROR (multi-scene)"
    cls[verdict].append(g)
    print(f"  g{g}: {len(fs)}f, {nc} masked-subscenes, cross-rep={[round(c,2) for c in cross]} -> {verdict}")
    cells="".join(f'<div class=t><img src="data:image/jpeg;base64,{thumb(f)}"><div class=b>B{feats(f)[2]:.0f}</div></div>' for f in sorted(fs,key=lambda f:feats(f)[2]))
    rows.append(f'<div class=grp><div class=h>OVER-SPLIT g{g} — {verdict}</div><div class=r>{cells}</div></div>')

print("\n=== OVER-MERGE clusters: are the merged groups DIFFERENT scenes? ===")
for clu in OVER_MERGE_CLUSTERS:
    reps={g:well_rep(g2f[g]) for g in clu}
    pairs=[(a,b,mz(feats(reps[a]),feats(reps[b]))) for i,a in enumerate(clu) for b in clu[i+1:] if reps[a] and reps[b]]
    mx=max((p[2] for p in pairs),default=-1)
    # different scenes (low cross-sim) => algorithm over-merge, fixable by better discrimination
    # same scene (high)            => reference over-split these into separate groups = GT error
    verdict="FIXABLE (different scenes, algo over-merges)" if mx<0.45 else "GROUND-TRUTH ERROR (reference over-split same scene)"
    for g in clu: cls[verdict].append(g)
    print(f"  cluster {clu}: max cross-group rep masked={mx:.2f} pairs={[(a,b,round(s,2)) for a,b,s in pairs]} -> {verdict}")
    cells=""
    for gi,g in enumerate(clu):
        for f in sorted(g2f[g],key=lambda f:feats(f)[2]):
            cells+=f'<div class=t><img src="data:image/jpeg;base64,{thumb(f)}"><div class=b style="background:{["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"][gi%5]}">g{g} B{feats(f)[2]:.0f}</div></div>'
    rows.append(f'<div class=grp><div class=h>OVER-MERGE {clu} — {verdict} (color=ref group)</div><div class=r>{cells}</div></div>')

print("\n=== SUMMARY ===")
for v,gs in cls.items(): print(f"  {len(gs)} groups: {v}\n     {sorted(set(gs))}")
html=f"""<!doctype html><meta charset=utf-8><title>Remaining missed-fixable inspection</title>
<style>body{{font:13px system-ui;background:#111;color:#ddd;margin:20px}}.grp{{margin:14px 0;border:1px solid #333;border-radius:8px;padding:10px;background:#1a1a1a}}
.h{{margin-bottom:8px;color:#bbb}}.r{{display:flex;flex-wrap:wrap;gap:5px}}.t{{text-align:center}}.t img{{width:80px;height:80px;object-fit:cover;border-radius:4px;display:block}}
.b{{font-size:9px;color:#fff;border-radius:0 0 4px 4px;padding:1px;background:#444}}</style>
<h2>Remaining missed-fixable groups — algorithm failure vs reference error</h2>{''.join(rows)}"""
open("verify_missed.html","w",encoding="utf-8").write(html)
print("\nwrote verify_missed.html")
