"""
Render the low-coherence-but-NOT-flagged groups (the green points sitting in the
outlier zone) so we can judge: genuine mislabel, or legitimately-varied group?
Sub-scene color-coded, well-exposed frames, with the coherence value.
"""
import csv, json, base64
from collections import defaultdict
import numpy as np, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA="data/large"; VLO,VHI=8,247
m=np.load(f"{DATA}/outlier_metrics.npz",allow_pickle=True); Gids=list(m["G"]); C={g:c for g,c in zip(m["G"],m["C"])}
g2f=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8")): g2f[r["group_id"]].append(r["filename"])
flagged=set(json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"].keys())
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True); RAW=raw["imgs"]; files=list(raw["files"]); idx={f:i for i,f in enumerate(files)}
clahe=cv2.createCLAHE(3.0,(8,8))
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def gm(i):
    g=clahe.apply(RAW[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((RAW[i]>=VLO)&(RAW[i]<=VHI))
def mz(i,j):
    a,va=gm(i); b,vb=gm(j); v=(va&vb).ravel()
    if v.sum()<300: return -1.0
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))
def subscenes(mem):
    ix=[idx[f] for f in mem]; k=len(ix); A=np.zeros((k,k),bool)
    for a in range(k):
        for b in range(a+1,k):
            if mz(ix[a],ix[b])>=0.45: A[a,b]=A[b,a]=True
    nc,lab=connected_components(csr_matrix(A),directed=False); return nc,lab
def thumb(f,col):
    im=cv2.resize(readbgr(f),(120,120)); ok,b=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,85])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" style="width:120px;border-radius:6px;border:3px solid {col};display:block">'

cand=sorted([g for g in Gids if g not in flagged and C[g]<0.55], key=lambda g:C[g])
print(f"low-coherence (<0.55) but NOT flagged: {len(cand)} groups")
PAL=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
rows=[]
for g in cand:
    mem=g2f[g]; nc,lab=subscenes(mem)
    print(f"  g{g}: coherence={C[g]:.2f}, {len(mem)} frames, {nc} sub-scenes")
    cells="".join(thumb(f,PAL[lab[i]%len(PAL)]) for i,f in enumerate(sorted(mem,key=lambda f:RAW[idx[f]].mean())))
    rows.append(f'<div class=grp><div class=h>group {g} — coherence <b>{C[g]:.2f}</b> · {len(mem)} frames · {nc} sub-scenes (color = sub-scene)</div><div class=r>{cells}</div></div>')
html=f"""<!doctype html><meta charset=utf-8><title>Low-coherence non-flagged groups</title>
<style>body{{font:13px system-ui;background:#111;color:#ddd;margin:20px}}.grp{{margin:14px 0;border:1px solid #333;border-radius:8px;padding:10px;background:#1a1a1a}}
.h{{margin-bottom:8px;color:#bbb}}.r{{display:flex;flex-wrap:wrap;gap:5px}}.r img{{object-fit:cover}}</style>
<h2>Low-coherence groups we did NOT flag ({len(cand)})</h2>
<p>These are the green points sitting in the outlier zone. If the sub-scenes (colors) are genuinely different rooms → mislabel we under-flagged. If they're the same room with variation → legitimately-varied, correctly not flagged.</p>{''.join(rows)}"""
open("borderline_gallery.html","w",encoding="utf-8").write(html)
print("wrote borderline_gallery.html")
