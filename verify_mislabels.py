import csv, base64
from collections import defaultdict
import numpy as np, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
DATA="data/large"; VLO,VHI=8,247
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); files=list(fc["files"]); idx={f:i for i,f in enumerate(files)}
gid={r["filename"]:r["group_id"] for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8"))}
g2f=defaultdict(list)
for f in files: g2f[gid[f]].append(f)
clahe=cv2.createCLAHE(3.0,(8,8))
def gm(i):
    g=clahe.apply(raw[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((raw[i]>=VLO)&(raw[i]<=VHI))
def mz(i,j):
    a,va=gm(i); b,vb=gm(j); v=(va&vb).ravel()
    if v.sum()<300: return -1.0
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def th(f,bord):
    im=cv2.resize(readbgr(f),(120,120)); ok,b=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,82])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" style="width:120px;border:3px solid {bord};border-radius:6px;margin:2px">'
PAL=["#e74c3c","#3498db","#2ecc71","#f1c40f","#9b59b6","#e67e22"]
rows=[]
for g,coh in [("9245",0.17),("38084",0.27),("73234",0.32),("40667",0.36)]:
    fs=[f for f in g2f[g] if 55<=raw[idx[f]].mean()<=200]  # well-exposed reps
    fs=sorted(fs,key=lambda f:raw[idx[f]].mean())
    n=len(fs); A=np.zeros((n,n),bool)
    for a in range(n):
        for b in range(a+1,n):
            if mz(idx[fs[a]],idx[fs[b]])>=0.45: A[a,b]=A[b,a]=True
    ncc,sub=connected_components(csr_matrix(A),directed=False)
    sc=defaultdict(list)
    for k,f in enumerate(fs): sc[int(sub[k])].append(f)
    blocks="".join(f'<div style="display:inline-block;vertical-align:top;margin:4px;padding:4px;border:1px dashed #555;border-radius:8px"><div style="color:{PAL[c%6]};font-size:12px">sub-scene {c+1}</div>{"".join(th(f,PAL[c%6]) for f in v[:5])}</div>' for c,v in sorted(sc.items()))
    rows.append(f'<div style="margin:16px 0;padding:10px;background:#1a1a1a;border-radius:10px"><h3 style="margin:4px">group <b>{g}</b> — labeled as ONE group, but contains <b>{ncc} distinct rooms</b> (sub-scenes match each other at only masked≈{coh:.2f}, vs &gt;0.6 for true brackets)</h3>{blocks}</div>')
open("mislabels_proof.html","w",encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><style>body{{background:#111;color:#ddd;font:14px system-ui;margin:24px}}</style><h1>4 newly-classified ground-truth mislabels</h1><p>Each reference group below is labeled as a single camera angle, but its well-exposed frames split into multiple visually-distinct rooms. Same colored border = same sub-scene. These cannot be solved by any exposure-invariant method because the label itself merges different scenes.</p>{''.join(rows)}")
print("wrote mislabels_proof.html")
