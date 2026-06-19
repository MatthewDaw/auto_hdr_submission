import csv, base64
from collections import defaultdict
import numpy as np, cv2
DATA="data/large"; VLO,VHI=8,247
fc=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=fc["M"]; files=list(fc["files"]); idx={f:i for i,f in enumerate(files)}
raw=np.load(f"{DATA}/raw256.npz",allow_pickle=True)["imgs"]
gid={r["filename"]:r["group_id"] for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8"))}
g2f=defaultdict(list)
for f in files: g2f[gid[f]].append(f)
gidarr=np.array([gid[f] for f in files])
clahe=cv2.createCLAHE(3.0,(8,8))
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def gm(i):
    g=clahe.apply(raw[i]).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((raw[i]>=VLO)&(raw[i]<=VHI))
def mz(i,j):
    a,va=gm(i); b,vb=gm(j); v=(va&vb).ravel()
    if v.sum()<300: return -1.0
    x=a.ravel()[v]-a.ravel()[v].mean(); y=b.ravel()[v]-b.ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))
def well(g): 
    c=[(abs(raw[idx[f]].mean()-128),f) for f in g2f[g]]; return min(c)[1]
sim=(M@M.T).astype(np.float32)
rows=[]
for g in ["17184","82871","88673"]:
    ix=[idx[f] for f in g2f[g]]; sub=sim[ix]; same=(gidarr[ix][:,None]==gidarr[None,:])
    sub=np.where(same,-1,sub); bj=np.unravel_index(sub.argmax(),sub.shape)[1]; partner=gidarr[bj]
    wp=well(g); wq=well(partner); m=mz(idx[wp],idx[wq])
    print(f"g{g}: best cross-group partner=g{partner}  gradient={sub.max():.2f}  masked(well-reps)={m:.2f}")
    def th(f):
        im=cv2.resize(readbgr(f),(150,150)); ok,b=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,85]); return f'<img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" style="width:150px;border-radius:6px;margin:2px">'
    rows.append(f'<div style="margin:10px 0"><div style="color:#bbb">group <b>{g}</b> &nbsp;↔&nbsp; group <b>{partner}</b> &nbsp; masked={m:.2f} (near-identical if &gt;0.8)</div>'
                f'<div style="display:flex;gap:6px"><div style="border:3px solid #e74c3c;border-radius:8px;padding:2px">{"".join(th(f) for f in sorted(g2f[g],key=lambda f:raw[idx[f]].mean())[:3])}</div>'
                f'<div style="border:3px solid #3498db;border-radius:8px;padding:2px">{"".join(th(f) for f in sorted(g2f[partner],key=lambda f:raw[idx[f]].mean())[:3])}</div></div></div>')
open("dup_check.html","w",encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><style>body{{background:#111;color:#ddd;font:14px system-ui;margin:20px}}</style><h2>Suspected near-duplicate groups (red vs blue = two reference groups)</h2>{''.join(rows)}")
print("wrote dup_check.html")
