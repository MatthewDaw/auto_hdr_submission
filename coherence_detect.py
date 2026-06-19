"""
Detect genuinely-unfixable reference groups: a group is unfixable if its own
members cannot be connected even with the best clipping-robust metric (co-valid
MASKED gradient ZNCC, which allows exposure-ladder chaining). If members form
2+ visually-disconnected sub-scenes IN ISOLATION, no appearance-based method can
group them correctly -> it's label noise / non-visual grouping / large drift.

Writes unfixable_<dataset>.json (group ids + reason) and an HTML gallery.
Usage: coherence_detect.py <data_dir>
"""
import sys, csv, json, base64
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA=Path(sys.argv[1] if len(sys.argv)>1 else "data/large")
VLO,VHI=8,247; THR=0.45   # masked-ZNCC edge threshold for "same scene"
raw=np.load(DATA/"raw256.npz",allow_pickle=True)["imgs"]
col=np.load(DATA/"img128c.npz",allow_pickle=True)["imgs"]
files=list(np.load(DATA/"raw256.npz",allow_pickle=True)["files"])
idxof={f:i for i,f in enumerate(files)}
g2f=defaultdict(list)
for r in csv.DictReader(open(DATA/"public_manifest.csv", encoding="utf-8")): g2f[r["group_id"]].append(r["filename"])

clahe=cv2.createCLAHE(3.0,(8,8))
def grad_mask(r):
    g=clahe.apply(r).astype(np.float32)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((r>=VLO)&(r<=VHI))
def mzncc(m1,v1,m2,v2):
    v=(v1&v2).ravel()
    if v.mean()<0.02: return -1.0
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def plain_zncc(m1,m2):
    a=m1.ravel()-m1.mean(); b=m2.ravel()-m2.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

WELL=(55,200)  # well-exposed brightness band
def components(mem):
    # 1) sub-scenes via MASKED zncc (bridges exposures within one scene)
    gms=[grad_mask(raw[idxof[f]]) for f in mem]
    br=np.array([raw[idxof[f]].mean() for f in mem])
    k=len(mem); A=np.zeros((k,k),bool)
    for a in range(k):
        for b in range(a+1,k):
            if mzncc(*gms[a],*gms[b])>=THR: A[a,b]=A[b,a]=True
    nc,lab=connected_components(csr_matrix(A),directed=False)
    if nc<2: return nc,lab
    # 2) merge sub-scenes UNLESS their WELL-EXPOSED frames are genuinely different.
    #    Connect A,B if best masked-zncc over well-exposed cross pairs >= 0.40, OR
    #    if either sub-scene has NO well-exposed frame (clipping -> benefit of doubt).
    well={c:[i for i in range(k) if lab[i]==c and WELL[0]<=br[i]<=WELL[1]] for c in range(nc)}
    RA=np.zeros((nc,nc),bool)
    for a in range(nc):
        for b in range(a+1,nc):
            if not well[a] or not well[b]:
                RA[a,b]=RA[b,a]=True; continue            # can't judge -> assume connectable
            mx=max(mzncc(*gms[ia],*gms[ib]) for ia in well[a] for ib in well[b])
            if mx>=0.40: RA[a,b]=RA[b,a]=True
    mnc,mlab=connected_components(csr_matrix(RA),directed=False)
    return mnc,np.array([mlab[lab[i]] for i in range(k)])

def thumb(f):
    im=col[idxof[f]]; im=cv2.resize(im,(80,80))
    ok,buf=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,70])
    return base64.b64encode(buf).decode()

unfix={}
for g,mem in g2f.items():
    if len(mem)<2: continue
    nc,lab=components(mem)
    if nc>=2:
        drone=any("DJI" in f for f in mem)
        reason="drone-drift" if drone else "multi-scene/label-noise"
        unfix[g]={"reason":reason,"frames":len(mem),"subscenes":int(nc),
                  "members":[(f,float(raw[idxof[f]].mean()),int(lab[i])) for i,f in enumerate(mem)]}

print(f"{DATA}: {len(unfix)}/{len(g2f)} groups genuinely unfixable")
by=defaultdict(int)
for v in unfix.values(): by[v["reason"]]+=1
print("  by reason:",dict(by))
# spot-check known groups
for g in ["33501","40599","40615","94278","11393","22994","25789"]:
    if g in g2f: print(f"  group {g}: {'UNFIXABLE('+unfix[g]['reason']+', '+str(unfix[g]['subscenes'])+' scenes)' if g in unfix else 'coherent/fixable'}")

json.dump({"threshold":THR,"groups":unfix}, open(DATA/"unfixable.json","w"), indent=1)
print(f"  wrote {DATA}/unfixable.json")

# HTML gallery
rows=[]
PAL=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
for g,info in sorted(unfix.items(), key=lambda kv:-kv[1]["subscenes"]):
    cells=""
    for f,br,sc in sorted(info["members"], key=lambda x:(x[2],x[1])):
        color=PAL[sc%len(PAL)]
        cells+=f'<div class="t"><img src="data:image/jpeg;base64,{thumb(f)}"><div class="b" style="background:{color}">B{br:.0f} · s{sc}</div></div>'
    rows.append(f'<div class="grp"><div class="h">group {g} — <b>{info["reason"]}</b> · {info["frames"]} frames · {info["subscenes"]} sub-scenes (color = disconnected sub-scene)</div><div class="r">{cells}</div></div>')
html=f"""<!doctype html><meta charset=utf-8><title>Unfixable groups — {DATA}</title>
<style>body{{font:13px system-ui;background:#111;color:#ddd;margin:20px}}
.grp{{margin:18px 0;border:1px solid #333;border-radius:8px;padding:10px;background:#1a1a1a}}
.h{{margin-bottom:8px;color:#bbb}} .r{{display:flex;flex-wrap:wrap;gap:6px}}
.t{{text-align:center}} .t img{{width:80px;height:80px;object-fit:cover;border-radius:4px;display:block}}
.b{{font-size:10px;color:#fff;border-radius:0 0 4px 4px;padding:1px}}</style>
<h2>{DATA}: {len(unfix)} genuinely-unfixable groups (members can't be visually connected)</h2>
<p>Each row is one reference group. Thumbnails colored by disconnected sub-scene — a group with multiple colors contains visually-unrelated scenes the ground truth lumps together (or a drone shot that moved). Sorted by sub-scene count.</p>
{''.join(rows)}"""
open(DATA/"unfixable_gallery.html","w",encoding="utf-8").write(html)
print(f"  wrote {DATA}/unfixable_gallery.html ({len(unfix)} groups)")
