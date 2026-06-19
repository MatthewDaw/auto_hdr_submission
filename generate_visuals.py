"""
Generate a compact visual for (nearly) every slide and inject into
algorithm_slideshow.html. Uses real photos from the dataset + a couple of
hand-drawn charts (threshold curve, results bars). Self-contained base64.
"""
import csv, base64
from collections import defaultdict
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA

DATA="data/large"
g2f=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8")):
    g2f[r["group_id"]].append(r["filename"])
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def brt(f):
    im=readbgr(f); return None if im is None else float(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).mean())
clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
def b64(im,png=False):
    ok,buf=cv2.imencode(".png" if png else ".jpg", im, [] if png else [cv2.IMWRITE_JPEG_QUALITY,86])
    return base64.b64encode(buf).decode()
def thumb(f,w=104,border=None):
    im=readbgr(f); h=int(im.shape[0]*w/im.shape[1]); im=cv2.resize(im,(w,h))
    st=f"width:{w}px;border-radius:6px;display:block" + (f";border:3px solid {border}" if border else ";border:1px solid #30363d")
    return f'<img src="data:image/jpeg;base64,{b64(im)}" style="{st}">'
def edgemap(f,w=104):
    g=clahe.apply(cv2.cvtColor(readbgr(f),cv2.COLOR_BGR2GRAY)); g=cv2.resize(g,(w,int(g.shape[0]*w/g.shape[1])))
    gx=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,1,0,3); gy=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,0,1,3)
    m=cv2.magnitude(gx,gy); m=np.clip(m/(np.percentile(m,99)+1e-6)*255,0,255).astype(np.uint8); return m
def imgtag(im,w=104,png=True,cap=None,col="#58a6ff"):
    mime="png" if png else "jpeg"
    t=f'<img src="data:image/{mime};base64,{b64(im,png)}" style="width:{w}px;border-radius:6px;border:1px solid #30363d;display:block">'
    if cap: t=f'<figure style="margin:0;text-align:center">{t}<figcaption style="font-size:.62em;color:{col};margin-top:.2em">{cap}</figcaption></figure>'
    return t
def strip(items,gap=".4em"):
    return f'<div style="display:flex;gap:{gap};flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:.5em 0">{"".join(items)}</div>'

# ---- pick example groups ----
def spread_group(lo_max=70,hi_min=140,n=(4,6)):
    for g,fs in g2f.items():
        if not(n[0]<=len(fs)<=n[1]) or "DJI" in fs[0]: continue
        bs=sorted((brt(f),f) for f in fs)
        if bs[0][0] and bs[0][0]<lo_max and bs[-1][0]>hi_min: return g,[f for _,f in bs]
    return None,None
LADG,ladder=spread_group()
print("ladder group",LADG,[int(brt(f)) for f in ladder])

# distinct groups for problem strip + embedding map
distinct=[g for g,fs in g2f.items() if 3<=len(fs)<=5 and "DJI" not in fs[0]]
emap_groups=distinct[:6]

# ---- model for embedding map ----
class Model(nn.Module):
    def __init__(s,d=128):
        super().__init__(); m=mobilenet_v3_small(weights=None)
        s.backbone=nn.Sequential(m.features,m.avgpool,nn.Flatten()); s.proj=nn.Sequential(nn.Linear(576,256),nn.ReLU(),nn.Linear(256,d))
    def forward(s,x): return F.normalize(s.proj(s.backbone(x)),dim=1)
DEV="cuda" if torch.cuda.is_available() else "cpu"
mc=Model().to(DEV); mc.load_state_dict(torch.load("embed2_best.pt")); mc.eval()
MEAN=torch.tensor([0.5]*3).view(1,3,1,1).to(DEV); STD=torch.tensor([0.25]*3).view(1,3,1,1).to(DEV)
@torch.no_grad()
def embed(f):
    bgr=cv2.resize(readbgr(f),(128,128)); L=cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab); L[:,:,0]=clahe.apply(L[:,:,0])
    x=torch.from_numpy(L).float().to(DEV)/255.0; x=x.permute(2,0,1).unsqueeze(0)
    return mc((x-MEAN)/STD).cpu().numpy()[0]

# =================== build images ===================
FRAG={}

# Slide: problem — mixed photos -> two grouped
mix=[ (g, g2f[g][0]) for g in distinct[:3] for _ in [0] ]
items=[thumb(g2f[distinct[0]][0],border="#e74c3c"),thumb(g2f[distinct[0]][min(1,len(g2f[distinct[0]])-1)],border="#e74c3c"),
       thumb(g2f[distinct[1]][0],border="#3498db"),thumb(g2f[distinct[2]][0],border="#2ecc71"),thumb(g2f[distinct[3]][0],border="#f39c12")]
FRAG['<h2>The problem</h2>']='<div style="font-size:.62em;color:#8b949e">same border = same group →</div>'+strip(items)

# Slide: twist — exposure ladder
FRAG['<h2>The twist</h2>']=('<div style="font-size:.62em;color:#8b949e">one camera angle, ladder of exposures (all ONE group):</div>'
    +strip([thumb(f,96) for f in ladder]))

# Slide: key insight — ladder + identical edges
FRAG['<h2>Key insight</h2>']=('<div style="font-size:.62em;color:#8b949e">raw (very different brightness):</div>'
    +strip([thumb(f,92) for f in ladder[:4]])
    +'<div style="font-size:.62em;color:#3fb950">edge maps (nearly identical → structure survives exposure):</div>'
    +strip([imgtag(edgemap(f,92),92) for f in ladder[:4]]))

# Slide: A learned complement — embedding 2D map
allf=[(g,f) for g in emap_groups for f in g2f[g]]
embs=np.stack([embed(f) for g,f in allf]); xy=PCA(n_components=2).fit_transform(embs)
xy-=xy.min(0); xy/=(xy.max(0)+1e-9)
CW,CH,TS=760,300,52; canvas=np.full((CH,CW,3),22,np.uint8)
pal=[(76,60,231),(219,152,52),(80,175,76),(34,153,210),(182,89,155),(96,125,139)]  # BGR
for (g,f),p in zip(allf,xy):
    cx=int(28+p[0]*(CW-2*28-TS)); cy=int(20+p[1]*(CH-2*20-TS))
    im=cv2.resize(readbgr(f),(TS,TS)); col=pal[emap_groups.index(g)%len(pal)]
    canvas[cy-2:cy+TS+2,cx-2:cx+TS+2]=col; canvas[cy:cy+TS,cx:cx+TS]=im
FRAG['<h2>A learned complement</h2>']=('<div style="font-size:.62em;color:#8b949e">each photo placed by its embedding (color = true room) — same room clusters together:</div>'
    +strip([imgtag(canvas,560)]))

# Slide: threshold curve
d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); M=d["M"]; sim=(M@M.T).astype(np.float32)
ths=np.arange(0.40,0.80,0.02); counts=[]
for t in ths:
    A=sim>=t; np.fill_diagonal(A,False); nc,_=connected_components(csr_matrix(A),directed=False); counts.append(nc)
counts=np.array(counts,float)
PW,PH=720,300; pc=np.full((PH,PW,3),22,np.uint8)
x0,y0,x1,y1=70,30,PW-20,PH-50
cv2.rectangle(pc,(x0,y0),(x1,y1),(60,60,60),1)
def px(i): return int(x0+(i/(len(ths)-1))*(x1-x0))
def py(v): return int(y1-(v-counts.min())/(counts.max()-counts.min()+1e-9)*(y1-y0))
pts=[(px(i),py(counts[i])) for i in range(len(ths))]
cv2.polylines(pc,[np.array(pts,np.int32)],False,(255,166,88),2)
# plateau marker near where slope min
slope=np.gradient(counts); pi=int(np.argmin(np.abs(slope[3:-3]))+3)
cv2.circle(pc,pts[pi],6,(80,175,76),-1)
cv2.putText(pc,"plateau = chosen threshold",(pts[pi][0]-40,pts[pi][1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,175,76),1,cv2.LINE_AA)
cv2.putText(pc,"# groups",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1,cv2.LINE_AA)
cv2.putText(pc,"threshold ->",(x1-110,PH-18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1,cv2.LINE_AA)
FRAG['<h2>Per-run threshold (no labels)</h2>']=strip([imgtag(pc,560)])

# Slide: three passes — over-split & over-merge examples
osg="70877"; omg=["4986","33301"]
FRAG['<h2>Three refinement passes</h2>']=(
    '<div style="display:flex;gap:1.5em;flex-wrap:wrap;justify-content:center;margin-top:.4em">'
    +f'<div style="text-align:center"><div style="color:#d29922;font-size:.7em">over-split: one group, clipped frame breaks off</div>'
    +strip([thumb(f,72) for f in sorted(g2f[osg],key=brt)[:5]],gap=".25em")+'</div>'
    +f'<div style="text-align:center"><div style="color:#d29922;font-size:.7em">over-merge: two different rooms joined</div>'
    +strip([thumb(g2f[omg[0]][0],80,"#e74c3c"),thumb(g2f[omg[1]][0],80,"#3498db")],gap=".25em")+'</div></div>')

# Slide: FIX 2 — clipped frame + valid mask + neighbor
clf=min(g2f[osg],key=brt); nbr=sorted(g2f[osg],key=lambda f:abs(brt(f)-110))[0]
def rsz(f,w=104):
    im=readbgr(f); return cv2.resize(im,(w,int(im.shape[0]*w/im.shape[1])))
raw=rsz(clf); g0=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
mask=cv2.cvtColor((((g0>=8)&(g0<=247)).astype(np.uint8)*255),cv2.COLOR_GRAY2BGR)
FRAG['<h2><span class="tag fix">FIX 2</span> Exposure-ladder re-attach</h2>']=strip([
    imgtag(raw,104,False,"clipped frame"),
    imgtag(mask,104,True,"valid (non-saturated) pixels","#d29922"),
    imgtag(rsz(nbr),104,False,"brightness-neighbor","#3fb950")],
    gap=".5em")+'<div style="font-size:.62em;color:#8b949e">compare only the co-valid pixels → re-attach the orphan</div>'

# Slide: FIX 4 — two pieces of one scene merge
half=sorted(ladder,key=brt)
FRAG['<h2><span class="tag fix">FIX 4</span> Cluster merge</h2>']=(
    '<div style="display:flex;gap:1em;align-items:center;justify-content:center;margin-top:.4em">'
    +f'<div style="text-align:center"><div style="font-size:.62em;color:#8b949e">dark piece</div>{strip([thumb(f,72) for f in half[:2]],gap=".25em")}</div>'
    +'<div style="color:#3fb950;font-size:1.6em">+</div>'
    +f'<div style="text-align:center"><div style="font-size:.62em;color:#8b949e">bright piece</div>{strip([thumb(f,72) for f in half[-2:]],gap=".25em")}</div>'
    +'<div style="color:#3fb950;font-size:1.6em">→ 1 scene</div></div>')

# Slide: FIX 5 — 5 different rooms wrongly merged
five=["4986","33301","56922","63126","90309"]
FRAG['<h2><span class="tag fix">FIX 5</span> High-resolution split</h2>']=(
    '<div style="font-size:.62em;color:#8b949e">these 5 LOOK alike to the coarse 64px descriptor — but are different rooms (256px reveals it):</div>'
    +strip([thumb(g2f[g][0],86,pal_hex) for g,pal_hex in zip(five,["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"])]))

# Slide: ground-truth error — a multi-scene mislabeled group
import json
uf=list(json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"].keys())
gte=next((g for g in ["40599","94278","9245"] if g in g2f),uf[0])
FRAG['<h2>Knowing what\'s impossible</h2>']=('<div style="font-size:.62em;color:#8b949e">one reference group, but visually-unrelated scenes (mislabeled → excluded):</div>'
    +strip([thumb(f,80) for f in sorted(g2f[gte],key=brt)[:6]]))

# Slide: results bar chart
RW,RH=560,260; rc=np.full((RH,RW,3),22,np.uint8)
bars=[("500-set",1.000,(80,175,76)),("5,000-set",0.992,(80,175,76)),("baseline",0.087,(120,120,120))]
bw=110; gap=70; bx=80
for name,val,col in bars:
    bh=int(val*(RH-80)); cv2.rectangle(rc,(bx,RH-40-bh),(bx+bw,RH-40),col,-1)
    cv2.putText(rc,f"{val:.3f}",(bx+8,RH-48-bh),cv2.FONT_HERSHEY_SIMPLEX,0.6,(230,230,230),2,cv2.LINE_AA)
    cv2.putText(rc,name,(bx-4,RH-16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1,cv2.LINE_AA)
    bx+=bw+gap
FRAG['<h2>Results</h2>']=strip([imgtag(rc,460)])

# new OVERLAY slide content (same vs different red/green)
def overlay(fa,fb):
    ea=edgemap(fa,150).astype(np.float32); eb=edgemap(fb,150).astype(np.float32)
    h=min(ea.shape[0],eb.shape[0]); ea=ea[:h]; eb=eb[:h]
    o=np.zeros((h,150,3),np.uint8); o[:,:,2]=ea.astype(np.uint8); o[:,:,1]=eb.astype(np.uint8)  # R=ea,G=eb (BGR: idx2=R,1=G)
    return o
ov_same=overlay(ladder[0],ladder[-1]); ov_diff=overlay(ladder[len(ladder)//2], g2f[distinct[1]][0])
overlay_frag=('<div style="display:flex;gap:1.6em;justify-content:center;margin:.5em 0">'
    +f'<div style="text-align:center">{imgtag(ov_same,190,True)}<div style="color:#3fb950;font-size:.78em;margin-top:.3em">same angle → mostly <span style="color:#d29922">yellow</span> (aligned)</div></div>'
    +f'<div style="text-align:center">{imgtag(ov_diff,190,True)}<div style="color:#f85149;font-size:.78em;margin-top:.3em">different room → red &amp; green separate</div></div></div>')
NEW_SLIDE=('\n  <section class="slide" data-narr="Here is the intuition made literal. Tint one photo\'s edges red and the other\'s green, then overlay them. Where the outlines fall in the same place, red and green combine into yellow. The same angle at two very different exposures lights up almost entirely yellow — the outlines match. A different room stays mostly separate red and green. That overlap is exactly what the correlation score measures.">\n'
    '    <h2>Gradient-ZNCC, made literal</h2>\n'
    '    <p class="mut">Tint one photo\'s edges <span style="color:#f85149">red</span>, the other\'s <span style="color:#3fb950">green</span>. Overlap → <span style="color:#d29922">yellow</span>.</p>\n'
    f'    {overlay_frag}\n  </section>\n')

# =================== inject ===================
html=open("algorithm_slideshow.html",encoding="utf-8").read()
for h2,frag in FRAG.items():
    if h2 in html: html=html.replace(h2, h2+"\n    "+frag, 1)
    else: print("ANCHOR MISSING:",h2)
# insert new overlay slide right after the core-signal slide
anchor="Pure arithmetic, no model.</p>\n  </section>"
html=html.replace(anchor, anchor+NEW_SLIDE, 1)
open("algorithm_slideshow.html","w",encoding="utf-8").write(html)
print("done. file bytes:",len(html))
