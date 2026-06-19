"""
Generate the per-step demo images for slide 6 and inject them into
algorithm_slideshow.html at the <!--DEMOIMGS--> marker. Self-contained base64.
Shows ONE photo through: raw -> grayscale -> CLAHE -> edge map; then edge
correlation (same-angle different-exposure = high ZNCC; different room = low).
"""
import csv, base64
from collections import defaultdict
import numpy as np, cv2

DATA="data/large"
g2f=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv", encoding="utf-8")):
    g2f[r["group_id"]].append(r["filename"])
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)

# pick a nice interior bracket: a group of size>=3 spanning exposures, none fully clipped
def bright(f):
    im=readbgr(f); return None if im is None else float(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).mean())
main_g=None
for g,fs in g2f.items():
    if not(3<=len(fs)<=6): continue
    bs=sorted((bright(f),f) for f in fs)
    if bs[0][0] is None: continue
    lo,hi=bs[0][0],bs[-1][0]
    if 25<lo<90 and 120<hi<210 and "DJI" not in fs[0]:    # interior, decent spread, not drone
        main_g=g; ordered=[f for _,f in bs]; break
# different-angle frame from another group of similar brightness
midf=ordered[len(ordered)//2]; midb=bright(midf)
other=None
for g,fs in g2f.items():
    if g==main_g or "DJI" in fs[0]: continue
    for f in fs:
        b=bright(f)
        if b and abs(b-midb)<25: other=f; break
    if other: break
print(f"demo group={main_g} frames={ordered}  partner(same)={ordered[-1]}  different={other}")

W=300
clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
def fit(im):
    h,w=im.shape[:2]; s=W/w; return cv2.resize(im,(W,int(h*s)),interpolation=cv2.INTER_AREA)
def b64(im,png=False):
    ext=".png" if png else ".jpg"; p=[cv2.IMWRITE_JPEG_QUALITY,88] if not png else []
    ok,buf=cv2.imencode(ext,im,p); return base64.b64encode(buf).decode()
def gray(f): return cv2.cvtColor(readbgr(f),cv2.COLOR_BGR2GRAY)
def edge(g):  # g grayscale (already CLAHE'd) -> normalized edge map (white on black)
    gx=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,1,0,3); gy=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,0,1,3)
    m=cv2.magnitude(gx,gy); m=np.clip(m/ (np.percentile(m,99)+1e-6)*255,0,255).astype(np.uint8); return m
def zncc(ea,eb):
    a=cv2.resize(ea,(64,64)).ravel().astype(np.float32); b=cv2.resize(eb,(64,64)).ravel().astype(np.float32)
    a-=a.mean(); b-=b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

# main pipeline on the mid frame
raw=fit(readbgr(midf)); gr=fit(gray(midf)); cl=fit(clahe.apply(gray(midf))); ed=fit(edge(clahe.apply(gray(midf))))
# correlation set (edge maps on CLAHE'd)
eA=edge(clahe.apply(gray(midf)))
eB=edge(clahe.apply(gray(ordered[-1])))      # same angle, brightest exposure
eC=edge(clahe.apply(gray(other)))            # different room
zAB=zncc(eA,eB); zAC=zncc(eA,eC)
imgA=fit(eA); imgB=fit(eB); imgC=fit(eC)
rawB=fit(readbgr(ordered[-1])); rawC=fit(readbgr(other))

def cell(title,im,png=False,sub=""):
    s=f'<div style="font-size:.62em;color:#8b949e;margin-bottom:.2em">{sub}</div>' if sub else ''
    return (f'<figure style="margin:0;text-align:center">'
            f'<img src="data:image/{"png" if png else "jpeg"};base64,{b64(im,png)}" '
            f'style="width:150px;border-radius:8px;border:1px solid #30363d;display:block">'
            f'<figcaption style="font-size:.72em;color:#58a6ff;margin-top:.3em">{title}</figcaption>{s}</figure>')
arr='<div style="color:#8b949e;align-self:center;font-size:1.4em">→</div>'

row1=('<div style="display:flex;gap:.5em;flex-wrap:wrap;align-items:flex-start;justify-content:center">'
      + cell("raw",raw) + arr + cell("grayscale",gr,True) + arr
      + cell("CLAHE",cl,True) + arr + cell("edge map",ed,True) + '</div>')

corr=(f'<div style="display:flex;gap:1.2em;flex-wrap:wrap;justify-content:center;margin-top:.7em">'
      f'<div style="text-align:center"><div style="display:flex;gap:.4em">{cell("",imgA,True)}{cell("",imgB,True)}</div>'
      f'<div style="margin-top:.2em;color:#3fb950;font-size:.8em"><b>ZNCC = {zAB:.2f}</b> ✓ same angle, diff exposure</div></div>'
      f'<div style="text-align:center"><div style="display:flex;gap:.4em">{cell("",imgA,True)}{cell("",imgC,True)}</div>'
      f'<div style="margin-top:.2em;color:#f85149;font-size:.8em"><b>ZNCC = {zAC:.2f}</b> ✗ different room</div></div>'
      f'</div>')

frag=('<div style="font-size:.7em;color:#8b949e;margin:.1em 0 .3em">①  one photo through the pipeline:</div>'
      + row1
      + '<div style="font-size:.7em;color:#8b949e;margin:.7em 0 .1em">②  compare two edge maps (ZNCC):</div>'
      + corr)

html=open("algorithm_slideshow.html",encoding="utf-8").read()
html=html.replace("<!--DEMOIMGS-->",frag)
open("algorithm_slideshow.html","w",encoding="utf-8").write(html)
print(f"injected demo images (ZNCC same={zAB:.2f}, diff={zAC:.2f})")
