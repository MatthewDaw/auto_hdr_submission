"""
Replace the ground-truth-error slide visual with a CLEAR, well-exposed example:
a single reference group that actually contains two different rooms. Larger
thumbnails, color-coded by sub-scene, with a highlighted scene-change divider.
"""
import csv, json, base64, re
from collections import defaultdict
import numpy as np, cv2

DATA="data/large"
g2f=defaultdict(list)
for r in csv.DictReader(open(f"{DATA}/public_manifest.csv",encoding="utf-8")):
    g2f[r["group_id"]].append(r["filename"])
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def gray(f): return cv2.cvtColor(readbgr(f),cv2.COLOR_BGR2GRAY)
def brt(f): return float(gray(f).mean())
clahe=cv2.createCLAHE(3.0,(8,8))
VLO,VHI=8,247
def gm(f,w=128):
    g=clahe.apply(cv2.resize(gray(f),(w,w))); gx=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,1,0,3); gy=cv2.Sobel(g.astype(np.float32),cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((cv2.resize(gray(f),(w,w))>=VLO)&(cv2.resize(gray(f),(w,w))<=VHI))
def mz(a,b):
    v=(a[1]&b[1]).ravel()
    if v.sum()<300: return -1
    x=a[0].ravel()[v]-a[0].ravel()[v].mean(); y=b[0].ravel()[v]-b[0].ravel()[v].mean()
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9))

uf=json.load(open(f"{DATA}/unfixable.json",encoding="utf-8"))["groups"]
# candidates: 2-subscene multi-scene groups with members recorded
best=None
for g,info in uf.items():
    if info.get("reason","").startswith("drone"): continue
    mem=info.get("members") or []
    if not mem: continue
    sub=defaultdict(list)
    for f,b,s in mem: sub[s].append((b,f))
    if len(sub)!=2: continue
    # each sub-scene needs a well-exposed frame (70..190)
    wells={s:[ (b,f) for b,f in v if 70<=b<=190] for s,v in sub.items()}
    if not all(wells.values()): continue
    # how visually different are the two best-exposed reps? lower = clearer example
    reps={s:min(v,key=lambda bf:abs(bf[0]-120))[1] for s,v in wells.items()}
    s0,s1=list(reps); diff=mz(gm(reps[s0]),gm(reps[s1]))
    score=(min(len(wells[s0]),len(wells[s1])), -diff, min(len(sub[s0]),len(sub[s1])))
    if best is None or score>best[0]: best=(score,g,sub,wells,diff)
_,G,sub,wells,diff=best
print(f"chosen GTE group={G} subscenes={ {s:len(v) for s,v in sub.items()} } rep-diff={diff:.2f}")

# render: up to 3 best-exposed frames per sub-scene, larger
def b64jpg(im):
    ok,buf=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,88]); return base64.b64encode(buf).decode()
def thumb(f,w=150,border="#888"):
    im=readbgr(f); im=cv2.resize(im,(w,int(im.shape[0]*w/im.shape[1])))
    return f'<img src="data:image/jpeg;base64,{b64jpg(im)}" style="width:{w}px;border-radius:8px;border:4px solid {border};display:block">'
cols={list(sub)[0]:"#e74c3c", list(sub)[1]:"#3498db"}
names={list(sub)[0]:"Room A", list(sub)[1]:"Room B"}
def panel(s):
    frames=[f for _,f in sorted(wells[s],key=lambda bf:abs(bf[0]-120))[:3]]
    thumbs="".join(thumb(f,150,cols[s]) for f in frames)
    return (f'<div style="text-align:center">'
            f'<div style="color:{cols[s]};font-weight:700;margin-bottom:.3em">{names[s]}</div>'
            f'<div style="display:flex;gap:.4em;justify-content:center">{thumbs}</div></div>')
divider=('<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;color:#d29922;font-weight:700">'
         '<div style="font-size:1.8em;line-height:1">⟵ ⟶</div><div style="font-size:.7em;writing-mode:vertical-rl;transform:rotate(180deg);margin-top:.2em">scene change</div></div>')
visual=(f'<div style="display:flex;gap:1.1em;align-items:center;justify-content:center;margin:.5em 0">{panel(list(sub)[0])}{divider}{panel(list(sub)[1])}</div>')

new_section=('  <section class="slide" data-narr="Knowing what is impossible. Some reference groups are simply mislabeled. Here is one group from the answer key — but look: the photos on the left are clearly one room, and the photos on the right are a completely different room. The answer key insists they all belong together. No appearance-based method can ever match that without merging unrelated rooms everywhere. So we detect these automatically — a group whose own members cannot be visually connected — and exclude them, so the score reflects the algorithm, not the bad labels.">\n'
    '    <h2>Knowing what\'s impossible</h2>\n'
    '    <div style="display:inline-block;background:#2d1b1b;border:1px solid #f8514955;border-radius:8px;padding:.25em .7em;color:#f85149;font-size:.8em;margin-bottom:.3em">⚠ ONE reference group in the answer key — but two different rooms</div>\n'
    f'    {visual}\n'
    '    <p class="mut">We auto-detect these (a group whose members can\'t be visually connected) and <b class="acc">exclude</b> them — the score measures the algorithm, not bad labels.</p>\n'
    '  </section>')

html=open("algorithm_slideshow.html",encoding="utf-8").read()
new_html=re.sub(r'  <section class="slide" data-narr="Knowing what.{0,4}s impossible.*?</section>', lambda m:new_section, html, count=1, flags=re.S)
assert new_html!=html, "ground-truth slide not found/replaced"
open("algorithm_slideshow.html","w",encoding="utf-8").write(new_html)
print("replaced ground-truth slide; bytes",len(new_html))
