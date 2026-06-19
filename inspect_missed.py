import base64
from collections import defaultdict
import numpy as np, cv2
DATA="data/large"; VLO,VHI=8,247
rd=np.load(f"{DATA}/raw256.npz",allow_pickle=True); raw=rd["imgs"]; files=list(rd["files"]); gid=rd["gid"]
idx={f:i for i,f in enumerate(files)}
g2i=defaultdict(list)
for i,f in enumerate(files): g2i[gid[i]].append(i)
B=np.array([raw[i].mean() for i in range(len(raw))])
def vf(i): return float(((raw[i]>=VLO)&(raw[i]<=VHI)).mean())
def readbgr(f):
    with open(f"{DATA}/images/{f}","rb") as fh: b=fh.read()
    return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
def th(i):
    dead = vf(i)<0.03
    bord = "#f85149" if dead else ("#d29922" if (B[i]>245 or B[i]<12) else "#30363d")
    im=cv2.resize(readbgr(files[i]),(110,110)); ok,b=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,82])
    lab=f"B{B[i]:.0f}·vf{vf(i):.2f}"
    return f'<figure style="margin:1px;display:inline-block;text-align:center"><img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" style="width:110px;border:3px solid {bord};border-radius:5px"><figcaption style="font-size:10px;color:#888">{lab}</figcaption></figure>'
rows=[]
for g in ["10370","35098","82871","56453","56460","70877","22901"]:
    mem=sorted(g2i[g],key=lambda i:B[i])
    rows.append(f'<div style="margin:14px 0;padding:8px;background:#161b22;border-radius:8px"><h3 style="margin:3px">group {g} — {len(mem)} frames (sorted dark→bright). <span style="color:#f85149">red</span>=structurally dead (&lt;3% valid px), <span style="color:#d29922">amber</span>=clipped</h3>{"".join(th(i) for i in mem)}</div>')
open("missed_inspect.html","w",encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><style>body{{background:#0d1117;color:#ddd;font:14px system-ui;margin:18px}}</style><h1>Inspecting the 7 still-missed groups</h1><p>vf = valid-pixel fraction (pixels in 8–247, the only ones carrying structure). A frame with vf≈0 is pure white/black — no recoverable content.</p>{''.join(rows)}")
print("wrote missed_inspect.html")
