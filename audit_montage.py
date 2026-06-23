"""Render thumbnail montages (from the 128px color cache) for the regressed groups,
so a human can judge: is each GT group really one scene (true regression) or two
scenes wrongly merged in GT (a real GT correction)? Frames are labeled with their
brightness and which cluster OUR prediction put them in."""
import csv, base64, cv2
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

data = Path("data/large")
raw = np.load(data/"raw256.npz", allow_pickle=True)
files = list(raw["files"]); gray = raw["imgs"]
col = np.load(data/"img128c.npz", allow_pickle=True)
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
color = np.stack([cimgs[cidx[f]] for f in files])
idx = {f: i for i, f in enumerate(files)}
gt = defaultdict(list)
with open(data/"public_manifest.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh): gt[row["group_id"]].append(row["filename"])

pred = ImageGrouper().group(Photoshoot(list(files), gray, color))
f2c = {}
for ci, grp in enumerate(pred):
    for f in grp: f2c[f] = ci

def thumb(f):
    im = cimgs[cidx[f]]
    ok, buf = cv2.imencode(".jpg", im)
    return base64.b64encode(buf).decode()

groups = ['23690','22534','49265','9635','42093','55384','63560','69279']
html = ["<html><body style='background:#222;color:#eee;font-family:sans-serif'>"]
for g in groups:
    html.append(f"<h3>GT {g}</h3><div style='display:flex;flex-wrap:wrap;gap:6px'>")
    for f in sorted(gt[g], key=lambda f: gray[idx[f]].mean()):
        b = round(float(gray[idx[f]].mean())); c = f2c.get(f, "?")
        html.append(f"<div style='text-align:center'><img src='data:image/jpeg;base64,{thumb(f)}' "
                    f"width=128><br>B={b} cl{c}</div>")
    html.append("</div>")
html.append("</body></html>")
Path("regression_montage.html").write_text("".join(html), encoding="utf-8")
print("wrote regression_montage.html")
