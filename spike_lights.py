"""SPIKE: pin LIGHT-SOURCE positions in near-black frames and match scenes by them.

A near-black frame is a few bright blobs (windows/bulbs) on black. Scene identity =
WHERE those lights are (position is exposure-invariant — a light sits at the same
pixel whatever the exposure). So detect the bright blobs, take their centroids, and
compare two frames by how well their light positions coincide. This pins the
discriminating signal instead of diluting it over a whole coarse grid.

Goal: does light-position matching cleanly separate FOREIGN near-black frames
(different scene, want MISMATCH) from genuine SAME-scene near-black frames (want
MATCH)?
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

ddir = Path("data/full_subset")
raw = np.load(ddir / "raw256.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
pred = json.load(open(ddir / "pred_labels.json"))
f2g = {}
for r in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
    f2g[r["filename"]] = r["group_id"]
gt = defaultdict(list)
for f, g in f2g.items():
    gt[g].append(f)

def lights(gray, min_area=3):
    """Centroids (y,x in [0,1]) + weight of bright blobs = the scene's light sources."""
    g = gray.astype(np.float32)
    # bright = clearly above this frame's own background, with an absolute floor so
    # pure sensor noise in a dead-black frame doesn't register as a light.
    thr = max(float(np.percentile(g, 99.5)), g.mean() + 4 * g.std(), 35.0)
    mask = (g >= thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            cx, cy = cent[k]
            out.append((cy / 256.0, cx / 256.0, int(stats[k, cv2.CC_STAT_AREA])))
    return out

def chamfer(la, lb):
    """Symmetric mean nearest-neighbour distance between two light-position sets
    (0 = identical positions; larger = lights are elsewhere). None if either empty."""
    if not la or not lb:
        return None
    pa = np.array([(y, x) for y, x, _ in la])
    pb = np.array([(y, x) for y, x, _ in lb])
    d_ab = np.mean([np.min(np.hypot(*(pb - p).T)) for p in pa])
    d_ba = np.mean([np.min(np.hypot(*(pa - p).T)) for p in pb])
    return 0.5 * (d_ab + d_ba)

def darkframes(g, hi=40):
    return sorted((f for f in gt[g] if f in idx and imgs[idx[f]].mean() <= hi),
                  key=lambda f: imgs[idx[f]].mean())

print("=== FOREIGN: foreign near-black frame vs MAIN scene near-black (want HIGH dist) ===")
for a, b in [("1038","10280"),("10464","10613"),("10886","10593"),
             ("14288","14037"),("14279","14983"),("10276","1038")]:
    # whichever group is the minority in their shared cluster is "foreign"
    cs = [{pred[f] for f in gt[g] if f in pred} for g in (a, b)]
    sh = set.intersection(*cs) if all(cs) else set()
    if not sh:
        print(f"  {a}+{b}: not co-merged"); continue
    cl = next(iter(sh))
    fr = [f for f in pred if pred[f] == cl]
    from collections import Counter
    cnt = Counter(f2g[f] for f in fr if f in f2g)
    minor = min(cnt, key=cnt.get); major = max(cnt, key=cnt.get)
    fdark = [f for f in fr if f2g.get(f) == minor and f in idx and imgs[idx[f]].mean() <= 40]
    mdark = [f for f in fr if f2g.get(f) == major and f in idx and imgs[idx[f]].mean() <= 40]
    if not fdark or not mdark:
        print(f"  {a}+{b}: minor={minor} no dark pair (fdark={len(fdark)},mdark={len(mdark)})"); continue
    d = min(filter(None, (chamfer(lights(imgs[idx[fa]]), lights(imgs[idx[ma]]))
                          for fa in fdark for ma in mdark)), default=None)
    nf = len(lights(imgs[idx[fdark[0]]])); nm = len(lights(imgs[idx[mdark[0]]]))
    print(f"  {a}+{b} (foreign={minor}): chamfer={d:.3f}  (#lights foreign={nf} main={nm})")

print("\n=== SAME scene near-black pairs (want LOW dist) ===")
for g in ["10463","13675","10147","10160","13093"]:
    df = darkframes(g, hi=60)
    if len(df) >= 2:
        d = chamfer(lights(imgs[idx[df[0]]]), lights(imgs[idx[df[1]]]))
        print(f"  {g}: chamfer={d}  (#lights={len(lights(imgs[idx[df[0]]]))})")
    else:
        # compare the darkest frame's lights to the next-darkest as same-scene proxy
        fs = sorted((f for f in gt[g] if f in idx), key=lambda f: imgs[idx[f]].mean())
        d = chamfer(lights(imgs[idx[fs[0]]]), lights(imgs[idx[fs[1]]])) if len(fs) >= 2 else None
        print(f"  {g}: chamfer={d} (proxy: 2 darkest)")
