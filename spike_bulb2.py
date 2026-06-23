"""Better bulb detector for B<30 frames, then re-test foreign-vs-same separation.

Premise: the lights that survive in a near-black frame are the scene's brightest,
most SATURATED points. In the well-exposed frame those same lights are also the
brightest (usually clipped to ~255) COMPACT blobs. Match saturated-compact blobs on
both sides instead of percentile regions (which pick dim noise in the clip and soft
window centroids in the template).

Detector:
  clip side  : connected blobs of pixels >= max(ABS_FLOOR, frame.max()*REL) that are
               COMPACT (area <= MAXA). Intensity-weighted centroids. These are the
               genuine surviving point-lights, not a percentile of dim noise.
  well side  : connected blobs of pixels >= SAT (near-saturation) AND compact — the
               physical light fixtures / blown highlights, point-localized (NOT the
               soft-window percentile centroid the old template used).

Coverage = fraction of clip lights whose translated position lands within TOL of a
well light (exhaustive 2-pt translation, same camera). Report foreign vs same-scene.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2

ABS_FLOOR = 45      # a real surviving light in a near-black frame clears this abs level
REL = 0.55          # ...or this fraction of the frame's own max (handles dim frames)
SAT = 240           # near-saturation on the well frame = a true light / blown highlight
MAXA = 400          # compact: a point-light blob, not a whole wall (px @256)
MINA = 2
TOL = 0.05
MAXT = 0.08
K = 8

def blobs(tile, thr, maxa=MAXA):
    mask = (tile >= thr).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for c in range(1, n):
        a = stats[c, cv2.CC_STAT_AREA]
        if a < MINA or a > maxa: continue
        ys, xs = np.where(lbl == c); w = tile[ys, xs].astype(np.float32)
        ws = w.sum()
        if ws <= 0: continue
        out.append((float(a)*float(w.mean()), (ys*w).sum()/ws/256.0, (xs*w).sum()/ws/256.0))
    out.sort(key=lambda t: -t[0])
    pts = []
    for _, y, x in out:
        if all(np.hypot(y-py, x-px) > 0.04 for py, px in pts): pts.append((y, x))
        if len(pts) >= K: break
    return np.array(pts, np.float32).reshape(-1, 2)

def clip_lights(tile):
    thr = max(ABS_FLOOR, float(tile.max()) * REL)
    return blobs(tile, thr)

def well_lights(tiles):
    allp = [blobs(t, SAT) for t in tiles]
    allp = [p for p in allp if len(p)]
    if not allp: return np.zeros((0,2), np.float32)
    # union then dedupe across well frames (same fixtures recur)
    P = np.vstack(allp); keep = []
    for y, x in P:
        if all(np.hypot(y-ky, x-kx) > 0.04 for ky, kx in keep): keep.append((y, x))
    return np.array(keep, np.float32)

def coverage(clip, wells):
    A = clip_lights(clip); B = well_lights(wells)
    if len(A) == 0 or len(B) == 0: return None, len(A), len(B)
    cands = [np.zeros(2, np.float32)] + [b-a for a in A for b in B if np.hypot(*(b-a)) <= MAXT]
    best = 0
    for t in cands:
        At = A + t
        used = set(); n = 0
        pr = sorted((float(np.hypot(*(At[i]-B[j]))), i, j) for i in range(len(At)) for j in range(len(B)))
        ui = set()
        for d, i, j in pr:
            if d > TOL: break
            if i in ui or j in used: continue
            ui.add(i); used.add(j); n += 1
        best = max(best, n)
    return best/len(A), len(A), len(B)

def load(d):
    raw=np.load(Path(d)/"raw256.npz",allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    gt=defaultdict(list)
    for r in csv.DictReader(open(Path(d)/"public_manifest.csv",encoding="utf-8")): gt[r["group_id"]].append(r["filename"])
    return imgs,idx,gt

def is_well(t): return 70 <= t.mean() <= 185
def darkest(g,gt,imgs,idx,lo=30):
    fr=[f for f in gt[g] if f in idx and imgs[idx[f]].mean()<lo]
    return sorted(fr,key=lambda f:imgs[idx[f]].mean())
def wells(g,gt,imgs,idx): return [imgs[idx[f]] for f in gt[g] if f in idx and is_well(imgs[idx[f]])]

imgs,idx,gt=load("data/full_subset")
print("=== FOREIGN (clip vs OTHER group template) — want LOW cov ===")
fvals=[]
for host,foreign in [("10280","1038"),("10613","10464"),("11533","11604"),("14037","14288"),("14983","14279")]:
    wt=wells(host,gt,imgs,idx); fr=darkest(foreign,gt,imgs,idx)
    if not fr or not wt: print(f"  {host}<-{foreign}: missing"); continue
    cov,na,nb=coverage(imgs[idx[fr[0]]],wt); fvals.append(cov)
    print(f"  {host}<-{foreign} B{imgs[idx[fr[0]]].mean():.0f}: cov={cov}  na={na} nb={nb}")

print("\n=== LINK positives (clip vs OWN template) — want HIGH cov ===")
for g in ["10463","19300"]:
    wt=wells(g,gt,imgs,idx); fr=darkest(g,gt,imgs,idx)
    if fr and wt:
        cov,na,nb=coverage(imgs[idx[fr[0]]],wt)
        print(f"  {g} B{imgs[idx[fr[0]]].mean():.0f}: cov={cov}  na={na} nb={nb}")

imgs2,idx2,gt2=load("data/large")
print("\n=== SAME-scene false-fire groups (clip vs OWN template) — want HIGH cov ===")
svals=[]
for g in ['60087','64473','9416','53608','75691','38250','47362','81802','86360',
          '51862','75587','50529','80204','13766','83477','27579','36696','51004','11175','62339','87479']:
    wt=wells(g,gt2,imgs2,idx2); fr=darkest(g,gt2,imgs2,idx2)
    if not fr or not wt: continue
    cov,na,nb=coverage(imgs2[idx2[fr[0]]],wt); svals.append(cov)
    print(f"  {g} B{imgs2[idx2[fr[0]]].mean():.0f}: cov={cov}  na={na} nb={nb}")

fv=[v for v in fvals if v is not None]; sv=[v for v in svals if v is not None]
print(f"\nSEPARATION: foreign max={max(fv) if fv else None}  same-scene min={min(sv) if sv else None}")
