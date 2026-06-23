"""Validate a STRICT clipped-vs-clipped foreign-frame split (pixel-only, no filenames).

Rule: within a cluster, a near-black frame (mean<30) with >= MIN_LIGHTS detected
lights is FOREIGN (split out) when its saturated-light pattern matches NONE of the
cluster's OTHER same-direction near-black frames (each also >= MIN_LIGHTS) — i.e.
max clipped-vs-clipped light-match == 0. Same-scene dark frames share light
positions (match > 0), so they are never split; a foreign room's dark frame shares
no lights (match 0).

We (1) confirm it fires on the target over-merges and (2) scan every same-scene
group's dark frames for false-fires (must be 0 to hold the 1302 gate).
"""
import csv, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import extreme_anchor as ea

MIN_LIGHTS = 2

def light_match(a_tile, b_tile, tol=0.06, max_t=0.10):
    A = ea.clip_lights(a_tile); B = ea.clip_lights(b_tile)
    if len(A) < MIN_LIGHTS or len(B) < MIN_LIGHTS:
        return None
    cands = [np.zeros(2, np.float32)] + [b - a for a in A for b in B if np.hypot(*(b - a)) <= max_t]
    best = 0
    for t in cands:
        At = A + t; ui = set(); uj = set(); n = 0
        for d, i, j in sorted((float(np.hypot(*(At[i] - B[j]))), i, j)
                              for i in range(len(At)) for j in range(len(B))):
            if d > tol: break
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j); n += 1
        best = max(best, n)
    return best / min(len(A), len(B))

def load(dd):
    raw = np.load(Path(dd)/"raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
    gt = defaultdict(list)
    for r in csv.DictReader(open(Path(dd)/"public_manifest.csv", encoding="utf-8")):
        gt[r["group_id"]].append(r["filename"])
    return imgs, idx, gt

# --- target over-merges: foreign frame vs the OTHER group's dark frames (want max==0)
imgs, idx, gt = load("data/full_subset")
def darks(g, src_imgs, src_idx, hi=30):
    return [f for f in gt[g] if f in src_idx and src_imgs[src_idx[f]].mean() < hi]
print("=== TARGET over-merges (foreign dark vs other group's darks, want max-match 0) ===")
for a, b in [("10280","1038"),("10464","10613"),("10886","10593")]:
    da = darks(a, imgs, idx); db = darks(b, imgs, idx)
    if not da or not db:
        print(f"  {a}/{b}: missing darks (a={len(da)} b={len(db)})"); continue
    # foreign = the group contributing the lone dark frame; test both directions
    cross = max((light_match(imgs[idx[x]], imgs[idx[y]]) or 0) for x in da for y in db)
    # same-scene self-match for each
    sa = max((light_match(imgs[idx[da[i]]], imgs[idx[da[j]]]) or 0) for i in range(len(da)) for j in range(i+1,len(da))) if len(da)>1 else None
    sb = max((light_match(imgs[idx[db[i]]], imgs[idx[db[j]]]) or 0) for i in range(len(db)) for j in range(i+1,len(db))) if len(db)>1 else None
    print(f"  {a}/{b}: cross-match={cross:.2f}  own-match a={sa} b={sb}")

# --- SAFETY scan on data/large: same-scene dark frame matching NONE of its siblings
imgs2, idx2, gt2 = load("data/large")
import json
unfix = set(json.load(open("data/large/unfixable.json"))["groups"].keys())
false_fires = []; n_checked = 0
for g, frs in gt2.items():
    if g in unfix: continue
    dk = [f for f in frs if f in idx2 and imgs2[idx2[f]].mean() < 30
          and len(ea.clip_lights(imgs2[idx2[f]])) >= MIN_LIGHTS]
    if len(dk) < 2: continue
    for k in dk:
        n_checked += 1
        others = [o for o in dk if o != k]
        mx = max((light_match(imgs2[idx2[k]], imgs2[idx2[o]]) or 0) for o in others)
        if mx == 0:  # matches NONE of its same-scene dark siblings -> would be split
            false_fires.append((g, round(float(imgs2[idx2[k]].mean())), k))
print(f"\n=== SAFETY scan (data/large fixable) ===")
print(f"checked {n_checked} same-scene dark frames; FALSE force-splits: {len(false_fires)}")
for ff in false_fires[:30]: print("  ", ff)
