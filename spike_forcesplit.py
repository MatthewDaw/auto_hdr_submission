"""Force-split discriminator probe for super-low-exposure (near-black) frames.

Question: can we FORCE a split when a near-black frame's light sources clearly do
NOT match the cluster's well-exposed template, WITHOUT false-firing on info-limited
same-scene frames (the 22 regressions the naive low-coverage rule broke)?

Key idea (from the user): only act on POSITIVE mismatch evidence —
  * the clipped frame must HAVE >= MIN_BULBS strong, separated light blobs,
  * the template must HAVE lights to match against,
  * and coverage of those bulbs onto the template must be clearly LOW.
If the frame is info-limited (few bulbs), ABSTAIN (don't split).

We print, for each frame:
  nb   = # strong bright blobs in the clipped frame (real light sources)
  nt   = # template light peaks
  cov  = coverage_score (fraction of clipped bulbs landing on a template light)
for two populations:
  FOREIGN (want to SPLIT): the foreign near-black frame vs the OTHER group's template
  SAME    (must NOT split): the lone near-black frame the naive rule wrongly spawned,
                            vs its OWN group's template
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import extreme_anchor as ea

def load(ddir):
    raw = np.load(Path(ddir)/"raw256.npz", allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return imgs, idx, gt

def nbulbs(tile, polarity="bright"):
    spots,_ = ea.extreme_spots(tile, polarity=polarity)
    return len(spots)

def stats(clip_tile, well_tiles, polarity="bright"):
    tmpl = ea.build_template(well_tiles) if well_tiles else None
    if tmpl is None: return None
    nb = nbulbs(clip_tile, polarity)
    nt = len(ea._template_spots(tmpl, polarity))
    cov = ea.coverage_score(clip_tile, tmpl, polarity)
    return nb, nt, cov

# ---- FOREIGN cases: (host_group, foreign_group) both in data/full_subset ----
FOREIGN = [("10280","1038"),("10613","10464"),("11533","11604"),
           ("14037","14288"),("14983","14279")]
imgs, idx, gt = load("data/full_subset")
def darkest(g, lo=30):
    fr=[f for f in gt[g] if f in idx and imgs[idx[f]].mean()<lo]
    return sorted(fr, key=lambda f: imgs[idx[f]].mean())
def wells(g):
    return [imgs[idx[f]] for f in gt[g] if f in idx and ea.is_well_exposed(imgs[idx[f]])]

print("=== FOREIGN near-black frame vs OTHER group's template (want SPLIT) ===")
print("   host  foreign   B   nb  nt  cov")
for host, foreign in FOREIGN:
    wt = wells(host)
    for f in darkest(foreign)[:2]:
        s = stats(imgs[idx[f]], wt)
        if s: print(f"   {host:>6} {foreign:>6}  {imgs[idx[f]].mean():4.0f}  {s[0]:2d}  {s[1]:2d}  {s[2]:.2f}")

# ---- SAME-scene info-limited frames (the 24 regressions) in data/large ------
imgs2, idx2, gt2 = load("data/large")
def darkest2(g, lo=30):
    fr=[f for f in gt2[g] if f in idx2 and imgs2[idx2[f]].mean()<lo]
    return sorted(fr, key=lambda f: imgs2[idx2[f]].mean())
def wells2(g):
    return [imgs2[idx2[f]] for f in gt2[g] if f in idx2 and ea.is_well_exposed(imgs2[idx2[f]])]

REG = ['23690','22534','49265','9635','40648','5332','65892','89342','63350',
       '34588','44888','54656','77535','5125','63560','69279','93065']
print("\n=== SAME-scene near-black frame vs its OWN template (must NOT split) ===")
print("   group     B   nb  nt  cov")
for g in REG:
    wt = wells2(g)
    for f in darkest2(g)[:1]:
        s = stats(imgs2[idx2[f]], wt)
        if s: print(f"   {g:>6}  {imgs2[idx2[f]].mean():4.0f}  {s[0]:2d}  {s[1]:2d}  {s[2]:.2f}")
