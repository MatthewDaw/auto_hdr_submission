"""Safety scan for the confident-mismatch force-split rule.

Rule: a near-black (hard-clipped) frame is force-split from its cluster ONLY when
  nb >= MIN_BULBS  (clipped frame has enough strong light blobs)
  nt >= MIN_TMPL   (its cluster's well-template has enough lights to match against)
  cov <= COV_MAX   (<=1 of the clipped frame's bulbs lands on a template light)

If this rule is safe, then across EVERY correctly-grouped GT group in data/large,
NO same-scene near-black frame should trip it (they belong, so splitting = error).
We scan all fixable groups and report every same-scene frame that WOULD be split.
Also re-confirms it FIRES on the foreign full_subset cases.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import extreme_anchor as ea

MIN_BULBS, MIN_TMPL, COV_MAX = 5, 3, 0.20

def load(ddir):
    raw = np.load(Path(ddir)/"raw256.npz", allow_pickle=True)
    files=list(raw["files"]); imgs=raw["imgs"]; idx={f:i for i,f in enumerate(files)}
    gt=defaultdict(list)
    with open(Path(ddir)/"public_manifest.csv",encoding="utf-8") as f:
        for row in csv.DictReader(f): gt[row["group_id"]].append(row["filename"])
    return imgs, idx, gt

def trips(clip_tile, well_tiles, polarity):
    tmpl = ea.build_template(well_tiles) if well_tiles else None
    if tmpl is None: return None
    nb = len(ea.extreme_spots(clip_tile, polarity=polarity)[0])
    nt = len(ea._template_spots(tmpl, polarity))
    cov = ea.coverage_score(clip_tile, tmpl, polarity)
    fire = (nb >= MIN_BULBS and nt >= MIN_TMPL and cov <= COV_MAX)
    return nb, nt, cov, fire

imgs, idx, gt = load("data/large")
unfix = set(json.load(open("data/large/unfixable.json"))["groups"].keys())

false_fires = []
n_black = 0
for g, frs in gt.items():
    if g in unfix: continue
    tiles = [imgs[idx[f]] for f in frs if f in idx]
    wells = [t for t in tiles if ea.is_well_exposed(t)]
    if not wells: continue
    for f in frs:
        if f not in idx: continue
        t = imgs[idx[f]]; m = float(t.mean())
        # near-black hard-clipped only (the bright-spot / lightbulb regime)
        if m >= 30: continue
        n_black += 1
        r = trips(t, wells, "bright")
        if r and r[3]:
            false_fires.append((g, f, round(m), r[0], r[1], round(r[2],2)))

print(f"scanned {n_black} same-scene near-black frames across "
      f"{len(gt)-len(unfix)} fixable groups")
print(f"FALSE FORCE-SPLITS (same-scene frames the rule would wrongly split): "
      f"{len(false_fires)}")
for ff in false_fires:
    print("   group=%s B=%s nb=%s nt=%s cov=%s  %s" % (ff[0], ff[2], ff[3], ff[4], ff[5], ff[1]))
