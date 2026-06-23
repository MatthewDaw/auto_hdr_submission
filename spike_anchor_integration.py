"""Spike: compare CURRENT clipped-vs-clipped anchor_match vs PROPOSED
clipped-vs-well-template coverage_score on the named link/split cases.

For each case we load the GT group(s)' frames from whichever data dir has them,
build the cluster's well-template from its well-exposed members, and for each
clipped frame print:
  - current best anchor_match to any cluster member (clipped-vs-clipped)
  - proposed coverage_score against each cluster's well-template
LINK cases: clipped orphan should cover its OWN scene's template best.
SPLIT cases: a frame from scene B should cover scene A's template LOW (and vice).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from autohdr.features import extreme_anchor as EA
from autohdr.refinement import anchor_splitter as AS

DATA = [Path("data/full_subset"), Path("data/large")]


def load_dir(d):
    raw = np.load(d / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    gt = defaultdict(list)
    with open(d / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["group_id"]].append(row["filename"])
    return imgs, idx, gt


DIRS = [(d,) + load_dir(d) for d in DATA if (d / "raw256.npz").exists()]


def frames_for(gid):
    for d, imgs, idx, gt in DIRS:
        fr = [f for f in gt.get(gid, []) if f in idx]
        if fr:
            return np.stack([imgs[idx[f]] for f in fr]), d.name
    return None, None


def well_template(tiles):
    well = [t for t in tiles if EA.is_well_exposed(t)]
    if not well:
        # fall back to nearest-mid-grey frames
        order = sorted(tiles, key=lambda t: abs(EA.brightness(t) - 128))
        well = order[:max(1, len(order) // 2)]
    return EA.build_template(well)


def clipped_idx(tiles):
    return [i for i, t in enumerate(tiles) if EA.is_clipped(t)]


def cur_anchor(a, b):
    return AS.anchor_match(a, b)


print("=" * 70)
print("LINK cases (clipped orphan should cover its OWN scene's template)")
print("=" * 70)
for gid in ["73234", "10125", "10129", "10463", "11992", "19300"]:
    tiles, src = frames_for(gid)
    if tiles is None:
        print(f"{gid}: NOT FOUND"); continue
    tmpl = well_template(list(tiles))
    cidxs = clipped_idx(tiles)
    print(f"\n[{gid}] ({src}) n={len(tiles)} clipped={len(cidxs)} "
          f"means={[round(EA.brightness(t)) for t in tiles]}")
    if tmpl is None:
        print("  no template"); continue
    for ci in cidxs:
        pol = EA.clip_polarity(tiles[ci])
        cov = EA.coverage_score(tiles[ci], tmpl, pol)
        # current: best clipped-vs-any-member
        cur = max(cur_anchor(tiles[ci], tiles[j]) for j in range(len(tiles)) if j != ci)
        print(f"  frame{ci} mean={round(EA.brightness(tiles[ci]))} pol={pol} "
              f"cov(own)={cov:.3f}  cur_anchor(best)={cur:.3f}")


print("\n" + "=" * 70)
print("SPLIT cases (cross-scene coverage should be LOW; own-scene HIGH)")
print("=" * 70)
splits = [("10280", "1038"), ("10464", "10613"), ("11533", "11604"),
          ("14037", "14288"), ("14279", "14983"), ("12226", "13169")]
for a, b in splits:
    ta, sa = frames_for(a)
    tb, sb = frames_for(b)
    if ta is None or tb is None:
        print(f"\n[{a}/{b}] missing ({a}:{ta is not None} {b}:{tb is not None})")
        continue
    tmA = well_template(list(ta))
    tmB = well_template(list(tb))
    print(f"\n[{a}/{b}] ({sa}/{sb}) nA={len(ta)} nB={len(tb)}")
    print(f"  A means={[round(EA.brightness(t)) for t in ta]}")
    print(f"  B means={[round(EA.brightness(t)) for t in tb]}")
    for label, tiles, own, other in [("A", ta, tmA, tmB), ("B", tb, tmB, tmA)]:
        for ci in clipped_idx(tiles):
            pol = EA.clip_polarity(tiles[ci])
            cov_own = EA.coverage_score(tiles[ci], own, pol) if own else float("nan")
            cov_oth = EA.coverage_score(tiles[ci], other, pol) if other else float("nan")
            # current cross-scene clipped-vs-clipped
            other_tiles = tb if label == "A" else ta
            cur_cross = max((cur_anchor(tiles[ci], ot) for ot in other_tiles), default=0.0)
            print(f"  {label}.frame{ci} mean={round(EA.brightness(tiles[ci]))} "
                  f"pol={pol} cov_own={cov_own:.3f} cov_other={cov_oth:.3f} "
                  f"cur_anchor_cross={cur_cross:.3f}")
