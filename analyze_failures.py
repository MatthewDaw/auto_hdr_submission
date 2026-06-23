"""Analyze the user's flagged model-failure groups: locate each in its dataset,
show how it fails, and compute TRANSITIVE partners (any other GT group sharing an
erroneous cluster — if our merge is wrong, that group's cluster is wrong too)."""
import csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

TARGETS = {
    "low_exposure": ["14309","15406","14372","14342","14333","14330","14279",
                     "14037","10464","10613","10593","10886"],
    "high_exposure": ["15399","11992"],
    "low_freq_scene": ["14332","14329"],
}
flat = [g for v in TARGETS.values() for g in v]

DATASETS = ["data/full_subset", "data/large", "sample"]


def load(d):
    p = Path(d)
    gt = {}
    with open(p / "public_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["filename"]] = r["group_id"]
    pred = json.load(open(p / "pred_labels.json"))
    raw = np.load(p / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    B = {f: float(imgs[idx[f]].mean()) for f in files if f in idx}
    return gt, pred, B


# locate each target's dataset
loc = {}
data = {}
for d in DATASETS:
    if not (Path(d) / "pred_labels.json").exists():
        continue
    gt, pred, B = load(d)
    data[d] = (gt, pred, B)
    groups = set(gt.values())
    for g in flat:
        if g in groups and g not in loc:
            loc[g] = d

print("=== location of each target ===")
for g in flat:
    print(f"  {g}: {loc.get(g, 'NOT FOUND')}")

# per dataset, build cluster->gtgroups and analyze each target + transitive partners
transitive = defaultdict(set)  # dataset -> set of extra gt groups pulled in
for d, (gt, pred, B) in data.items():
    gt_groups = defaultdict(list)
    for f, g in gt.items():
        gt_groups[g].append(f)
    clus = defaultdict(list)
    for f, c in pred.items():
        clus[c].append(f)
    targets_here = [g for g in flat if loc.get(g) == d]
    if not targets_here:
        continue
    print(f"\n========== {d} ==========")
    for g in targets_here:
        frames = gt_groups[g]
        cs = sorted({pred[f] for f in frames if f in pred})
        print(f"\n[{g}] {len(frames)} frames, B={sorted(round(B.get(f,-1)) for f in frames)}")
        partners = set()
        for c in cs:
            comp = Counter(gt[f] for f in clus[c])
            others = {gg: n for gg, n in comp.items() if gg != g}
            tag = "OURS-pure" if not others else f"MIXED with {dict(others)}"
            print(f"    cluster {c}: {len(clus[c])} frames, GT-composition={dict(comp)} -> {tag}")
            partners |= set(others.keys())
        # also: foreign frames sitting in clusters OWNED by g? captured above as MIXED
        if len(cs) > 1 and not partners:
            print(f"    => OVER-SPLIT into {len(cs)} own clusters (no other GT group)")
        if partners:
            print(f"    => TRANSITIVE partners (also wrong): {sorted(partners)}")
            transitive[d] |= partners

print("\n=== TRANSITIVE PARTNERS to ADD (not already in the flagged list) ===")
allflag = set(flat)
for d, parts in transitive.items():
    extra = sorted(parts - allflag)
    print(f"  {d}: {extra}")
