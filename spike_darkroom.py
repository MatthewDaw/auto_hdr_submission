"""DARK-ROOM spike: coverage numbers for each clipped LINK/SPLIT case.

For LINK groups: build the cluster's well-template from its well-exposed frames,
score each clipped frame's coverage onto it.
For SPLIT pairs: build each GT group's well-template, score the OTHER group's
clipped frames against it (foreign frames should score LOW).
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import extreme_anchor as _ea


def load(ddir):
    raw = np.load(Path(ddir) / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    gt = defaultdict(list)
    with open(Path(ddir) / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["group_id"]].append(row["filename"])
    return imgs, idx, gt


def tiles(g, imgs, idx, gt):
    return [imgs[idx[f]] for f in gt.get(g, []) if f in idx]


def describe(t):
    m = float(t.mean())
    pol = _ea.clip_polarity(t)
    tag = "WELL" if _ea.is_well_exposed(t) else ("CLIP-" + (pol or "?") if _ea.is_clipped(t) else "mid")
    return m, pol, tag


def link_case(name, ddir, imgs, idx, gt):
    ts = tiles(name, imgs, idx, gt)
    well = [t for t in ts if _ea.is_well_exposed(t)]
    tmpl = _ea.build_template(well) if well else None
    print(f"\nLINK {name} ({ddir}): {len(ts)} frames, {len(well)} well")
    if tmpl is None:
        print("  no template"); return
    for t in ts:
        m, pol, tag = describe(t)
        if _ea.is_clipped(t):
            cov = _ea.coverage_score(t, tmpl, pol)
            print(f"  mean={m:6.1f} {tag:11s} coverage={cov:.3f}")


def split_case(a, b, ddir, imgs, idx, gt):
    print(f"\nSPLIT {a} / {b} ({ddir})")
    for home, foreign in ((a, b), (b, a)):
        ts_h = tiles(home, imgs, idx, gt)
        well = [t for t in ts_h if _ea.is_well_exposed(t)]
        tmpl = _ea.build_template(well) if well else None
        if tmpl is None:
            print(f"  home {home}: no well template ({len(ts_h)} frames)"); continue
        # own clipped vs own template (should be HIGH)
        for t in ts_h:
            if _ea.is_clipped(t):
                pol = _ea.clip_polarity(t)
                print(f"  {home} own  clip mean={t.mean():6.1f} cov_own={_ea.coverage_score(t, tmpl, pol):.3f}")
        # foreign clipped vs home template (should be LOW)
        for t in tiles(foreign, imgs, idx, gt):
            if _ea.is_clipped(t):
                pol = _ea.clip_polarity(t)
                print(f"  {foreign}->{home} foreign clip mean={t.mean():6.1f} cov_foreign={_ea.coverage_score(t, tmpl, pol):.3f}")


LINK = [("73234", "data/large"), ("10125", "data/full_subset"),
        ("10129", "data/full_subset"), ("10463", "data/full_subset"),
        ("11992", "data/full_subset"), ("19300", "data/full_subset")]
SPLIT = [("10280", "1038"), ("10464", "10613"), ("11533", "11604"),
         ("14037", "14288"), ("14279", "14983")]

cache = {}
def get(ddir):
    if ddir not in cache:
        cache[ddir] = load(ddir)
    return cache[ddir]

for name, ddir in LINK:
    link_case(name, ddir, *get(ddir))
for a, b in SPLIT:
    split_case(a, b, "data/full_subset", *get("data/full_subset"))
