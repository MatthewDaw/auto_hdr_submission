"""SPIKE: separation benchmark for autohdr.features.extreme_anchor.

POSITIVE pair = a CLIPPED frame matched to a SceneTemplate built from its OWN
group's well-exposed frames (want HIGH coverage).
NEGATIVE pair = a CLIPPED frame matched to a SceneTemplate built from a DIFFERENT
group's well-exposed frames (want LOW coverage).

Covers both polarities (bright spots for near-black frames, dark spots for
near-white frames). Reports the coverage-score distributions and whether a clean
threshold-with-margin separates positives from negatives.

If 256 resolution overlaps, re-localize the clipped frame from a higher-res decode
of the original image and re-report.
"""
import csv, json, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autohdr.features import extreme_anchor as ea

random.seed(0); np.random.seed(0)

DATA_DIRS = [Path("data/full_subset"), Path("data/large")]


def load(d):
    raw = np.load(d / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    gt = defaultdict(list)
    for r in csv.DictReader(open(d / "public_manifest.csv", encoding="utf-8")):
        gt[r["group_id"]].append(r["filename"])
    return dict(dir=d, imgs=imgs, idx=idx, gt=gt,
                imgdir=d / "images")


DS = {d.name: load(d) for d in DATA_DIRS if (d / "raw256.npz").exists()}


def find_group(g):
    for ds in DS.values():
        if g in ds["gt"] and any(f in ds["idx"] for f in ds["gt"][g]):
            return ds
    return None


def tiles(ds, g, lo=None, hi=None):
    out = []
    for f in ds["gt"].get(g, []):
        if f not in ds["idx"]:
            continue
        t = ds["imgs"][ds["idx"][f]]
        b = float(t.mean())
        if lo is not None and b < lo:
            continue
        if hi is not None and b > hi:
            continue
        out.append((f, t, b))
    return out


def well_tiles(ds, g):
    return [t for f, t, b in tiles(ds, g) if ea.WELL_LO <= b <= ea.WELL_HI]


def clipped_tiles(ds, g):
    out = []
    for f, t, b in tiles(ds, g):
        pol = ea.clip_polarity(t)
        if pol:
            out.append((f, t, b, pol))
    return out


def decode_hires(ds, fname, size):
    p = ds["imgdir"] / fname
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ----- build benchmark cases -----------------------------------------------------
OVERSPLIT = ["10125", "10129", "10463", "11992", "19300", "73234"]
OVERMERGE = [("10280", "1038"), ("10464", "10613"), ("11533", "11604"),
             ("14037", "14288"), ("14279", "14983")]


def build_cases():
    pos, neg = [], []  # each: (label, clip_fname, clip_tile, polarity, ds_for_template, well_group)

    # explicit over-split positives: clipped frame vs own well-exposed template
    for g in OVERSPLIT:
        ds = find_group(g)
        if ds is None:
            continue
        wt = well_tiles(ds, g)
        if not wt:
            continue
        for f, t, b, pol in clipped_tiles(ds, g):
            pos.append((f"split:{g}", f, t, pol, ds, g))

    # explicit over-merge negatives: clipped frame of group A vs well frames of group B
    for a, bgrp in OVERMERGE:
        dsa, dsb = find_group(a), find_group(bgrp)
        if dsa is None or dsb is None:
            continue
        wt_b = well_tiles(dsb, bgrp)
        wt_a = well_tiles(dsa, a)
        cl_a = clipped_tiles(dsa, a)
        cl_b = clipped_tiles(dsb, bgrp)
        # clipped of A vs well of B (and own well if available, as positive)
        for f, t, b, pol in cl_a:
            if wt_b:
                neg.append((f"merge:{a}->{bgrp}", f, t, pol, dsb, bgrp))
            if wt_a:
                pos.append((f"merge_own:{a}", f, t, pol, dsa, a))
        for f, t, b, pol in cl_b:
            if wt_a:
                neg.append((f"merge:{bgrp}->{a}", f, t, pol, dsa, a))
            if wt_b:
                pos.append((f"merge_own:{bgrp}", f, t, pol, dsb, bgrp))

    # random positives: groups with both a clipped and a well-exposed frame
    ds = DS.get("full_subset") or next(iter(DS.values()))
    cand = []
    for g in ds["gt"]:
        ct = clipped_tiles(ds, g)
        wt = well_tiles(ds, g)
        if ct and wt:
            cand.append((g, ct, wt))
    random.shuffle(cand)
    rand_pos = cand[:120]
    for g, ct, wt in rand_pos:
        for f, t, b, pol in ct[:2]:
            pos.append((f"rand:{g}", f, t, pol, ds, g))

    # random negatives: clipped frame vs a DIFFERENT random group's well template
    glist = [g for g, ct, wt in cand]
    for g, ct, wt in rand_pos:
        other = random.choice(glist)
        while other == g:
            other = random.choice(glist)
        for f, t, b, pol in ct[:2]:
            neg.append((f"randneg:{g}->{other}", f, t, pol, ds, other))

    return pos, neg


def score_case(case, hires=0):
    label, f, t, pol, ds, well_g = case
    size = 256 if not hires else hires
    wt = well_tiles(ds, well_g)
    if not wt:
        return None
    if hires:
        names = [ff for ff, tt, bb in tiles(ds, well_g) if ea.WELL_LO <= bb <= ea.WELL_HI]
        wt = [decode_hires(ds, n, size) for n in names]
        wt = [w for w in wt if w is not None]
        if not wt:
            return None
    tmpl = ea.build_template(wt, size=size)
    if tmpl is None:
        return None
    clip = t
    if hires:
        h = decode_hires(ds, f, size)
        clip = h if h is not None else cv2.resize(t, (size, size))
    return ea.coverage_score(clip, tmpl, polarity=pol)


def report(pos, neg, hires=0):
    pv = defaultdict(list); nv = defaultdict(list)
    pv_all = {"bright": [], "dark": []}
    nv_all = {"bright": [], "dark": []}
    for c in pos:
        s = score_case(c, hires)
        if s is not None:
            pv_all[c[3]].append((s, c[0], c[1]))
    for c in neg:
        s = score_case(c, hires)
        if s is not None:
            nv_all[c[3]].append((s, c[0], c[1]))

    tag = f"HIRES {hires}" if hires else "256"
    print(f"\n================ RESOLUTION: {tag} ================")
    for pol in ("bright", "dark"):
        P = [s for s, _, _ in pv_all[pol]]
        N = [s for s, _, _ in nv_all[pol]]
        print(f"\n--- polarity={pol}  (n_pos={len(P)} n_neg={len(N)}) ---")
        if not P or not N:
            print("  insufficient samples"); continue
        pmin, nmax = min(P), max(N)
        gap = pmin - nmax
        thr = (pmin + nmax) / 2
        print(f"  POS: min={pmin:.3f} median={np.median(P):.3f} max={max(P):.3f}")
        print(f"  NEG: min={min(N):.3f} median={np.median(N):.3f} max={nmax:.3f}")
        print(f"  margin (pos_min - neg_max) = {gap:+.3f}   chosen_thr={thr:.3f}")
        # error counts at chosen thr
        fn = sum(s < thr for s in P); fp = sum(s >= thr for s in N)
        print(f"  at thr={thr:.3f}: false_neg={fn}/{len(P)}  false_pos={fp}/{len(N)}")
        # overlapping cases
        ov_p = sorted([(s, lbl, f) for s, lbl, f in pv_all[pol] if s <= nmax])[:6]
        ov_n = sorted([(s, lbl, f) for s, lbl, f in nv_all[pol] if s >= pmin], reverse=True)[:6]
        if ov_p:
            print("  low positives:", [(round(s, 2), lbl) for s, lbl, f in ov_p])
        if ov_n:
            print("  high negatives:", [(round(s, 2), lbl) for s, lbl, f in ov_n])
    return pv_all, nv_all


if __name__ == "__main__":
    pos, neg = build_cases()
    print(f"assembled: {len(pos)} positive pairs, {len(neg)} negative pairs")
    pv, nv = report(pos, neg, hires=0)

    # if 256 overlaps in either polarity, try hires on clipped localization
    overlap = False
    for pol in ("bright", "dark"):
        P = [s for s, _, _ in pv[pol]]; N = [s for s, _, _ in nv[pol]]
        if P and N and min(P) <= max(N):
            overlap = True
    if overlap:
        print("\n256 overlaps -> retrying with HIRES 512 decode")
        report(pos, neg, hires=512)
