"""Validate MotionSplitter (FIX10) on Cluster C: localized in-scene motion.

For each of the 5 motion cases we build the over-merged cluster (the two GT
groups' frames), run MotionSplitter in isolation, and confirm it partitions into
exactly the two correct GT groups. We then confirm the user's named safety set
(genuine single brackets + the praised 14258-14262 triplets) stays one cluster.
Finally we print the motion-score distribution: the 5 cases vs a genuine
same-scene SAME-LEVEL baseline (real same-scene pairs + self+sensor-noise), with
the threshold and margin.

Read-only on data. Checks data/full_subset and data/large; reports missing cases.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from autohdr.features import MaskedCorrelation
from autohdr.refinement.context import RefinementContext
from autohdr.refinement.motion_splitter import (
    MotionSplitter,
    _B_MIN_AREA,
    _B_MIN_DIM,
    _B_MIN_FILL,
    _B_STRONG_DIM,
    _B_STRONG_LOC,
    _B_WIDE_DIM,
    _B_WIDE_LOC,
    _A_ZNCC,
    _A_MIN_OVERLAP,
)

CASES = [
    ("10600", "10601", "curtain"),
    ("1087", "1093", "pool"),
    ("11573", "11574", "window"),
    ("13339", "13340", "door"),
    ("17879", "17880", "finger"),
]
SAFETY = [
    "10147", "10160", "13601", "13609", "12583", "13093", "13586", "13590",
    "14011", "14258", "14259", "14260", "14261", "14262",
]


def load(ddir: Path):
    raw = np.load(ddir / "raw256.npz", allow_pickle=True)
    files = list(raw["files"])
    imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    gt = defaultdict(list)
    with open(ddir / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["group_id"]].append(row["filename"])
    return files, imgs, idx, gt


def make_ctx(frames, imgs, idx):
    """Build a RefinementContext over just these frames (row order = list order)."""
    gray = np.stack([imgs[idx[f]] for f in frames])
    brightness = gray.reshape(len(frames), -1).mean(axis=1)
    return RefinementContext(
        brightness=brightness,
        embedding=None,
        masked=MaskedCorrelation(gray),
        filenames=list(frames),
        gray=gray,
        chroma=None,
    )


def run_case(a, b, gt, imgs, idx):
    fa = [f for f in gt.get(a, []) if f in idx]
    fb = [f for f in gt.get(b, []) if f in idx]
    if not fa or not fb:
        return None
    frames = fa + fb
    ctx = make_ctx(frames, imgs, idx)
    labels = np.zeros(len(frames), dtype=int)  # one merged cluster
    out = MotionSplitter().apply(labels, ctx)
    groups = defaultdict(set)
    for f, l in zip(frames, out):
        groups[int(l)].add(f)
    parts = list(groups.values())
    correct = (
        len(parts) == 2
        and {frozenset(p) for p in parts} == {frozenset(fa), frozenset(fb)}
    )
    return parts, correct, set(fa), set(fb)


def run_safety(g, gt, imgs, idx):
    frames = [f for f in gt.get(g, []) if f in idx]
    if not frames:
        return None
    ctx = make_ctx(frames, imgs, idx)
    labels = np.zeros(len(frames), dtype=int)
    out = MotionSplitter().apply(labels, ctx)
    return len(set(out.tolist()))


# --- motion-score diagnostics (mirror MotionSplitter's internals) -----------
def motion_scores(frames, imgs, idx):
    """Best (max) per-track scores over all same-level well pairs of a frame set."""
    ms = MotionSplitter()
    ctx = make_ctx(frames, imgs, idx)
    B = ctx.brightness
    best = {"zncc": 1.0, "area": 0, "loc": 0.0, "mindim": 0, "fill": 0.0}
    members = list(range(len(frames)))
    for i, j in ms._same_level_pairs(members, B):
        z, ov = ctx.masked.score(i, j)
        if ov >= _A_MIN_OVERLAP:
            best["zncc"] = min(best["zncc"], z)
        blob = ms._changed_blob(i, j, ctx.gray)
        if blob and blob["area"] > best["area"]:
            best.update(area=blob["area"], loc=blob["loc"],
                        mindim=blob["mindim"], fill=blob["fill"])
    return best


def main():
    for dname in ["data/full_subset", "data/large"]:
        ddir = Path(dname)
        if not (ddir / "raw256.npz").exists():
            print(f"\n##### {dname}: MISSING (no raw256.npz) #####")
            continue
        print(f"\n##### {dname} #####")
        _, imgs, idx, gt = load(ddir)

        print("\n=== CASES: split into the correct two GT groups? ===")
        n_present = n_ok = 0
        case_scores = {}
        for a, b, what in CASES:
            res = run_case(a, b, gt, imgs, idx)
            if res is None:
                print(f"  {what} (GT {a}+{b}): MISSING in {dname}")
                continue
            n_present += 1
            parts, correct, sa, sb = res
            sc = motion_scores([f for f in gt[a] + gt[b] if f in idx], imgs, idx)
            case_scores[what] = sc
            n_ok += correct
            sizes = "/".join(str(len(p)) for p in parts)
            print(f"  {what} (GT {a}+{b}): split={len(parts)} sizes={sizes} "
                  f"CORRECT={correct}  [trackA zncc={sc['zncc']:.2f} | "
                  f"trackB area={sc['area']} mindim={sc['mindim']} "
                  f"fill={sc['fill']:.2f} loc={sc['loc']:.1f}]")
        if n_present:
            print(f"  -> {n_ok}/{n_present} cases correct")

        print("\n=== SAFETY: each must stay 1 cluster ===")
        safe_ok = True
        for g in SAFETY:
            n = run_safety(g, gt, imgs, idx)
            if n is None:
                print(f"  {g}: (not in {dname})")
                continue
            ok = n == 1
            safe_ok &= ok
            print(f"  {g}: {n} cluster(s) {'OK' if ok else '*** SPLIT! ***'}")
        print(f"  -> safety {'ALL OK' if safe_ok else 'VIOLATED'}")

        # genuine-baseline (rich data only) + real per-cluster regression scan
        if dname.endswith("full_subset"):
            baseline_report(imgs, idx, gt, case_scores)
        cluster_regression_scan(ddir, imgs, idx)


def baseline_report(imgs, idx, gt, case_scores):
    """Genuine same-scene SAME-LEVEL baseline: real same-scene pairs + self+noise.
    Confirms the cases clear the threshold while genuine pairs stay below it."""
    ms = MotionSplitter()
    print("\n=== GENUINE same-scene SAME-LEVEL baseline (must NOT trip a track) ===")

    def B(f):
        return float(imgs[idx[f]].mean())

    # real same-scene same-level well pairs, evaluated per GT group (note: in the
    # live pipeline these pairs only co-occur in a cluster when GT merged them, so
    # an isolated-group trip here is NOT a regression — see cluster_regression_scan)
    a_trip = b_trip = total = 0
    worst_loc = 0.0
    worst_area = 0
    min_zncc = 1.0
    for g, fs in gt.items():
        fs = [f for f in fs if f in idx and 45 <= B(f) <= 215]
        if len(fs) < 2:
            continue
        ms._norm.clear()  # per-group: indices restart at 0, drop stale cache
        ctx = make_ctx(fs, imgs, idx)
        Bs = ctx.brightness
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if abs(Bs[i] - Bs[j]) > 35:
                    continue
                total += 1
                z, ov = ctx.masked.score(i, j)
                if ov >= _A_MIN_OVERLAP:
                    min_zncc = min(min_zncc, z)
                    if z < _A_ZNCC:
                        a_trip += 1
                blob = ms._changed_blob(i, j, ctx.gray)
                if blob:
                    worst_area = max(worst_area, blob["area"])
                    worst_loc = max(worst_loc, blob["loc"])
                    if ms._is_object(blob):
                        b_trip += 1
    print(f"  genuine same-level pairs scanned: {total}")
    print(f"  Track A: min genuine ZNCC={min_zncc:.2f} (thr <{_A_ZNCC}); "
          f"tripped {a_trip} pairs")
    print(f"  Track B: worst genuine area={worst_area} loc={worst_loc:.1f}; "
          f"tripped {b_trip} pairs")
    if case_scores:
        case_locs = [s["loc"] for s in case_scores.values()]
        case_zncc = [s["zncc"] for s in case_scores.values()]
        print(f"  cases: ZNCC range {min(case_zncc):.2f}-{max(case_zncc):.2f}, "
              f"loc range {min(case_locs):.1f}-{max(case_locs):.1f}")
    # self + sensor noise: a genuine still scene must never trip Track B
    rng = np.random.default_rng(0)
    worst = 0
    for g in SAFETY + ["11393", "14731", "10147"]:
        for f in [x for x in gt.get(g, []) if x in idx][:6]:
            base = imgs[idx[f]]
            if not (45 <= base.mean() <= 215):
                continue
            for s in (2.0, 3.0, 4.0):
                noisy = np.clip(base.astype(np.float32) + rng.normal(0, s, base.shape),
                                0, 255).astype(np.uint8)
                two = np.stack([base, noisy])
                ms._norm.clear()  # stale-cache guard: same indices, different imgs
                blob = ms._changed_blob(0, 1, two)
                if blob and ms._is_object(blob):
                    worst = max(worst, blob["area"])
    print(f"  self+noise Track-B trips: {'NONE' if worst == 0 else worst}")
    print(f"\n  THRESHOLDS  TrackA: zncc<{_A_ZNCC} @ overlap>={_A_MIN_OVERLAP}")
    print(f"              TrackB: area>={_B_MIN_AREA} fill>={_B_MIN_FILL} "
          f"mindim>={_B_MIN_DIM}; (loc>={_B_STRONG_LOC} & dim>={_B_STRONG_DIM}) "
          f"OR (dim>={_B_WIDE_DIM} & loc>={_B_WIDE_LOC})")


def cluster_regression_scan(ddir: Path, imgs, idx):
    """The metric that actually matters: run MotionSplitter on every REAL pipeline
    cluster (from pred_labels.json) and count single-GT-group clusters that split
    (= regressions) vs multi-GT clusters that split (= over-merges it correctly
    repairs). The case clusters are excluded (they are the intended targets)."""
    import json

    pred = ddir / "pred_labels.json"
    if not pred.exists():
        print("\n  (no pred_labels.json — skipping per-cluster regression scan)")
        return
    labels_map = json.load(open(pred))
    clusters = defaultdict(list)
    for f, l in labels_map.items():
        if f in idx:
            clusters[int(l)].append(f)
    case_clusters = {520, 773, 1368, 2901, 6730}
    unfix = set()
    up = ddir / "unfixable.json"
    if up.exists():
        unfix = set(json.load(open(up)).get("groups", {}).keys())
    ms = MotionSplitter()
    reg = correct = excluded = 0
    reg_groups = []
    for l, frames in clusters.items():
        if l in case_clusters or len(frames) < 4:
            continue
        ctx = make_ctx(frames, imgs, idx)
        out = ms.apply(np.zeros(len(frames), int), ctx)
        if len(set(out.tolist())) < 2:
            continue
        gts = sorted({f.split("_")[0] for f in frames})
        if len(gts) == 1:
            g = gts[0].lstrip("g")
            if g in unfix:
                excluded += 1  # already a known GT error, outside the 1302 gate
            else:
                reg += 1
                reg_groups.append((l, gts[0]))
        else:
            correct += 1
    print("\n=== REAL per-cluster scan (pred_labels.json) ===")
    print(f"  single-GT clusters split, FIXABLE (REGRESSIONS): {reg} "
          f"{reg_groups if reg else ''}")
    print(f"  single-GT clusters split, in unfixable/excluded set: {excluded}")
    print(f"  multi-GT clusters split (over-merge repairs): {correct}")


def make_ctx_arr(gray_arr):
    brightness = gray_arr.reshape(len(gray_arr), -1).mean(axis=1)
    return RefinementContext(
        brightness=brightness, embedding=None,
        masked=MaskedCorrelation(gray_arr),
        filenames=[f"x{i}.jpg" for i in range(len(gray_arr))],
        gray=gray_arr, chroma=None,
    )


if __name__ == "__main__":
    main()
