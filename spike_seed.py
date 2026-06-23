"""SPIKE (separated): seed-based scene assignment to beat THE WALL.

Connected-components fails on wrong-merges because one bridge edge (a bright frame
that's lost both edge and colour discrimination) reconnects two scenes. Seed-based
assignment is non-transitive:

  1. SEEDS: greedily collect mid-exposed frames that are pairwise "confidently a
     different scene" (low edge-ZNCC OR high same-exposure-level colour divergence).
     One seed => one scene, never split. Two+ => a real multi-scene cluster.
  2. ASSIGN: every frame goes to its NEAREST seed by masked edge-ZNCC; frames too
     clipped to score (bright/dark bridges) fall back to capture-order nearest seed.
     A bridge can only join ONE seed, so it can't re-merge the scenes.

Tested both directions: must SPLIT the wrong-merges, must NOT split single brackets.
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from autohdr.features import MaskedCorrelation, ChromaSignature

_WELL_LOW, _WELL_HIGH = 50, 205
_MID_LOW, _MID_HIGH = 70, 185
_EDGE_DIFF = 0.35      # edge-ZNCC below this (at overlap) => confidently different
_COLOR_LEVEL = 25      # only trust colour within this brightness gap
_COLOR_DIFF = 0.15     # same-level chroma divergence above this => different scene
_MIN_OVERLAP = 3000

def _seqnum(f):
    stem = re.sub(r"^g\d+_", "", f.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1

def seed_scenes(fr, gray, color):
    n = len(fr)
    B = gray.reshape(n, -1).mean(1)
    mc = MaskedCorrelation(gray); ch = ChromaSignature(color)
    def diff_score(i, j):
        """Evidence that i,j are different SCENES (0 = same): max of an edge term
        (only when confidently low) and a same-exposure-level colour term."""
        s = 0.0
        z, o = mc.score(i, j)
        if o >= _MIN_OVERLAP and z < _EDGE_DIFF:
            s = max(s, (_EDGE_DIFF - z) / _EDGE_DIFF)          # 0..1
        if _MID_LOW <= B[i] <= _MID_HIGH and _MID_LOW <= B[j] <= _MID_HIGH \
           and abs(B[i] - B[j]) <= _COLOR_LEVEL:
            d = ch.diverge(i, j)
            if d is not None and d >= _COLOR_DIFF:
                s = max(s, 1.0)                                # confident colour split
        return s
    mids = [k for k in range(n) if _MID_LOW <= B[k] <= _MID_HIGH]
    # SEEDS: start from the most-divergent mid pair (not greedily from centre — the
    # distinguishing evidence may live between two specific frames). Then add any
    # mid frame that is also different from every existing seed (k>2 scenes).
    best, bp = 0.0, None
    for a in range(len(mids)):
        for b in range(a + 1, len(mids)):
            s = diff_score(mids[a], mids[b])
            if s > best:
                best, bp = s, (mids[a], mids[b])
    if bp is None or best <= 0:
        return [list(range(n))]                  # one scene — no split
    seeds = list(bp)
    for k in mids:
        if k not in seeds and all(diff_score(k, s) > 0 for s in seeds):
            seeds.append(k)
    # assign every frame to its nearest seed
    groups = defaultdict(list)
    for i in range(n):
        scored = [(mc.score(i, s)[0], mc.score(i, s)[1], si) for si, s in enumerate(seeds)]
        usable = [(z, si) for z, o, si in scored if o >= _MIN_OVERLAP]
        if usable:
            best = max(usable)[1]
        else:  # too clipped to score — capture-order nearest seed
            best = min(range(len(seeds)), key=lambda si: abs(_seqnum(fr[i]) - _seqnum(fr[seeds[si]])))
        groups[best].append(i)
    return [g for g in groups.values() if g]

def load(d):
    d = Path(d)
    raw = np.load(d / "raw256.npz", allow_pickle=True); col = np.load(d / "img128c.npz", allow_pickle=True)
    return (list(raw["files"]), raw["imgs"], {f: i for i, f in enumerate(raw["files"])},
            {f: i for i, f in enumerate(col["files"])}, col["imgs"])

def run(d, fr, gt=None):
    files, imgs, idx, cidx, cimgs = d
    fr = [f for f in fr if f in idx and f in cidx]
    gray = np.stack([imgs[idx[f]] for f in fr]); color = np.stack([cimgs[cidx[f]] for f in fr])
    scenes = seed_scenes(fr, gray, color)
    desc = [sorted({gt[fr[i]] for i in s}) for s in scenes] if gt else [len(s) for s in scenes]
    return len(scenes), desc

try:
    FS = load("data/full_subset")
    pred = json.load(open("data/full_subset/pred_labels.json"))
    gtf = {}
    for row in csv.DictReader(open("data/full_subset/public_manifest.csv", encoding="utf-8")):
        gtf[row["filename"]] = row["group_id"]
    clmap = defaultdict(list)
    for f, c in pred.items(): clmap[c].append(f)
    print("WRONG-MERGES (want 2 scenes, cleanly split by GT group):")
    for a, b in [("12226", "13169"), ("10464", "10613"), ("10886", "10593")]:
        # find the cluster that currently holds frames from BOTH groups
        ca = {pred[f] for f in gtf if gtf[f] == a and f in pred}
        cb = {pred[f] for f in gtf if gtf[f] == b and f in pred}
        shared = ca & cb
        if not shared:
            print(f"  {a}+{b}: not merged in current labels (skip)"); continue
        cl = next(iter(shared))
        n, desc = run(FS, clmap[cl], gtf)
        print(f"  {a}+{b} (cluster {cl}, {len(clmap[cl])}f) -> {n} scenes: {desc}")
except Exception as e:
    print(f"[full_subset busy (sweep rewriting caches): {type(e).__name__}] — rerun after sweep")

LG = load("data/large")
gtl = {f: f.split("_", 1)[0][1:] for f in LG[0]}
gl = defaultdict(list)
for f in LG[0]: gl[f.split("_", 1)[0]].append(f)
print("SINGLE-BRACKET large groups (want 1 scene — no over-split):")
for g in ["g33301", "g48226", "g86981"]:
    n, desc = run(LG, sorted(gl[g]), gtl)
    print(f"  {g} -> {n} scenes")
