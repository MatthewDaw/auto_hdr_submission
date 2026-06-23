"""SPIKE (separated — no autohdr/ edits): fix the clipped-strand reattachment gap.

Root cause: OrphanReattachment._accept rejects any match whose exposure gap is
< _MIN_STEP (25), even when masked edge-ZNCC is near-certain (0.9). So a frame
that is unambiguously the same scene strands because its exposure step is small.

Proposed rule: a near-certain match (ZNCC >= 0.85 at large overlap) reattaches
regardless of exposure gap. Tested here by monkeypatching _accept at runtime and
re-running the full grouper on the strand groups.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

import autohdr.refinement.orphan_reattachment as orph

_orig = orph._accept
def _accept_patched(zncc, overlap, gap, step):
    # near-certain same scene: reattach regardless of exposure step
    if zncc >= 0.85 and overlap >= 15000:
        return True
    return _orig(zncc, overlap, gap, step)

from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot

ddir = Path("data/full_subset")
raw = np.load(ddir / "raw256.npz", allow_pickle=True)
col = np.load(ddir / "img128c.npz", allow_pickle=True)
files = list(raw["files"]); imgs = raw["imgs"]; idx = {f: i for i, f in enumerate(files)}
cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
gt = defaultdict(list)
for row in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
    gt[row["group_id"]].append(row["filename"])

def run(g, patched):
    orph._accept = _accept_patched if patched else _orig
    fr = [f for f in gt[g] if f in idx]
    gray = np.stack([imgs[idx[f]] for f in fr])
    color = np.stack([cimgs[cidx[f]] for f in fr]) if all(f in cidx for f in fr) else None
    out = ImageGrouper().group(Photoshoot(fr, gray, color))
    return len(fr), len(out)

for g in ["10125", "10129", "10463"]:
    n, before = run(g, False)
    _, after = run(g, True)
    tag = "FIXED->1" if after == 1 else f"{before}->{after}"
    print(f"{g}: {n} frames | baseline={before} clusters | patched={after} clusters  [{tag}]")
