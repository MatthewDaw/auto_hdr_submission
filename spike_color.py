"""SPIKE: does exposure-invariant rg-chromaticity separate wrong-merge scenes?

For each wrong-merge case (two different GT groups fused into one of our clusters),
compute a per-frame rg-chromaticity histogram on well-exposed pixels, then compare
WITHIN-group vs BETWEEN-group histogram distance. If between >> within, a color veto
would correctly refuse the merge that grayscale edge-ZNCC accepted.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ddir = Path("data/full_subset")
col = np.load(ddir / "img128c.npz", allow_pickle=True)
cfiles = list(col["files"]); cimgs = col["imgs"]  # (N,128,128,3) BGR or RGB
cidx = {f: i for i, f in enumerate(cfiles)}

gt = defaultdict(list)
for row in csv.DictReader(open(ddir / "public_manifest.csv", encoding="utf-8")):
    gt[row["group_id"]].append(row["filename"])

BINS = 8
def gray_of(img):
    return img.astype(np.float32).mean(2)

def chroma_hist(img):
    """rg-chromaticity histogram on MID-exposed pixels only. img is HxWx3 uint8."""
    x = img.astype(np.float32)
    s = x.sum(2) + 1e-6
    gray = x.mean(2)
    m = (gray > 70) & (gray < 185)          # mid band (tightest exposure invariance)
    if m.sum() < 200:
        return None
    r = (x[..., 0] / s)[m]; g = (x[..., 1] / s)[m]
    h, _, _ = np.histogram2d(r, g, bins=BINS, range=[[0, 1], [0, 1]])
    h = h.ravel(); h /= (h.sum() + 1e-9)
    return h

def chi2(a, b):
    return 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9))

def hist_for(f):
    if f not in cidx: return None
    return chroma_hist(cimgs[cidx[f]])

def mid_band(frames):
    """frames whose grayscale mean is in the mid band, sorted toward 128."""
    out = []
    for f in frames:
        if f not in cidx: continue
        gm = float(gray_of(cimgs[cidx[f]]).mean())
        if 70 < gm < 185: out.append((abs(gm - 128), f))
    return [f for _, f in sorted(out)]

# wrong-merge cases: (ours-cluster spans these two GT groups)
cases = [("12226", "13169"), ("10464", "10613"), ("10886", "10593")]
print(f"rg-chromaticity, MID-band only ({BINS}x{BINS} bins), same-exposure-level comparison\n")
for a, b in cases:
    fa, fb = mid_band(gt[a]), mid_band(gt[b])
    ha = [h for f in fa if (h := hist_for(f)) is not None]
    hb = [h for f in fb if (h := hist_for(f)) is not None]
    if len(ha) < 2 or len(hb) < 2:
        print(f"{a} vs {b}: insufficient mid-band frames ({len(ha)},{len(hb)})"); continue
    # within = spread among same-scene mid-band frames (already same exposure level)
    within_a = np.mean([chi2(ha[i], ha[j]) for i in range(len(ha)) for j in range(i+1, len(ha))])
    within_b = np.mean([chi2(hb[i], hb[j]) for i in range(len(hb)) for j in range(i+1, len(hb))])
    between = np.mean([chi2(x, y) for x in ha for y in hb])
    within = (within_a + within_b) / 2
    sep = between / (within + 1e-9)
    flag = "SEPARABLE" if sep > 2.0 else "weak"
    print(f"GT {a} ({len(ha)}f) vs GT {b} ({len(hb)}f): "
          f"within={within:.4f}  between={between:.4f}  ratio={sep:.2f}x  abs_between={between:.3f} -> {flag}")
