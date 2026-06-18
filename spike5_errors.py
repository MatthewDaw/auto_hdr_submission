"""
Spike 5: error analysis. At the best CLAHE pipeline, print EXACTLY which
reference groups we get wrong and how (over-split into pieces, or over-merged
with other groups). This tells us what to fix to reach perfection.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample"); IMG_DIR = SAMPLE / "images"
SIZE = 256; ZNCC = 64; THR = 0.44
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def gradmap(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = clahe.apply(g).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    z = cv2.resize(mag, (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); z /= (np.linalg.norm(z) + 1e-9)
    return z

def main():
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    Z = np.array([gradmap(IMG_DIR / f) for f in files])
    sim = Z @ Z.T
    A = sim >= THR; np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    predsets = [frozenset(v) for v in pred.values()]
    refsets = {gid: frozenset(v) for gid, v in groups.items()}
    predset_lookup = set(predsets)

    # which predicted cluster each file landed in
    file2pred = {}
    for cid, fs in pred.items():
        for f in fs: file2pred[f] = cid

    print(f"thr={THR}: {sum(1 for s in refsets.values() if s in predset_lookup)}/{len(refsets)} exact\n")
    short = lambda f: f.split('_',1)[0]  # gNNN prefix = true group, for readability
    for gid, rs in refsets.items():
        if rs in predset_lookup:
            continue
        # how did this ref group get partitioned across predicted clusters?
        clusters = defaultdict(list)
        for f in rs: clusters[file2pred[f]].append(f)
        print(f"REF group {gid} ({len(rs)} imgs) FAILED:")
        if len(clusters) > 1:
            print(f"  OVER-SPLIT into {len(clusters)} predicted clusters")
        for cid, fs in clusters.items():
            others = pred[cid] - rs
            tag = "" if not others else f"  +CONTAMINATED with {len(others)} imgs from groups {sorted(set(short(o) for o in others))}"
            print(f"    cluster {cid}: {len(fs)} of these imgs{tag}")
        # show intra-group sim range to see if it's a chaining (exposure) failure
        idx = [files.index(f) for f in rs]
        sub = sim[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        print(f"    intra-group ZNCC: min={sub[iu].min():.3f} mean={sub[iu].mean():.3f} (thr={THR})\n")

if __name__ == "__main__":
    main()
