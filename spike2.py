"""
Spike 2: error breakdown + tiled-ZNCC for local-change sensitivity.

Adds a tiled ZNCC that takes the MIN correlation over a grid of tiles, so a
small localized scene change (moved chair, open door) tanks the score even
though the rest of the frame is identical. Compares against global ZNCC.
"""
import csv, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample"); IMG_DIR = SAMPLE / "images"
SIZE = 256
TILES = 4          # 4x4 grid -> 16 tiles
TSZ = 64           # each tile resized to 64x64 gradient map

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def descriptors(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # per-tile zero-mean unit-norm gradient vectors, shape (TILES*TILES, TSZ*TSZ)
    cs = SIZE // TILES
    vecs = []
    for i in range(TILES):
        for j in range(TILES):
            t = mag[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            t = cv2.resize(t, (TSZ, TSZ), interpolation=cv2.INTER_AREA).ravel()
            t = t - t.mean(); t /= (np.linalg.norm(t) + 1e-9)
            vecs.append(t)
    return np.array(vecs, np.float32)  # (16, 4096)

def score_and_errors(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    exact = ref & pred
    return len(exact)/len(ref), len(exact), len(ref), len(pred)

def main():
    t0 = time.time()
    groups, f2g = load_manifest()
    files = sorted(f2g.keys()); n = len(files)
    V = np.array([descriptors(IMG_DIR / f) for f in files])  # (n,16,4096)
    print(f"{n} imgs, embedded in {time.time()-t0:.1f}s")

    # global ZNCC = mean over tiles ; tiled-min ZNCC = min over tiles
    # pairwise via einsum per-tile correlation
    # corr[t] = V[:,t,:] @ V[:,t,:].T  -> stack -> (16,n,n)
    corr = np.stack([V[:, t, :] @ V[:, t, :].T for t in range(V.shape[1])])  # (16,n,n)
    glob = corr.mean(0)
    tmin = corr.min(0)

    grp = np.array([f2g[f] for f in files])
    same = grp[:, None] == grp[None, :]
    iu = np.triu_indices(n, k=1)

    for name, sim, lo, hi in [("global-ZNCC(mean)", glob, 0.45, 0.85),
                              ("tiled-ZNCC(min)", tmin, -0.1, 0.6)]:
        s = sim[iu]; sp = same[iu]
        pos, neg = s[sp], s[~sp]
        print(f"\n=== {name} ===")
        print(f"  SAME p10={np.percentile(pos,10):.3f} p50={np.percentile(pos,50):.3f}")
        print(f"  DIFF p99={np.percentile(neg,99):.3f} max={neg.max():.3f}")
        best = (0,0,None)
        for thr in np.arange(lo, hi, 0.02):
            A = sim >= thr; np.fill_diagonal(A, False)
            _, lab = connected_components(csr_matrix(A), directed=False)
            pred = defaultdict(set)
            for i,f in enumerate(files): pred[lab[i]].add(f)
            sc, ex, nref, npred = score_and_errors(pred.values(), groups)
            if sc > best[1]: best = (round(thr,3), round(sc,4), (ex,nref,npred))
        ex,nref,npred = best[2]
        print(f"  best score {best[1]} at thr={best[0]}  ({ex}/{nref} exact, predicted {npred} groups vs {nref} ref)")

    print(f"\ntotal {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
