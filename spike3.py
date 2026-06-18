"""
Spike 3: validity-gated tiled ZNCC.

Per pair, only compare tiles where BOTH images have real gradient energy
(excludes exposure-killed blown/crushed tiles). Among valid tiles take a robust
LOW quantile so a single moved-chair tile (content present in both but
uncorrelated) drags the score down, while exposure clipping is ignored.

Goal: beat global-ZNCC's 0.942 AND widen the margin against moved-chair pairs.
"""
import csv, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SAMPLE = Path("sample"); IMG_DIR = SAMPLE / "images"
SIZE = 288; TILES = 6; TSZ = 48          # 6x6 = 36 tiles
ALPHA = 0.25       # tile valid if energy > ALPHA * image's median tile energy
MINVALID = 6       # need this many valid tiles else fall back to global mean
LOWQ = 0.10        # robust low quantile of valid-tile correlations

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(SAMPLE / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def descriptors(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    cs = SIZE // TILES
    vecs = np.zeros((TILES*TILES, TSZ*TSZ), np.float32)
    energy = np.zeros(TILES*TILES, np.float32)
    k = 0
    for i in range(TILES):
        for j in range(TILES):
            t = mag[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            t = cv2.resize(t, (TSZ, TSZ), interpolation=cv2.INTER_AREA).ravel()
            energy[k] = np.linalg.norm(t)
            tm = t - t.mean(); nrm = np.linalg.norm(tm)
            vecs[k] = tm / nrm if nrm > 0 else tm
            k += 1
    return vecs, energy

def score(pred_groups, ref_groups):
    ref = set(frozenset(v) for v in ref_groups.values())
    pred = set(frozenset(v) for v in pred_groups)
    return len(ref & pred)/len(ref), len(ref & pred), len(pred)

def main():
    t0 = time.time()
    groups, f2g = load_manifest()
    files = sorted(f2g.keys()); n = len(files)
    V = np.zeros((n, TILES*TILES, TSZ*TSZ), np.float32)
    E = np.zeros((n, TILES*TILES), np.float32)
    for idx, f in enumerate(files):
        v, e = descriptors(IMG_DIR / f); V[idx] = v; E[idx] = e
    print(f"{n} imgs embedded in {time.time()-t0:.1f}s")

    # per-image valid tile mask
    med = np.median(E, axis=1, keepdims=True)
    valid_img = E > (ALPHA * med)                       # (n, T)
    T = TILES*TILES

    # per-tile pairwise correlation stack (T, n, n)
    corr = np.stack([V[:, t, :] @ V[:, t, :].T for t in range(T)])
    glob = corr.mean(0)

    # pair validity & gated low-quantile
    cn = np.full((n, n), 0.0, np.float32)
    vc = np.zeros((n, n), np.int32)
    cmasked = corr.copy()
    # build score row-by-row to bound memory
    gated = np.zeros((n, n), np.float32)
    for i in range(n):
        vi = valid_img[i][:, None]                      # (T,1)
        pairvalid = vi & valid_img.T                    # (T,n)
        c = corr[:, i, :].copy()                        # (T,n)
        c[~pairvalid] = np.nan
        cnt = pairvalid.sum(0)                          # (n,)
        with np.errstate(all="ignore"):
            lq = np.nanquantile(c, LOWQ, axis=0)        # (n,)
        # fall back to global mean where too few valid tiles
        lq = np.where(cnt >= MINVALID, lq, glob[i])
        gated[i] = lq
    gated = np.nan_to_num(gated, nan=-1.0)

    grp = np.array([f2g[f] for f in files])
    same = grp[:, None] == grp[None, :]
    iu = np.triu_indices(n, k=1)

    for name, sim, lo, hi in [("global-ZNCC", glob, 0.45, 0.80),
                              ("validity-gated-lowq", gated, 0.05, 0.70)]:
        s = sim[iu]; sp = same[iu]
        pos, neg = s[sp], s[~sp]
        print(f"\n=== {name} ===")
        print(f"  SAME p10={np.percentile(pos,10):.3f} p50={np.percentile(pos,50):.3f}")
        print(f"  DIFF p99={np.percentile(neg,99):.3f} max={neg.max():.3f}")
        best = (0,0,0)
        for thr in np.arange(lo, hi, 0.01):
            A = sim >= thr; np.fill_diagonal(A, False)
            _, lab = connected_components(csr_matrix(A), directed=False)
            pred = defaultdict(set)
            for i,f in enumerate(files): pred[lab[i]].add(f)
            sc, ex, npred = score(pred.values(), groups)
            if sc > best[1]: best = (round(thr,3), round(sc,4), npred)
        print(f"  best {best[1]} at thr={best[0]} (predicted {best[2]} vs 69 ref)")

    print(f"\ntotal {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
