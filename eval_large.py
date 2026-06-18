"""
Eval harness on a full dataset dir. Caches descriptors (decode is the cost),
runs CLAHE+gradient-ZNCC + connected-components, reports best exact-set score,
over-split vs over-merge breakdown, and a DRONE analysis: do drone groups
(DJI in training filename) need looser thresholds than non-drone groups?
"""
import csv, sys, time
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
IMG = DATA / "images"; CACHE = DATA / "desc_cache.npz"
SIZE = 256; ZNCC = 64

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(DATA / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def imread_unicode(path):
    # cv2.imread fails on non-ASCII paths on Windows; decode via numpy buffer
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

def desc(fname):
    # worker: takes a filename, returns descriptor (each process builds its own CLAHE)
    img = imread_unicode(IMG / fname)
    if img is None:
        return np.zeros(ZNCC*ZNCC, np.float32)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = clahe.apply(g).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    z = cv2.resize(cv2.magnitude(gx, gy), (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); n = np.linalg.norm(z)
    return (z / n if n > 0 else z).astype(np.float32)

def build_descriptors(files):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if list(d["files"]) == files:
            print(f"loaded cached descriptors ({len(files)})"); return d["M"]
    t0 = time.time()
    nproc = max(1, cpu_count() - 1)
    cv2.setNumThreads(1)  # avoid oversubscription: parallelism is across images
    print(f"decoding {len(files)} imgs on {nproc} processes...")
    with Pool(nproc) as pool:
        M = np.array(list(pool.imap(desc, files, chunksize=16)), np.float32)
    bad = int((np.linalg.norm(M, axis=1) == 0).sum())
    print(f"embedded in {time.time()-t0:.0f}s ({bad} undecodable)")
    np.savez(CACHE, M=M, files=np.array(files))
    return M

def score_breakdown(sim, thr, files, groups, f2g):
    A = sim >= thr; np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    refsets = {g: frozenset(v) for g, v in groups.items()}
    predlk = set(frozenset(v) for v in pred.values())
    exact = sum(1 for s in refsets.values() if s in predlk)
    # classify each ref group's miss
    file2pred = {f: lab[i] for i, f in enumerate(files)}
    oversplit = overmerge = 0
    for g, rs in refsets.items():
        if rs in predlk: continue
        clusters = set(file2pred[f] for f in rs)
        contaminated = any((pred[c] - rs) for c in clusters)
        if contaminated: overmerge += 1
        else: oversplit += 1
    return exact/len(refsets), len(pred), exact, len(refsets), oversplit, overmerge

def main():
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    print(f"{len(files)} imgs, {len(groups)} groups")
    M = build_descriptors(files)
    t0 = time.time(); sim = M @ M.T; print(f"similarity matmul {time.time()-t0:.1f}s")

    best = None
    for thr in np.arange(0.34, 0.60, 0.02):
        sc, npred, ex, nref, osp, omg = score_breakdown(sim, thr, files, groups, f2g)
        print(f"  thr={thr:.2f}: score={sc:.4f} ({ex}/{nref})  pred={npred}  over-split={osp} over-merge={omg}")
        if best is None or sc > best[0]: best = (sc, thr)
    print(f"BEST {best[0]:.4f} @ thr={best[1]:.2f}")

    # DRONE analysis: intra-group similarity, drone vs non-drone
    drone_grp = {g for g, fs in groups.items() if any("DJI" in f for f in fs)}
    idxof = {f: i for i, f in enumerate(files)}
    def intra_stats(gset):
        mins = []
        for g in gset:
            ix = [idxof[f] for f in groups[g]]
            if len(ix) < 2: continue
            sub = sim[np.ix_(ix, ix)]; u = np.triu_indices(len(ix), 1)
            mins.append(sub[u].min())
        return np.array(mins)
    dm = intra_stats(drone_grp); nm = intra_stats(set(groups) - drone_grp)
    print(f"\nDRONE groups: {len(drone_grp)}/{len(groups)}")
    print(f"  drone     intra-min: p10={np.percentile(dm,10):.3f} p50={np.percentile(dm,50):.3f} frac<0.44={np.mean(dm<0.44):.2f}")
    print(f"  non-drone intra-min: p10={np.percentile(nm,10):.3f} p50={np.percentile(nm,50):.3f} frac<0.44={np.mean(nm<0.44):.2f}")

if __name__ == "__main__":
    main()
