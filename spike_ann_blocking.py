"""SPIKE: O(n) tier-1 clustering via banded-SimHash blocking, validated to
reproduce the dense FusionClusterer labels on data/large (so 1302 is untouched).

The dense path (clustering.py) materializes an N x N similarity matrix and runs
connected_components ~70x to pick the plateau threshold. At N=95k the matrix is
~36 GB *per view* and cannot allocate. This spike proves we can get the SAME
clustering from a SPARSE candidate-edge graph built without ever forming N x N.

Key identity (scaled-concat trick)
-----------------------------------
    S_ij = 0.65*(g_i.g_j) + 0.35*(e_i.e_j) = v_i . v_j ,  v = [sqrt.65 g, sqrt.35 e]
Both g and e are L2-normalized, so v is unit-norm and v_i.v_j is a cosine in
[-1,1]. SimHash (sign of random hyperplane projections) therefore applies: two
points collide more often the smaller their angle. We band the bit signature so
two points become *candidates* if ANY band matches, score only candidates
exactly, threshold, and connected-component. No N x N is ever built.

Determinism: the random hyperplanes use a FIXED seed -> reproducible, no real
randomness. Pure matmul + bit packing + sort -> portable to a browser.

Usage:
    python spike_ann_blocking.py data/large           # equivalence + bench
    python spike_ann_blocking.py data/large 6 28       # band-width w, n-bands r
    python spike_ann_blocking.py --scale 30000 95000   # synthetic scale bench
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

from autohdr.clustering import FusionClusterer, W_GRADIENT
from autohdr.features import GradientDescriptor, WaveletEmbedding
from autohdr.image_loader import Photoshoot

_SEED = 1234567          # fixed -> deterministic hyperplanes
_FLOOR = 0.20            # candidate edges scored/kept at/above this (grid floor)
_DEFAULT_W = 6           # bits per band (bucket = 2^w per band)
_DEFAULT_R = 28          # number of bands; candidate if ANY band collides


def stacked_vectors(gradient: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """v_i = [sqrt(W) g_i, sqrt(1-W) e_i]  =>  v_i.v_j == fused similarity."""
    a = np.sqrt(W_GRADIENT) * gradient
    b = np.sqrt(1.0 - W_GRADIENT) * embedding
    return np.ascontiguousarray(np.hstack([a, b]).astype(np.float32))


# --------------------------------------------------------------------------- #
#  Tier-1: banded SimHash -> sparse candidate edge list (weighted)            #
# --------------------------------------------------------------------------- #
def simhash_edges(v: np.ndarray, w: int, r: int, floor: float = _FLOOR,
                  seed: int = _SEED) -> coo_matrix:
    """Sparse symmetric weighted edge list of candidate pairs with sim >= floor.

    Never forms N x N. Cost ~ O(N*r*w) for hashing + O(sum bucket^2 * D) for the
    block scoring of candidates (buckets are small for sensible w).
    """
    n, d = v.shape
    rng = np.random.default_rng(seed)
    bits_total = w * r
    planes = rng.standard_normal((d, bits_total)).astype(np.float32)
    sign = (v @ planes) > 0                       # (N, w*r) bool
    pow2 = (1 << np.arange(w)).astype(np.int64)   # pack w bits -> bucket key

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for band in range(r):
        sub = sign[:, band * w:(band + 1) * w]
        keys = sub.astype(np.int64) @ pow2        # (N,) bucket id in [0, 2^w)
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        # contiguous runs of equal key = one bucket
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for members in np.split(order, bounds):
            m = len(members)
            if m < 2:
                continue
            block = v[members] @ v[members].T     # m x m exact sim (small m)
            iu, ju = np.triu_indices(m, k=1)
            s = block[iu, ju]
            keep = s >= floor
            if not keep.any():
                continue
            rows.append(members[iu[keep]])
            cols.append(members[ju[keep]])
            vals.append(s[keep])

    if not rows:
        return coo_matrix((n, n), dtype=np.float32)
    ri = np.concatenate(rows); ci = np.concatenate(cols); vv = np.concatenate(vals)
    # a pair can collide in several bands; its exact sim is identical each time,
    # so dedup by canonical (lo,hi) id and keep one representative value.
    lo = np.minimum(ri, ci).astype(np.int64)
    hi = np.maximum(ri, ci).astype(np.int64)
    pid = lo * n + hi
    _, idx = np.unique(pid, return_index=True)
    return coo_matrix((vv[idx], (lo[idx], hi[idx])), shape=(n, n), dtype=np.float32)


def sparse_plateau_threshold(edges: coo_matrix, n: int) -> float:
    """Mirror FusionClusterer._plateau_threshold but on the SPARSE graph.

    Same grid + same knee rule; connected_components on a sparse csr is
    O(N + E) per threshold, not O(N^2).
    """
    r, c, s = edges.row, edges.col, edges.data
    grid = np.arange(0.20, 0.90, 0.01)

    def count_at(t: float) -> int:
        keep = s >= t
        g = csr_matrix((np.ones(keep.sum(), np.int8), (r[keep], c[keep])), shape=(n, n))
        return connected_components(g, directed=False)[0]

    counts = np.array([count_at(t) for t in grid], float)
    window = 3
    slope = np.full(len(grid), np.inf)
    for i in range(window, len(grid) - window):
        slope[i] = (counts[i + window] - counts[i - window]) / (2 * window)
    slope[counts <= 0.5 * counts.max()] = np.inf
    smallest = slope[np.isfinite(slope)].min()
    cut = 1.3 * smallest + 0.5
    knee = int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])
    return grid[knee]


def sparse_labels(edges: coo_matrix, n: int, threshold: float) -> np.ndarray:
    keep = edges.data >= threshold
    g = csr_matrix((np.ones(keep.sum(), np.int8),
                    (edges.row[keep], edges.col[keep])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def groups_of(labels: np.ndarray) -> set:
    members: dict[int, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        members[int(l)].append(i)
    return {frozenset(m) for m in members.values()}


# --------------------------------------------------------------------------- #
def run_equivalence(data: Path, w: int, r: int) -> None:
    raw = np.load(data / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); gray = raw["imgs"]
    print(f"[load] {len(files)} images from {data}")

    t = time.perf_counter()
    gradient = GradientDescriptor().encode(gray)
    embedding = WaveletEmbedding().encode(gray)
    print(f"[encode] gradient {gradient.shape} embedding {embedding.shape} "
          f"in {time.perf_counter()-t:.1f}s")

    n = len(files)

    # ---- dense reference -------------------------------------------------- #
    t = time.perf_counter()
    dense_graph = FusionClusterer(gradient, embedding)
    dense_thr = dense_graph._plateau_threshold()
    dense_lab = dense_graph.initial_graph().labels()
    dense_t = time.perf_counter() - t
    dense_mem_gb = 3 * (n * n * 4) / 1e9  # 3 N x N float32 matrices
    print(f"[dense]  thr={dense_thr:.3f}  clusters={len(set(dense_lab))}  "
          f"{dense_t:.2f}s  ~{dense_mem_gb:.2f}GB N^2")

    # ---- sparse tier-1 ---------------------------------------------------- #
    t = time.perf_counter()
    v = stacked_vectors(gradient, embedding)
    edges = simhash_edges(v, w, r)
    build_t = time.perf_counter() - t
    nnz = edges.nnz
    t = time.perf_counter()
    sp_thr = sparse_plateau_threshold(edges, n)
    sp_lab = sparse_labels(edges, n, sp_thr)
    clust_t = time.perf_counter() - t
    sparse_mem_gb = (v.nbytes + nnz * 12) / 1e9  # vectors + COO(i,j,val)
    print(f"[sparse] thr={sp_thr:.3f}  clusters={len(set(sp_lab))}  "
          f"edges={nnz} ({nnz/n:.1f}/node)  build={build_t:.2f}s "
          f"cluster={clust_t:.2f}s  ~{sparse_mem_gb:.2f}GB")

    # ---- equivalence ------------------------------------------------------ #
    dg, sg = groups_of(dense_lab), groups_of(sp_lab)
    same = dg & sg
    print(f"\n[EQUIVALENCE] dense groups={len(dg)} sparse groups={len(sg)} "
          f"identical={len(same)}  ({len(same)/len(dg)*100:.2f}% of dense)")

    # recall of dense edges at/above the dense threshold (why any mismatch)
    dense_sim = (W_GRADIENT * (gradient @ gradient.T)
                 + (1 - W_GRADIENT) * (embedding @ embedding.T)).astype(np.float32)
    np.fill_diagonal(dense_sim, 0.0)
    true_mask = np.triu(dense_sim >= dense_thr, k=1)
    n_true = int(true_mask.sum())
    found = set(zip(*[a.tolist() for a in
                      (np.minimum(edges.row, edges.col), np.maximum(edges.row, edges.col))]))
    ti, tj = np.where(true_mask)
    hit = sum((int(a), int(b)) in found for a, b in zip(ti, tj))
    print(f"[edge recall] dense edges>=thr: {n_true}  found by LSH: {hit} "
          f"({hit/max(n_true,1)*100:.2f}%)")
    miss = sorted(dg - sg, key=len, reverse=True)[:6]
    if miss:
        print(f"[mismatch] {len(dg-sg)} dense groups not reproduced; "
              f"largest sizes={[len(m) for m in miss]}")


def run_scale(sizes: list[int], _w: int, r: int) -> None:
    """Systems benchmark on synthetic unit vectors: shows sparse stays
    subquadratic where dense N^2 explodes. Mimics the real descriptor dim and
    the tight-bracket structure (clusters of ~8 near-identical frames).

    Bit-width is ADAPTIVE: w = log2(N / target_bucket) keeps buckets ~constant
    so candidate volume does not blow up with N. Recall is measured against the
    planted clusters (a planted co-member pair is a true edge).
    """
    d = 4096 + 127
    target_bucket = 80
    for n in sizes:
        rng = np.random.default_rng(_SEED)
        k = n // 8 + 1                                   # number of planted brackets
        owner = rng.integers(0, k, n)
        base = rng.standard_normal((k, d)).astype(np.float32)
        v = base[owner] + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v = v.astype(np.float32)
        w = max(6, int(round(np.log2(n / target_bucket))))  # adaptive buckets
        dense_gb = 3 * (n * n * 4) / 1e9
        t = time.perf_counter()
        edges = simhash_edges(v, w, r)
        bt = time.perf_counter() - t
        sp_gb = (v.nbytes + edges.nnz * 12) / 1e9
        # recall: fraction of planted co-member pairs that became candidate edges
        same = owner[edges.row] == owner[edges.col]
        found_intra = int(same.sum())
        _, counts = np.unique(owner, return_counts=True)
        true_intra = int((counts * (counts - 1) // 2).sum())
        feasible = "OK" if dense_gb < 8 else "OOM(>8GB)"
        print(f"N={n:>6}  w={w} r={r}  dense N^2 ~{dense_gb:7.1f}GB [{feasible:>9}]  "
              f"sparse build={bt:6.2f}s  mem~{sp_gb:5.2f}GB  edges={edges.nnz:>8}  "
              f"intra-recall={found_intra/max(true_intra,1)*100:5.1f}%")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--scale":
        sizes = [int(x) for x in args[1:]] or [5000, 30000, 95000]
        run_scale(sizes, _DEFAULT_W, _DEFAULT_R)
        return
    data = Path(args[0]) if args else Path("data/large")
    w = int(args[1]) if len(args) > 1 else _DEFAULT_W
    r = int(args[2]) if len(args) > 2 else _DEFAULT_R
    run_equivalence(data, w, r)


if __name__ == "__main__":
    main()
