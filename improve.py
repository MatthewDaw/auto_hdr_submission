"""
Apply staged fixes on top of the gradient-ZNCC baseline and measure each.

Stage A: baseline (CLAHE+gradient-ZNCC + connected components)
Stage B: drone-aware stricter threshold (drone-drone pairs need higher bar)
Stage C: orphan re-attachment of clipped exposure-tail clusters via census argmax

Decode is parallel + Unicode/CMYK-safe (np.fromfile->imdecode, PIL fallback).
Drone flag uses the training filename ('DJI') as an ORACLE for now — a real
solution needs a pixel-based drone classifier (measured separately if this helps).
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
IMG = DATA / "images"; FEAT = DATA / "feat_cache.npz"
SIZE = 256; ZNCC = 64; CEN = 96

def load_manifest():
    groups = defaultdict(set); f2g = {}
    for r in csv.DictReader(open(DATA / "public_manifest.csv")):
        groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
    return groups, f2g

def read_gray(path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if im is not None: return im
    except Exception:
        pass
    try:  # fallback: PIL handles CMYK / some truncated JPEGs
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None

def features(fname):
    im = read_gray(IMG / fname)
    if im is None:
        return np.zeros(ZNCC*ZNCC, np.float32), np.zeros(CEN*CEN, np.uint8), -1.0
    bright = float(im.mean())
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gc = clahe.apply(g).astype(np.float32)
    gx = cv2.Sobel(gc, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gc, cv2.CV_32F, 0, 1, ksize=3)
    z = cv2.resize(cv2.magnitude(gx, gy), (ZNCC, ZNCC), interpolation=cv2.INTER_AREA).ravel()
    z = z - z.mean(); n = np.linalg.norm(z); z = (z/n if n > 0 else z).astype(np.float32)
    c = cv2.resize(gc, (CEN, CEN)).astype(np.int16)
    code = np.zeros((CEN, CEN), np.uint8); bit = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            sh = np.roll(np.roll(c, di, 0), dj, 1)
            code |= ((c > sh).astype(np.uint8) << bit); bit += 1
    return z, code.ravel(), bright

def build(files):
    if FEAT.exists():
        d = np.load(FEAT, allow_pickle=True)
        if list(d["files"]) == files:
            print(f"loaded feat cache ({len(files)})"); return d["M"], d["C"], d["B"]
    t0 = time.time(); cv2.setNumThreads(1)
    print(f"extracting {len(files)} on {cpu_count()-1} procs...")
    with Pool(max(1, cpu_count()-1)) as pool:
        res = list(pool.imap(features, files, chunksize=16))
    M = np.array([r[0] for r in res], np.float32)
    C = np.array([r[1] for r in res], np.uint8)
    B = np.array([r[2] for r in res], np.float32)
    bad = int((B < 0).sum()); print(f"extracted in {time.time()-t0:.0f}s ({bad} undecodable)")
    np.savez(FEAT, M=M, C=C, B=B, files=np.array(files))
    return M, C, B

def clusters_from_adj(A, n):
    np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    return lab

def score(lab, files, groups):
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    refsets = set(frozenset(v) for v in groups.values())
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk) / len(refsets), len(pred)

def census_sim_rows(C, idxs, allidx):
    # similarity of each row in idxs to each col in allidx ; (len(idxs), len(allidx))
    out = np.zeros((len(idxs), len(allidx)), np.float32)
    P8 = C.shape[1] * 8.0
    Csub = C[allidx]
    for k, i in enumerate(idxs):
        x = np.bitwise_xor(C[i][None, :], Csub)
        out[k] = 1.0 - np.unpackbits(x, axis=1).sum(1) / P8
    return out

def main():
    groups, f2g = load_manifest(); files = sorted(f2g.keys())
    n = len(files); print(f"{n} imgs, {len(groups)} groups")
    M, C, B = build(files)
    sim = M @ M.T
    drone = np.array([("DJI" in f) for f in files])  # ORACLE drone flag
    THR = 0.58

    # ---- Stage A: baseline ----
    lab = clusters_from_adj(sim >= THR, n)
    scA, npA = score(lab, files, groups)
    print(f"A baseline           : {scA:.4f} (pred {npA})")

    # ---- Stage B: drone-aware stricter threshold ----
    best = (scA, THR, None)
    for dthr in [0.70, 0.75, 0.80, 0.85]:
        A = sim >= THR
        dd = drone[:, None] & drone[None, :]      # both endpoints drone
        A = A & ~(dd & (sim < dthr))              # drone-drone must clear dthr
        lab2 = clusters_from_adj(A, n)
        sc, npq = score(lab2, files, groups)
        if sc > best[0]: best = (sc, dthr, lab2)
    scB, dthrB, labB = best
    print(f"B + drone-strict     : {scB:.4f} (drone_thr={dthrB})")
    labB = labB if labB is not None else lab

    # ---- Stage C: orphan re-attachment of clipped tails (on top of B) ----
    cl = defaultdict(list)
    for i, l in enumerate(labB): cl[l].append(i)
    # orphan pieces: small clusters whose members are brightness-extreme (clipped)
    EXTREME = (B < 45) | (B > 210)
    moved = 0
    labC = labB.copy()
    # medoid per cluster for matching target
    for l, members in list(cl.items()):
        if len(members) > 2: continue                 # only small/orphan clusters
        if not any(EXTREME[i] for i in members): continue
        for i in members:
            others = [j for j in range(n) if labC[j] != labC[i]]
            cs = census_sim_rows(C, [i], others)[0]
            # best target cluster by max census sim, require margin + brightness gap
            order = np.argsort(cs)[::-1]
            bestj = others[order[0]]
            # 2nd-best from a DIFFERENT cluster for margin
            secj = next((others[k] for k in order[1:] if labC[others[k]] != labC[bestj]), None)
            margin = cs[order[0]] - (census_sim_rows(C, [i], [secj])[0,0] if secj is not None else 0)
            bgap = abs(B[i] - B[bestj])
            if cs[order[0]] >= 0.62 and margin >= 0.03 and bgap >= 25:
                labC[i] = labC[bestj]; moved += 1
    scC, npC = score(labC, files, groups)
    print(f"C + orphan re-attach : {scC:.4f} (moved {moved})")

if __name__ == "__main__":
    main()
