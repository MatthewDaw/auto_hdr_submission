"""
Direction 2: learned edge classifier (browser-portable tree).

For candidate pairs (gradient-ZNCC >= 0.45) compute cheap features that include
GLOBAL GRAPH CONTEXT (common-neighbor count, degrees, mutual-NN rank, local
density ratios) -- the very signals that distinguish a within-group chain link
from a between-group bridge. Train HistGradientBoosting on manifest labels with
a GROUP-DISJOINT split (no leakage), then cluster the held-out groups using the
predicted edge probability and compare to the gradient-threshold baseline.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
d = np.load(DATA / "feat_cache.npz", allow_pickle=True)
M, C, B = d["M"], d["C"], d["B"]; files = list(d["files"]); n = len(files)
groups = defaultdict(set); f2g = {}
for r in csv.DictReader(open(DATA / "public_manifest.csv")):
    groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
gid = np.array([f2g[f] for f in files])

FLOOR = 0.45
sim = (M @ M.T).astype(np.float32)
np.fill_diagonal(sim, -1)
smax = sim.max(1)
A = sim >= FLOOR
deg = A.sum(1)
Asp = csr_matrix(A.astype(np.int8))
common = (Asp @ Asp).toarray().astype(np.float32)   # shared-neighbor counts
# neighbor rank: rank of j in i's sorted neighbor list (0=best)
rank = np.argsort(np.argsort(-sim, axis=1), axis=1).astype(np.float32)

def census_pair(i, j):
    x = np.bitwise_xor(C[i], C[j])
    return 1.0 - np.unpackbits(x).sum() / (C.shape[1]*8.0)

# candidate pairs (i<j, sim>=FLOOR)
ii, jj = np.where(np.triu(A, 1))
print(f"{len(ii)} candidate pairs")
feat = np.zeros((len(ii), 11), np.float32)
feat[:, 0] = sim[ii, jj]
feat[:, 1] = [census_pair(i, j) for i, j in zip(ii, jj)]
feat[:, 2] = np.abs(B[ii] - B[jj])
feat[:, 3] = common[ii, jj]
feat[:, 4] = np.minimum(rank[ii, jj], rank[jj, ii])
feat[:, 5] = np.maximum(rank[ii, jj], rank[jj, ii])
feat[:, 6] = np.minimum(smax[ii], smax[jj])
feat[:, 7] = sim[ii, jj] / np.maximum(smax[ii], 1e-6)
feat[:, 8] = sim[ii, jj] / np.maximum(smax[jj], 1e-6)
feat[:, 9] = np.minimum(deg[ii], deg[jj])
feat[:, 10] = np.maximum(deg[ii], deg[jj])
label = (gid[ii] == gid[jj]).astype(int)
print(f"positives {label.sum()} / {len(label)} ({label.mean():.3f})")

# group-disjoint split
uniq = sorted(groups.keys())
rng = np.argsort([hash(g) & 0xffff for g in uniq])  # deterministic shuffle
train_g = set(np.array(uniq)[rng[:int(0.6*len(uniq))]])
tr = np.array([gid[i] in train_g for i in ii])
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=4)
clf.fit(feat[tr], label[tr])
prob = clf.predict_proba(feat)[:, 1]

refsets = set(frozenset(v) for v in groups.values())
def cluster_score(edge_mask, node_subset):
    sub = sorted(node_subset)
    idxmap = {g: k for k, g in enumerate(sub)}
    rows = [idxmap[ii[e]] for e in range(len(ii)) if edge_mask[e] and ii[e] in node_subset and jj[e] in node_subset]
    cols = [idxmap[jj[e]] for e in range(len(ii)) if edge_mask[e] and ii[e] in node_subset and jj[e] in node_subset]
    G = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sub), len(sub)))
    _, lab = connected_components(G, directed=False)
    pred = defaultdict(set)
    for k, node in enumerate(sub): pred[lab[k]].add(files[node])
    predlk = set(frozenset(v) for v in pred.values())
    testref = set(frozenset(v) for g, v in groups.items() if g not in train_g)
    return len(testref & predlk) / len(testref)

test_nodes = set(i for i in range(n) if gid[i] not in train_g)
# learned: sweep prob threshold
best = (0, 0)
for p in np.arange(0.30, 0.85, 0.05):
    sc = cluster_score(prob >= p, test_nodes)
    if sc > best[0]: best = (sc, round(p, 2))
print(f"\nLEARNED edge clf : best {best[0]:.4f} @ p>={best[1]} (held-out groups)")
# baseline: gradient threshold on same held-out nodes
gb = (0, 0)
for t in np.arange(0.50, 0.70, 0.02):
    sc = cluster_score(sim[ii, jj] >= t, test_nodes)
    if sc > gb[0]: gb = (sc, round(t, 2))
print(f"BASELINE gradient: best {gb[0]:.4f} @ thr={gb[1]} (same held-out groups)")
print("\nfeature importances (permutation skipped; using split gains proxy):")
for name, im in sorted(zip(
    ["grad","census","bgap","common","rankmin","rankmax","smaxmin","ratioi","ratioj","degmin","degmax"],
    clf.feature_importances_ if hasattr(clf,'feature_importances_') else [0]*11), key=lambda x:-x[1])[:6]:
    print(f"  {name}: {im:.3f}")
