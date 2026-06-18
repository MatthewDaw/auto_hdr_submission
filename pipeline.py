"""
Consolidated pipeline: auto-threshold (plateau) + connected-components +
orphan re-attachment. Then dump REMAINING failures, categorized, with the
key diagnostic for over-merges: the bridge similarity between a ref group and
its contaminants (high => genuinely confusable lookalike; marginal => weak bridge).
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
d = np.load(DATA / "feat_cache.npz", allow_pickle=True)
M, C, B = d["M"], d["C"], d["B"]; files = list(d["files"]); n = len(files)
groups = defaultdict(set); f2g = {}
for r in csv.DictReader(open(DATA / "public_manifest.csv")):
    groups[r["group_id"]].add(r["filename"]); f2g[r["filename"]] = r["group_id"]
sim = M @ M.T
refsets = set(frozenset(v) for v in groups.values())
idxof = {f: i for i, f in enumerate(files)}

def cluster(thr):
    A = sim >= thr; np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    return lab

def score(lab):
    pred = defaultdict(set)
    for i, f in enumerate(files): pred[lab[i]].add(f)
    predlk = set(frozenset(v) for v in pred.values())
    return len(refsets & predlk)/len(refsets), len(pred)

# --- auto threshold (plateau leading edge) ---
grid = np.arange(0.35, 0.75, 0.01); counts = []
for t in grid:
    lab = cluster(t); counts.append(len(set(lab)))
counts = np.array(counts, float)
W = 3; slope = np.full(len(grid), np.inf)
for i in range(W, len(grid)-W): slope[i] = (counts[i+W]-counts[i-W])/(2*W)
slope[counts <= 0.5*counts.max()] = np.inf
cut = 1.3*slope[np.isfinite(slope)].min() + 0.5
thr = grid[int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])]
lab = cluster(thr)
print(f"auto-threshold = {thr:.2f}  baseline score = {score(lab)[0]:.4f}")

# --- orphan re-attach ---
def census_rows(i, allidx):
    x = np.bitwise_xor(C[i][None, :], C[allidx])
    return 1.0 - np.unpackbits(x, axis=1).sum(1) / (C.shape[1]*8.0)
cl = defaultdict(list)
for i, l in enumerate(lab): cl[l].append(i)
EXTREME = (B < 45) | (B > 210); moved = 0
for l, members in list(cl.items()):
    if len(members) > 2 or not any(EXTREME[i] for i in members): continue
    for i in members:
        others = [j for j in range(n) if lab[j] != lab[i]]
        cs = census_rows(i, others); order = np.argsort(cs)[::-1]
        bestj = others[order[0]]
        secj = next((others[k] for k in order[1:] if lab[others[k]] != lab[bestj]), None)
        sec = census_rows(i, [secj])[0] if secj is not None else 0
        if cs[order[0]] >= 0.62 and cs[order[0]]-sec >= 0.03 and abs(B[i]-B[bestj]) >= 25:
            lab[i] = lab[bestj]; moved += 1
sc, npred = score(lab); print(f"+ orphan re-attach   = {sc:.4f} (moved {moved}, pred {npred})")

# --- remaining failure taxonomy ---
pred = defaultdict(set)
for i, f in enumerate(files): pred[lab[i]].add(f)
predlk = set(frozenset(v) for v in pred.values())
file2pred = {f: lab[idxof[f]] for f in files}
osplit = omerge = 0; merge_bridges = []
rows = []
for g, members in groups.items():
    rs = frozenset(members)
    if rs in predlk: continue
    ix = [idxof[f] for f in members]; drone = any("DJI" in f for f in members)
    sub = sim[np.ix_(ix, ix)]; u = np.triu_indices(len(ix), 1)
    imin = sub[u].min() if len(ix) > 1 else float('nan')
    clusters = set(file2pred[f] for f in members)
    contam = set()
    for c in clusters: contam |= (pred[c] - members)
    if contam:
        omerge += 1; kind = "MERGE"
        # bridge sim: max similarity between a member and a contaminant
        cix = [idxof[f] for f in contam]
        bridge = sim[np.ix_(ix, cix)].max()
        merge_bridges.append(bridge)
        rows.append((g, len(members), drone, kind, round(bridge,3), len(contam)))
    else:
        osplit += 1
        rows.append((g, len(members), drone, "SPLIT", round(imin,3), len(clusters)))

print(f"\nREMAINING: {len(rows)} failures | over-split={osplit} over-merge={omerge}")
mb = np.array(merge_bridges)
if len(mb):
    print(f"over-merge bridge sim: p25={np.percentile(mb,25):.3f} p50={np.percentile(mb,50):.3f} p75={np.percentile(mb,75):.3f}")
    print(f"  weak bridges (<thr+0.06={thr+0.06:.2f}): {int((mb < thr+0.06).sum())}/{len(mb)}  strong(confusable): {int((mb>=thr+0.06).sum())}")
print(f"\n{'group':>8} {'sz':>3} {'drn':>4} {'kind':>6} {'bridge/imin':>11} {'extra':>5}")
for g, sz, dr, kind, val, ex in sorted(rows, key=lambda r: (r[3], -r[1]))[:55]:
    print(f"{g:>8} {sz:>3} {str(dr)[0]:>4} {kind:>6} {val:>11} {ex:>5}")
