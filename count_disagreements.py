"""Disagreement breakdown for a full_subset pred_labels file vs the manifest."""
import json, csv, sys
from collections import defaultdict, Counter
ddir = "data/full_subset"
pred = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ddir + "/pred_labels.json"))
f2g = {}; gt = defaultdict(set)
for row in csv.DictReader(open(ddir + "/public_manifest.csv", encoding="utf-8")):
    gt[row["group_id"]].add(row["filename"]); f2g[row["filename"]] = row["group_id"]
predsets = defaultdict(set)
for f, c in pred.items(): predsets[c].add(f)
predfz = {frozenset(s) for s in predsets.values()}
dis = [g for g in gt if frozenset(gt[g]) not in predfz]
owner = {c: Counter(f2g[f] for f in s if f in f2g).most_common(1)[0][0] for c, s in predsets.items()}
osplit = leak = fmm = 0
for g in dis:
    cs = {pred[f] for f in gt[g] if f in pred}
    own = [c for c in cs if owner.get(c) == g]; lk = [c for c in cs if owner.get(c) != g]
    if len(own) >= 2 and not lk: osplit += 1
    elif lk: leak += 1
    else: fmm += 1
print(f"groups={len(gt)} exact={len(gt)-len(dis)} DISAGREEMENTS={len(dis)}")
print(f"  over-splits={osplit}  wrong-merges/leaks={leak}  frame-mismatch={fmm}")
