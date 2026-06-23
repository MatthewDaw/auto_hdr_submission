"""Audit every case the user called out, against current full_subset pred_labels.

Three classes:
  OVER-SPLIT  (must now be MERGED = 1 cluster, GT right): we previously over-split.
  WRONG-MERGE (must now be SPLIT apart: each GT group owns its own cluster): color veto.
  STRAND      (clipped-frame splits color can't reach — honest not-fixed tracking).
"""
import json, csv
from collections import defaultdict
ddir = "data/full_subset"
pred = json.load(open(ddir + "/pred_labels.json"))
gt = defaultdict(set)
for row in csv.DictReader(open(ddir + "/public_manifest.csv", encoding="utf-8")):
    gt[row["group_id"]].add(row["filename"])

def clusters_of(g):
    return {pred[f] for f in gt[g] if f in pred}

def exact(g):
    fr = frozenset(gt[g])
    cl = clusters_of(g)
    if len(cl) != 1:
        return False
    c = next(iter(cl))
    return frozenset(f for f in pred if pred[f] == c) == fr

# over-split cases the user said GT-is-right / merge
OVER = ["3221","10147","10149","10150","10152","10153","10159","10160","11574",
        "12583","13093","13360","13586","13590","13601","13609","14011","11964",
        "12150"]
# wrong-merges the user flagged (we fused two different GT scenes)
WRONG = [("12226","13169"),("10464","10613"),("10886","10593")]
# clipped-frame strands color can't address (honest tracking)
STRAND = ["10125","10129","10463","10276","1087"]

print("=== OVER-SPLIT (want: exact match to GT, 1 cluster) ===")
ok = 0
for g in OVER:
    if g not in gt: print(f"  {g}: (not in 30k subset)"); continue
    e = exact(g); ok += e
    print(f"  {g}: clusters={len(clusters_of(g))} exact_GT={'YES' if e else 'NO'}")
print(f"  -> {ok}/{sum(1 for g in OVER if g in gt)} exact\n")

print("=== WRONG-MERGE (want: the two GT groups separated, each exact) ===")
for a, b in WRONG:
    if a not in gt or b not in gt: print(f"  {a}+{b}: (missing)"); continue
    shared = clusters_of(a) & clusters_of(b)
    ea, eb = exact(a), exact(b)
    verdict = "SEPARATED" if not shared else "STILL MERGED"
    print(f"  {a}+{b}: {verdict}  exact[{a}]={'Y' if ea else 'N'} exact[{b}]={'Y' if eb else 'N'}")
print()

print("=== STRAND (clipped — expected NOT fixed by color) ===")
for g in STRAND:
    if g not in gt: print(f"  {g}: (not in subset)"); continue
    print(f"  {g}: clusters={len(clusters_of(g))} exact_GT={'YES' if exact(g) else 'NO'}")
