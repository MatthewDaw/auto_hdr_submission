"""Audit EVERY user-flagged case against the current full_subset pred_labels."""
import json, csv
from collections import defaultdict
ddir = "data/full_subset"
pred = json.load(open(ddir + "/pred_labels.json"))
gt = defaultdict(set)
for r in csv.DictReader(open(ddir + "/public_manifest.csv", encoding="utf-8")):
    gt[r["group_id"]].add(r["filename"])
predsets = defaultdict(set)
for f, c in pred.items():
    predsets[c].add(f)
predfz = {frozenset(s) for s in predsets.values()}

def clusters_of(g):
    return {pred[f] for f in gt[g] if f in pred}
def exact(g):
    return g in gt and frozenset(gt[g]) in predfz
def present(g):
    return g in gt and any(f in pred for f in gt[g])

print("=== Cluster A: link across exposure (want EXACT 1 cluster) ===")
for g in ["10370", "10463", "14055", "19300", "13675"]:
    if not present(g): print(f"  {g}: not in 30k subset"); continue
    print(f"  {g}: {len(clusters_of(g))} cluster(s)  exact_GT={'YES' if exact(g) else 'NO'}")

print("\n=== Cluster B: split wrong-merges (want SEPARATED, each exact) ===")
for a, b in [("10276","1038"),("1038","10280"),("10464","10613"),("10886","10593"),
             ("11533","11604"),("11534","11605"),("14288","14037"),("14279","14983"),
             ("17040","17060")]:
    if not (present(a) and present(b)): print(f"  {a}+{b}: missing"); continue
    shared = clusters_of(a) & clusters_of(b)
    print(f"  {a}+{b}: {'SEPARATED' if not shared else 'STILL MERGED'}  "
          f"exact[{a}]={'Y' if exact(a) else 'N'} exact[{b}]={'Y' if exact(b) else 'N'}")

print("\n=== Cluster C: split on motion (want SEPARATED, each exact) ===")
for a, b in [("10600","10601"),("1087","1093"),("11573","11574"),
             ("13339","13340"),("17879","17880")]:
    if not (present(a) and present(b)): print(f"  {a}+{b}: missing"); continue
    shared = clusters_of(a) & clusters_of(b)
    print(f"  {a}+{b}: {'SEPARATED' if not shared else 'STILL MERGED'}  "
          f"exact[{a}]={'Y' if exact(a) else 'N'} exact[{b}]={'Y' if exact(b) else 'N'}")

print("\n=== Praise / safety (want 1 cluster, unchanged) ===")
for g in ["14258","14259","14260","14261","14262"]:
    if not present(g): print(f"  {g}: not in subset"); continue
    print(f"  {g}: {len(clusters_of(g))} cluster(s)  exact_GT={'YES' if exact(g) else 'NO'}")
