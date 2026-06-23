"""Fast cached evaluator — mirrors evaluate.py's scoring but loads the decoded
raw256.npz + img128c.npz caches instead of re-decoding images/ each run.

Usage: python eval_cached.py data/large
Reports exact-set and (if unfixable.json present) fixable-only, plus missed list.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot


def main() -> None:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
    raw = np.load(data / "raw256.npz", allow_pickle=True)
    files = list(raw["files"]); gray = raw["imgs"]
    col = np.load(data / "img128c.npz", allow_pickle=True)
    cidx = {f: i for i, f in enumerate(col["files"])}; cimgs = col["imgs"]
    color = np.stack([cimgs[cidx[f]] for f in files]) if all(f in cidx for f in files) else None

    ps = Photoshoot(list(files), gray, color)
    predicted = ImageGrouper().group(ps)
    predicted_sets = {frozenset(g) for g in predicted}

    reference: dict[str, set] = defaultdict(set)
    with open(data / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reference[row["group_id"]].add(row["filename"])
    matched = {g for g, m in reference.items() if frozenset(m) in predicted_sets}
    print(f"{data}: exact-set {len(matched)}/{len(reference)} = "
          f"{len(matched)/len(reference):.4f}")

    upath = data / "unfixable.json"
    if upath.exists():
        unfixable = set(json.load(open(upath))["groups"].keys())
        fixable = set(reference) - unfixable
        ok = matched & fixable
        print(f"  fixable-only {len(ok)}/{len(fixable)} = {len(ok)/len(fixable):.4f} "
              f"({len(unfixable)} excluded)")
        missed = sorted(fixable - matched)
        if missed:
            print(f"  still-missed fixable groups: {missed}")


if __name__ == "__main__":
    main()
