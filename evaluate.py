"""Local evaluation harness — reproduces the exact-set score on a labeled set.

Usage:
    python evaluate.py <data_dir>

``<data_dir>`` must contain:
    images/                folder of photos
    public_manifest.csv    filename,group_id answer key
    unfixable.json         (optional) groups that are ground-truth label errors

Scoring matches the competition: a predicted group counts only if its filename
set exactly equals a reference group. The optional "fixable-only" line reports
the score with known ground-truth-error groups excluded — the learnable ceiling.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from autohdr import ImageGrouper
from autohdr.image_loader import ImageLoader


def load_reference(data_dir: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with open(data_dir / "public_manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["group_id"]].add(row["filename"])
    return groups


def load_unfixable(data_dir: Path) -> set[str]:
    path = data_dir / "unfixable.json"
    if not path.exists():
        return set()
    return set(json.load(open(path))["groups"].keys())


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sample")

    photoshoot = ImageLoader(data_dir / "images").load()
    predicted = ImageGrouper().group(photoshoot)
    predicted_sets = {frozenset(group) for group in predicted}

    reference = load_reference(data_dir)
    matched = {g for g, members in reference.items() if frozenset(members) in predicted_sets}
    print(f"{data_dir}: exact-set {len(matched)}/{len(reference)} = "
          f"{len(matched) / len(reference):.4f}")

    unfixable = load_unfixable(data_dir)
    if unfixable:
        fixable = set(reference) - unfixable
        ok = matched & fixable
        print(f"  fixable-only {len(ok)}/{len(fixable)} = {len(ok) / len(fixable):.4f} "
              f"({len(unfixable)} ground-truth-error groups excluded)")
        missed = sorted(fixable - matched)
        if missed:
            print(f"  still-missed fixable groups: {missed}")


if __name__ == "__main__":
    main()
