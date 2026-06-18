"""
Local evaluation harness for the AutoHDR grouping challenge.

Runs solution.group_images() over a local dataset folder and scores the
result against that dataset's public_manifest.csv using the exact-match
metric from SCORING.md:

    score = |reference_groups & predicted_groups| / |reference_groups|

where each group is compared as a frozenset of filenames (order, group-id
values, and row order are all irrelevant).

Usage:
    python evaluate.py data/large
    python evaluate.py data/large --csv predictions.csv   # also dump predictions

The dataset folder must contain:
    images/                 the .jpg files
    public_manifest.csv     columns: group_id, filename
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import solution


SUPPORTED = {".jpg", ".jpeg", ".png"}


def groups_to_frozensets(group_to_files):
    """dict[group_id] -> set(filenames)  ==>  set of frozensets."""
    return {frozenset(files) for files in group_to_files.values()}


def load_reference(manifest_path):
    buckets = defaultdict(set)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            buckets[row["group_id"]].add(row["filename"])
    return groups_to_frozensets(buckets)


def run_solution(images_dir):
    """Call the user's group_images() and return set of frozensets + raw groups."""
    image_paths = sorted(
        str(p) for p in Path(images_dir).iterdir()
        if p.suffix.lower() in SUPPORTED
    )
    print(f"Loaded {len(image_paths)} images from {images_dir}")

    t0 = time.time()
    groups = solution.group_images(image_paths)
    elapsed = time.time() - t0
    print(f"Predicted {len(groups)} groups in {elapsed:.1f}s")

    # groups is a list of lists of basenames
    predicted = {frozenset(os.path.basename(f) for f in g) for g in groups}
    return predicted, groups


def score(reference, predicted):
    matches = reference & predicted
    return len(matches) / len(reference), len(matches)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset folder (contains images/ and public_manifest.csv)")
    ap.add_argument("--csv", help="optional path to also write predictions.csv")
    args = ap.parse_args()

    root = Path(args.dataset)
    images_dir = root / "images"
    manifest = root / "public_manifest.csv"
    if not images_dir.is_dir():
        sys.exit(f"Missing images dir: {images_dir}")
    if not manifest.is_file():
        sys.exit(f"Missing manifest: {manifest}")

    reference = load_reference(manifest)
    predicted, groups = run_solution(images_dir)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "group_id"])
            for gid, group in enumerate(groups):
                for fn in group:
                    w.writerow([os.path.basename(fn), gid])
        print(f"Wrote predictions to {args.csv}")

    s, n_match = score(reference, predicted)
    print("-" * 40)
    print(f"Reference groups : {len(reference)}")
    print(f"Predicted groups : {len(predicted)}")
    print(f"Exact matches    : {n_match}")
    print(f"SCORE            : {s:.4f}")


if __name__ == "__main__":
    main()
