"""Order-robustness sweep. Production keeps the sorted-filename order, but here we
feed the images in many RANDOM orders and check that every learnable (fixable)
group still groups correctly. Any fixable group that fails under some permutation
is order-fragile — a candidate to harden (or, if truly information-limited, to
move to the unfixable list).

Usage: sweep_order.py <data_dir> <n_trials>
Runs from raw256.npz (cached gray) so each trial skips disk decode.
"""
import sys, json, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    raw = np.load(data / "raw256.npz", allow_pickle=True)
    files = list(map(str, raw["files"]))
    gray = raw["imgs"]

    gt = defaultdict(set)
    for r in csv.DictReader(open(data / "public_manifest.csv", encoding="utf-8")):
        gt[r["group_id"]].add(r["filename"])
    unfixable = set(json.load(open(data / "unfixable.json"))["groups"].keys())
    fixable = {g for g in gt if g not in unfixable and f"g{g}" not in unfixable}

    def score(order):
        ps = Photoshoot([files[i] for i in order], gray[order])
        predsets = {frozenset(g) for g in ImageGrouper().group(ps)}
        return {g for g in fixable if frozenset(gt[g]) in predsets}

    base = score(sorted(range(len(files)), key=lambda i: files[i]))
    print(f"{data.name}: {len(fixable)} fixable groups | sorted-order baseline "
          f"{len(base)}/{len(fixable)}", flush=True)

    ever_failed = set(fixable) - base
    for t in range(trials):
        rng = np.random.RandomState(1000 + t)
        order = np.arange(len(files)); rng.shuffle(order)
        matched = score(order)
        failed = set(fixable) - matched
        ever_failed |= failed
        print(f"  trial {t:2d}: {len(matched)}/{len(fixable)} fixable matched"
              f"{'' if not failed else '  FAILED: ' + ', '.join(sorted(failed))}",
              flush=True)

    print(f"\nfixable groups that failed under >=1 order ({len(ever_failed)}): "
          f"{sorted(ever_failed)}", flush=True)


if __name__ == "__main__":
    main()
