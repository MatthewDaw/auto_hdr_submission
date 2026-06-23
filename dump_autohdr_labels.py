"""Run the real autohdr.ImageGrouper on a dataset and dump per-frame cluster
labels -> <data>/pred_labels.json {filename: cluster_id}. This is the submission
model's actual output (incl. the HighResSplitter), so the unfixable report shows
what we really predict. Usage: dump_autohdr_labels.py <data_dir>"""
import sys, json
import multiprocessing
from pathlib import Path
from autohdr import ImageGrouper
from autohdr.image_loader import ImageLoader


def main():
    DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
    photoshoot = ImageLoader(DATA / "images").load()
    groups = ImageGrouper().group(photoshoot)
    labels = {}
    for cid, group in enumerate(groups):
        for fn in group:
            labels[fn] = cid
    json.dump(labels, open(DATA / "pred_labels.json", "w"))
    print(f"wrote {DATA}/pred_labels.json ({len(labels)} frames, {len(groups)} clusters)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
