"""Process a (possibly still-downloading) image folder into a report-ready dataset.

Freezes the current file list, derives ground-truth groups from the g<gid>_
filename prefix, decodes the gray + color caches, and runs the grouping model in
GROUP-PRESERVING chunks (so the dense N×N similarity never exceeds one chunk and
no group is ever split across chunks). Writes, into <data_dir>:
    public_manifest.csv   filename,group_id   (group_id = the g<gid> prefix)
    raw256.npz            gray tiles  (brightness + masked correlation)
    img128c.npz           color tiles (report thumbnails)
    pred_labels.json      {filename: cluster_id}

Usage: process_subset.py <data_dir> [chunk_size]
"""
import sys, csv, json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autohdr import ImageGrouper
from autohdr.image_loader import Photoshoot, _to_gray256


def _color128(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((128, 128, 3), np.uint8)
    return cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)


def _gid(name):
    return name.split("_", 1)[0][1:]  # 'g1000_x.jpg' -> '1000'


def main():
    data = Path(sys.argv[1])
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = no cap
    img_dir = data / "images"

    files = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images and len(files) > max_images:
        # keep whole groups only: cut at the last complete g<gid> boundary <= cap
        cut = max_images
        while cut > 0 and _gid(files[cut - 1]) == _gid(files[min(cut, len(files) - 1)]):
            cut -= 1
        files = files[:cut]
        print(f"capped to {len(files)} images (memory-safe, whole groups)", flush=True)
    print(f"frozen file list: {len(files)} images, {len({_gid(f) for f in files})} groups",
          flush=True)
    paths = [img_dir / f for f in files]

    # manifest from prefixes
    with open(data / "public_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["filename", "group_id"])
        for f in files:
            w.writerow([f, _gid(f)])

    cv2.setNumThreads(1)
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        gray = np.asarray(list(pool.imap(_to_gray256, paths, chunksize=32)), np.uint8)
        print(f"decoded gray {gray.shape}", flush=True)
        color = np.asarray(list(pool.imap(_color128, paths, chunksize=32)), np.uint8)
        print(f"decoded color {color.shape}", flush=True)

    gid = np.array([_gid(f) for f in files])
    np.savez_compressed(data / "raw256.npz", imgs=gray, files=np.array(files), gid=gid)
    np.savez_compressed(data / "img128c.npz", imgs=color, files=np.array(files), gid=gid)
    print("wrote raw256.npz + img128c.npz", flush=True)

    # group-preserving chunks
    by_group = {}
    for i, f in enumerate(files):
        by_group.setdefault(_gid(f), []).append(i)
    chunks, cur = [], []
    for members in by_group.values():
        if cur and len(cur) + len(members) > chunk_size:
            chunks.append(cur); cur = []
        cur += members
    if cur:
        chunks.append(cur)
    print(f"{len(chunks)} group-preserving chunks "
          f"(sizes {[len(c) for c in chunks]})", flush=True)

    labels, next_id = {}, 0
    for ci, idxs in enumerate(chunks):
        ps = Photoshoot([files[i] for i in idxs], gray[idxs], color[idxs])
        groups = ImageGrouper().group(ps)
        for grp in groups:
            for fn in grp:
                labels[fn] = next_id
            next_id += 1
        print(f"  chunk {ci}: {len(idxs)} imgs -> {len(groups)} clusters", flush=True)

    json.dump(labels, open(data / "pred_labels.json", "w"))
    print(f"wrote pred_labels.json ({len(labels)} frames, {next_id} clusters)", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
