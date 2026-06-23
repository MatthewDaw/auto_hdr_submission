"""AutoHDR Challenge submission entrypoint.

Container contract:
    Input:  /input/images/          JPEG/PNG from one photoshoot (read-only)
    Output: /output/predictions.csv  filename,group_id

The grouping algorithm lives in the ``autohdr`` package; this file only wires
the container's filesystem contract to it.
"""
import csv
from pathlib import Path

from autohdr import ImageGrouper
from autohdr.image_loader import ImageLoader

INPUT_DIR = Path("/input/images")
OUTPUT_DIR = Path("/output")


def write_predictions(groups: list[list[str]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id"])
        for group_id, group in enumerate(groups):
            for filename in sorted(group):
                writer.writerow([filename, group_id])
                written += 1
    return written


def main() -> None:
    photoshoot = ImageLoader(INPUT_DIR).load()
    print(f"Loaded {photoshoot.count} images from {INPUT_DIR}")

    groups = ImageGrouper().group(photoshoot)
    print(f"Predicted {len(groups)} groups")

    out_path = OUTPUT_DIR / "predictions.csv"
    written = write_predictions(groups, out_path)
    print(f"Wrote {written} predictions to {out_path}")


if __name__ == "__main__":
    main()
