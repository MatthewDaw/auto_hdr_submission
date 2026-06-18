"""
AutoHDR Challenge — Sample Submission

This is a starter Docker submission. It reads test images from /input/images/
and writes predictions.csv to /output/.

To use:
    1. Replace the group_images() logic with your algorithm
    2. Add any pip packages you need to the Dockerfile
    3. Build:  docker build -t yourusername/autohdr-solution:v1 .
    4. Test:   docker run -v /path/to/test/images:/input/images:ro -v /tmp/output:/output yourusername/autohdr-solution:v1
    5. Push:   docker push yourusername/autohdr-solution:v1
    6. Submit your image name on the challenge platform

Contract:
    Input:  /input/images/  — JPEG images from a single photoshoot (read-only)
    Output: /output/predictions.csv — your grouping predictions

predictions.csv format:
    filename,group_id
    IMG_001.jpg,0
    IMG_002.jpg,0
    IMG_003.jpg,1
    ...

Images in the same group share a camera angle. group_id can be any string/number,
it just needs to be consistent within a group.
"""

import csv
import os
from pathlib import Path

INPUT_DIR = Path("/input/images")
OUTPUT_DIR = Path("/output")
SUPPORTED = {".jpg", ".jpeg", ".png"}


def group_images(image_paths: list[str]) -> list[list[str]]:
    """
    Group images by camera angle.

    Args:
        image_paths: List of absolute file paths to images.

    Returns:
        List of groups. Each group is a list of filenames (basenames only).

    Replace this with your algorithm!
    """
    # -------------------------------------------
    # BASELINE: each image in its own group
    # This scores ~25% — replace with your algo
    # -------------------------------------------
    return [[os.path.basename(p)] for p in image_paths]


def main():
    # Load images
    images = sorted([
        str(p) for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED
    ])
    print(f"Loaded {len(images)} images from {INPUT_DIR}")

    # Run grouping
    groups = group_images(images)
    print(f"Predicted {len(groups)} groups")

    # Write predictions.csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id"])
        for group_id, group in enumerate(groups):
            for filename in sorted(group):
                writer.writerow([os.path.basename(filename), group_id])

    print(f"Wrote {sum(len(g) for g in groups)} predictions to {out_path}")


if __name__ == "__main__":
    main()
