"""Post-clustering refinement passes.

The fusion clustering is good but makes two characteristic mistakes that the
masked edge correlation can fix:

  * over-splits — a heavily clipped bracket (near-black or near-white) loses the
    edge content the descriptor relies on, so it strands as a singleton, or a
    scene splits into two pieces across an exposure step;
  * over-merges — two different rooms with a similar layout look alike to the
    coarse 64x64 descriptor.

The passes run in order, each consuming the previous result:

  1. OrphanReattachment  (FIX2)  — clipped orphans rejoin via an exposure ladder
  2. ClusterMerging      (FIX4)  — embedding-near split pieces re-merge
  3. HighResSplitter     (FIX5)  — coarse over-merges split by 256px masked ZNCC
  4. ClippedReattachment (FIX2c) — pure-clipped singletons rejoin by uniqueness
  5. CaptureRunSplitter  (FIX6)  — over-merged distinct scenes split by capture run
"""

from .anchor_reattachment import AnchorReattachment
from .anchor_splitter import AnchorSplitter
from .capture_run_splitter import CaptureRunSplitter
from .scene_change_splitter import SceneChangeSplitter
from .clipped_foreign_splitter import ClippedForeignSplitter
from .clipped_reattachment import ClippedReattachment
from .cluster_merging import ClusterMerging
from .color_merge import ColorMerge
from .contiguity_reattachment import ContiguityReattachment
from .drift_reattachment import DriftReattachment
from .context import RefinementContext
from .highres_splitter import HighResSplitter
from .light_mismatch_splitter import LightMismatchSplitter
from .motion_splitter import MotionSplitter
from .orphan_reattachment import OrphanReattachment
from .seed_scene_splitter import SeedSceneSplitter
from .well_orphan_reattachment import WellOrphanReattachment

__all__ = [
    "RefinementContext",
    "OrphanReattachment",
    "ClusterMerging",
    "HighResSplitter",
    "ClippedReattachment",
    "CaptureRunSplitter",
    "SceneChangeSplitter",
    "SeedSceneSplitter",
    "ColorMerge",
    "MotionSplitter",
    "AnchorSplitter",
    "AnchorReattachment",
    "LightMismatchSplitter",
    "ContiguityReattachment",
    "WellOrphanReattachment",
    "DriftReattachment",
    "ClippedForeignSplitter",
]
