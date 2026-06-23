"""End-to-end grouping: decoded photoshoot -> groups of filenames."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .clustering import FusionClusterer
from .features import ChromaSignature, GradientDescriptor, MaskedCorrelation, WaveletEmbedding
from .image_loader import Photoshoot
from .refinement import (
    AnchorReattachment,
    AnchorSplitter,
    CaptureRunSplitter,
    ClippedReattachment,
    ClusterMerging,
    ColorMerge,
    ClippedForeignSplitter,
    ContiguityReattachment,
    DriftReattachment,
    HighResSplitter,
    LightMismatchSplitter,
    MotionSplitter,
    OrphanReattachment,
    RefinementContext,
    WellOrphanReattachment,
    SceneChangeSplitter,
    SeedSceneSplitter,
)


class ImageGrouper:
    """Groups a photoshoot's images by camera angle.

    Stateless and training-free: each call fits its descriptors, threshold, and
    refinement entirely from the photoshoot passed in.
    """

    def group(self, photoshoot: Photoshoot) -> list[list[str]]:
        gray = photoshoot.gray

        gradient = GradientDescriptor().encode(gray)
        embedding = WaveletEmbedding().encode(gray)

        graph = FusionClusterer(gradient, embedding).initial_graph()
        ctx = RefinementContext(
            brightness=photoshoot.brightness,
            embedding=embedding,
            masked=MaskedCorrelation(gray),
            filenames=photoshoot.filenames,
            gray=gray,
            chroma=ChromaSignature(photoshoot.color),
        )

        # adjacency-editing passes
        OrphanReattachment().apply(graph, ctx)
        ClusterMerging().apply(graph, ctx)

        # label-editing passes
        labels = graph.labels()
        labels = HighResSplitter().apply(labels, ctx)
        labels = ClippedReattachment().apply(labels, ctx)
        labels = CaptureRunSplitter().apply(labels, ctx)
        labels = SceneChangeSplitter().apply(labels, ctx)
        # FIX10-12 — clipped-frame anchor + localized-motion passes (run last):
        # splits first (separate wrongly-merged scenes), then the reattach merge.
        labels = MotionSplitter().apply(labels, ctx)      # FIX10 (C) localized motion
        labels = AnchorSplitter().apply(labels, ctx)      # FIX11 (B) anchor mismatch
        # FIX17 (B'') split a foreign near-black frame whose surviving lights match
        # NONE of the cluster's other dark frames (clipped-vs-clipped, pixel-only).
        labels = ClippedForeignSplitter().apply(labels, ctx)
        labels = AnchorReattachment().apply(labels, ctx)  # FIX12 (A) clipped reattach
        # FIX16 (A''') rejoin a frame stranded by a small CAMERA DRIFT: phase-correlate
        # to recover the few-pixel shift, and reattach when the drift-aligned edge ZNCC
        # is high though the raw (unshifted) ZNCC was too low for the masked passes.
        labels = DriftReattachment().apply(labels, ctx)
        # FIX8/FIX9 (colour merge + seed split) DISABLED — colour regresses the 1302
        # benchmark in BOTH directions: as a separator it over-splits same-scene
        # brackets (within-bracket colour varies >0.20), as a merger it over-merges
        # different scenes with similar colour (-27). Within/between colour
        # distributions overlap too much on this data. Infra kept for future work.
        # labels = SeedSceneSplitter().apply(labels, ctx)
        # labels = ColorMerge().apply(labels, ctx)

        return self._labels_to_groups(labels, photoshoot.filenames)

    @staticmethod
    def _labels_to_groups(labels: np.ndarray, filenames: list[str]) -> list[list[str]]:
        groups: dict[int, list[str]] = defaultdict(list)
        for label, filename in zip(labels, filenames):
            groups[int(label)].append(filename)
        return list(groups.values())
