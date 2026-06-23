"""AutoHDR image grouping — group real-estate photos by camera angle.

The public entry point is :class:`ImageGrouper`, which turns a folder of images
into a list of groups (each group a list of filenames sharing a camera angle,
across exposure brackets). The algorithm is training-free and label-free: every
threshold and basis is fit per photoshoot from the photoshoot's own images.

Pipeline (see module docstrings for detail):

    images
      -> GradientDescriptor   (Sobel edge signature, exposure-robust)
      -> WaveletEmbedding     (per-run PCA of wavelet detail bands)
      -> FusionClusterer      (fuse the two, label-free plateau threshold, CC)
      -> refinement passes    (exposure-ladder + masked-correlation cleanup)
      -> groups
"""

from .grouper import ImageGrouper

__all__ = ["ImageGrouper"]
