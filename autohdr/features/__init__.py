"""Per-image feature extractors and the masked-correlation comparator."""

from .chroma import ChromaSignature
from .gradient_descriptor import GradientDescriptor
from .masked_correlation import MaskedCorrelation
from .wavelet_embedding import WaveletEmbedding

__all__ = ["GradientDescriptor", "WaveletEmbedding", "MaskedCorrelation", "ChromaSignature"]
