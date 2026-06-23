"""Fuse the two similarity views, pick a label-free threshold, and cluster.

The gradient descriptor and the wavelet embedding capture complementary
evidence, so their similarity matrices are blended (``W_GRADIENT`` weight on the
gradient view). The cut threshold is chosen *without labels* from the shape of
the predicted-group-count vs threshold curve: the correct operating point sits
at the knee of that curve, where the clustering is most stable.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

W_GRADIENT = 0.65  # weight on the gradient view; wavelet gets the remainder


class AdjacencyGraph:
    """A symmetric boolean "same-group" graph over image indices.

    Refinement passes link nodes (add edges) or read the induced clusters; the
    grouping is always the connected components of the current edge set.
    """

    def __init__(self, adjacency: np.ndarray) -> None:
        self.adjacency = adjacency

    def link(self, i: int, j: int) -> None:
        self.adjacency[i, j] = self.adjacency[j, i] = True

    def labels(self) -> np.ndarray:
        np.fill_diagonal(self.adjacency, False)
        _, labels = connected_components(
            csr_matrix(self.adjacency), directed=False
        )
        return labels

    def clusters(self) -> dict[int, list[int]]:
        members: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels()):
            members[label].append(index)
        return members


class FusionClusterer:
    def __init__(self, gradient: np.ndarray, embedding: np.ndarray) -> None:
        gradient_sim = (gradient @ gradient.T).astype(np.float32)
        embedding_sim = (embedding @ embedding.T).astype(np.float32)
        self.similarity = (
            W_GRADIENT * gradient_sim + (1 - W_GRADIENT) * embedding_sim
        ).astype(np.float32)

    def _graph_at(self, threshold: float) -> np.ndarray:
        adjacency = self.similarity >= threshold
        np.fill_diagonal(adjacency, False)
        return adjacency

    def _plateau_threshold(self) -> float:
        """Label-free cut: the leading edge of the count-curve's flat plateau."""
        grid = np.arange(0.20, 0.90, 0.01)
        counts = np.array(
            [connected_components(csr_matrix(self._graph_at(t)), directed=False)[0]
             for t in grid],
            float,
        )
        window = 3
        slope = np.full(len(grid), np.inf)
        for i in range(window, len(grid) - window):
            slope[i] = (counts[i + window] - counts[i - window]) / (2 * window)
        # ignore the low-threshold floor where everything is merged together
        slope[counts <= 0.5 * counts.max()] = np.inf
        smallest = slope[np.isfinite(slope)].min()
        # the knee = lowest threshold whose slope is within 1.3x (+0.5) of flattest
        cut = 1.3 * smallest + 0.5
        knee = int(np.where(np.isfinite(slope) & (slope <= cut))[0][0])
        return grid[knee]

    def initial_graph(self) -> AdjacencyGraph:
        threshold = self._plateau_threshold()
        return AdjacencyGraph(self._graph_at(threshold))
