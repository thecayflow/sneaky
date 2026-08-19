"""
src/axes/hierarchical.py

Builds a single hierarchical clustering tree over the dataset's CLIP
embeddings, and lets the caller "cut" that tree at any number of clusters
(k = number of radar axes). Because it's the same tree at every k, going
from k axes to k+1 subdivides one existing cluster instead of reshuffling
everything — which is what makes the +/- axis buttons feel coherent in
the UI, unlike re-running flat KMeans at each k.

Uses Ward linkage (minimizes within-cluster variance) on a PCA-reduced
version of the embeddings. Average/complete linkage on raw cosine distance
tends to "chain" in high-dimensional CLIP embedding space — one giant
cluster absorbs almost everything while a handful of outlier images form
tiny singleton clusters. Ward + PCA gives much more balanced, semantically
meaningful clusters in practice.

scipy and scikit-learn are imported lazily (inside fit()/get_clusters()) —
importing this module shouldn't pay their import cost until clustering
actually runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MIN_AXES = 3  # matches the product decision: minimum of 3 auto-detected axes
DEFAULT_PCA_COMPONENTS = 50


@dataclass
class Cluster:
    """One detected semantic cluster (a candidate radar axis)."""

    cluster_id: int
    member_indices: np.ndarray  # indices into the original embeddings array
    centroid: np.ndarray  # L2-normalized mean embedding of the cluster's members (original space)

    @property
    def size(self) -> int:
        return len(self.member_indices)


class HierarchicalAxisEngine:
    """
    Computes the full hierarchical clustering tree once, then exposes cuts
    of that tree at any k (number of axes) via `get_clusters(k)`.
    """

    def __init__(
        self,
        linkage_method: str = "ward",
        pca_components: int | None = DEFAULT_PCA_COMPONENTS,
    ) -> None:
        self.linkage_method = linkage_method
        self.pca_components = pca_components
        self._embeddings: np.ndarray | None = None  # original-space embeddings (for centroids)
        self._Z: np.ndarray | None = None  # scipy linkage matrix

    @property
    def Z(self) -> np.ndarray | None:
        """The raw scipy linkage matrix — useful for persisting to disk."""
        return self._Z

    @classmethod
    def from_cache(
        cls,
        embeddings: np.ndarray,
        Z: np.ndarray,
        linkage_method: str = "ward",
        pca_components: int | None = DEFAULT_PCA_COMPONENTS,
    ) -> "HierarchicalAxisEngine":
        """
        Reconstruct an engine from a previously computed linkage matrix
        (e.g. loaded from cache), skipping the (relatively cheap, but not
        free) PCA + linkage computation in `fit()`.
        """
        engine = cls(linkage_method=linkage_method, pca_components=pca_components)
        engine._embeddings = embeddings
        engine._Z = Z
        return engine

    def fit(self, embeddings: np.ndarray) -> "HierarchicalAxisEngine":
        """
        Build the hierarchical tree once for the given (n_samples, dim)
        L2-normalized embeddings.
        """
        from scipy.cluster.hierarchy import linkage
        from sklearn.decomposition import PCA

        n = embeddings.shape[0]
        self._embeddings = embeddings

        clustering_space = embeddings
        # PCA can't ask for more components than min(n_samples, n_features)
        # — clamp automatically instead of erroring out on small datasets
        # (e.g. a folder with only a handful of images).
        effective_pca_components = None
        if self.pca_components is not None:
            effective_pca_components = min(self.pca_components, embeddings.shape[1], n)

        if effective_pca_components is not None and effective_pca_components < embeddings.shape[1]:
            if effective_pca_components < self.pca_components:
                logger.info(
                    "Requested pca_components=%d but only %d samples available — "
                    "using pca_components=%d instead",
                    self.pca_components,
                    n,
                    effective_pca_components,
                )
            logger.info(
                "Reducing %d-dim embeddings to %d dims via PCA before clustering",
                embeddings.shape[1],
                effective_pca_components,
            )
            pca = PCA(n_components=effective_pca_components, random_state=42)
            clustering_space = pca.fit_transform(embeddings)
            explained = pca.explained_variance_ratio_.sum()
            logger.info("PCA retains %.1f%% of variance", explained * 100)

        logger.info(
            "Building hierarchical tree over %d embeddings (method=%s)",
            n,
            self.linkage_method,
        )
        # Ward (and centroid/median) require the raw coordinates, not a
        # precomputed distance matrix — scipy computes euclidean distances
        # internally as part of the ward criterion itself.
        self._Z = linkage(clustering_space, method=self.linkage_method)

        logger.info("Hierarchical tree built.")
        return self

    def get_clusters(self, k: int) -> list[Cluster]:
        """
        Cut the tree at `k` clusters and return them with their centroids.
        `k` is clamped to [MIN_AXES, n_samples].
        """
        if self._Z is None or self._embeddings is None:
            raise RuntimeError("Call fit(embeddings) before get_clusters(k).")

        from scipy.cluster.hierarchy import fcluster

        n = self._embeddings.shape[0]
        k = max(MIN_AXES, min(k, n))

        labels = fcluster(self._Z, t=k, criterion="maxclust")  # 1-indexed cluster ids

        clusters: list[Cluster] = []
        for cluster_id in np.unique(labels):
            member_indices = np.where(labels == cluster_id)[0]
            centroid = self._embeddings[member_indices].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            clusters.append(
                Cluster(
                    cluster_id=int(cluster_id),
                    member_indices=member_indices,
                    centroid=centroid,
                )
            )

        # Largest clusters first — usually what you want to show/label first.
        clusters.sort(key=lambda c: c.size, reverse=True)

        logger.info(
            "Cut tree at k=%d: %d clusters, sizes=%s",
            k,
            len(clusters),
            [c.size for c in clusters],
        )
        return clusters


if __name__ == "__main__":
    # Quick manual test, chained with ingestion + embeddings:
    #   python src\axes\hierarchical.py "E:\dataset_unificado"
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.embeddings.clip_embedder import ClipEmbedder
    from src.ingestion.loader import scan_folder

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python hierarchical.py <folder_path>")
        sys.exit(1)

    scan = scan_folder(sys.argv[1], recursive=True)
    embedder = ClipEmbedder()
    result = embedder.embed_images(scan.valid_images, batch_size=32)

    engine = HierarchicalAxisEngine().fit(result.embeddings)

    for k in (3, 5, 7):
        clusters = engine.get_clusters(k)
        print(f"\nk={k}:")
        for c in clusters:
            print(f"  cluster {c.cluster_id}: {c.size} images")
