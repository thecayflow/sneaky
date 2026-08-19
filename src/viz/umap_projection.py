"""
src/viz/umap_projection.py

Reduces the full embedding space down to 2D via UMAP — the second
projection method alongside t-SNE, cached independently so switching
between them never invalidates or recomputes the other.

umap and scikit-learn's PCA are imported lazily (inside the function that
uses them) — umap in particular has a real import cost (it compiles parts
of itself via numba on first use), so importing this module shouldn't pay
that cost until a UMAP projection is actually requested.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np

from src.persistence import cache

logger = logging.getLogger(__name__)

DEFAULT_PCA_COMPONENTS = 50  # matches hierarchical.py's clustering PCA step


def compute_umap_projection(
    embeddings: np.ndarray,
    random_state: int = 42,
    pca_components: int | None = DEFAULT_PCA_COMPONENTS,
) -> np.ndarray:
    """
    Reduce (n_samples, dim) embeddings to (n_samples, 2) via UMAP.

    A PCA pre-reduction step (e.g. 768 -> 50 dims) runs first by default,
    same rationale as tsne_projection.py — mainly a speed win here (UMAP's
    approximate nearest-neighbor search is already fairly robust to high
    dimensionality on its own, less so than t-SNE benefits, but distance
    computations are still cheaper in fewer dimensions). Pass
    pca_components=None to skip it and run UMAP directly on the full
    embeddings.
    """
    import umap
    from sklearn.decomposition import PCA

    n_samples = embeddings.shape[0]
    # UMAP's n_neighbors must be less than n_samples, same idea as t-SNE's
    # perplexity clamp — avoids errors on very small datasets.
    n_neighbors = min(15, max(2, n_samples - 1))

    working_space = embeddings
    if pca_components is not None:
        effective_components = min(pca_components, embeddings.shape[1], n_samples)
        if effective_components < embeddings.shape[1]:
            logger.info(
                "Reducing %d-dim embeddings to %d dims via PCA before UMAP",
                embeddings.shape[1],
                effective_components,
            )
            pca = PCA(n_components=effective_components, random_state=random_state)
            working_space = pca.fit_transform(embeddings)

    logger.info(
        "Computing UMAP projection for %d embeddings (n_neighbors=%d)", n_samples, n_neighbors
    )
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
    coords = reducer.fit_transform(working_space)
    logger.info("UMAP projection complete")
    return coords


def get_or_compute_umap(root_path: str, embeddings: np.ndarray) -> np.ndarray:
    """
    Cache-aware wrapper: returns the cached projection if present, otherwise
    computes it and saves it for next time.
    """
    coords = cache.load_umap(root_path)
    if coords is not None and coords.shape[0] == embeddings.shape[0]:
        return coords

    coords = compute_umap_projection(embeddings)
    cache.save_umap(root_path, coords)
    return coords


if __name__ == "__main__":
    # Quick manual test, chained with the cached pipeline:
    #   python src\viz\umap_projection.py "E:\dataset_unificado"
    import logging as _logging

    from src.persistence import cache as _cache

    _logging.basicConfig(level=_logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python umap_projection.py <folder_path>")
        sys.exit(1)

    root = sys.argv[1]
    loaded = _cache.load_scan_and_embeddings(root)
    if loaded is None:
        print("No cached embeddings found — run pipeline.py first.")
        sys.exit(1)
    _, embeddings, _ = loaded

    coords = get_or_compute_umap(root, embeddings)
    print(f"\nProjection shape: {coords.shape}")
    print(f"First 5 points:\n{coords[:5]}")
