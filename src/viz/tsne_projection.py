"""
src/viz/tsne_projection.py

Reduces the full embedding space (768D with ViT-L-14) down to 2D via t-SNE,
for the scatter-plot alternative to the radar. Cached per dataset (not per
k or linkage method — it only depends on the embeddings themselves), since
t-SNE is too slow to recompute on every interaction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.persistence import cache

logger = logging.getLogger(__name__)

DEFAULT_PCA_COMPONENTS = 50  # matches hierarchical.py's clustering PCA step


def compute_tsne_projection(
    embeddings: np.ndarray,
    random_state: int = 42,
    pca_components: int | None = DEFAULT_PCA_COMPONENTS,
) -> np.ndarray:
    """
    Reduce (n_samples, dim) embeddings to (n_samples, 2) via t-SNE.

    A PCA pre-reduction step (e.g. 768 -> 50 dims) runs first by default —
    standard practice before t-SNE/UMAP on high-dimensional embeddings: it's
    considerably faster (distance computations scale with dimensionality)
    and discards low-variance dimensions that are mostly noise for this
    purpose. Pass pca_components=None to skip it and run t-SNE directly on
    the full embeddings.

    perplexity must be less than n_samples (scikit-learn requirement) — it's
    clamped down automatically for small datasets instead of erroring out.
    """
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(2, n_samples - 1))

    working_space = embeddings
    if pca_components is not None:
        # Same small-dataset guard as hierarchical.py: PCA can't ask for
        # more components than min(n_samples, n_features).
        effective_components = min(pca_components, embeddings.shape[1], n_samples)
        if effective_components < embeddings.shape[1]:
            logger.info(
                "Reducing %d-dim embeddings to %d dims via PCA before t-SNE",
                embeddings.shape[1],
                effective_components,
            )
            pca = PCA(n_components=effective_components, random_state=random_state)
            working_space = pca.fit_transform(embeddings)

    logger.info(
        "Computing t-SNE projection for %d embeddings (perplexity=%d)", n_samples, perplexity
    )
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    coords = tsne.fit_transform(working_space)
    logger.info("t-SNE projection complete")
    return coords


def get_or_compute_tsne(root_path: str, embeddings: np.ndarray) -> np.ndarray:
    """
    Cache-aware wrapper: returns the cached projection if present, otherwise
    computes it and saves it for next time.
    """
    coords = cache.load_tsne(root_path)
    if coords is not None and coords.shape[0] == embeddings.shape[0]:
        return coords

    coords = compute_tsne_projection(embeddings)
    cache.save_tsne(root_path, coords)
    return coords


if __name__ == "__main__":
    # Quick manual test, chained with the cached pipeline:
    #   python src\viz\tsne_projection.py "E:\dataset_unificado"
    import logging as _logging

    from src.persistence import cache as _cache

    _logging.basicConfig(level=_logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python tsne_projection.py <folder_path>")
        sys.exit(1)

    root = sys.argv[1]
    loaded = _cache.load_scan_and_embeddings(root)
    if loaded is None:
        print("No cached embeddings found — run pipeline.py first.")
        sys.exit(1)
    _, embeddings, _ = loaded

    coords = get_or_compute_tsne(root, embeddings)
    print(f"\nProjection shape: {coords.shape}")
    print(f"First 5 points:\n{coords[:5]}")
