"""
src/scoring/dataset_similarity.py

Dataset-level (not image-level) distributional comparison: "how similar
is this feed's overall visual/semantic makeup to that other feed's?" —
a single summary number, complementary to the per-image similarity work
in src/similarity/phash.py and the axis-by-axis radar comparison.

Implements CLIP-based Maximum Mean Discrepancy (CMMD), following
Jayasumana et al. (Google, 2023) — proposed as a more robust, more
sample-efficient alternative to FID for comparing sets of images. Reuses
the CLIP embeddings we already compute for every image in both feeds, so
this adds no new embedding pass — just a comparison over data we already
have.

scipy is imported lazily (inside the function that uses it) — same
lazy-import pattern as the rest of the project, see BACKLOG.md.
"""

from __future__ import annotations

import numpy as np


def compute_clip_mmd(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """
    Maximum Mean Discrepancy between two sets of CLIP embeddings, using a
    Gaussian RBF kernel. Lower = more similar overall distributions (0 in
    the limit of infinite, identical-distribution samples); no fixed
    upper bound, so this is best read as a RELATIVE number — e.g. useful
    for comparing several candidate second feeds against the same primary
    feed — rather than judged against some universal "good/bad" threshold.

    Uses the unbiased U-statistic MMD² estimator (Gretton et al., 2012):
        MMD² = mean_{i≠j} k(x_i,x_j) + mean_{i≠j} k(y_i,y_j) - 2*mean k(x_i,y_j)
    with the kernel bandwidth set via the median heuristic (a standard,
    parameter-free way to scale an RBF kernel: set it to the median
    pairwise distance in the pooled sample) — avoids needing to hand-tune
    a bandwidth per dataset.

    embeddings_a / embeddings_b: (n_images, dim) arrays, expected
    L2-normalized (as ClipEmbedder already produces).
    """
    from scipy.spatial.distance import cdist

    pooled = np.vstack([embeddings_a, embeddings_b])
    pooled_sq_dists = cdist(pooled, pooled, metric="sqeuclidean")
    nonzero = pooled_sq_dists[pooled_sq_dists > 0]
    median_sq_dist = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = 1.0 / median_sq_dist if median_sq_dist > 0 else 1.0

    def _mean_kernel(X: np.ndarray, Y: np.ndarray, exclude_diagonal: bool) -> float:
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        K = np.exp(-gamma * sq_dists)
        if exclude_diagonal:
            n = X.shape[0]
            if n < 2:
                return 0.0
            np.fill_diagonal(K, 0.0)
            return float(K.sum() / (n * (n - 1)))
        return float(K.mean())

    k_aa = _mean_kernel(embeddings_a, embeddings_a, exclude_diagonal=True)
    k_bb = _mean_kernel(embeddings_b, embeddings_b, exclude_diagonal=True)
    k_ab = _mean_kernel(embeddings_a, embeddings_b, exclude_diagonal=False)

    mmd_squared = k_aa + k_bb - 2 * k_ab
    # Floating-point noise can push a near-zero value slightly negative —
    # clip before the square root.
    return float(max(mmd_squared, 0.0)) ** 0.5


def compute_self_split_mmd(embeddings: np.ndarray, random_state: int = 42) -> float:
    """
    A same-dataset "noise floor" reference for compute_clip_mmd: splits
    `embeddings` into two random halves and computes the MMD between them.
    Since both halves come from the exact same underlying distribution,
    this reflects pure sampling variability — a natural, dataset-specific
    baseline to compare a cross-feed MMD against, instead of an external
    "universal" similarity threshold (there isn't a well-established one
    to cite honestly).
    """
    rng = np.random.default_rng(random_state)
    n = embeddings.shape[0]
    if n < 4:
        return 0.0
    idx = rng.permutation(n)
    half = n // 2
    return compute_clip_mmd(embeddings[idx[:half]], embeddings[idx[half:]])


if __name__ == "__main__":
    # Quick manual sanity check — no dataset needed, just synthetic vectors.
    import numpy as _np

    rng = _np.random.default_rng(0)

    def _random_unit_vectors(n, dim):
        v = rng.normal(size=(n, dim))
        return v / _np.linalg.norm(v, axis=1, keepdims=True)

    # Same distribution vs. same distribution -> should be close to 0.
    a = _random_unit_vectors(500, 768)
    b = _random_unit_vectors(500, 768)
    print("Same distribution, MMD:", compute_clip_mmd(a, b))

    # Clearly different distributions (shifted mean direction) -> should
    # be noticeably higher.
    c = _random_unit_vectors(500, 768) + 0.5
    c = c / _np.linalg.norm(c, axis=1, keepdims=True)
    print("Shifted distribution, MMD:", compute_clip_mmd(a, c))

    # Self-split baseline: splitting ONE dataset in half should give a
    # "noise floor" value, much lower than comparing genuinely different
    # distributions.
    combined = _np.vstack([a, b])
    print("Self-split baseline (noise floor):", compute_self_split_mmd(combined))
