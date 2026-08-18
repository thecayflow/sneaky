"""
src/scoring/scoring.py

Scores every image against every axis (cluster) — the numbers that
ultimately feed the radar chart.

Because both image embeddings and axis centroids are L2-normalized, cosine
similarity is just a dot product. This works identically for:
  - axes coming from hierarchical clustering (their centroid is the mean of
    the cluster's member embeddings), and
  - future custom text axes (their "centroid" would be a CLIP text
    embedding) — the scoring layer doesn't need to know the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np

from src.persistence.cache import AxisRecord


def compute_score_matrix(embeddings: np.ndarray, axes: list[AxisRecord]) -> np.ndarray:
    """
    Compute cosine similarity of every image against every axis centroid.

    Returns an (n_images, n_axes) array, columns in the same order as `axes`.
    """
    centroids = np.array([axis.centroid for axis in axes])  # (n_axes, dim)
    return embeddings @ centroids.T


def get_radar_values(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
) -> dict[str, float]:
    """
    Axis label -> raw mean cosine similarity across ALL images in the
    dataset. Kept as a simple reference/debugging view — the radar itself
    uses get_radar_values_by_dominance or get_radar_values_normalized,
    which are directly comparable to the "images: N" counts shown
    alongside them; this raw mean isn't.
    """
    score_matrix = compute_score_matrix(embeddings, axes)
    means = score_matrix.mean(axis=0)
    return {axis.label: float(v) for axis, v in zip(axes, means)}


def get_radar_values_normalized(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
) -> dict[str, float]:
    """
    Mean similarity per axis, but min-max normalized PER AXIS to its own
    observed range across the dataset before averaging — a corrected
    alternative to get_radar_values(method="mean") for mixed sets of
    auto-detected (image-derived) and custom (text-derived) axes.

    Why: raw cosine similarity for text-derived axes runs on a
    systematically different scale than for image-derived axes, even for
    genuinely well-matching content — CLIP's "modality gap" (see
    get_axis_counts_by_dominance for the fuller explanation). Comparing raw
    means directly across axis types makes custom axes look artificially
    weaker than auto-detected ones on the radar, regardless of how well
    they actually fit the dataset.

    Normalizing each axis to its own [min, max] range before averaging
    reframes the question from "what's this axis's absolute similarity?"
    to "where does this dataset sit, on average, within THIS axis's own
    observed range?" — which is comparable across axis types, since it's
    always relative to what that specific axis itself considers a
    weak-vs-strong match.
    """
    score_matrix = compute_score_matrix(embeddings, axes)
    col_min = score_matrix.min(axis=0, keepdims=True)
    col_max = score_matrix.max(axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # guard against a degenerate axis with zero spread
    normalized = (score_matrix - col_min) / col_range
    means = normalized.mean(axis=0)
    return {axis.label: float(v) for axis, v in zip(axes, means)}


def get_dominant_axis_per_image(score_matrix: np.ndarray, axes: list[AxisRecord]) -> list[str]:
    """
    For each image, the label of the axis it scores highest against.
    Not needed for the radar itself, but this is the piece the future
    t-SNE/UMAP toggle will use to color points by dominant theme.
    """
    dominant_indices = score_matrix.argmax(axis=1)
    return [axes[i].label for i in dominant_indices]


def _standardize_scores(score_matrix: np.ndarray) -> np.ndarray:
    """Z-score each axis's column — see get_axis_counts_by_dominance for why."""
    means = score_matrix.mean(axis=0, keepdims=True)
    stds = score_matrix.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0  # guard against a degenerate axis with zero variance
    return (score_matrix - means) / stds


OTHER_LABEL = "Other"
# Standardized (z-score) similarity an image needs to reach on its BEST
# axis to count as belonging there at all. Below this, even the "winning"
# axis isn't a real match — the image is likely a genuine outlier — so it
# goes to OTHER_LABEL instead of being force-assigned to the least-bad
# option. 0.0 means "at least average" for that axis; negative values are
# more permissive. Pass other_threshold=None anywhere below to disable
# this and always force-assign (the old behavior).
DEFAULT_OTHER_THRESHOLD = -0.3


def get_dominant_labels(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
    other_threshold: float | None = DEFAULT_OTHER_THRESHOLD,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    For each image, the label of its best-matching axis — or OTHER_LABEL
    if even its best (standardized) score doesn't clear `other_threshold`,
    meaning it doesn't clearly belong to any currently active axis rather
    than being forced into the least-bad one.

    Returns (labels, score_matrix, standardized_matrix) — the two matrices
    are returned alongside so callers that also need raw similarity (e.g.
    ranking images within an axis) don't have to recompute them.
    """
    score_matrix = compute_score_matrix(embeddings, axes)
    standardized = _standardize_scores(score_matrix)

    dominant_indices = standardized.argmax(axis=1)
    best_scores = standardized.max(axis=1)

    labels = []
    for i, axis_idx in enumerate(dominant_indices):
        if other_threshold is not None and best_scores[i] < other_threshold:
            labels.append(OTHER_LABEL)
        else:
            labels.append(axes[axis_idx].label)

    return labels, score_matrix, standardized


def get_axis_counts_by_dominance(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
    other_threshold: float | None = DEFAULT_OTHER_THRESHOLD,
) -> dict[str, int]:
    """
    Recompute "how many images belong to each axis" by dominant similarity
    across the FULL current set of active axes (auto-detected + any custom
    text axes), instead of using each axis's original hard cluster size.
    Images that don't clearly belong anywhere land in OTHER_LABEL instead
    of being forced into the least-bad axis — see get_dominant_labels.

    Scores are standardized per axis (z-score) before comparing across
    axes, rather than compared as raw cosine similarity. This corrects for
    CLIP's well-documented "modality gap": image-derived centroids
    (clustering axes) systematically score higher in raw cosine similarity
    than text-derived centroids (custom axes), even for genuinely relevant
    images — image-image similarities simply run higher than image-text
    similarities on average. Comparing "how unusually high is this image's
    score for axis A" (z-score) instead of the raw magnitude puts axes of
    both kinds on a fair footing.

    This is what makes custom axes behave correctly: adding a new axis
    like "sky" can pull images away from whichever existing axis they used
    to dominate, without ever rebuilding the hierarchical tree — the tree
    only ever produced the *initial proposal*, this is the up-to-date
    count.
    """
    from collections import Counter

    labels, _, _ = get_dominant_labels(embeddings, axes, other_threshold=other_threshold)
    counts = Counter(labels)

    result = {axis.label: counts.get(axis.label, 0) for axis in axes}
    if other_threshold is not None and counts.get(OTHER_LABEL):
        result[OTHER_LABEL] = counts[OTHER_LABEL]
    return result


def get_ranked_images_for_axis(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
    paths: list[Path],
    axis_label: str,
    other_threshold: float | None = DEFAULT_OTHER_THRESHOLD,
) -> list[tuple[Path, float]]:
    """
    Images assigned to `axis_label` (same rule as get_axis_counts_by_dominance,
    including OTHER_LABEL), ordered so the most relevant ones come first.

    For a normal axis: ordered by raw cosine similarity to its centroid,
    closest first. For OTHER_LABEL: there's no centroid to rank against, so
    images are ordered by their best (standardized) score ascending — the
    ones that fit LEAST well anywhere come first, since those are the
    clearest outliers.
    """
    labels, score_matrix, standardized = get_dominant_labels(
        embeddings, axes, other_threshold=other_threshold
    )
    matches = np.array([i for i, lbl in enumerate(labels) if lbl == axis_label])
    if len(matches) == 0:
        return []

    if axis_label == OTHER_LABEL:
        best_scores = standardized.max(axis=1)[matches]
        order = np.argsort(best_scores)  # ascending: least-fitting first
        return [(paths[matches[i]], float(best_scores[i])) for i in order]

    axis_index = next(i for i, a in enumerate(axes) if a.label == axis_label)
    raw_scores = score_matrix[matches, axis_index]
    order = np.argsort(-raw_scores)  # descending: closest to centroid first
    return [(paths[matches[i]], float(raw_scores[i])) for i in order]


def get_radar_values_by_dominance(
    embeddings: np.ndarray,
    axes: list[AxisRecord],
    other_threshold: float | None = DEFAULT_OTHER_THRESHOLD,
) -> dict[str, float]:
    """
    Axis label -> fraction of the dataset for which this axis is the
    dominant match (see get_axis_counts_by_dominance). May include
    OTHER_LABEL — exclude it before plotting on the radar, since it has no
    real centroid/direction; it belongs in the axis list, not the chart.

    This is what the radar actually plots: the raw mean-similarity value
    (get_radar_values) and the "images: N" dominance count measure
    genuinely different things — average presence across the WHOLE
    dataset vs. a hard "wins the competition" count — so an axis can have
    a high mean similarity but a modest dominance count, or vice versa.
    Plotting the dominance fraction instead keeps the chart consistent
    with the image counts shown alongside it.
    """
    counts = get_axis_counts_by_dominance(embeddings, axes, other_threshold=other_threshold)
    total = len(embeddings)
    if total == 0:
        return {label: 0.0 for label in counts}
    return {label: count / total for label, count in counts.items()}


if __name__ == "__main__":
    # Quick manual test, chained with the full cached pipeline:
    #   python src\scoring\scoring.py "E:\dataset_unificado" 8
    import logging

    from src.pipeline import run_pipeline
    from src.persistence import cache

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python scoring.py <folder_path> [k]")
        sys.exit(1)

    root = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    axes = run_pipeline(root, k=k)

    cached = cache.load_scan_and_embeddings(root)
    if cached is None:
        print("No cached embeddings found — run pipeline.py first.")
        sys.exit(1)
    _, embeddings, _ = cached

    score_matrix = compute_score_matrix(embeddings, axes)

    print("\nRadar values (raw mean similarity per axis):")
    for label, value in get_radar_values(embeddings, axes).items():
        print(f"  '{label}': {value:.4f}")

    print("\nRadar values (normalized, per-axis min-max):")
    for label, value in get_radar_values_normalized(embeddings, axes).items():
        print(f"  '{label}': {value:.4f}")

    dominant = get_dominant_axis_per_image(score_matrix, axes)
    print("\nDominant axis counts (sanity check vs cluster sizes):")
    from collections import Counter

    for label, count in Counter(dominant).most_common():
        print(f"  '{label}': {count} images")
