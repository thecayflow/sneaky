"""
src/report/metrics.py

Computes the "at a glance" summary metrics for the Dataset Report section
— the numbers that go in the overview cards, independent of how they get
rendered (Streamlit dashboard or PDF export share this module).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.persistence.cache import AxisRecord
from src.scoring.scoring import OTHER_LABEL


@dataclass
class DatasetOverviewMetrics:
    total_images: int
    semantic_coverage_pct: float
    unclassified_pct: float
    unclassified_count: int
    visual_duplicates_pct: float
    visual_duplicates_count: int
    n_semantic_axes: int
    largest_cluster_label: str
    largest_cluster_pct: float
    smallest_cluster_label: str
    smallest_cluster_pct: float
    axis_sizes: dict[str, int]  # label -> image count, semantic axes only (no "Other")


def compute_overview_metrics(
    axis_counts: dict[str, int],
    total_images: int,
    duplicate_image_count: int,
) -> DatasetOverviewMetrics:
    """
    `axis_counts` is the same dict already computed elsewhere
    (scoring.get_axis_counts_by_dominance) — label -> image count, may
    include OTHER_LABEL. `duplicate_image_count` is the number of images
    belonging to a multi-image near-duplicate cluster (from
    phash.compute_duplicate_stats).
    """
    other_count = axis_counts.get(OTHER_LABEL, 0)
    unclassified_pct = (other_count / total_images * 100) if total_images else 0.0
    semantic_coverage_pct = 100.0 - unclassified_pct

    axis_sizes = {label: count for label, count in axis_counts.items() if label != OTHER_LABEL}

    if axis_sizes:
        largest_label = max(axis_sizes, key=axis_sizes.get)
        smallest_label = min(axis_sizes, key=axis_sizes.get)
        largest_pct = axis_sizes[largest_label] / total_images * 100 if total_images else 0.0
        smallest_pct = axis_sizes[smallest_label] / total_images * 100 if total_images else 0.0
    else:
        largest_label, smallest_label, largest_pct, smallest_pct = "—", "—", 0.0, 0.0

    duplicates_pct = (duplicate_image_count / total_images * 100) if total_images else 0.0

    return DatasetOverviewMetrics(
        total_images=total_images,
        semantic_coverage_pct=semantic_coverage_pct,
        unclassified_pct=unclassified_pct,
        unclassified_count=other_count,
        visual_duplicates_pct=duplicates_pct,
        visual_duplicates_count=duplicate_image_count,
        n_semantic_axes=len(axis_sizes),
        largest_cluster_label=largest_label,
        largest_cluster_pct=largest_pct,
        smallest_cluster_label=smallest_label,
        smallest_cluster_pct=smallest_pct,
        axis_sizes=axis_sizes,
    )


def get_representative_images_by_axis(
    axes: list[AxisRecord], top_n_axes: int | None = None, images_per_axis: int = 1
) -> list[tuple[str, Path]]:
    """
    (axis_label, image_path) pairs for a quick visual mosaic — by default,
    EVERY axis that has a representative image (i.e. every auto-detected
    semantic axis; custom text axes have no image of their own, since
    they're not derived from a cluster of real images, so they simply
    don't contribute an entry here). Pass top_n_axes to cap it instead.
    """
    sorted_axes = sorted(axes, key=lambda a: -a.size)
    if top_n_axes is not None:
        sorted_axes = sorted_axes[:top_n_axes]
    pairs: list[tuple[str, Path]] = []
    for axis in sorted_axes:
        for path_str in axis.representative_paths[:images_per_axis]:
            pairs.append((axis.label, Path(path_str)))
    return pairs
