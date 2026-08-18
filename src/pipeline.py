"""
src/pipeline.py

Orchestrates the full pipeline (ingestion -> embeddings -> hierarchical
clustering -> labeling), using the disk cache at every expensive step so
re-running on the same folder — or just trying a different k — doesn't
repeat work that's already been done.

This is the entry point the future Streamlit UI will call.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np

from src.axes.hierarchical import HierarchicalAxisEngine
from src.axes.labeling import ClusterLabeler
from src.embeddings.clip_embedder import ClipEmbedder
from src.ingestion.loader import scan_folder
from src.persistence import cache

logger = logging.getLogger(__name__)


def _update_embeddings_incrementally(root_path: str, recursive: bool):
    """
    Scan the folder fresh and reconcile against whatever is cached:
      - no cache at all -> embed everything (first run for this folder)
      - cache exists, nothing changed -> return the cache as-is, untouched
      - cache exists, images added/removed -> keep embeddings for images
        that are still present, embed ONLY the new ones, drop the rest.
        The hierarchical tree and per-k labels are invalidated, since they
        were computed from the old composition of the dataset.

    Returns (paths, embeddings, skipped).
    """
    scan = scan_folder(root_path, recursive=recursive)
    current_paths_set = {str(p) for p in scan.valid_images}

    cached = cache.load_scan_and_embeddings(root_path)

    if cached is None:
        logger.info("No cache found — embedding all %d images", len(scan.valid_images))
        embedder = ClipEmbedder()
        result = embedder.embed_images(scan.valid_images, batch_size=32)
        paths, embeddings = result.paths, result.embeddings
        skipped = scan.skipped_files + result.failed
        cache.save_scan_and_embeddings(root_path, paths, embeddings, skipped)
        return paths, embeddings, skipped

    old_paths, old_embeddings, _old_skipped = cached
    old_paths_set = {str(p) for p in old_paths}

    added_paths = [p for p in scan.valid_images if str(p) not in old_paths_set]
    keep_mask = [str(p) in current_paths_set for p in old_paths]
    n_removed = keep_mask.count(False)

    if not added_paths and n_removed == 0:
        logger.info("Using cached embeddings (%d images), no changes detected", len(old_paths))
        return old_paths, old_embeddings, scan.skipped_files

    logger.info(
        "Detected changes in %s: %d new image(s), %d removed — updating incrementally "
        "instead of recomputing everything",
        root_path,
        len(added_paths),
        n_removed,
    )

    kept_paths = [p for p, keep in zip(old_paths, keep_mask) if keep]
    kept_embeddings = old_embeddings[np.array(keep_mask, dtype=bool)]

    if added_paths:
        embedder = ClipEmbedder()
        result = embedder.embed_images(added_paths, batch_size=32)
        new_paths, new_embeddings, new_failed = result.paths, result.embeddings, result.failed
    else:
        new_paths, new_failed = [], []
        new_embeddings = np.empty((0, kept_embeddings.shape[1]), dtype=kept_embeddings.dtype)

    paths = kept_paths + new_paths
    embeddings = (
        np.concatenate([kept_embeddings, new_embeddings], axis=0) if new_paths else kept_embeddings
    )
    # scan.skipped_files already reflects a full, up-to-date validation pass
    # over the current folder contents, so it alone is the correct skipped
    # list — no need to merge with whatever was skipped last time.
    skipped = scan.skipped_files + new_failed

    cache.save_scan_and_embeddings(root_path, paths, embeddings, skipped)
    cache.invalidate_tree_and_axes(root_path)

    return paths, embeddings, skipped


def get_embeddings_only(root_path: str, recursive: bool = True):
    """
    Public entry point for a "lightweight" pipeline run: scan + reconcile
    embeddings only, no clustering or labeling. This is what a comparison
    feed (Fase 2) needs — its images get scored against the PRIMARY feed's
    already-computed axes, so there's no need to cluster or caption a
    second time for it.

    Returns (paths, embeddings, skipped) — same shape as
    cache.load_scan_and_embeddings.
    """
    return _update_embeddings_incrementally(root_path, recursive)


def run_pipeline(
    root_path: str, k: int = 5, recursive: bool = True, linkage_method: str = "ward"
) -> list[cache.AxisRecord]:
    """
    Full pipeline for one dataset folder, using cache wherever possible.
    Returns the list of labeled axes (clusters) for the requested k.

    `linkage_method` ("ward" or "average") is cached independently — trying
    both on the same dataset never invalidates or overwrites the other, so
    switching back and forth is always fast after the first time.
    """
    # Step 1: scan + reconcile embeddings — the expensive step, but now
    # incremental: only genuinely new images get embedded.
    paths, embeddings, skipped = _update_embeddings_incrementally(root_path, recursive)

    # Step 2: hierarchical tree — cached independently of k (but per
    # linkage_method), since the same tree is reused no matter how many
    # axes you end up asking for. Gets invalidated automatically above if
    # the image set changed.
    Z = cache.load_tree(root_path, linkage_method=linkage_method)
    if Z is not None:
        engine = HierarchicalAxisEngine.from_cache(
            embeddings, Z, linkage_method=linkage_method
        )
        logger.info("Using cached hierarchical tree (%s)", linkage_method)
    else:
        logger.info("No cached tree (%s) — building hierarchical clustering", linkage_method)
        engine = HierarchicalAxisEngine(linkage_method=linkage_method).fit(embeddings)
        cache.save_tree(root_path, engine.Z, linkage_method=linkage_method)

    clusters = engine.get_clusters(k)

    # Step 3: labeled axes — cached per (linkage_method, k), since
    # captioning is the one step that's genuinely expensive to repeat.
    # Also gets invalidated automatically above if the image set changed.
    records = cache.load_axes(root_path, k, linkage_method=linkage_method)
    if records is not None:
        logger.info("Using cached labels for method=%s, k=%d", linkage_method, k)
        return records

    logger.info("No cached labels for method=%s, k=%d — running captioning", linkage_method, k)
    labeler = ClusterLabeler()
    cluster_labels = labeler.label_clusters(clusters, embeddings, paths)

    clusters_by_id = {c.cluster_id: c for c in clusters}
    records = [
        cache.AxisRecord(
            cluster_id=cl.cluster_id,
            label=cl.label,
            size=clusters_by_id[cl.cluster_id].size,
            member_indices=clusters_by_id[cl.cluster_id].member_indices.tolist(),
            centroid=clusters_by_id[cl.cluster_id].centroid.tolist(),
            captions=cl.captions,
            representative_paths=[str(p) for p in cl.representative_paths],
        )
        for cl in cluster_labels
    ]
    cache.save_axes(root_path, k, records, linkage_method=linkage_method)
    return records


if __name__ == "__main__":
    # Usage: python src\pipeline.py "E:\dataset_unificado" 8
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <folder_path> [k]")
        sys.exit(1)

    root = sys.argv[1]
    k_value = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    start = time.time()
    axes = run_pipeline(root, k=k_value)
    elapsed = time.time() - start

    print(f"\nAxes for k={k_value} (took {elapsed:.1f}s):")
    for r in axes:
        print(f"  '{r.label}' ({r.size} images)")
