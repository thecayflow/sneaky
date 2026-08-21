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


def _current_mtimes(paths: list[Path]) -> dict[str, float]:
    """{path_str: modification_time} for every path, read via stat() —
    cheap (no file content read), used both to detect changes on later
    runs and to record what to compare against next time."""
    mtimes = {}
    for p in paths:
        try:
            mtimes[str(p)] = p.stat().st_mtime
        except OSError:
            continue  # gone between the scan and here — skip, harmless
    return mtimes


def _update_embeddings_incrementally(root_path: str, recursive: bool, embedder: ClipEmbedder | None = None):
    """
    Scan the folder fresh and reconcile against whatever is cached:
      - no cache at all -> embed everything (first run for this folder)
      - cache exists, nothing changed -> return the cache as-is, untouched
      - cache exists, images added/removed/modified -> keep embeddings for
        images that are still present AND unchanged, re-embed anything new
        OR whose modification time changed since it was cached (edited in
        place, same filename — content changes alone, with no path added
        or removed, used to go undetected entirely). The hierarchical
        tree and per-k labels are invalidated, since they were computed
        from the old composition of the dataset.

    A path with no cached mtime (dataset was cached before this feature
    existed) is treated as unchanged, not as a change — this doesn't
    retroactively catch past silent staleness for datasets cached
    earlier, but starts tracking correctly from this point on.

    embedder: reuse an already-loaded ClipEmbedder instead of loading the
    model weights again — pass the same @st.cache_resource-backed instance
    from app.py's get_embedder() when calling this from Streamlit, so
    switching between multiple analyzed datasets in the same session
    doesn't reload CLIP's weights each time new images need embedding.
    None (the default) instantiates one locally, exactly as before — for
    any non-Streamlit caller (scripts, tests).

    Returns (paths, embeddings, skipped).
    """
    scan = scan_folder(root_path, recursive=recursive)
    current_paths_set = {str(p) for p in scan.valid_images}
    current_path_by_str = {str(p): p for p in scan.valid_images}

    cached = cache.load_scan_and_embeddings(root_path)

    if cached is None:
        logger.info("No cache found — embedding all %d images", len(scan.valid_images))
        active_embedder = embedder if embedder is not None else ClipEmbedder()
        result = active_embedder.embed_images(scan.valid_images, batch_size=32)
        paths, embeddings = result.paths, result.embeddings
        skipped = scan.skipped_files + result.failed
        cache.save_scan_and_embeddings(root_path, paths, embeddings, skipped, mtimes=_current_mtimes(paths))
        return paths, embeddings, skipped

    old_paths, old_embeddings, _old_skipped = cached
    old_paths_set = {str(p) for p in old_paths}
    cached_mtimes = cache.load_mtimes(root_path) or {}

    added_paths = [p for p in scan.valid_images if str(p) not in old_paths_set]

    # A "kept" path (still present) whose mtime has changed since it was
    # cached is treated the same as a removed-and-re-added path: its old
    # embedding is dropped and it's re-embedded, instead of silently
    # reusing an embedding computed from stale content. See load_mtimes
    # for why a MISSING cached mtime means "assume unchanged", not
    # "changed".
    modified_path_strs: set[str] = set()
    for p_str in old_paths_set & current_paths_set:
        cached_mtime = cached_mtimes.get(p_str)
        if cached_mtime is None:
            continue
        try:
            current_mtime = current_path_by_str[p_str].stat().st_mtime
        except OSError:
            continue
        if current_mtime != cached_mtime:
            modified_path_strs.add(p_str)

    keep_mask = [
        (str(p) in current_paths_set) and (str(p) not in modified_path_strs) for p in old_paths
    ]
    n_removed_or_modified = keep_mask.count(False)

    if not added_paths and not modified_path_strs and n_removed_or_modified == 0:
        logger.info("Using cached embeddings (%d images), no changes detected", len(old_paths))
        return old_paths, old_embeddings, scan.skipped_files

    modified_paths = [current_path_by_str[p_str] for p_str in modified_path_strs]
    paths_to_embed = added_paths + modified_paths
    n_removed = n_removed_or_modified - len(modified_paths)

    logger.info(
        "Detected changes in %s: %d new image(s), %d modified, %d removed — updating "
        "incrementally instead of recomputing everything",
        root_path,
        len(added_paths),
        len(modified_paths),
        n_removed,
    )

    kept_paths = [p for p, keep in zip(old_paths, keep_mask) if keep]
    kept_embeddings = old_embeddings[np.array(keep_mask, dtype=bool)]

    if paths_to_embed:
        active_embedder = embedder if embedder is not None else ClipEmbedder()
        result = active_embedder.embed_images(paths_to_embed, batch_size=32)
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

    # Carry forward mtimes for KEPT paths (unchanged, no need to re-stat)
    # and record fresh ones for whatever just got (re-)embedded.
    kept_mtimes = {p_str: cached_mtimes[p_str] for p_str in (str(p) for p in kept_paths) if p_str in cached_mtimes}
    new_mtimes = _current_mtimes(new_paths)
    mtimes = {**kept_mtimes, **new_mtimes}

    cache.save_scan_and_embeddings(root_path, paths, embeddings, skipped, mtimes=mtimes)
    cache.invalidate_tree_and_axes(root_path)

    return paths, embeddings, skipped


def get_embeddings_only(root_path: str, recursive: bool = True, embedder: ClipEmbedder | None = None):
    """
    Public entry point for a "lightweight" pipeline run: scan + reconcile
    embeddings only, no clustering or labeling. This is what a comparison
    feed (Fase 2) needs — its images get scored against the PRIMARY feed's
    already-computed axes, so there's no need to cluster or caption a
    second time for it.

    embedder: see _update_embeddings_incrementally — pass a cached
    instance to avoid reloading CLIP's weights.

    Returns (paths, embeddings, skipped) — same shape as
    cache.load_scan_and_embeddings.
    """
    return _update_embeddings_incrementally(root_path, recursive, embedder=embedder)


def run_pipeline(
    root_path: str,
    k: int = 5,
    recursive: bool = True,
    linkage_method: str = "ward",
    embedder: ClipEmbedder | None = None,
    labeler: ClusterLabeler | None = None,
) -> list[cache.AxisRecord]:
    """
    Full pipeline for one dataset folder, using cache wherever possible.
    Returns the list of labeled axes (clusters) for the requested k.

    `linkage_method` ("ward" or "average") is cached independently — trying
    both on the same dataset never invalidates or overwrites the other, so
    switching back and forth is always fast after the first time.

    embedder / labeler: reuse already-loaded instances instead of loading
    CLIP/BLIP's model weights again — pass the same @st.cache_resource-
    backed instances from app.py (get_embedder()/get_labeler()) when
    calling this from Streamlit. Trying several k or linkage_method
    values on the same dataset within one session still needs to redo the
    actual clustering/captioning for each new combination (that part is
    genuinely different work), but the underlying model weights only get
    loaded once per session either way. None (the default) instantiates
    locally, exactly as before — for any non-Streamlit caller.
    """
    # Step 1: scan + reconcile embeddings — the expensive step, but now
    # incremental: only genuinely new images get embedded.
    paths, embeddings, skipped = _update_embeddings_incrementally(root_path, recursive, embedder=embedder)

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
    active_labeler = labeler if labeler is not None else ClusterLabeler()
    cluster_labels = active_labeler.label_clusters(clusters, embeddings, paths)

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
