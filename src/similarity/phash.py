"""
src/similarity/phash.py

Pixel-level visual similarity via perceptual hashing (pHash) — deliberately
independent of CLIP/semantic embeddings entirely. Two images can be
semantically identical ("a dog in a park") but visually very different, or
near-duplicate visually while CLIP treats them as merely "similar" — this
is a different, complementary lens: near-identical crops, recompressions,
minor color/exposure edits, actual duplicates.

Comparing two pHashes is just their Hamming distance (bit differences) —
essentially free computationally, no GPU or model needed.

scipy is imported lazily (inside the functions that use it) — importing
this module shouldn't pay that cost until it's actually needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import imagehash
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from src.persistence import cache

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times.
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

DEFAULT_CHAIN_MAX_LENGTH = 200
DEFAULT_GROUP_THRESHOLD_BITS = 10  # near-duplicate cutoff, out of 64 bits total


def compute_phashes(
    paths: list[Path],
    progress_callback=None,
) -> dict[str, imagehash.ImageHash]:
    """
    Compute a perceptual hash per image. Skips (and logs) unreadable files.

    `progress_callback`, if given, is called as progress_callback(done, total)
    after every image — lets a caller (e.g. a Streamlit progress bar) show
    real incremental progress instead of a single opaque spinner, since this
    can take a while for large datasets and Streamlit's spinner doesn't
    always render reliably for one long blocking call.
    """
    hashes: dict[str, imagehash.ImageHash] = {}
    total = len(paths)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                hashes[str(path)] = imagehash.phash(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to hash %s: %s", path, exc)
        if progress_callback is not None:
            progress_callback(i + 1, total)
    return hashes


def get_or_compute_phashes(
    root_path: str, paths: list[Path], progress_callback=None
) -> dict[str, imagehash.ImageHash]:
    """
    Cache-aware wrapper. Recomputes (for everyone, not incrementally) if the
    cached hash set doesn't exactly match the current path list — pHash is
    cheap enough (no neural net) that a full recompute on a changed dataset
    is acceptable, unlike CLIP embeddings.
    """
    current_path_strs = {str(p) for p in paths}
    cached = cache.load_phashes(root_path)

    if cached is not None and set(cached.keys()) == current_path_strs:
        return {path_str: imagehash.hex_to_hash(hex_str) for path_str, hex_str in cached.items()}

    logger.info("Computing perceptual hashes for %d images", len(paths))
    hashes = compute_phashes(paths, progress_callback=progress_callback)
    # The hashes themselves are about to change — anything derived from the
    # OLD ones (global order, duplicate stats) is now stale too. Clear
    # first, then save the fresh phashes, so nothing gets deleted right
    # after being written.
    cache.clear_similarity_cache(root_path)
    cache.save_phashes(root_path, {path_str: str(h) for path_str, h in hashes.items()})
    return hashes


def build_similarity_chain(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    start_index: int = 0,
    max_length: int = DEFAULT_CHAIN_MAX_LENGTH,
) -> list[tuple[Path, int | None]]:
    """
    Greedy nearest-neighbor walk: start at `paths[start_index]`, repeatedly
    jump to the closest not-yet-visited image (by Hamming distance), until
    `max_length` images are visited or none remain.

    Returns a list of (path, distance_from_previous) — the first entry's
    distance is None (nothing to compare it to).
    """
    available = [p for p in paths if str(p) in hashes]
    if not available:
        return []

    start_index = min(start_index, len(available) - 1)
    current = available.pop(start_index)
    chain: list[tuple[Path, int | None]] = [(current, None)]

    while available and len(chain) < max_length:
        current_hash = hashes[str(current)]
        best_idx, best_distance = None, None
        for i, candidate in enumerate(available):
            distance = current_hash - hashes[str(candidate)]
            if best_distance is None or distance < best_distance:
                best_idx, best_distance = i, distance

        current = available.pop(best_idx)
        chain.append((current, best_distance))

    return chain


def compute_global_order(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
) -> list[tuple[Path, int | None]]:
    """
    A single globally-smooth ordering over ALL hashed images — not just a
    greedy "nearest so far" walk (build_similarity_chain), which can jump
    abruptly once a tight cluster of near-duplicates is exhausted.

    Uses scipy's optimal_leaf_ordering: build a hierarchical clustering tree
    over the hashes' Hamming distances, then reorder its leaves to minimize
    the sum of distances between CONSECUTIVE images in the final sequence —
    the actual "traveling salesman"-style objective this feature wants,
    approximated well (not exactly solved — that's computationally
    infeasible at this scale — but the leaf-ordering approximation is a
    well-established standard technique for exactly this kind of problem).

    Can be noticeably slower than the greedy chain for large datasets —
    cache the result (see cache.save_global_order).
    """
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    available = [p for p in paths if str(p) in hashes]
    if len(available) < 2:
        return [(p, None) for p in available]

    # Each ImageHash wraps a 2D boolean array — flatten to a feature vector
    # per image so scipy can compute all pairwise Hamming distances at once
    # (vectorized), instead of looping in Python over every pair.
    vectors = np.array([hashes[str(p)].hash.flatten() for p in available])

    logger.info("Computing pairwise distances for %d images", len(available))
    condensed = pdist(vectors, metric="hamming")  # fraction of differing bits

    logger.info("Building hierarchical tree and optimal leaf ordering")
    Z = linkage(condensed, method="average")
    Z_ordered = optimal_leaf_ordering(Z, condensed)
    order = leaves_list(Z_ordered)

    ordered_paths = [available[i] for i in order]

    chain: list[tuple[Path, int | None]] = [(ordered_paths[0], None)]
    for i in range(1, len(ordered_paths)):
        prev_hash = hashes[str(ordered_paths[i - 1])]
        curr_hash = hashes[str(ordered_paths[i])]
        chain.append((ordered_paths[i], int(curr_hash - prev_hash)))

    logger.info("Global ordering complete")
    return chain


def get_or_compute_global_order(
    root_path: str,
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
) -> list[tuple[Path, int | None]]:
    """Cache-aware wrapper around compute_global_order."""
    current_path_strs = {str(p) for p in paths if str(p) in hashes}
    cached = cache.load_global_order(root_path)
    if cached is not None and {p for p, _ in cached} == current_path_strs:
        return [(Path(p), d) for p, d in cached]

    chain = compute_global_order(paths, hashes)
    cache.save_global_order(root_path, [(str(p), d) for p, d in chain])
    return chain


def _compute_duplicate_groups(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    threshold_bits: int,
) -> tuple[list[Path], dict[int, list[int]]]:
    """
    Shared grouping logic behind build_grouped_chain and
    compute_duplicate_stats — connected components on the "close enough"
    (Hamming distance <= threshold_bits) graph.

    Returns (available_paths, groups) where `groups` maps an internal group
    id to a list of indices into `available_paths`.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform

    available = [p for p in paths if str(p) in hashes]
    if not available:
        return available, {}

    vectors = np.array([hashes[str(p)].hash.flatten() for p in available])
    hash_bits = vectors.shape[1]
    threshold_fraction = threshold_bits / hash_bits

    condensed = pdist(vectors, metric="hamming")
    square = squareform(condensed)
    adjacency = square <= threshold_fraction
    np.fill_diagonal(adjacency, False)

    _, labels = connected_components(csr_matrix(adjacency), directed=False)

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    return available, groups


def compute_cross_dataset_matches(
    paths_a: list[Path],
    hashes_a: dict[str, imagehash.ImageHash],
    paths_b: list[Path],
    hashes_b: dict[str, imagehash.ImageHash],
    threshold_bits: int = DEFAULT_GROUP_THRESHOLD_BITS,
) -> list[list[Path]]:
    """
    Near-duplicate groups that span BOTH datasets — i.e. images from
    dataset A and dataset B close enough (Hamming distance <=
    threshold_bits) to be the same underlying photo. Reuses the same
    connected-components grouping as compute_duplicate_stats/
    build_grouped_chain, just fed the POOLED hashes of both datasets —
    groups that turn out to contain paths from only one dataset are
    filtered out here, since those are ordinary within-dataset
    duplicates, not the cross-dataset signal this function is for.

    Returns a list of groups, each a list of Paths (from either
    dataset) — the caller can check `path in paths_a` / `path in
    paths_b` to tell which dataset a given member came from.
    """
    pooled_paths = list(paths_a) + list(paths_b)
    pooled_hashes = {**hashes_a, **hashes_b}
    available, groups = _compute_duplicate_groups(pooled_paths, pooled_hashes, threshold_bits)

    set_a = {str(p) for p in paths_a}
    cross_groups: list[list[Path]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_paths = [available[i] for i in members]
        has_a = any(str(p) in set_a for p in member_paths)
        has_b = any(str(p) not in set_a for p in member_paths)
        if has_a and has_b:
            cross_groups.append(member_paths)
    return cross_groups


def compute_duplicate_stats(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    threshold_bits: int = DEFAULT_GROUP_THRESHOLD_BITS,
) -> dict:
    """
    Dataset-wide near-duplicate stats — used by the Dataset Report, over
    ALL hashed images (not capped at DEFAULT_CHAIN_MAX_LENGTH like the
    Visual similarity chain display).

    Returns: {"n_duplicate_images": int, "n_groups": int, "group_sizes": list[int]}
    — n_duplicate_images counts images that belong to a group of size > 1
    (i.e. excludes "unique" images with no near-duplicate at all).
    """
    available, groups = _compute_duplicate_groups(paths, hashes, threshold_bits)
    multi_groups = [members for members in groups.values() if len(members) > 1]
    n_duplicate_images = sum(len(members) for members in multi_groups)
    return {
        "n_duplicate_images": n_duplicate_images,
        "n_groups": len(multi_groups),
        "group_sizes": sorted((len(m) for m in multi_groups), reverse=True),
    }


def get_or_compute_duplicate_stats(
    root_path: str,
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    threshold_bits: int = DEFAULT_GROUP_THRESHOLD_BITS,
) -> dict:
    """
    Cache-aware wrapper around compute_duplicate_stats — this is an O(n²)
    pairwise comparison, expensive enough that recomputing it on every
    Streamlit rerun (e.g. every time some unrelated dialog closes and
    forces a full-page rerun) was making the whole app feel sluggish.
    Invalidated automatically whenever get_or_compute_phashes recomputes
    fresh hashes (dataset composition changed).
    """
    cached = cache.load_duplicate_stats(root_path)
    if cached is not None:
        return cached

    stats = compute_duplicate_stats(paths, hashes, threshold_bits)
    cache.save_duplicate_stats(root_path, stats)
    return stats


def get_duplicate_sample_paths(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    threshold_bits: int = DEFAULT_GROUP_THRESHOLD_BITS,
    n_samples: int = 6,
) -> list[Path]:
    """
    A handful of images from the LARGEST near-duplicate group — used to
    illustrate the "Near duplicates" section of the Dataset Report PDF with
    an actual visual example, not just a number.
    """
    available, groups = _compute_duplicate_groups(paths, hashes, threshold_bits)
    multi_groups = [members for members in groups.values() if len(members) > 1]
    if not multi_groups:
        return []
    largest_group = max(multi_groups, key=len)
    return [available[i] for i in largest_group[:n_samples]]


def build_grouped_chain(
    paths: list[Path],
    hashes: dict[str, imagehash.ImageHash],
    threshold_bits: int = DEFAULT_GROUP_THRESHOLD_BITS,
    max_length: int = DEFAULT_CHAIN_MAX_LENGTH,
) -> list[tuple[Path, str]]:
    """
    Group images into near-duplicate clusters (e.g. burst-mode phone shots),
    largest cluster first, smallest last — images with no close match to
    anything else ("unique") come last of all. Within a cluster, images are
    ordered by a small internal nearest-neighbor walk.

    Two images are considered part of the same cluster if their Hamming
    distance is <= threshold_bits (out of 64 total bits for a standard
    pHash) — found via connected components on the "close enough" graph,
    which is the standard efficient way to do this kind of thresholded
    grouping (much faster than optimal_leaf_ordering, no caching needed).

    Returns (path, label) pairs — label is "group of N" for the first image
    of a multi-image cluster, "Δ <bits>" for subsequent images in that same
    cluster, or "unique" for images with no near-duplicate at all.
    """
    available, groups = _compute_duplicate_groups(paths, hashes, threshold_bits)
    if not available:
        return []

    vectors = np.array([hashes[str(p)].hash.flatten() for p in available])

    multi_groups = sorted(
        (gid for gid, members in groups.items() if len(members) > 1),
        key=lambda gid: -len(groups[gid]),
    )
    singleton_groups = [gid for gid, members in groups.items() if len(members) == 1]

    def _order_within_group(indices: list[int]) -> list[int]:
        remaining = indices.copy()
        ordered = [remaining.pop(0)]
        while remaining:
            last_vec = vectors[ordered[-1]]
            best_i, best_dist = None, None
            for i in remaining:
                dist = int(np.sum(vectors[i] != last_vec))
                if best_dist is None or dist < best_dist:
                    best_i, best_dist = i, dist
            ordered.append(best_i)
            remaining.remove(best_i)
        return ordered

    chain: list[tuple[Path, str]] = []

    for gid in multi_groups:
        if len(chain) >= max_length:
            break
        ordered_indices = _order_within_group(groups[gid])
        group_size = len(ordered_indices)
        for j, idx in enumerate(ordered_indices):
            if len(chain) >= max_length:
                break
            if j == 0:
                label = f"group of {group_size}"
            else:
                prev_idx = ordered_indices[j - 1]
                distance = int(np.sum(vectors[idx] != vectors[prev_idx]))
                label = f"Δ {distance}"
            chain.append((available[idx], label))

    for gid in singleton_groups:
        if len(chain) >= max_length:
            break
        idx = groups[gid][0]
        chain.append((available[idx], "unique"))

    logger.info(
        "Grouped into %d multi-image cluster(s) and %d unique image(s)",
        len(multi_groups),
        len(singleton_groups),
    )
    return chain


if __name__ == "__main__":
    # Quick manual test, chained with the cached pipeline:
    #   python src\similarity\phash.py "E:\dataset_unificado"
    import logging as _logging

    from src.persistence import cache as _cache

    _logging.basicConfig(level=_logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python phash.py <folder_path>")
        sys.exit(1)

    root = sys.argv[1]
    loaded = _cache.load_scan_and_embeddings(root)
    if loaded is None:
        print("No cached embeddings found — run pipeline.py first.")
        sys.exit(1)
    paths, _, _ = loaded

    hashes = get_or_compute_phashes(root, paths)
    chain = build_similarity_chain(paths, hashes, start_index=0, max_length=20)

    print(f"\nChain of {len(chain)} images:")
    for path, distance in chain:
        dist_str = "start" if distance is None else f"distance={distance}"
        print(f"  {path.name}  ({dist_str})")
