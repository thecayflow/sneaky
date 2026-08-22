"""
src/persistence/cache.py

Disk cache for the pipeline, keyed by the dataset's folder path. Avoids
recomputing embeddings (the ~7 minute step for ~3k images), the clustering
tree, and captions/labels (both non-trivial) every time the same folder is
processed again.

Cache layout (relative to project root):
    cache/<dataset_hash>/
        meta.json           # root path, image count, timestamp
        paths.json          # image paths, aligned 1:1 with embeddings.npz
        mtimes.json          # {path_str: modification_time}, for detecting in-place edits
        skipped.json        # images that failed to load/embed, with reason
        embeddings.npz       # CLIP embeddings array
        tree.npz              # hierarchical clustering linkage matrix (Z)
        axes_k<K>.json         # per-k: labeled clusters (centroids, members, captions)
        phashes.json         # perceptual hashes, for near-duplicate detection
        wavelet_paths.json   # image paths, aligned 1:1 with wavelet_features.npz
        wavelet_features.npz  # wavelet texture feature vectors, for Wavelet-MMD

<dataset_hash> is derived from the resolved absolute folder path, so
pointing the app at the same folder twice reuses the same cache entry, and
different folders (e.g. a future feed 01 vs feed 02 comparison) get
separate entries automatically — nothing is copied, only the path itself
is hashed to name the cache folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = _PROJECT_ROOT / "cache"


@dataclass
class AxisRecord:
    """A fully computed, labeled axis (cluster) — everything scoring/viz need."""

    cluster_id: int
    label: str
    size: int
    member_indices: list[int]
    centroid: list[float]
    captions: list[str]
    representative_paths: list[str]


def _dataset_cache_dir(root_path: str | Path) -> Path:
    resolved = str(Path(root_path).resolve())
    dataset_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    cache_dir = CACHE_ROOT / dataset_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ---------------------------------------------------------------------------
# Scan + embeddings
# ---------------------------------------------------------------------------

def save_scan_and_embeddings(
    root_path: str | Path,
    paths: list[Path],
    embeddings: np.ndarray,
    skipped: list[tuple[Path, str]],
    mtimes: dict[str, float] | None = None,
) -> None:
    """
    mtimes: optional {path_str: modification_time} for every path in
    `paths` — lets _update_embeddings_incrementally detect when an
    EXISTING file's content changed (edited in place, same filename) on
    a later run, not just files added or removed. Written to its own
    mtimes.json (not merged into paths.json) so this stays a purely
    additive, backward-compatible change — a cache written before this
    existed simply has no mtimes.json, and load_mtimes returns None for
    it (see there for how that case is handled).
    """
    cache_dir = _dataset_cache_dir(root_path)

    np.savez_compressed(cache_dir / "embeddings.npz", embeddings=embeddings)
    (cache_dir / "paths.json").write_text(
        json.dumps([str(p) for p in paths], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cache_dir / "skipped.json").write_text(
        json.dumps([[str(p), reason] for p, reason in skipped], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "root_path": str(Path(root_path).resolve()),
                "n_images": len(paths),
                "n_skipped": len(skipped),
                "cached_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if mtimes is not None:
        (cache_dir / "mtimes.json").write_text(
            json.dumps(mtimes, indent=2), encoding="utf-8"
        )
    logger.info("Cached scan + embeddings for %s at %s", root_path, cache_dir)


def load_mtimes(root_path: str | Path) -> dict[str, float] | None:
    """
    {path_str: modification_time}, or None if this dataset's cache
    predates mtime tracking (no mtimes.json yet) — the caller should
    treat None (and a missing entry for any individual path) as "unknown,
    assume unchanged" rather than as evidence of a change, so existing
    caches aren't forced through a one-time full re-embed just because
    they don't have this metadata yet.
    """
    cache_dir = _dataset_cache_dir(root_path)
    mtimes_file = cache_dir / "mtimes.json"
    if not mtimes_file.exists():
        return None
    return json.loads(mtimes_file.read_text(encoding="utf-8"))


def load_scan_and_embeddings(
    root_path: str | Path,
) -> tuple[list[Path], np.ndarray, list[tuple[Path, str]]] | None:
    cache_dir = _dataset_cache_dir(root_path)
    embeddings_file = cache_dir / "embeddings.npz"
    paths_file = cache_dir / "paths.json"

    if not embeddings_file.exists() or not paths_file.exists():
        return None

    embeddings = np.load(embeddings_file)["embeddings"]
    paths = [Path(p) for p in json.loads(paths_file.read_text(encoding="utf-8"))]

    skipped_file = cache_dir / "skipped.json"
    skipped: list[tuple[Path, str]] = []
    if skipped_file.exists():
        skipped = [
            (Path(p), reason) for p, reason in json.loads(skipped_file.read_text(encoding="utf-8"))
        ]

    logger.info("Loaded cached scan + embeddings for %s from %s", root_path, cache_dir)
    return paths, embeddings, skipped


# ---------------------------------------------------------------------------
# Hierarchical tree
# ---------------------------------------------------------------------------

def save_tree(root_path: str | Path, Z: np.ndarray, linkage_method: str = "ward") -> None:
    cache_dir = _dataset_cache_dir(root_path)
    np.savez_compressed(cache_dir / f"tree_{linkage_method}.npz", Z=Z)
    logger.info("Cached hierarchical tree (%s) for %s", linkage_method, root_path)


def load_tree(root_path: str | Path, linkage_method: str = "ward") -> np.ndarray | None:
    cache_dir = _dataset_cache_dir(root_path)
    tree_file = cache_dir / f"tree_{linkage_method}.npz"
    if not tree_file.exists():
        return None
    Z = np.load(tree_file)["Z"]
    logger.info("Loaded cached hierarchical tree (%s) for %s", linkage_method, root_path)
    return Z


# ---------------------------------------------------------------------------
# Labeled axes (per k, per linkage method)
# ---------------------------------------------------------------------------

def save_axes(
    root_path: str | Path, k: int, records: list[AxisRecord], linkage_method: str = "ward"
) -> None:
    cache_dir = _dataset_cache_dir(root_path)
    axes_file = cache_dir / f"axes_{linkage_method}_k{k}.json"
    payload = [
        {
            "cluster_id": r.cluster_id,
            "label": r.label,
            "size": r.size,
            "member_indices": r.member_indices,
            "centroid": r.centroid,
            "captions": r.captions,
            "representative_paths": r.representative_paths,
        }
        for r in records
    ]
    axes_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Cached %d labeled axes (method=%s, k=%d) for %s", len(records), linkage_method, k, root_path
    )


def load_axes(
    root_path: str | Path, k: int, linkage_method: str = "ward"
) -> list[AxisRecord] | None:
    cache_dir = _dataset_cache_dir(root_path)
    axes_file = cache_dir / f"axes_{linkage_method}_k{k}.json"
    if not axes_file.exists():
        return None

    payload = json.loads(axes_file.read_text(encoding="utf-8"))
    records = [
        AxisRecord(
            cluster_id=item["cluster_id"],
            label=item["label"],
            size=item["size"],
            member_indices=item["member_indices"],
            centroid=item["centroid"],
            captions=item["captions"],
            representative_paths=item["representative_paths"],
        )
        for item in payload
    ]
    logger.info("Loaded cached labeled axes (method=%s, k=%d) for %s", linkage_method, k, root_path)
    return records


# ---------------------------------------------------------------------------
# t-SNE projection (2D, independent of k and linkage method — depends only
# on the embeddings themselves)
# ---------------------------------------------------------------------------

def save_tsne(root_path: str | Path, coords: np.ndarray) -> None:
    cache_dir = _dataset_cache_dir(root_path)
    np.savez_compressed(cache_dir / "tsne.npz", coords=coords)
    logger.info("Cached t-SNE projection for %s", root_path)


def load_tsne(root_path: str | Path) -> np.ndarray | None:
    cache_dir = _dataset_cache_dir(root_path)
    tsne_file = cache_dir / "tsne.npz"
    if not tsne_file.exists():
        return None
    coords = np.load(tsne_file)["coords"]
    logger.info("Loaded cached t-SNE projection for %s", root_path)
    return coords


# ---------------------------------------------------------------------------
# UMAP projection (2D, same role as t-SNE above — a second projection
# method, cached separately so switching between them never invalidates
# the other)
# ---------------------------------------------------------------------------

def save_umap(root_path: str | Path, coords: np.ndarray) -> None:
    cache_dir = _dataset_cache_dir(root_path)
    np.savez_compressed(cache_dir / "umap.npz", coords=coords)
    logger.info("Cached UMAP projection for %s", root_path)


def load_umap(root_path: str | Path) -> np.ndarray | None:
    cache_dir = _dataset_cache_dir(root_path)
    umap_file = cache_dir / "umap.npz"
    if not umap_file.exists():
        return None
    coords = np.load(umap_file)["coords"]
    logger.info("Loaded cached UMAP projection for %s", root_path)
    return coords


# ---------------------------------------------------------------------------
# Perceptual hashes (pHash) — pixel-level visual similarity, independent of
# CLIP/semantic embeddings entirely. Cheap enough to just store as JSON.
# ---------------------------------------------------------------------------

def save_phashes(root_path: str | Path, hashes: dict[str, str]) -> None:
    """`hashes` maps image path (str) -> hex string of its perceptual hash."""
    cache_dir = _dataset_cache_dir(root_path)
    (cache_dir / "phashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cached %d perceptual hashes for %s", len(hashes), root_path)


def load_phashes(root_path: str | Path) -> dict[str, str] | None:
    cache_dir = _dataset_cache_dir(root_path)
    phashes_file = cache_dir / "phashes.json"
    if not phashes_file.exists():
        return None
    hashes = json.loads(phashes_file.read_text(encoding="utf-8"))
    logger.info("Loaded %d cached perceptual hashes for %s", len(hashes), root_path)
    return hashes


def save_wavelet_features(root_path: str | Path, paths: list[Path], features: np.ndarray) -> None:
    """
    `features` is an (n_images, n_features) array, row-aligned 1:1 with
    `paths` — same array-plus-paths-file convention as embeddings.npz/
    paths.json, kept as its own separate pair of files (rather than reusing
    paths.json) since wavelet feature extraction can legitimately skip an
    image that the main embedding pass didn't (or vice versa), so the two
    path lists aren't guaranteed to match row-for-row.
    """
    cache_dir = _dataset_cache_dir(root_path)
    np.savez_compressed(cache_dir / "wavelet_features.npz", features=features)
    (cache_dir / "wavelet_paths.json").write_text(
        json.dumps([str(p) for p in paths], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cached %d wavelet feature vectors for %s", len(paths), root_path)


def load_wavelet_features(root_path: str | Path) -> tuple[list[Path], np.ndarray] | None:
    cache_dir = _dataset_cache_dir(root_path)
    features_file = cache_dir / "wavelet_features.npz"
    paths_file = cache_dir / "wavelet_paths.json"
    if not features_file.exists() or not paths_file.exists():
        return None
    features = np.load(features_file)["features"]
    paths = [Path(p) for p in json.loads(paths_file.read_text(encoding="utf-8"))]
    logger.info("Loaded %d cached wavelet feature vectors for %s", len(paths), root_path)
    return paths, features


def save_global_order(root_path: str | Path, chain: list[tuple[str, int | None]]) -> None:
    """
    `chain` is the full globally-ordered sequence: (path_str, hamming_distance
    from the previous entry — None for the first). Optimal leaf ordering can
    be noticeably slower than the greedy chain, so this is cached separately.
    """
    cache_dir = _dataset_cache_dir(root_path)
    payload = [{"path": p, "distance": d} for p, d in chain]
    (cache_dir / "global_order.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cached global visual order (%d images) for %s", len(chain), root_path)


def load_global_order(root_path: str | Path) -> list[tuple[str, int | None]] | None:
    cache_dir = _dataset_cache_dir(root_path)
    order_file = cache_dir / "global_order.json"
    if not order_file.exists():
        return None
    payload = json.loads(order_file.read_text(encoding="utf-8"))
    chain = [(item["path"], item["distance"]) for item in payload]
    logger.info("Loaded cached global visual order (%d images) for %s", len(chain), root_path)
    return chain


def save_duplicate_stats(root_path: str | Path, stats: dict) -> None:
    """
    `stats` is the dict from phash.compute_duplicate_stats — a full O(n²)
    pairwise comparison, so this is cached to avoid recomputing it on every
    single Streamlit rerun (e.g. every time a dialog closes elsewhere in
    the app forces a full-page rerun).
    """
    cache_dir = _dataset_cache_dir(root_path)
    (cache_dir / "duplicate_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Cached duplicate stats for %s", root_path)


def load_duplicate_stats(root_path: str | Path) -> dict | None:
    cache_dir = _dataset_cache_dir(root_path)
    stats_file = cache_dir / "duplicate_stats.json"
    if not stats_file.exists():
        return None
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    logger.info("Loaded cached duplicate stats for %s", root_path)
    return stats


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def invalidate_tree_and_axes(root_path: str | Path) -> None:
    """
    Delete every cached hierarchical tree, labeled-axes file, and
    projection (t-SNE, UMAP) for this dataset (all linkage methods, all k
    values), WITHOUT touching the cached embeddings themselves.

    Call this whenever the underlying image set changes (images added or
    removed): every tree_*.npz, axes_*_k*.json, tsne.npz, and umap.npz was
    computed from the old composition of the dataset and is no longer
    valid, but the embeddings for images that are still present are still
    perfectly good — only the embeddings for genuinely new images need
    recomputing.
    """
    cache_dir = _dataset_cache_dir(root_path)

    n_trees_removed = 0
    for tree_file in cache_dir.glob("tree_*.npz"):
        tree_file.unlink()
        n_trees_removed += 1

    n_axes_removed = 0
    for axes_file in cache_dir.glob("axes_*_k*.json"):
        axes_file.unlink()
        n_axes_removed += 1

    tsne_file = cache_dir / "tsne.npz"
    tsne_removed = tsne_file.exists()
    if tsne_removed:
        tsne_file.unlink()

    umap_file = cache_dir / "umap.npz"
    umap_removed = umap_file.exists()
    if umap_removed:
        umap_file.unlink()

    order_file = cache_dir / "global_order.json"
    order_removed = order_file.exists()
    if order_removed:
        order_file.unlink()

    logger.info(
        "Invalidated %d cached tree(s), %d cached axes file(s), tsne=%s, umap=%s, "
        "global_order=%s for %s (dataset composition changed)",
        n_trees_removed,
        n_axes_removed,
        tsne_removed,
        umap_removed,
        order_removed,
        root_path,
    )


def clear_similarity_cache(root_path: str | Path) -> None:
    """Delete the cached perceptual hashes, global visual order, and
    duplicate stats for this dataset."""
    cache_dir = _dataset_cache_dir(root_path)
    for filename in ("phashes.json", "global_order.json", "duplicate_stats.json"):
        f = cache_dir / filename
        if f.exists():
            f.unlink()
    logger.info("Cleared visual similarity cache for %s", root_path)


def clear_all(root_path: str | Path) -> None:
    """
    Delete the ENTIRE cache directory for this dataset — embeddings
    included. The next analysis of this folder starts completely from
    scratch (re-scan, re-embed, re-cluster, re-caption, re-project).
    """
    import shutil

    cache_dir = _dataset_cache_dir(root_path)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cleared ALL cache for %s", root_path)


def get_cache_info(root_path: str | Path) -> list[tuple[str, int]]:
    """List (filename, size_in_bytes) for every file currently cached for this dataset."""
    cache_dir = _dataset_cache_dir(root_path)
    return sorted(
        ((f.name, f.stat().st_size) for f in cache_dir.iterdir() if f.is_file()),
        key=lambda item: item[0],
    )


# ---------------------------------------------------------------------------
# Combined-axes cache (two datasets analyzed together) — a separate
# location from the single-dataset one above, keyed by the PAIR of root
# paths (order-independent: comparing A vs B hits the same cache entry as
# B vs A). Used when a comparison feed is loaded: rather than only
# scoring the second dataset against the primary's own axes, the axes
# themselves get re-clustered over BOTH datasets' pooled embeddings, so a
# theme present only in the second dataset (e.g. many horses when the
# primary has none at all) can still surface as its own axis.
# ---------------------------------------------------------------------------

def _pair_cache_dir(root_path_a: str | Path, root_path_b: str | Path) -> Path:
    resolved_a = str(Path(root_path_a).resolve())
    resolved_b = str(Path(root_path_b).resolve())
    # sorted() makes this order-independent — comparing A-vs-B and B-vs-A
    # resolve to the exact same cache directory.
    combined_key = "||".join(sorted([resolved_a, resolved_b]))
    pair_hash = hashlib.sha256(combined_key.encode("utf-8")).hexdigest()[:16]
    cache_dir = CACHE_ROOT / "pairs" / pair_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_pair_combined_paths(
    root_path_a: str | Path, root_path_b: str | Path, paths: list[Path]
) -> None:
    """
    The exact combined path list (dataset A's paths, then dataset B's)
    the pair's cached tree/axes were computed from — compared against a
    freshly-recomputed combined path list on each run_combined_pipeline
    call to detect whether EITHER dataset changed since, mirroring the
    single-dataset invalidation pattern (added/removed/modified images)
    but at the pair level.
    """
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    (cache_dir / "combined_paths.json").write_text(
        json.dumps([str(p) for p in paths], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_pair_combined_paths(root_path_a: str | Path, root_path_b: str | Path) -> list[Path] | None:
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    paths_file = cache_dir / "combined_paths.json"
    if not paths_file.exists():
        return None
    return [Path(p) for p in json.loads(paths_file.read_text(encoding="utf-8"))]


def save_pair_tree(
    root_path_a: str | Path, root_path_b: str | Path, Z: np.ndarray, linkage_method: str = "ward"
) -> None:
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    np.savez_compressed(cache_dir / f"tree_{linkage_method}.npz", Z=Z)
    logger.info("Cached combined hierarchical tree (%s) for pair (%s, %s)", linkage_method, root_path_a, root_path_b)


def load_pair_tree(
    root_path_a: str | Path, root_path_b: str | Path, linkage_method: str = "ward"
) -> np.ndarray | None:
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    tree_file = cache_dir / f"tree_{linkage_method}.npz"
    if not tree_file.exists():
        return None
    return np.load(tree_file)["Z"]


def save_pair_axes(
    root_path_a: str | Path,
    root_path_b: str | Path,
    k: int,
    records: list[AxisRecord],
    linkage_method: str = "ward",
) -> None:
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    axes_file = cache_dir / f"axes_{linkage_method}_k{k}.json"
    payload = [
        {
            "cluster_id": r.cluster_id,
            "label": r.label,
            "size": r.size,
            "member_indices": r.member_indices,
            "centroid": r.centroid,
            "captions": r.captions,
            "representative_paths": r.representative_paths,
        }
        for r in records
    ]
    axes_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Cached %d combined labeled axes (method=%s, k=%d) for pair (%s, %s)",
        len(records), linkage_method, k, root_path_a, root_path_b,
    )


def load_pair_axes(
    root_path_a: str | Path, root_path_b: str | Path, k: int, linkage_method: str = "ward"
) -> list[AxisRecord] | None:
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    axes_file = cache_dir / f"axes_{linkage_method}_k{k}.json"
    if not axes_file.exists():
        return None
    payload = json.loads(axes_file.read_text(encoding="utf-8"))
    return [
        AxisRecord(
            cluster_id=item["cluster_id"],
            label=item["label"],
            size=item["size"],
            member_indices=item["member_indices"],
            centroid=item["centroid"],
            captions=item["captions"],
            representative_paths=item["representative_paths"],
        )
        for item in payload
    ]


def invalidate_pair_tree_and_axes(root_path_a: str | Path, root_path_b: str | Path) -> None:
    """Same idea as invalidate_tree_and_axes, but for a combined-axes pair
    — call whenever EITHER dataset's own embeddings have changed."""
    cache_dir = _pair_cache_dir(root_path_a, root_path_b)
    n_trees_removed = 0
    for tree_file in cache_dir.glob("tree_*.npz"):
        tree_file.unlink()
        n_trees_removed += 1
    n_axes_removed = 0
    for axes_file in cache_dir.glob("axes_*_k*.json"):
        axes_file.unlink()
        n_axes_removed += 1
    logger.info(
        "Invalidated %d cached combined tree(s), %d cached combined axes file(s) for pair (%s, %s)",
        n_trees_removed, n_axes_removed, root_path_a, root_path_b,
    )
