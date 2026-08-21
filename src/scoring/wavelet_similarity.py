"""
src/scoring/wavelet_similarity.py

Dataset-level (not image-level) distributional comparison — same idea as
dataset_similarity.py's CLIP-MMD, but over a WAVELET texture feature space
instead of CLIP's semantic embedding space. Complementary, not a
replacement: two datasets can look identical semantically (same objects,
same scenes — CLIP-MMD near zero) while differing sharply in low-level
visual statistics (sensor noise, compression, rendering engine, film
grain vs. digital) that CLIP was never trained to be sensitive to, or
vice versa. Neither number feeds into axis clustering or per-image
scoring anywhere — both are purely a summary "how different are these
two datasets overall" figure, shown side by side in the PDF report.

Unlike CLIP-MMD (which reuses embeddings already computed for the main
pipeline — effectively free), this needs its own pass reading every
image in both feeds the FIRST time — real, if modest, added cost (no GPU
needed). get_or_compute_wavelet_features caches the result to disk per
dataset (same convention as get_or_compute_phashes), so generating the
PDF again for the same dataset doesn't repeat that pass.

Requires the `PyWavelets` package (import name `pywt`), imported lazily
inside the functions that use it — same lazy-import convention as the
rest of the project (see BACKLOG.md).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.persistence import cache
from src.scoring.dataset_similarity import compute_clip_mmd, compute_self_split_mmd

logger = logging.getLogger(__name__)

DEFAULT_RESIZE = 256
DEFAULT_WAVELET = "db4"
DEFAULT_LEVELS = 3


def extract_wavelet_features(
    image_path: Path,
    resize_to: int = DEFAULT_RESIZE,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_LEVELS,
) -> np.ndarray:
    """
    One image -> a fixed-length texture "fingerprint" vector, independent
    of the image's own resolution or aspect ratio (always resized to a
    square first, same as CLIP's own preprocessing needs a fixed input
    size — the point here isn't to preserve the image, only to compare
    its texture statistics against other images on equal footing).

    Converts to grayscale (texture/frequency structure, not color, is
    the target — color statistics are a separate concern this doesn't
    try to capture) and resized to (resize_to, resize_to), then runs a
    multi-level 2D discrete wavelet transform. Rather than using the raw
    coefficients directly (far too high-dimensional and position-
    sensitive to be useful for a whole-dataset comparison), this reduces
    each sub-band to two summary statistics — mean absolute value (an
    energy proxy: how much texture/edge activity at this scale and
    orientation) and standard deviation — the standard "wavelet energy
    signature" approach to texture description.

    With the defaults (level=3, giving 3 detail sub-bands per level:
    horizontal/vertical/diagonal, plus one final approximation band),
    this returns a 20-value vector: 2 stats x (3 levels x 3 orientations
    + 1 approximation) = 2 x 10.
    """
    from PIL import Image
    import pywt

    with Image.open(image_path) as im:
        im = im.convert("L").resize((resize_to, resize_to))
        arr = np.asarray(im, dtype=np.float64) / 255.0

    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=levels)

    features: list[float] = []
    approximation = coeffs[0]
    features.append(float(np.mean(np.abs(approximation))))
    features.append(float(np.std(approximation)))
    for level_bands in coeffs[1:]:  # each is (cH, cV, cD), coarsest level first
        for band in level_bands:
            features.append(float(np.mean(np.abs(band))))
            features.append(float(np.std(band)))

    return np.array(features, dtype=np.float64)


def extract_wavelet_features_batch(
    paths: list[Path],
    resize_to: int = DEFAULT_RESIZE,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_LEVELS,
) -> tuple[list[Path], np.ndarray]:
    """
    extract_wavelet_features for every path — skips any image that fails
    to open or process (same tolerant convention as the rest of the
    pipeline, e.g. ClipEmbedder.embed_images' own `failed` list), logging
    a warning rather than aborting the whole batch over one bad file.

    Returns (successful_paths, features) — successful_paths is the
    SUBSET of the input that actually produced a feature vector, row-
    aligned 1:1 with `features`. Returning both together (rather than
    just the array) is what lets a caller reliably cache the result: a
    failure isn't necessarily the last image in the list, so there's no
    safe way to reconstruct which paths succeeded from the array alone.
    """
    successful_paths: list[Path] = []
    features: list[np.ndarray] = []
    for p in paths:
        try:
            features.append(
                extract_wavelet_features(p, resize_to=resize_to, wavelet=wavelet, levels=levels)
            )
            successful_paths.append(p)
        except Exception:
            logger.warning("Skipping %s — could not extract wavelet features", p, exc_info=True)
    if not features:
        return [], np.empty((0, 0))
    return successful_paths, np.array(features)


def get_or_compute_wavelet_features(root_path: str, paths: list[Path]) -> tuple[list[Path], np.ndarray]:
    """
    Cache-aware wrapper — same convention as phash.py's
    get_or_compute_phashes. Recomputes for everyone (not incrementally) if
    the cached path set doesn't exactly match the current one; wavelet
    extraction is cheap enough (no neural net, no GPU) that a full
    recompute on a changed dataset is acceptable, same reasoning as pHash.

    Returns (paths, features), row-aligned — paths may be a SUBSET of the
    input if any images failed extraction (see extract_wavelet_features_
    batch), so the two are always returned together rather than assuming
    the caller's original path list still lines up.
    """
    current_path_strs = {str(p) for p in paths}
    cached = cache.load_wavelet_features(root_path)

    if cached is not None:
        cached_paths, cached_features = cached
        if {str(p) for p in cached_paths} == current_path_strs:
            return cached_paths, cached_features

    logger.info("Computing wavelet texture features for %d images", len(paths))
    successful_paths, features = extract_wavelet_features_batch(paths)
    if successful_paths:
        cache.save_wavelet_features(root_path, successful_paths, features)
    return successful_paths, features


def compute_wavelet_mmd(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Maximum Mean Discrepancy between two sets of wavelet texture feature
    vectors — same estimator, same median-heuristic kernel bandwidth as
    compute_clip_mmd (the math is entirely generic over the feature
    space; only the vectors passed in differ). Kept as its own named
    function (rather than calling compute_clip_mmd directly at each call
    site) so it's self-documenting which comparison a call site means,
    and so the two could diverge in kernel choice later without touching
    the shared implementation.
    """
    return compute_clip_mmd(features_a, features_b)


def compute_wavelet_self_split_mmd(features: np.ndarray, random_state: int = 42) -> float:
    """Same-dataset "noise floor" baseline for compute_wavelet_mmd — see
    compute_self_split_mmd's own docstring for why this matters."""
    return compute_self_split_mmd(features, random_state=random_state)
