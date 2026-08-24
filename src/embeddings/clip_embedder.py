"""
src/embeddings/clip_embedder.py

Generates CLIP image embeddings for a list of image paths, in batches,
using the GPU when available. Embeddings are L2-normalized so downstream
cosine similarity is a plain dot product.

torch and open_clip are imported lazily (inside functions/methods, not at
module level) — they're heavy (multi-second import cost on their own),
and importing this module (e.g. transitively, just by app.py starting up)
shouldn't pay that cost until an embedding actually needs to happen. See
BACKLOG.md — this is one of several modules converted this way to speed
up the app's startup time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from tqdm import tqdm

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times. This
# one is lightweight (no torch/heavy deps), so it stays at module level.
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

# Default number of background processes that decode/preprocess images in
# parallel while the GPU works on the previous batch. Set to 0 (single
# process, no parallelism) after a real Windows failure: PyTorch's
# DataLoader multiprocessing (num_workers>0) uses the 'spawn' start method
# on Windows, which re-launches a whole new Python interpreter per worker —
# under a Streamlit-launched process this occasionally failed with
# `OSError: [WinError 6] Controlador no válido` inside
# multiprocessing/spawn.py. Single-process loading is slower but doesn't
# depend on Windows handle duplication working correctly in this
# environment. If you have a Linux/macOS setup (or a Windows one where
# parallel loading is confirmed to work reliably), raising this back up is
# safe to try — just watch for the same "spawn"-related error returning.
DEFAULT_NUM_WORKERS = 0

# Default model — good speed/quality balance on a single consumer GPU.
# Upgraded from ViT-B-32 to ViT-L-14 for higher-quality embeddings, at the
# cost of noticeably slower embedding (roughly 3-4x) and a bigger download.
# NOTE: this changes the embedding dimension (512 -> 768) — cached
# embeddings from ViT-B-32 are NOT compatible and must be recomputed
# (delete the dataset's cache/<hash>/ folder to force that).
DEFAULT_MODEL_NAME = "ViT-L-14"
DEFAULT_PRETRAINED = "laion2b_s32b_b82k"


@dataclass
class EmbeddingResult:
    """Aligned embeddings + the paths they correspond to (same order, 1:1)."""

    paths: list[Path]
    embeddings: np.ndarray  # shape: (n_valid_images, embedding_dim), L2-normalized
    failed: list[tuple[Path, str]]  # images that failed to load/encode, with reason


class ClipEmbedder:
    """Wraps an open_clip model to turn images into L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model %s (%s) on %s", model_name, pretrained, self.device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        model.to(self.device)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim = model.visual.output_dim

    def embed_images(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
        num_workers: int = DEFAULT_NUM_WORKERS,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """
        Compute L2-normalized CLIP embeddings for a list of image paths.

        Images are decoded/preprocessed in parallel by `num_workers`
        background processes (via a DataLoader) while the GPU works on the
        previous batch, instead of loading them one at a time on the main
        process — this was the actual bottleneck (~7 img/s on an RTX 3060
        despite the GPU being mostly idle). Pass num_workers=0 to fall back
        to the original single-process behavior if this causes issues.

        Images that fail to load are skipped and reported separately
        rather than crashing the whole run.

        on_progress: optional (done, total) callback, called as batches
        complete — throttled to roughly once per percentage point (or
        every batch, whichever is less frequent) rather than every single
        image, so a UI progress bar doesn't get flooded with updates on a
        large dataset. None (the default): no callback, exactly today's
        behavior — this stays optional so every non-Streamlit caller
        (scripts, tests) is unaffected. Purely additive to the existing
        tqdm progress bar below, which keeps working the same either way.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        # Defined here (not at module level) so importing this module
        # doesn't require torch to already be loaded — see module docstring.
        class _ImagePathDataset(Dataset):
            """
            Loads + preprocesses one image per __getitem__ call. Designed to
            be driven by a DataLoader with num_workers>0, so this
            decode/resize work happens in parallel background processes
            instead of blocking the GPU.

            Failures are caught here (not raised) — a DataLoader worker
            process crashing on one bad file would otherwise take down the
            whole batch.
            """

            def __init__(self, paths: list[Path], preprocess) -> None:
                self.paths = paths
                self.preprocess = preprocess

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, idx: int):
                path = self.paths[idx]
                try:
                    with Image.open(path) as img:
                        # Apply EXIF orientation before anything else —
                        # Pillow loads raw pixel data as-is and ignores the
                        # EXIF orientation tag by default, so a photo stored
                        # sideways-with-a-rotate-flag would otherwise get
                        # embedded (and thumbnailed) rotated incorrectly.
                        img = ImageOps.exif_transpose(img)
                        img = img.convert("RGB")
                        tensor = self.preprocess(img)
                    return path, tensor, None
                except Exception as exc:  # noqa: BLE001
                    return path, None, str(exc)

        def _collate_batch(batch):
            """Split a batch into (successfully loaded) vs (failed) items."""
            paths, tensors, errors = zip(*batch)

            valid = [(p, t) for p, t, e in zip(paths, tensors, errors) if e is None]
            failed = [(p, e) for p, t, e in zip(paths, tensors, errors) if e is not None]

            if valid:
                valid_paths, valid_tensors = zip(*valid)
                stacked = torch.stack(valid_tensors)
                valid_paths = list(valid_paths)
            else:
                valid_paths, stacked = [], None

            return valid_paths, stacked, failed

        dataset = _ImagePathDataset(image_paths, self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_batch,
            pin_memory=(self.device == "cuda"),
        )

        all_embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []
        failed: list[tuple[Path, str]] = []
        total_images = len(image_paths)
        # Throttled to roughly once per percentage point (never less often
        # than once per batch either way) — calling on_progress on every
        # single image would flood a UI progress bar with far more updates
        # than it needs on a large dataset, without any benefit.
        _last_reported_pct = -1

        with torch.no_grad(), tqdm(
            total=len(image_paths), desc="Embedding images", unit="img"
        ) as pbar:
            for batch_paths, batch_tensor, batch_failed in loader:
                for path, reason in batch_failed:
                    logger.warning("Failed to load %s for embedding: %s", path, reason)
                failed.extend(batch_failed)
                pbar.update(len(batch_paths) + len(batch_failed))

                if on_progress is not None:
                    done = pbar.n
                    pct = int(100 * done / total_images) if total_images else 100
                    if pct != _last_reported_pct or done == total_images:
                        on_progress(done, total_images)
                        _last_reported_pct = pct

                if batch_tensor is None:
                    continue

                batch_input = batch_tensor.to(self.device, non_blocking=True)
                features = self.model.encode_image(batch_input)
                features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

                all_embeddings.append(features.cpu().numpy())
                valid_paths.extend(batch_paths)

        if all_embeddings:
            embeddings_array = np.concatenate(all_embeddings, axis=0)
        else:
            embeddings_array = np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info(
            "Embedding complete: %d succeeded, %d failed",
            len(valid_paths),
            len(failed),
        )

        return EmbeddingResult(paths=valid_paths, embeddings=embeddings_array, failed=failed)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single free-text label (e.g. "sky") into the same space as
        the image embeddings, L2-normalized so it's directly comparable via
        dot product. Used for user-defined custom axes.
        """
        import torch

        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()[0]


if __name__ == "__main__":
    # Quick manual test, chained with the ingestion module:
    #   python src\embeddings\clip_embedder.py "E:\dataset_unificado"
    import sys
    import time

    sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow `from src...` imports
    from src.ingestion.loader import scan_folder

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python clip_embedder.py <folder_path>")
        sys.exit(1)

    scan = scan_folder(sys.argv[1], recursive=True)
    print(f"Scanned {scan.total_found} valid images, embedding now...")

    embedder = ClipEmbedder()
    start = time.time()
    result = embedder.embed_images(scan.valid_images, batch_size=32)
    elapsed = time.time() - start

    print(f"Embeddings shape: {result.embeddings.shape}")
    print(f"Failed: {len(result.failed)}")
    print(f"Elapsed: {elapsed:.1f}s ({len(result.paths) / elapsed:.1f} img/s)")
