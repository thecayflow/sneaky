"""
src/embeddings/clip_embedder.py

Generates CLIP image embeddings for a list of image paths, in batches,
using the GPU when available. Embeddings are L2-normalized so downstream
cosine similarity is a plain dot product.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import pillow_heif
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times.
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

# Default number of background processes that decode/preprocess images in
# parallel while the GPU works on the previous batch. On a Windows/Streamlit
# setup this uses real subprocesses (not threads) — if you hit odd errors
# mentioning "spawn" or the app seems to relaunch itself, pass
# num_workers=0 to ClipEmbedder.embed_images(...) to fall back to the
# original single-process behavior while we investigate.
DEFAULT_NUM_WORKERS = 4

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


class _ImagePathDataset(Dataset):
    """
    Loads + preprocesses one image per __getitem__ call. Designed to be
    driven by a DataLoader with num_workers>0, so this decode/resize work
    happens in parallel background processes instead of blocking the GPU.

    Failures are caught here (not raised) — a DataLoader worker process
    crashing on one bad file would otherwise take down the whole batch.
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
                # Apply EXIF orientation before anything else — Pillow loads
                # raw pixel data as-is and ignores the EXIF orientation tag
                # by default, so a photo stored sideways-with-a-rotate-flag
                # would otherwise get embedded (and thumbnailed) rotated
                # incorrectly.
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


class ClipEmbedder:
    """Wraps an open_clip model to turn images into L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
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

    @torch.no_grad()
    def embed_images(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
        num_workers: int = DEFAULT_NUM_WORKERS,
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
        """
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

        with tqdm(total=len(image_paths), desc="Embedding images", unit="img") as pbar:
            for batch_paths, batch_tensor, batch_failed in loader:
                for path, reason in batch_failed:
                    logger.warning("Failed to load %s for embedding: %s", path, reason)
                failed.extend(batch_failed)
                pbar.update(len(batch_paths) + len(batch_failed))

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

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single free-text label (e.g. "sky") into the same space as
        the image embeddings, L2-normalized so it's directly comparable via
        dot product. Used for user-defined custom axes.
        """
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
