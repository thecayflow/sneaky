"""
src/ingestion/loader.py

Scans a given folder path (provided at runtime, not fixed) and returns
the list of valid, readable images found inside it. Supports recursive
search through subfolders.

This module does NOT copy or move any files — it only reads references
to images that live wherever the user points the app to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
import pillow_heif

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times.
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

# Extensions currently supported.
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


@dataclass
class ScanResult:
    """Result of scanning a folder for images."""

    root_path: Path
    valid_images: list[Path] = field(default_factory=list)
    skipped_files: list[tuple[Path, str]] = field(default_factory=list)  # (path, reason)

    @property
    def total_found(self) -> int:
        return len(self.valid_images)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped_files)


def _iter_candidate_files(root_path: Path, recursive: bool) -> list[Path]:
    """List files under root_path whose extension matches SUPPORTED_EXTENSIONS."""
    pattern_iter = root_path.rglob("*") if recursive else root_path.glob("*")
    return [
        p for p in pattern_iter
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _is_valid_image(path: Path) -> tuple[bool, str | None]:
    """
    Verify that a file is actually a readable, non-corrupted image.
    Returns (is_valid, error_reason).
    """
    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check, doesn't decode full pixel data
        return True, None
    except Exception as exc:  # noqa: BLE001 - we want to catch any decode failure
        return False, str(exc)


def scan_folder(root_path: str | Path, recursive: bool = True) -> ScanResult:
    """
    Scan `root_path` for supported image files.

    Args:
        root_path: Any folder path on disk, provided dynamically by the user
                    (e.g. from a Streamlit text input or folder picker).
        recursive: If True, also search subfolders.

    Returns:
        ScanResult with the list of valid image paths and any skipped files
        (with the reason they were skipped), so the caller/UI can report
        corrupted or unreadable files without crashing the whole pipeline.
    """
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {root}")

    result = ScanResult(root_path=root)
    candidates = _iter_candidate_files(root, recursive=recursive)

    logger.info("Found %d candidate files under %s", len(candidates), root)

    for path in candidates:
        is_valid, reason = _is_valid_image(path)
        if is_valid:
            result.valid_images.append(path)
        else:
            logger.warning("Skipping unreadable/corrupted file %s (%s)", path, reason)
            result.skipped_files.append((path, reason or "unknown error"))

    logger.info(
        "Scan complete: %d valid images, %d skipped",
        result.total_found,
        result.total_skipped,
    )
    return result


if __name__ == "__main__":
    # Quick manual test — run directly with:
    #   python src\ingestion\loader.py "E:\dataset_unificado"
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python loader.py <folder_path>")
        sys.exit(1)

    scan = scan_folder(sys.argv[1], recursive=True)
    print(f"Valid images: {scan.total_found}")
    print(f"Skipped: {scan.total_skipped}")
    if scan.skipped_files:
        print("First few skipped files:")
        for path, reason in scan.skipped_files[:5]:
            print(f"  - {path}: {reason}")
