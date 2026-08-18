"""
src/axes/custom.py

Builds a "custom axis" from a free-text label the user types in — e.g.
"sky" — using CLIP's text encoder so its centroid lives in the exact same
embedding space as the image embeddings and the auto-detected axes.

Unlike clustering-derived axes, a custom axis has no hard image membership
from the start — which images "count" toward it is decided at display
time, by comparing it against every other active axis (see
scoring.get_axis_counts_by_dominance). This is what lets a new custom axis
"take" images away from existing axes it's more relevant to, without ever
rebuilding the hierarchical tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np

from src.embeddings.clip_embedder import ClipEmbedder
from src.persistence.cache import AxisRecord

# Keeps custom axis cluster_ids well clear of the small integers used by
# hierarchical clustering (1, 2, 3...), so they never collide.
CUSTOM_AXIS_ID_OFFSET = 100_000

# CLIP text embeddings are known to work noticeably better on natural
# phrases than on a bare word ("prompt engineering", per the original CLIP
# paper). Averaging a few templates and renormalizing gives a more robust
# centroid than relying on a single phrasing.
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
    "a photo of a {}",
]


def create_custom_axis(embedder: ClipEmbedder, label: str, index: int) -> AxisRecord:
    """
    Build an AxisRecord for a user-defined text axis.

    `index` should be unique among the custom axes added in the same
    session (0, 1, 2...) — used only to generate a stable, non-colliding
    cluster_id. `size` and `member_indices` are left empty/zero: a custom
    axis has no hard cluster membership, only a centroid to score against.
    """
    template_embeddings = [
        embedder.embed_text(template.format(label)) for template in PROMPT_TEMPLATES
    ]
    centroid = np.mean(template_embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    return AxisRecord(
        cluster_id=CUSTOM_AXIS_ID_OFFSET + index,
        label=label,
        size=0,
        member_indices=[],
        centroid=centroid.tolist(),
        captions=[],
        representative_paths=[],
    )
