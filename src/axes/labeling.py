"""
src/axes/labeling.py

Turns each detected cluster (from hierarchical.py) into a short, human
readable axis label — e.g. "sunset", "selfie", "food" — without relying on
a fixed predefined vocabulary.

Approach:
  1. Pick the images closest to each cluster centroid (most representative).
  2. Caption each of them with BLIP (image captioning model).
  3. Extract a distinctive keyword per cluster using TF-IDF across all
     clusters' captions — i.e. a word that's frequent in THIS cluster's
     captions but rare across the OTHER clusters' captions.

Why TF-IDF instead of simple word frequency (v1): with only a handful of
captions per cluster, raw frequency tends to surface generic words that
are common in BLIP captions in general ("man", "people", "white") rather
than words that actually distinguish one cluster from another. TF-IDF
fixes this by comparing clusters against each other, not just counting
within one cluster in isolation.

This TF-IDF heuristic is still a stand-in for a smarter approach. A future
improvement (tracked in BACKLOG.md) is to replace it with an actual LLM
summarization call for more natural labels — that requires an API key, so
this heuristic keeps the pipeline fully local/offline for now.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: make `src.*` importable whether this file is run directly
# (`python src\axes\labeling.py`) or as a module (`python -m src.axes.labeling`).
# Must happen BEFORE any `from src...` import below, not just inside
# `if __name__ == "__main__":` — module-level imports run first.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import spacy
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)

from src.axes.hierarchical import Cluster

DEFAULT_SPACY_MODEL = "en_core_web_sm"
_NLP = None  # lazy-loaded singleton — spaCy model load isn't free


def _get_nlp():
    """
    Load the spaCy POS-tagging pipeline once and reuse it. Only the
    tagger/attribute_ruler components are needed (not parser or NER), so
    those are disabled for speed — captions are short, but no need to pay
    for dependency parsing we never use.
    """
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(DEFAULT_SPACY_MODEL, disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{DEFAULT_SPACY_MODEL}' not found. Install it with:\n"
                f"  python -m spacy download {DEFAULT_SPACY_MODEL}"
            ) from exc
    return _NLP


def _get_noun_words(captions: list[str]) -> set[str]:
    """Words (lowercased) that spaCy tags as NOUN or PROPN in any of these captions."""
    nlp = _get_nlp()
    nouns: set[str] = set()
    for doc in nlp.pipe(captions):
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                nouns.add(token.text.lower())
    return nouns


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_TOP_N_REPRESENTATIVES = 8

# Small stopword list — enough to filter out captioning boilerplate words.
# Not meant to be exhaustive, just enough for short BLIP-style captions.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "is", "are",
    "to", "for", "by", "it", "its", "this", "that", "some", "there", "as",
    "photo", "picture", "image", "close", "up", "shot", "view", "background",
    # Generic person nouns — near-ubiquitous in personal photo captions,
    # so they rarely help distinguish one semantic cluster from another.
    # "people" is deliberately kept: as a collective it can denote a real
    # scene type (crowds/gatherings), unlike singular "man"/"boy"/etc.
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "baby", "babies", "kid", "kids", "child", "children", "person",
}


@dataclass
class ClusterLabel:
    """Result of labeling one cluster."""

    cluster_id: int
    label: str
    captions: list[str]
    representative_paths: list[Path]


def _tokenize(caption: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", caption.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _extract_keywords_tfidf(captions_per_cluster: list[list[str]]) -> list[str]:
    """
    Pick one distinctive keyword per cluster using TF-IDF across clusters:
    a word scores high if it's frequent in this cluster's captions but
    appears in few other clusters' captions. Among candidates, nouns
    (NOUN/PROPN per spaCy) are always preferred over other parts of speech
    — e.g. "window" over "open" — since a noun is a more natural, concrete
    axis label than an adjective that happened to score well. TF-IDF score
    only breaks ties within the noun/non-noun groups. Ties/collisions
    between clusters wanting the same word are resolved greedily (clusters
    earlier in the input list — normally the larger ones — get first
    pick), so the returned labels are always unique across the input
    clusters.
    """
    n_clusters = len(captions_per_cluster)

    all_captions_flat = [c for captions in captions_per_cluster for c in captions]
    noun_words = _get_noun_words(all_captions_flat)

    # Term frequency per cluster, and document frequency (how many
    # clusters' caption-sets contain the term at all).
    term_freq_per_cluster: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()

    for captions in captions_per_cluster:
        # Count each word at most once per caption (not per raw occurrence).
        # Without this, a single degenerate/repetitive caption (a known BLIP
        # failure mode — e.g. "...an aurora aurora aurora aurora...") can
        # dominate the term frequency on its own and hijack the label.
        tf: Counter[str] = Counter()
        for caption in captions:
            unique_words_in_caption = set(_tokenize(caption))
            tf.update(unique_words_in_caption)
        term_freq_per_cluster.append(tf)
        for term in tf:
            doc_freq[term] += 1

    labels: list[str] = []
    for tf in term_freq_per_cluster:
        if not tf:
            labels.append([])
            continue

        scored_terms = []
        for term, freq in tf.items():
            idf = math.log((n_clusters + 1) / (doc_freq[term] + 1)) + 1.0
            score = freq * idf
            scored_terms.append((score, term))
        # Sort by (is_noun, score) — nouns first, then highest TF-IDF score
        # within each group. Both components are "higher is better", so a
        # single reverse=True sort handles it.
        scored_terms.sort(key=lambda x: (x[1] in noun_words, x[0]), reverse=True)
        labels.append([term for _, term in scored_terms])

    # Greedy conflict resolution: clusters are processed in their given
    # order (largest cluster first, per HierarchicalAxisEngine.get_clusters),
    # so bigger/more prominent clusters get first pick of their favorite
    # word if two clusters would otherwise want the same label.
    used_labels: set[str] = set()
    final_labels: list[str] = []
    for ranked_candidates in labels:
        chosen = next((term for term in ranked_candidates if term not in used_labels), None)
        if chosen is None:
            chosen = ranked_candidates[0] if ranked_candidates else "unlabeled"
        used_labels.add(chosen)
        final_labels.append(chosen)

    return final_labels


class ClusterLabeler:
    """Wraps a BLIP captioning model to generate labels for clusters."""

    def __init__(self, model_name: str = DEFAULT_CAPTION_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading captioning model %s on %s", model_name, self.device)

        self.processor = BlipProcessor.from_pretrained(model_name)
        # use_safetensors=True avoids transformers' torch.load safety block
        # (torch < 2.6 can't load pickle .bin checkpoints) by always fetching
        # the .safetensors weights instead, which don't use pickle.
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, use_safetensors=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _caption_image(self, path: Path) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            inputs = self.processor(img, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=30,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    @staticmethod
    def _pick_representative_paths(
        cluster: Cluster,
        embeddings: np.ndarray,
        paths: list[Path],
        top_n: int,
    ) -> list[Path]:
        """Pick the `top_n` images in the cluster closest to its centroid."""
        member_embeddings = embeddings[cluster.member_indices]
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        similarities = member_embeddings @ cluster.centroid
        order = np.argsort(-similarities)  # descending: most similar first
        top_local_indices = order[:top_n]
        top_global_indices = cluster.member_indices[top_local_indices]
        return [paths[i] for i in top_global_indices]

    def label_clusters(
        self,
        clusters: list[Cluster],
        embeddings: np.ndarray,
        paths: list[Path],
        top_n: int = DEFAULT_TOP_N_REPRESENTATIVES,
    ) -> list[ClusterLabel]:
        """
        Caption representative images for every cluster first, THEN derive
        labels jointly via TF-IDF — a cluster's label depends on what all
        the OTHER clusters look like too, so this can't be done one cluster
        at a time.
        """
        all_representative_paths: list[list[Path]] = []
        all_captions: list[list[str]] = []

        for cluster in clusters:
            representative_paths = self._pick_representative_paths(
                cluster, embeddings, paths, top_n
            )
            captions = [self._caption_image(p) for p in representative_paths]
            all_representative_paths.append(representative_paths)
            all_captions.append(captions)
            logger.info(
                "Cluster %d (%d images) captions: %s",
                cluster.cluster_id,
                cluster.size,
                captions,
            )

        labels = _extract_keywords_tfidf(all_captions)

        results = [
            ClusterLabel(
                cluster_id=cluster.cluster_id,
                label=label,
                captions=captions,
                representative_paths=representative_paths,
            )
            for cluster, label, captions, representative_paths in zip(
                clusters, labels, all_captions, all_representative_paths
            )
        ]

        for r in results:
            logger.info("Cluster %d labeled '%s'", r.cluster_id, r.label)

        return results


if __name__ == "__main__":
    # Quick manual test, chained with the full pipeline so far:
    #   python src\axes\labeling.py "E:\dataset_unificado"
    from src.axes.hierarchical import HierarchicalAxisEngine
    from src.embeddings.clip_embedder import ClipEmbedder
    from src.ingestion.loader import scan_folder

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python labeling.py <folder_path>")
        sys.exit(1)

    scan = scan_folder(sys.argv[1], recursive=True)
    embedder = ClipEmbedder()
    result = embedder.embed_images(scan.valid_images, batch_size=32)

    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    engine = HierarchicalAxisEngine().fit(result.embeddings)
    clusters = engine.get_clusters(k=k)

    labeler = ClusterLabeler()
    labels = labeler.label_clusters(clusters, result.embeddings, result.paths)

    print("\nFinal axes:")
    for cl in labels:
        print(f"  '{cl.label}' — {cl.captions}")
