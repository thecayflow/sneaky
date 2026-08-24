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

torch, spacy, and transformers are imported lazily (inside the functions/
methods that use them, not at module level) — all three are heavy, and
importing this module shouldn't pay that cost until labeling actually runs.
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

import contextlib
import io
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from src.axes.hierarchical import Cluster

DEFAULT_SPACY_MODEL = "en_core_web_sm"
_NLP = None  # lazy-loaded singleton — spaCy model load isn't free


def _get_nlp():
    """
    Load the spaCy pipeline once and reuse it. NER is disabled (never used
    here), but the parser MUST stay enabled — `doc.noun_chunks` (used for
    the phrase-based labels below) is derived from the dependency parse
    and returns nothing without it. Captions are short, so the extra cost
    of running the parser is negligible in practice.
    """
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load(DEFAULT_SPACY_MODEL, disable=["ner"])
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


def _clean_noun_chunk(chunk_text: str) -> str | None:
    """
    Lowercase a spaCy noun_chunk, strip leading determiners/possessives,
    cap it at MAX_LABEL_WORDS, and reject it entirely if what's left isn't
    a useful label (empty, a single stopword, or headed by a generic
    person noun like "man"/"person" — same reasoning as the single-word
    heuristic, just applied to the chunk's head noun instead of a bare word).
    """
    words = chunk_text.lower().split()
    while words and words[0] in LEADING_DETERMINERS:
        words = words[1:]
    if not words:
        return None
    if len(words) > MAX_LABEL_WORDS:
        words = words[:MAX_LABEL_WORDS]
    if words[-1] in GENERIC_PERSON_NOUNS:
        return None
    if len(words) == 1 and (words[0] in STOPWORDS or len(words[0]) <= 2):
        return None
    return " ".join(words)


def _get_noun_chunks(captions: list[str]) -> list[list[str]]:
    """
    For each caption, its cleaned noun chunks — spaCy's dependency-parse
    -derived noun phrases (e.g. "a young child" -> "young child"), richer
    than single-word nouns since a phrase like "orange shirts" captures a
    visual attribute that a single word ("orange" or "shirts" alone) would
    lose. Requires the parser (see _get_nlp).
    """
    nlp = _get_nlp()
    all_chunks: list[list[str]] = []
    for doc in nlp.pipe(captions):
        cleaned = []
        for chunk in doc.noun_chunks:
            cleaned_text = _clean_noun_chunk(chunk.text)
            if cleaned_text is not None:
                cleaned.append(cleaned_text)
        all_chunks.append(cleaned)
    return all_chunks


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_TOP_N_REPRESENTATIVES = 8
MAX_LABEL_WORDS = 3  # keeps phrase labels compact enough for the radar/mosaic/PDF

# Generic person nouns — near-ubiquitous in personal photo captions, so
# they rarely help distinguish one semantic cluster from another. "people"
# is deliberately excluded from this set: as a collective it can denote a
# real scene type (crowds/gatherings), unlike singular "man"/"boy"/etc.
GENERIC_PERSON_NOUNS = {
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "baby", "babies", "kid", "kids", "child", "children", "person",
}

# Leading words to strip from a noun chunk before using it as a label —
# spaCy's noun_chunks include determiners/possessives ("a young child" ->
# "young child"), which read awkwardly as a short axis label.
LEADING_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "many", "several", "few", "each",
    "every", "any", "no",
}

# Small stopword list — enough to filter out captioning boilerplate words.
# Not meant to be exhaustive, just enough for short BLIP-style captions.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "is", "are",
    "to", "for", "by", "it", "its", "this", "that", "some", "there", "as",
    "photo", "picture", "image", "close", "up", "shot", "view", "background",
    *GENERIC_PERSON_NOUNS,
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


def _extract_keywords_noun_chunks(captions_per_cluster: list[list[str]]) -> list[str | None]:
    """
    Pick one distinctive SHORT PHRASE per cluster (spaCy noun chunks, e.g.
    "young child" or "orange shirts") instead of a single word — phrases
    capture visual attributes ("running race") that a lone word ("race" or
    "running") loses on its own. Same cross-cluster TF-IDF idea as the
    word-level version above: a phrase scores high if it's frequent in
    this cluster's captions but rare in other clusters' captions.

    Returns None (not a fallback string) for any cluster where no usable
    noun chunk was found at all — the caller (_extract_keywords) decides
    how to fill that gap, so this function stays a pure "phrase attempt".
    """
    n_clusters = len(captions_per_cluster)

    chunks_per_cluster = [_get_noun_chunks(captions) for captions in captions_per_cluster]

    term_freq_per_cluster: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()

    for chunks_per_caption in chunks_per_cluster:
        # Count each phrase at most once per caption — same rationale as
        # the word-level version: one degenerate/repetitive caption
        # shouldn't be able to dominate the term frequency on its own.
        tf: Counter[str] = Counter()
        for chunks_in_one_caption in chunks_per_caption:
            tf.update(set(chunks_in_one_caption))
        term_freq_per_cluster.append(tf)
        for term in tf:
            doc_freq[term] += 1

    ranked_per_cluster: list[list[str]] = []
    for tf in term_freq_per_cluster:
        if not tf:
            ranked_per_cluster.append([])
            continue
        scored_terms = []
        for term, freq in tf.items():
            idf = math.log((n_clusters + 1) / (doc_freq[term] + 1)) + 1.0
            score = freq * idf
            # With only a handful of captions per cluster, exact-phrase
            # ties are common (several distinct candidates land on the
            # exact same freq/doc_freq). A mild per-word bonus (capped by
            # MAX_LABEL_WORDS, so at most +20% for a 3-word phrase) breaks
            # those ties toward the more descriptive multi-word candidate
            # instead of an incidental short word winning by default —
            # e.g. "plate of food" over "table" for a food-themed cluster.
            n_words = len(term.split())
            score *= 1 + 0.10 * (n_words - 1)
            scored_terms.append((score, term))
        scored_terms.sort(key=lambda x: -x[0])
        ranked_per_cluster.append([term for _, term in scored_terms])

    # Same greedy conflict resolution as the word-level version: clusters
    # are processed in their given order (largest first), so bigger
    # clusters get first pick when two would otherwise want the same phrase.
    used_labels: set[str] = set()
    final_labels: list[str | None] = []
    for ranked_candidates in ranked_per_cluster:
        chosen = next((term for term in ranked_candidates if term not in used_labels), None)
        final_labels.append(chosen)
        if chosen is not None:
            used_labels.add(chosen)

    return final_labels


def _extract_keywords(captions_per_cluster: list[list[str]]) -> list[str]:
    """
    Public entry point: one distinctive label per cluster — a short noun
    phrase where possible (e.g. "orange shirts", "running race"), falling
    back to the single-distinctive-word heuristic for any cluster where no
    usable noun chunk could be extracted at all, so no cluster is ever
    left unlabeled.
    """
    phrase_labels = _extract_keywords_noun_chunks(captions_per_cluster)
    if all(label is not None for label in phrase_labels):
        return phrase_labels  # type: ignore[return-value]

    word_labels = _extract_keywords_tfidf(captions_per_cluster)
    used = {label for label in phrase_labels if label is not None}
    final: list[str] = []
    for phrase_label, word_label in zip(phrase_labels, word_labels):
        if phrase_label is not None:
            final.append(phrase_label)
            continue
        candidate = word_label
        if candidate in used:
            # Extremely unlikely (a fallback word colliding with an
            # already-chosen phrase) — disambiguate rather than silently
            # duplicate a label.
            candidate = f"{candidate} (2)"
        used.add(candidate)
        final.append(candidate)
    return final


class ClusterLabeler:
    """Wraps a BLIP captioning model to generate labels for clusters."""

    def __init__(self, model_name: str = DEFAULT_CAPTION_MODEL, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading captioning model %s on %s", model_name, self.device)

        # transformers prints some of its own internal noise directly to
        # stdout/stderr somewhere in this import + loading block — NOT
        # through Python's logging module at all, confirmed by testing:
        # these specific messages (about processor Kwargs documentation
        # for models this project never uses — DeepSeek VL, Kimi, PaddleOCR
        # VL — unrelated to BLIP) survived raising
        # logging.getLogger("transformers") all the way to CRITICAL, the
        # highest level there is. Redirecting stdout/stderr to a throwaway
        # buffer for just this block catches it regardless of which exact
        # line triggers it, rather than guessing at (and depending on) a
        # specific mechanism that might move between transformers
        # versions. A genuine failure here still raises a normal Python
        # exception, which isn't affected by redirecting stdout/stderr and
        # prints its own traceback normally once it propagates past this
        # block — nothing here can hide a real error, only discard
        # harmless print()-based noise.
        _discard = io.StringIO()
        with contextlib.redirect_stdout(_discard), contextlib.redirect_stderr(_discard):
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(model_name)
            # use_safetensors=True avoids transformers' torch.load safety
            # block (torch < 2.6 can't load pickle .bin checkpoints) by
            # always fetching the .safetensors weights instead, which
            # don't use pickle.
            #
            # low_cpu_mem_usage=False: with accelerate installed,
            # transformers defaults to a "meta device" loading path
            # (builds the model skeleton without real data first, fills
            # it in afterward) to save RAM during loading. This BLIP
            # checkpoint is missing one parameter
            # (text_decoder.cls.predictions.decoder.bias — newly
            # initialized, not part of the checkpoint), and that specific
            # parameter can end up left on the meta device instead of
            # properly materialized, making the .to(self.device) call
            # below fail with "Cannot copy out of meta tensor; no data!".
            # Disabling the meta-device path avoids this entirely — this
            # model is small enough that the extra RAM during loading
            # doesn't matter.
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name, use_safetensors=True, low_cpu_mem_usage=False
            )
            # Explicitly re-resolve tied weights (BLIP's text decoder head
            # shares/ties its output bias with another layer) before
            # moving off the meta device — without this, the tied
            # parameter can stay a meta placeholder even with
            # low_cpu_mem_usage=False, and .to() then fails with "Cannot
            # copy out of meta tensor; no data!".
            self.model.tie_weights()
            self.model = self.model.to(self.device)
        self.model.eval()

    def _caption_image(self, path: Path) -> str:
        import torch

        with Image.open(path) as img:
            img = img.convert("RGB")
            inputs = self.processor(img, return_tensors="pt").to(self.device)
        # Same stdout/stderr redirection as __init__'s model-loading block,
        # and for the same reason: transformers' own print()-based noise
        # (confirmed NOT to be Python logging — surviving both raising the
        # logger to CRITICAL and redirecting stdout/stderr around loading)
        # can apparently also fire lazily on the FIRST real inference call,
        # not just at load time — this is the actual call that generates a
        # caption, so it's the other place this needs covering. A genuine
        # failure here still raises a normal Python exception, unaffected
        # by the redirection.
        _discard = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stdout(_discard), contextlib.redirect_stderr(_discard):
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
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ClusterLabel]:
        """
        Caption representative images for every cluster first, THEN derive
        labels jointly via TF-IDF — a cluster's label depends on what all
        the OTHER clusters look like too, so this can't be done one cluster
        at a time.

        on_progress: optional (done, total_clusters) callback, called once
        per cluster as its captioning finishes — no throttling needed here
        (unlike ClipEmbedder.embed_images' own on_progress), since the
        number of clusters is typically small (a few dozen at most), so
        every update is already infrequent and meaningful. None (the
        default): no callback, exactly today's behavior.
        """
        all_representative_paths: list[list[Path]] = []
        all_captions: list[list[str]] = []
        total_clusters = len(clusters)

        for i, cluster in enumerate(clusters):
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
            if on_progress is not None:
                on_progress(i + 1, total_clusters)

        labels = _extract_keywords(all_captions)

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
