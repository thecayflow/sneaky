"""
app.py

Streamlit UI tying the whole Fase 1 (+ thumbnail preview) pipeline
together: point it at any local folder of images, pick how many
auto-detected axes you want, add your own custom text axes on top, see
the semantic radar, and browse the images behind any axis via its "View
images" button. No fixed ingestion folder — the path is whatever you type
in, and changing it (or the axis count) re-runs the pipeline, using the
on-disk cache wherever possible.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

import numpy as np

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times.
pillow_heif.register_heif_opener()

from src.axes.custom import create_custom_axis
from src.axes.hierarchical import MIN_AXES
from src.axes.labeling import ClusterLabeler
from src.embeddings.clip_embedder import ClipEmbedder
from src.persistence import cache
from src.pipeline import get_embeddings_only, run_pipeline
from src.scoring.scoring import (
    OTHER_LABEL,
    compute_score_matrix,
    get_axis_counts_by_dominance,
    get_dominant_labels,
    get_radar_values_by_dominance,
    get_radar_values_normalized,
    get_ranked_images_for_axis,
)
from src.viz.radar import build_radar_figure, build_stacked_radar_figure
from src.viz.scatter import build_scatter_figure
from src.viz.tsne_projection import compute_tsne_projection, get_or_compute_tsne
from src.similarity.phash import (
    DEFAULT_CHAIN_MAX_LENGTH,
    DEFAULT_GROUP_THRESHOLD_BITS,
    build_grouped_chain,
    build_similarity_chain,
    compute_global_order,
    get_all_duplicate_groups_combined,
    get_or_compute_duplicate_stats,
    get_or_compute_global_order,
    get_or_compute_phashes,
)
from src.report.metrics import compute_overview_metrics, get_representative_images_by_axis
from src.report.pdf_report import generate_pdf_report
from src.scoring.dataset_similarity import compute_clip_mmd, compute_self_split_mmd
from src.scoring.wavelet_similarity import (
    compute_wavelet_mmd,
    compute_wavelet_self_split_mmd,
    get_or_compute_wavelet_features,
)
from src.viz.umap_projection import compute_umap_projection, get_or_compute_umap

MAX_AXES = 25  # soft cap for the + button; raising axis count re-runs captioning
THUMB_BATCH_SIZE = 24  # how many more thumbnails "Load more" reveals each click
THUMB_COLUMNS = 4
SIMILARITY_CHAIN_THUMB_SIZE = 140  # px, for the horizontal-scroll chain

st.set_page_config(page_title="sneakyReport™ — Visual Dataset Intelligence", layout="centered")

st.title("sneakyReport™ — Visual Dataset Intelligence")
st.caption(
    "Point this at any local folder of images to see its semantic makeup as a radar chart. "
    "Use the 'View images' button next to any axis to browse the images behind it."
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> ClipEmbedder:
    """Loaded once per Streamlit session and reused for custom axis text embeddings."""
    return ClipEmbedder()


@st.cache_resource(show_spinner=False)
def get_labeler() -> ClusterLabeler:
    """Loaded once per Streamlit session — avoids reloading BLIP's model
    weights every time a new (k, linkage_method) combination needs
    captioning within the same session (each new combination still does
    its own captioning work, just without re-loading the model first).

    show_spinner=False on both this and get_embedder(): the caller already
    wraps the whole operation (Analyze / Load comparison feed) in its own
    st.spinner with a more specific message — without this, Streamlit's
    own generic "Running get_embedder()." spinner briefly overlaps with
    it the first time each is genuinely loaded in a session, which reads
    as two spinners for one wait. Purely cosmetic — no change to what
    gets cached or when."""
    return ClusterLabeler()


def _image_to_base64_thumb(path: Path, max_size: int = SIMILARITY_CHAIN_THUMB_SIZE) -> str | None:
    """Small JPEG thumbnail, base64-encoded, for embedding directly in raw HTML
    (a local file:// path won't reliably load in an <img> tag on a page served
    over http://localhost — embedding the bytes directly sidesteps that)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


if "k" not in st.session_state:
    st.session_state.k = 10
if "linkage_method" not in st.session_state:
    st.session_state.linkage_method = "ward"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "custom_axis_labels" not in st.session_state:
    st.session_state.custom_axis_labels = []
if "excluded_axis_labels" not in st.session_state:
    st.session_state.excluded_axis_labels = set()
if "viewing_axis" not in st.session_state:
    st.session_state.viewing_axis = None
if "viewing_axis_scope" not in st.session_state:
    st.session_state.viewing_axis_scope = "primary"
if "thumb_shown_count" not in st.session_state:
    st.session_state.thumb_shown_count = THUMB_BATCH_SIZE
if "zoomed_image_path" not in st.session_state:
    st.session_state.zoomed_image_path = None
if "scatter_selected_info" not in st.session_state:
    st.session_state.scatter_selected_info = None
if "scatter_dismissed_path" not in st.session_state:
    st.session_state.scatter_dismissed_path = None
if "compare_result" not in st.session_state:
    st.session_state.compare_result = None
if "pdf_report_bytes" not in st.session_state:
    st.session_state.pdf_report_bytes = None
if "similarity_chain_cache_key" not in st.session_state:
    st.session_state.similarity_chain_cache_key = None
if "similarity_chain_html" not in st.session_state:
    st.session_state.similarity_chain_html = None
if "scatter_data_cache_key" not in st.session_state:
    st.session_state.scatter_data_cache_key = None
if "scatter_coords" not in st.session_state:
    st.session_state.scatter_coords = None
if "scatter_dominant_labels" not in st.session_state:
    st.session_state.scatter_dominant_labels = None
if "scatter_cluster_numbers" not in st.session_state:
    st.session_state.scatter_cluster_numbers = None
if "scatter_similarities" not in st.session_state:
    st.session_state.scatter_similarities = None
if "scatter_paths" not in st.session_state:
    st.session_state.scatter_paths = None
if "scatter_dataset_origin" not in st.session_state:
    # None = single dataset. Otherwise a "primary"/"comparison" list,
    # same length/order as the other scatter_* arrays — see the Scatter
    # view section for how this feeds build_scatter_figure's dataset
    # legend/marker-shape logic.
    st.session_state.scatter_dataset_origin = None
if "scatter_dataset_display_names" not in st.session_state:
    st.session_state.scatter_dataset_display_names = None
if "copied_sources" not in st.session_state:
    # dest folder (resolved absolute path string) -> set of resolved source
    # image paths already copied there this session, so re-copying an axis
    # into the same destination never duplicates a file that's already there.
    st.session_state.copied_sources = {}


def run_analysis(path: str, k: int, linkage_method: str) -> None:
    """Run the pipeline for `path`/`k`/`linkage_method` and store the result (or error)."""
    already_cached = cache.load_axes(path, k, linkage_method=linkage_method) is not None
    spinner_msg = (
        "Loading cached results..."
        if already_cached
        else "Analyzing images — first run on this folder (or this linkage method) can take "
        "several minutes (embeddings + captioning). Later runs will be much faster."
    )

    with st.spinner(spinner_msg):
        try:
            axes = run_pipeline(
                path, k=k, linkage_method=linkage_method, embedder=get_embedder(), labeler=get_labeler()
            )
            loaded = cache.load_scan_and_embeddings(path)
            if loaded is None:
                st.error("Pipeline ran but no cached embeddings were found — please retry.")
                return
            paths, embeddings, _ = loaded
        except (FileNotFoundError, NotADirectoryError) as exc:
            st.session_state.last_result = None
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_result = None
            st.error(f"Something went wrong while analyzing this folder: {exc}")
            return

    st.session_state.last_result = {
        "path": path,
        "dataset_name": Path(path).name,
        "k": k,
        "base_axes": axes,  # auto-detected axes only, from the hierarchical tree
        "embeddings": embeddings,
        "paths": paths,
    }
    st.session_state.viewing_axis = None


def _open_axis_dialog(label: str, scope: str = "primary") -> None:
    """
    scope: "primary" (default — every existing call site keeps this,
    unchanged behavior), "comparison" (Separated mode's second row —
    reuses this same dialog with the comparison feed's data instead, no
    new dialog logic needed for that), or "aggregated" (Aggregated mode —
    shows both datasets' images together, tagged by origin).
    """
    st.session_state.viewing_axis = label
    st.session_state.viewing_axis_scope = scope
    st.session_state.thumb_shown_count = THUMB_BATCH_SIZE


def _close_axis_dialog() -> None:
    st.session_state.viewing_axis = None
    st.session_state.zoomed_image_path = None


def _load_more_thumbs(new_count: int) -> None:
    st.session_state.thumb_shown_count = new_count


def _zoom_image(path_str: str) -> None:
    st.session_state.zoomed_image_path = path_str


def _unzoom_image() -> None:
    st.session_state.zoomed_image_path = None


def _close_scatter_selection() -> None:
    # Remember what we just dismissed, so the still-selected point on the
    # Plotly chart (selections persist across reruns until a NEW click)
    # doesn't immediately re-open the same dialog.
    if st.session_state.scatter_selected_info:
        st.session_state.scatter_dismissed_path = st.session_state.scatter_selected_info["path"]
    st.session_state.scatter_selected_info = None


def _show_similar_from_scatter(axis_label: str) -> None:
    """'Show similar' — closes this single-image dialog and opens the axis
    mosaic instead, reusing the exact same escape mechanism as closing."""
    if st.session_state.scatter_selected_info:
        st.session_state.scatter_dismissed_path = st.session_state.scatter_selected_info["path"]
    st.session_state.scatter_selected_info = None
    st.session_state.viewing_axis = axis_label
    st.session_state.thumb_shown_count = THUMB_BATCH_SIZE


@st.dialog("Image", width="large", dismissible=False)
def show_single_image_dialog(info: dict) -> None:
    current = st.session_state.scatter_selected_info
    if current is None or current["path"] != info["path"]:
        # Same escape trick as show_axis_images_dialog below: a rerun
        # triggered from inside a dialog only re-runs the dialog itself
        # (fragment behavior), so force a full app rerun to actually close.
        st.rerun(scope="app")
        return

    path_str = info["path"]

    header_col, close_col = st.columns([10, 1])
    with header_col:
        st.subheader(info["filename"])
    with close_col:
        st.button(
            "✕", key="close_scatter_dialog", on_click=_close_scatter_selection, help="Close"
        )

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Dominant axis", info["axis"])
    meta_col2.metric("Similarity", f"{info['similarity']:.2f}")
    meta_col3.metric("Cluster", f"#{info['cluster']:02d}")

    with st.spinner("Loading image..."):
        try:
            with Image.open(path_str) as im:
                im = ImageOps.exif_transpose(im)
                # Downscale before sending to the browser — the dialog only
                # displays it at a fraction of native resolution anyway, and
                # large originals (e.g. big Wikimedia/DSLR files) made
                # st.image() itself take several seconds to encode + send,
                # which is what was causing the visible delay.
                im.thumbnail((1600, 1600))
                st.image(im, width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't load {path_str}: {exc}")

    st.button(
        "Show similar",
        on_click=_show_similar_from_scatter,
        args=(info["axis"],),
        help="Browse the mosaic of images that dominate this same axis.",
    )


def _pooled_score_reference(result, compare_result, full_axes):
    """
    Raw (unstandardized) score matrix pooling BOTH datasets' images
    against full_axes — the shared z-score reference so images from
    either dataset are judged in the SAME frame when comparing. See
    scoring.py::get_dominant_labels's standardize_reference docstring for
    why each dataset standardizing against only its own scores is a real
    bug (an axis with genuinely no matches in one dataset could still
    "win" for an unrelated image there, since that axis's score
    distribution is unusually tight/low within that one dataset alone).

    None when there's no comparison feed loaded — every call site below
    passes this straight through as standardize_reference, and None
    there means "use the embeddings' own scores", exactly the prior,
    single-dataset behavior.
    """
    if compare_result is None:
        return None
    primary_scores = compute_score_matrix(result["embeddings"], full_axes)
    compare_scores = compute_score_matrix(compare_result["embeddings"], full_axes)
    return np.concatenate([primary_scores, compare_scores], axis=0)


def _score_and_pool_for_scatter(result, compare_result, full_axes, other_threshold):
    """
    Shared by the live Scatter view and the PDF's UMAP page: scores every
    image (the primary feed, and the comparison feed too if one is
    loaded) against `full_axes`, and — when comparing — prepares the
    pooled embeddings/paths/dataset-origin arrays needed to project both
    feeds onto the same 2D space together (t-SNE has no way to place new
    points into an already-fit projection, so both datasets need to go
    through projection together, not one after the other).

    Returns (embeddings_for_projection, dominant_labels, cluster_numbers,
    similarities, paths, dataset_origin, dataset_display_names) — the
    caller still has to actually run the projection (t-SNE/UMAP) on the
    returned embeddings. dataset_origin/dataset_display_names are None
    when there's no comparison feed loaded (single dataset).
    """
    axis_index_by_label = {axis.label: i + 1 for i, axis in enumerate(full_axes)}
    standardize_reference = _pooled_score_reference(result, compare_result, full_axes)

    def _score_against_axes(embeddings_arr):
        dom_labels, score_mat, _ = get_dominant_labels(
            embeddings_arr, full_axes, other_threshold=other_threshold, standardize_reference=standardize_reference
        )
        clusters, sims = [], []
        for i, lbl in enumerate(dom_labels):
            if lbl == OTHER_LABEL:
                clusters.append(0)
                sims.append(float(score_mat[i].max()))
            else:
                axis_idx = axis_index_by_label[lbl]
                clusters.append(axis_idx)
                sims.append(float(score_mat[i, axis_idx - 1]))
        return list(dom_labels), clusters, sims

    if compare_result is None:
        dominant_labels, cluster_numbers, similarities = _score_against_axes(result["embeddings"])
        return (
            result["embeddings"],
            dominant_labels,
            cluster_numbers,
            similarities,
            list(result["paths"]),
            None,
            None,
        )

    pooled_embeddings = np.concatenate([result["embeddings"], compare_result["embeddings"]], axis=0)
    primary_labels, primary_clusters, primary_sims = _score_against_axes(result["embeddings"])
    compare_labels, compare_clusters, compare_sims = _score_against_axes(compare_result["embeddings"])
    dominant_labels = primary_labels + compare_labels
    cluster_numbers = primary_clusters + compare_clusters
    similarities = primary_sims + compare_sims
    all_paths = list(result["paths"]) + list(compare_result["paths"])
    dataset_origin = ["primary"] * len(result["paths"]) + ["comparison"] * len(compare_result["paths"])
    dataset_display_names = {
        "primary": result["dataset_name"],
        "comparison": compare_result["dataset_name"],
    }
    return (
        pooled_embeddings,
        dominant_labels,
        cluster_numbers,
        similarities,
        all_paths,
        dataset_origin,
        dataset_display_names,
    )


def _copy_axis_images_to(
    label: str,
    dest_folder: str,
    embeddings,
    axes,
    paths,
    other_threshold: float,
    standardize_reference=None,
) -> str:
    """
    Copies every image currently assigned to `label` into dest_folder
    (created automatically if needed), skipping any source image already
    copied there earlier this session — so clicking "Copy images to..."
    again for the same axis/destination combo is always safe and never
    duplicates a file.

    standardize_reference: see scoring.py::get_dominant_labels — pass the
    pooled score matrix of both datasets when comparing, so which images
    count as belonging to `label` is consistent with what's shown
    everywhere else (the axis list, View images, the radar/scatter).

    Returns the summary message; the caller displays it (kept separate so
    the caller can decide exactly when/how to show it).
    """
    dest_dir = Path(dest_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ranked = get_ranked_images_for_axis(
        embeddings, axes, paths, label, other_threshold, standardize_reference=standardize_reference
    )
    source_paths = [p for p, _ in ranked]

    dest_key = str(dest_dir.resolve())
    already_copied = st.session_state.copied_sources.setdefault(dest_key, set())
    existing_names = {p.name for p in dest_dir.iterdir() if p.is_file()}

    to_copy = [p for p in source_paths if str(p.resolve()) not in already_copied]
    skipped_count = len(source_paths) - len(to_copy)
    total = len(to_copy)

    if total > 0:
        progress_bar = st.progress(0, text=f"Copying images from '{label}'...")
        for i, src in enumerate(to_copy, start=1):
            # Two different source folders can contain a same-named file —
            # de-duplicate on the way in rather than silently overwrite one
            # image with another.
            candidate = src.name
            counter = 1
            while candidate in existing_names:
                candidate = f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            shutil.copy2(src, dest_dir / candidate)
            existing_names.add(candidate)
            already_copied.add(str(src.resolve()))
            progress_bar.progress(i / total, text=f"Copying images from '{label}'... ({i}/{total})")
        progress_bar.empty()

    if total == 0 and skipped_count > 0:
        # Explicit, unambiguous message for the "nothing new" case — the
        # de-dup safeguard is deliberate (see docstring), but a silent
        # "0 copied" would look identical to something actually failing.
        message = (
            f"0 new images copied from '{label}' — all {skipped_count} were "
            f"already copied to this folder earlier this session. Delete "
            f"them from the destination folder yourself if you want a "
            f"fresh copy."
        )
    else:
        message = f"Copied {total} images from '{label}' to {dest_dir}"
        if skipped_count:
            message += f" ({skipped_count} already copied here before, skipped)."
    return message


def _copy_axis_images_to_aggregated(
    label: str, dest_folder: str, result, compare_result, full_axes, other_threshold: float
) -> str:
    """
    Aggregated-mode "Copy images to..." — copies this axis's images from
    BOTH datasets into the same destination folder. Just calls the
    existing single-dataset _copy_axis_images_to twice in sequence (once
    per dataset) rather than a new merged code path — that function's
    de-duplication is already keyed by each image's own source path, so
    calling it back-to-back for two different datasets is safe as-is; no
    changes needed there.
    """
    standardize_reference = _pooled_score_reference(result, compare_result, full_axes)
    msg_primary = _copy_axis_images_to(
        label,
        dest_folder,
        result["embeddings"],
        full_axes,
        result["paths"],
        other_threshold,
        standardize_reference=standardize_reference,
    )
    msg_compare = _copy_axis_images_to(
        label,
        dest_folder,
        compare_result["embeddings"],
        full_axes,
        compare_result["paths"],
        other_threshold,
        standardize_reference=standardize_reference,
    )
    return f"{result['dataset_name']}: {msg_primary}\n{compare_result['dataset_name']}: {msg_compare}"


def _render_axis_management_row(
    label: str,
    tag: str,
    separated: bool,
    axis_counts: dict,
    compare_axis_counts: dict,
    result,
    compare_result,
    full_axes,
    other_threshold: float,
    standardize_reference,
    copy_destination: str,
    on_remove,
    key_suffix: str,
    is_removable: bool = True,
) -> None:
    """
    Renders one axis's row(s) in the Axes management UI — a single
    combined row in Aggregated mode, or a bordered two-row block (🔵
    primary dataset / 🟠 comparison dataset) in Separated mode. Shared by
    BOTH the main "Axes" list (auto-detected axes + Other) and the
    "Custom axes" section — previously the Custom axes section had its
    own separate, older, comparison-unaware listing (plain "View images"
    only, no dataset split, no Copy button), which also meant custom
    axes appeared TWICE on the page: once correctly (with the dataset
    split) in the main Axes list above, and once again — redundantly,
    without the split — under "Custom axes" below.

    on_remove: a zero-arg callable invoked when "Remove axis" is clicked
    — different for auto-detected axes (add to excluded_axis_labels) vs.
    custom axes (remove from custom_axis_labels), so this stays a
    parameter rather than being hardcoded here.

    key_suffix: makes widget keys unique between the two call sites
    (e.g. "auto" vs "custom") even when the same label could in
    principle appear in both (it can't today, but keeps this robust).
    """
    if not separated:
        count = axis_counts.get(label, 0) + compare_axis_counts.get(label, 0)
        c1, c2, c3, c4 = st.columns([2.4, 1.3, 1.6, 1])
        c1.write(f"- **{label}**{tag} — {count} images")
        c2.button(
            "View images",
            key=f"view_{key_suffix}_{label}",
            on_click=_open_axis_dialog,
            args=(label, "aggregated" if compare_result is not None else "primary"),
        )
        if c3.button("Copy images to...", key=f"copy_{key_suffix}_{label}"):
            if not copy_destination:
                st.error("Enter a destination folder below first.")
            elif compare_result is not None:
                message = _copy_axis_images_to_aggregated(
                    label, copy_destination, result, compare_result, full_axes, other_threshold
                )
                st.toast(message, icon="✅")
            else:
                message = _copy_axis_images_to(
                    label,
                    copy_destination,
                    result["embeddings"],
                    full_axes,
                    result["paths"],
                    other_threshold,
                    standardize_reference=standardize_reference,
                )
                st.toast(message, icon="✅")
        if is_removable:
            if c4.button("Remove axis", key=f"remove_{key_suffix}_{label}"):
                on_remove()
    else:
        with st.container(border=True):
            primary_count = axis_counts.get(label, 0)
            c1, c2, c3, c4 = st.columns([2.4, 1.3, 1.6, 1])
            c1.write(f"🔵 **{label}**{tag} — {primary_count} images ({result['dataset_name']})")
            c2.button(
                "View images",
                key=f"view_{key_suffix}_{label}_primary",
                on_click=_open_axis_dialog,
                args=(label, "primary"),
            )
            if c3.button("Copy images to...", key=f"copy_{key_suffix}_{label}_primary"):
                if not copy_destination:
                    st.error("Enter a destination folder below first.")
                else:
                    message = _copy_axis_images_to(
                        label,
                        copy_destination,
                        result["embeddings"],
                        full_axes,
                        result["paths"],
                        other_threshold,
                        standardize_reference=standardize_reference,
                    )
                    st.toast(message, icon="✅")
            if is_removable:
                if c4.button("Remove axis", key=f"remove_{key_suffix}_{label}_shared"):
                    on_remove()

            compare_count = compare_axis_counts.get(label, 0)
            c1b, c2b, c3b, c4b = st.columns([2.4, 1.3, 1.6, 1])
            c1b.write(f"🟠 **{label}**{tag} — {compare_count} images ({compare_result['dataset_name']})")
            c2b.button(
                "View images",
                key=f"view_{key_suffix}_{label}_compare",
                on_click=_open_axis_dialog,
                args=(label, "comparison"),
            )
            if c3b.button("Copy images to...", key=f"copy_{key_suffix}_{label}_compare"):
                if not copy_destination:
                    st.error("Enter a destination folder below first.")
                else:
                    message = _copy_axis_images_to(
                        label,
                        copy_destination,
                        compare_result["embeddings"],
                        full_axes,
                        compare_result["paths"],
                        other_threshold,
                        standardize_reference=standardize_reference,
                    )
                    st.toast(message, icon="✅")
            # c4b intentionally left blank — Remove axis already shown
            # once above, applies to both rows.


@st.dialog("Axis images", width="large", dismissible=False)
def show_axis_images_dialog(
    label: str,
    embeddings,
    axes,
    paths,
    other_threshold: float,
    extra_source: dict | None = None,
    primary_dataset_name: str | None = None,
    standardize_reference=None,
) -> None:
    """
    extra_source / primary_dataset_name are only used by Aggregated-mode
    comparisons (see _open_axis_dialog's "aggregated" scope) — every
    existing call site omits them, which keeps this function's behavior
    completely unchanged from before: extra_source=None means no second
    dataset gets merged in, and primary_dataset_name=None means no
    per-image dataset tag gets added to captions.

    extra_source, when given: {"embeddings", "paths", "dataset_name"} for
    the SECOND dataset — scored against the same `axes`/`other_threshold`
    as the primary, then its ranked images are appended after the
    primary's own (not interleaved by score — simplest predictable
    ordering: this dataset's images first, then the other's).

    standardize_reference: see scoring.py::get_dominant_labels — pass the
    pooled score matrix of both datasets whenever this dialog might show
    the comparison feed's images (scope="comparison" or "aggregated"), so
    which images end up assigned to `label` is consistent with everywhere
    else (the axis list, Copy images, the radar/scatter) — without this,
    an axis with genuinely no matches in one dataset could still "win"
    for an unrelated image in that dataset alone.
    """
    # Close button rendered FIRST, unconditionally, before any other logic
    # — guarantees there's always a manual way to close this dialog, no
    # matter what state led to it being open. The automatic self-close
    # rerun below (for when the Close button itself was clicked) has
    # proven unreliable in some environments, so this is the real safety
    # net, not just a nicety.
    header_col, close_col = st.columns([10, 1])
    with close_col:
        st.button("✕", key="close_axis_dialog", on_click=_close_axis_dialog, help="Close")

    if st.session_state.viewing_axis != label:
        # We've been asked to close (Close button already cleared
        # viewing_axis via its callback), or something else cleared it
        # out from under us. A normal rerun triggered from inside a
        # dialog only re-runs the dialog itself (it behaves like a
        # fragment), which would just redraw this same content — so we
        # force a full app-level rerun instead, which re-checks the
        # top-level "should this dialog even be shown" condition and
        # actually closes it. If that somehow doesn't take effect, the
        # Close button above still works as a manual fallback.
        st.rerun(scope="app")
        return

    ranked = get_ranked_images_for_axis(
        embeddings, axes, paths, label, other_threshold=other_threshold, standardize_reference=standardize_reference
    )
    # (path, score, dataset name or None) — the dataset name stays None
    # for every entry in the single-dataset case (extra_source is None,
    # primary_dataset_name typically None too), which keeps captions
    # below in their original, unchanged format.
    tagged = [(p, s, primary_dataset_name) for p, s in ranked]
    if extra_source is not None:
        compare_ranked = get_ranked_images_for_axis(
            extra_source["embeddings"],
            axes,
            extra_source["paths"],
            label,
            other_threshold=other_threshold,
            standardize_reference=standardize_reference,
        )
        tagged += [(p, s, extra_source["dataset_name"]) for p, s in compare_ranked]

    total = len(tagged)
    shown = min(st.session_state.thumb_shown_count, total)

    with header_col:
        st.subheader(label)
        if label == OTHER_LABEL:
            st.caption(f"{total} images — ordered worst-fit first (clearest outliers)")
        else:
            st.caption(f"{total} images — ordered by similarity to the axis, closest first")
        if extra_source is not None:
            st.caption(f"🔵 {primary_dataset_name}   🟠 {extra_source['dataset_name']}")

    if st.session_state.zoomed_image_path:
        st.button("◀ Back to grid", key="back_to_grid", on_click=_unzoom_image)
        with st.spinner("Loading image..."):
            try:
                with Image.open(st.session_state.zoomed_image_path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((1600, 1600))
                    st.image(im, width="stretch")
            except Exception:  # noqa: BLE001
                st.error(f"Couldn't load {st.session_state.zoomed_image_path}")
        return

    visible = tagged[:shown]
    for row_start in range(0, len(visible), THUMB_COLUMNS):
        row_items = visible[row_start : row_start + THUMB_COLUMNS]
        cols = st.columns(THUMB_COLUMNS)
        for col, (img_path, score, ds_name) in zip(cols, row_items):
            with col:
                try:
                    with Image.open(img_path) as im:
                        im = ImageOps.exif_transpose(im)
                        caption = f"similarity: {score:.3f}"
                        if ds_name is not None:
                            dot = "🔵" if ds_name == primary_dataset_name else "🟠"
                            caption = f"{dot} {ds_name} · {caption}"
                        st.image(im, width="stretch", caption=caption)
                    st.button(
                        "🔍 View full size",
                        key=f"zoom_{img_path}",
                        on_click=_zoom_image,
                        args=(str(img_path),),
                    )
                except Exception:  # noqa: BLE001
                    st.caption(f"⚠️ couldn't load\n{img_path.name}")

    if shown < total:
        st.button(
            f"Load {min(THUMB_BATCH_SIZE, total - shown)} more",
            key="load_more_thumbs",
            on_click=_load_more_thumbs,
            args=(shown + THUMB_BATCH_SIZE,),
        )
    else:
        st.caption("— end of results —")


path = st.text_input(
    "Dataset folder path",
    placeholder=r"C:\Projects\sneaky\dataset_samples\sample_01",
    help="Any local folder — subfolders are searched too. Nothing is copied.",
)

with st.expander("Compare with a second feed (optional)"):
    st.caption(
        "The comparison feed doesn't get its own axes — its images are scored "
        "against the axes of the primary feed above, so the radar shows how "
        "differently it fits the same categories."
    )
    compare_path = st.text_input(
        "Second feed folder path",
        key="compare_path_input",
        placeholder=r"C:\Projects\sneaky\dataset_samples\sample_02",
    )
    if st.button("Load comparison feed"):
        if not compare_path:
            st.error("Please enter a folder path.")
        elif not Path(compare_path).exists() or not Path(compare_path).is_dir():
            st.error(f"'{compare_path}' is not a valid folder.")
        else:
            with st.spinner(
                "Loading comparison feed — embeddings only, no clustering needed "
                "since it reuses the primary feed's axes."
            ):
                try:
                    c_paths, c_embeddings, _ = get_embeddings_only(compare_path, embedder=get_embedder())
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Something went wrong loading this feed: {exc}")
                    c_paths = None
            if c_paths is not None:
                st.session_state.compare_result = {
                    "path": compare_path,
                    "dataset_name": Path(compare_path).name,
                    "paths": c_paths,
                    "embeddings": c_embeddings,
                }
                st.rerun()
    if st.session_state.compare_result is not None:
        st.success(
            f"Comparing against: {st.session_state.compare_result['dataset_name']} "
            f"({len(st.session_state.compare_result['paths'])} images)"
        )
        if st.button("Remove comparison feed"):
            st.session_state.compare_result = None
            st.rerun()

new_k = st.number_input(
    "Axis number:",
    min_value=MIN_AXES,
    max_value=MAX_AXES,
    value=st.session_state.k,
    step=1,
    help=f"Min {MIN_AXES}, max {MAX_AXES}. Type a value directly and press Enter, "
    "or use the +/- steppers.",
)
if new_k != st.session_state.k:
    st.session_state.k = new_k
    if path and st.session_state.last_result and st.session_state.last_result["path"] == path:
        run_analysis(path, st.session_state.k, st.session_state.linkage_method)

linkage_choice = st.radio(
    "Clustering method",
    options=["ward", "average"],
    index=["ward", "average"].index(st.session_state.linkage_method),
    horizontal=True,
    help=(
        "'ward' (default) gives balanced, compact clusters. 'average' tends to "
        "produce one giant cluster plus tiny outlier ones on typical photo "
        "datasets — mainly here for comparison, not recommended for regular use."
    ),
)
if linkage_choice != st.session_state.linkage_method:
    st.session_state.linkage_method = linkage_choice
    if path and st.session_state.last_result and st.session_state.last_result["path"] == path:
        run_analysis(path, st.session_state.k, st.session_state.linkage_method)

if st.button("Analyze", type="primary"):
    if not path:
        st.error("Please enter a folder path.")
    elif not Path(path).exists() or not Path(path).is_dir():
        st.error(f"'{path}' is not a valid folder.")
    else:
        run_analysis(path, st.session_state.k, st.session_state.linkage_method)

result = st.session_state.last_result
if result is not None:
    embedder = get_embedder()

    # Build the full active axis set: auto-detected (minus any the user
    # excluded) + custom, in that order.
    full_axes = [
        a for a in result["base_axes"] if a.label not in st.session_state.excluded_axis_labels
    ]
    for i, label in enumerate(st.session_state.custom_axis_labels):
        full_axes.append(create_custom_axis(embedder, label, i))

    compare_result = st.session_state.compare_result

    # Computed ONCE here, reused by every scoring call below (axis counts,
    # radar, scatter, View images, Copy images, PDF metrics) — see
    # _pooled_score_reference's docstring for why each dataset judging
    # itself against only its OWN score distribution is a real bug when
    # comparing two datasets, not just a cosmetic inconsistency.
    standardize_reference = _pooled_score_reference(result, compare_result, full_axes)

    # Recomputed every time against the FULL current axis set, so a new
    # custom axis can pull images away from whichever axis it now beats.
    other_threshold = st.slider(
        "'Other' strictness",
        min_value=-2.0,
        max_value=1.0,
        value=0.90,
        step=0.1,
        key="other_threshold_slider",
        help=(
            "How confidently an image must match its best axis to count there. "
            "Higher = stricter (more images end up in 'Other'). "
            "Lower = more permissive (fewer 'Other', more force-assigned to the "
            "closest axis even if it's a weak match). 0.0 means 'at least average' "
            "for that axis."
        ),
    )
    axis_counts = get_axis_counts_by_dominance(
        result["embeddings"], full_axes, other_threshold=other_threshold, standardize_reference=standardize_reference
    )

    radar_mode_options = ["Dominance (% of images)", "Normalized similarity"]
    if compare_result is not None:
        # Only meaningful once there's a second feed to compare against —
        # with a single feed it's mathematically redundant with Dominance %
        # (dividing every axis by the same constant doesn't change the
        # radar's shape at all, since the chart auto-scales).
        radar_mode_options.append("Absolute count (images)")

    radar_mode = st.radio(
        "Radar value",
        options=radar_mode_options,
        horizontal=True,
        help=(
            "'Normalized similarity' averages each axis after scaling it to its "
            "own observed range in this dataset — this keeps custom text axes "
            "(e.g. 'sky') comparable to auto-detected axes, correcting for CLIP's "
            "'modality gap' (raw image-text similarity runs lower than "
            "image-image similarity even for a genuinely good match)."
            + (
                " 'Absolute count' plots the raw number of images per axis, "
                "unnormalized — use this when comparing two feeds of very "
                "different sizes, since the percentage-based modes can make a "
                "handful of images in a small feed look as 'big' as hundreds in "
                "a large one."
                if compare_result is not None
                else ""
            )
        ),
    )
    if radar_mode == "Dominance (% of images)":
        radar_values = get_radar_values_by_dominance(
            result["embeddings"],
            full_axes,
            other_threshold=other_threshold,
            standardize_reference=standardize_reference,
        )
        value_label, value_format = "share of dataset", ".1%"
    elif radar_mode == "Normalized similarity":
        radar_values = get_radar_values_normalized(result["embeddings"], full_axes)
        value_label, value_format = "normalized similarity", ".2f"
    else:
        radar_values = {label: float(count) for label, count in axis_counts.items()}
        value_label, value_format = "images", ".0f"

    # "Other" has no real centroid/direction, so it doesn't belong on the
    # radar itself — it's still shown in the axis list below, just not
    # plotted as a spoke.
    chart_values = {label: v for label, v in radar_values.items() if label != OTHER_LABEL}

    radar_datasets = {result["dataset_name"]: chart_values}
    radar_counts = {result["dataset_name"]: axis_counts}

    if compare_result is not None:
        # Scored against the SAME full_axes (primary feed's axes) — this
        # feed never gets its own clustering/labels, so both series are
        # directly comparable on the same radar.
        compare_axis_counts = get_axis_counts_by_dominance(
            compare_result["embeddings"],
            full_axes,
            other_threshold=other_threshold,
            standardize_reference=standardize_reference,
        )
        if radar_mode == "Dominance (% of images)":
            compare_values = get_radar_values_by_dominance(
                compare_result["embeddings"],
                full_axes,
                other_threshold=other_threshold,
                standardize_reference=standardize_reference,
            )
        elif radar_mode == "Normalized similarity":
            compare_values = get_radar_values_normalized(compare_result["embeddings"], full_axes)
        else:
            compare_values = {label: float(count) for label, count in compare_axis_counts.items()}
        compare_chart_values = {
            label: v for label, v in compare_values.items() if label != OTHER_LABEL
        }
        radar_datasets[compare_result["dataset_name"]] = compare_chart_values
        radar_counts[compare_result["dataset_name"]] = compare_axis_counts

    if compare_result is not None:
        stacked_available = radar_mode == "Absolute count (images)"
        display_mode = st.radio(
            "Comparison display",
            options=["Overlay", "Stacked (sum)"],
            horizontal=True,
            disabled=not stacked_available,
            help=(
                "'Stacked' draws the second feed's wedge starting where the first "
                "one ends, per axis, so the total reflects the combined image count "
                "across both feeds — useful for checking if a second feed helps "
                "reach a target total. Only available with 'Absolute count' "
                "selected above — summing percentages or normalized similarities "
                "wouldn't mean anything coherent."
            ),
        )
        if not stacked_available:
            st.caption("Stacked view needs 'Absolute count (images)' selected as the radar value.")
            display_mode = "Overlay"
    else:
        display_mode = "Overlay"

    title = f"Semantic radar — {result['dataset_name']}" + (
        f" vs {compare_result['dataset_name']}" if compare_result is not None else ""
    )

    if display_mode == "Stacked (sum)":
        radar_fig = build_stacked_radar_figure(
            radar_datasets,
            title=title,
            value_label=value_label,
            value_format=value_format,
        )
    else:
        radar_fig = build_radar_figure(
            radar_datasets,
            counts=radar_counts,
            title=title,
            value_label=value_label,
            value_format=value_format,
        )

    st.plotly_chart(
        radar_fig,
        key="radar_chart",
        width="stretch",
    )

    if st.session_state.viewing_axis:
        scope = st.session_state.viewing_axis_scope
        if scope == "comparison" and compare_result is not None:
            # Separated mode's second row — same dialog, just fed the
            # comparison feed's own data instead. No new dialog logic
            # needed for this case at all.
            show_axis_images_dialog(
                st.session_state.viewing_axis,
                compare_result["embeddings"],
                full_axes,
                compare_result["paths"],
                other_threshold,
                standardize_reference=standardize_reference,
            )
        elif scope == "aggregated" and compare_result is not None:
            show_axis_images_dialog(
                st.session_state.viewing_axis,
                result["embeddings"],
                full_axes,
                result["paths"],
                other_threshold,
                extra_source={
                    "embeddings": compare_result["embeddings"],
                    "paths": compare_result["paths"],
                    "dataset_name": compare_result["dataset_name"],
                },
                primary_dataset_name=result["dataset_name"],
                standardize_reference=standardize_reference,
            )
        else:
            show_axis_images_dialog(
                st.session_state.viewing_axis,
                result["embeddings"],
                full_axes,
                result["paths"],
                other_threshold,
                standardize_reference=standardize_reference,
            )

    st.subheader("Axes")
    st.caption("Click 'View images' to browse the images behind any axis.")
    st.caption(
        "'Remove axis' only removes that axis from the radar/scatter/PDF "
        "— it never deletes or modifies any of your image files."
    )
    st.caption(
        "Tip: close any open 'View images' window before using "
        "'Copy images to...' — the two aren't meant to be used at the "
        "same time."
    )

    # Comparison-only controls: only shown/computed when a second feed is
    # loaded — with a single dataset, "Aggregated" is the only meaningful
    # mode anyway, so the toggle would be a no-op distraction.
    compare_axis_counts: dict[str, int] = {}
    aggregation_mode = "Aggregated"
    if compare_result is not None:
        compare_axis_counts = get_axis_counts_by_dominance(
            compare_result["embeddings"],
            full_axes,
            other_threshold=other_threshold,
            standardize_reference=standardize_reference,
        )
        aggregation_mode = st.radio(
            "Datasets",
            options=["Aggregated", "Separated"],
            horizontal=True,
            key="axes_aggregation_mode",
            help=(
                "'Aggregated' combines both datasets into one row per axis "
                "(counts summed; View images/Copy images act on both at "
                "once, tagged 🔵/🟠 by origin). 'Separated' shows each "
                "dataset in its own row, with its own dedicated View/Copy "
                "buttons acting only on that one dataset."
            ),
        )

    # Read the destination value already stored in session_state (set by
    # the text_input rendered at the END of this block, below) —
    # Streamlit widget values persist in session_state across reruns, so
    # this is safe to read here even though the widget itself appears
    # later on the page.
    copy_destination = st.session_state.get("copy_destination_input", "")

    # Sort by TOTAL count (both datasets combined when comparing, so the
    # row order doesn't jump around between Aggregated/Separated) — same
    # descending-by-size convention as before. Custom axes are excluded
    # here — they get their own dedicated rendering under "Custom axes"
    # below (with the same Aggregated/Separated-aware row logic), so they
    # aren't shown twice.
    combined_counts_for_sort = {
        lbl: axis_counts.get(lbl, 0) + compare_axis_counts.get(lbl, 0)
        for lbl in set(axis_counts) | set(compare_axis_counts)
        if lbl not in st.session_state.custom_axis_labels
    }
    sorted_labels = [lbl for lbl, _ in sorted(combined_counts_for_sort.items(), key=lambda item: -item[1])]

    separated = compare_result is not None and aggregation_mode == "Separated"

    # Two SEPARATE, always-present placeholders (not one shared container
    # that swaps its whole content type) — each mode renders into its own
    # placeholder and explicitly empties the OTHER one first. This avoids
    # asking Streamlit to reconcile two very differently-shaped widget
    # trees against each other on a mode switch, which is what caused
    # "ghost" dimmed leftover buttons to briefly appear (worse still
    # while other slow sections like the scatter/similarity chain hadn't
    # finished their own first computation yet).
    aggregated_placeholder = st.empty()
    separated_placeholder = st.empty()

    def _remove_auto_axis(lbl):
        st.session_state.excluded_axis_labels.add(lbl)
        st.rerun()

    if not separated:
        separated_placeholder.empty()
        with aggregated_placeholder.container():
            for label in sorted_labels:
                tag = " _(unclassified — no clear match)_" if label == OTHER_LABEL else ""
                _render_axis_management_row(
                    label,
                    tag,
                    separated=False,
                    axis_counts=axis_counts,
                    compare_axis_counts=compare_axis_counts,
                    result=result,
                    compare_result=compare_result,
                    full_axes=full_axes,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                    copy_destination=copy_destination,
                    on_remove=lambda lbl=label: _remove_auto_axis(lbl),
                    key_suffix="auto",
                    is_removable=label != OTHER_LABEL,
                )
    else:
        aggregated_placeholder.empty()
        with separated_placeholder.container():
            for label in sorted_labels:
                tag = " _(unclassified — no clear match)_" if label == OTHER_LABEL else ""
                _render_axis_management_row(
                    label,
                    tag,
                    separated=True,
                    axis_counts=axis_counts,
                    compare_axis_counts=compare_axis_counts,
                    result=result,
                    compare_result=compare_result,
                    full_axes=full_axes,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                    copy_destination=copy_destination,
                    on_remove=lambda lbl=label: _remove_auto_axis(lbl),
                    key_suffix="auto",
                    is_removable=label != OTHER_LABEL,
                )

    st.text_input(
        "Copy destination folder",
        key="copy_destination_input",
        placeholder=r"C:\Projects\sneaky\dataset_samples\dataset_copy",
        help="Used by every 'Copy images to...' button above — set it "
        "once, then click 'Copy images to...' on as many axes as you "
        "like; they all land in this same folder (created automatically "
        "if it doesn't exist yet). Change this to start filling a "
        "different folder instead.",
    )

    if st.session_state.excluded_axis_labels:
        with st.expander(f"Excluded axes ({len(st.session_state.excluded_axis_labels)})"):
            st.caption(
                "Removed from the active axis set — the radar, scatter, and PDF "
                "report are all recomputed without them. Useful for validating "
                "against only your custom axes, for example."
            )
            for label in sorted(st.session_state.excluded_axis_labels):
                rc1, rc2 = st.columns([4, 1.3])
                rc1.write(f"- {label}")
                if rc2.button("Restore", key=f"restore_{label}"):
                    st.session_state.excluded_axis_labels.discard(label)
                    st.rerun()
            if st.button("Restore all"):
                st.session_state.excluded_axis_labels.clear()
                st.rerun()

    st.subheader("Custom axes")
    new_axis = st.text_input("Add a custom axis", key="new_axis_input", placeholder="e.g. sky")
    if st.button("+ Add axis"):
        label = new_axis.strip()
        label_lower = label.lower()
        # Case-insensitive, and only checked against currently ACTIVE
        # axes — an auto-detected axis you've already removed via "Remove
        # axis" no longer blocks reusing its name here; once you've
        # excluded it, its slot is free to reclaim as a custom axis. A
        # custom axis behaves identically to an auto-detected one for
        # scoring/image attribution either way (same get_dominant_labels
        # competition), so there's no meaningful difference to protect
        # against here beyond avoiding two simultaneously-active axes
        # with the same name.
        active_auto_labels = {
            a.label.lower()
            for a in result["base_axes"]
            if a.label not in st.session_state.excluded_axis_labels
        }
        active_custom_labels = {lbl.lower() for lbl in st.session_state.custom_axis_labels}
        if not label:
            st.warning("Type a word or short phrase first.")
        elif label_lower in active_custom_labels or label_lower in active_auto_labels:
            st.warning(f"'{label}' is already an active axis.")
        else:
            st.session_state.custom_axis_labels.append(label)
            st.rerun()

    if st.session_state.custom_axis_labels:

        def _remove_custom_axis(lbl):
            st.session_state.custom_axis_labels.remove(lbl)
            st.rerun()

        # Same Aggregated/Separated-aware rendering as the main Axes list
        # above (see _render_axis_management_row) — and the same
        # two-placeholder pattern to avoid the "ghost button" reconciliation
        # issue on a mode switch, since this section is now mode-aware too.
        custom_aggregated_placeholder = st.empty()
        custom_separated_placeholder = st.empty()

        if not separated:
            custom_separated_placeholder.empty()
            with custom_aggregated_placeholder.container():
                for label in list(st.session_state.custom_axis_labels):
                    _render_axis_management_row(
                        label,
                        "",
                        separated=False,
                        axis_counts=axis_counts,
                        compare_axis_counts=compare_axis_counts,
                        result=result,
                        compare_result=compare_result,
                        full_axes=full_axes,
                        other_threshold=other_threshold,
                        standardize_reference=standardize_reference,
                        copy_destination=copy_destination,
                        on_remove=lambda lbl=label: _remove_custom_axis(lbl),
                        key_suffix="custom",
                    )
        else:
            custom_aggregated_placeholder.empty()
            with custom_separated_placeholder.container():
                for label in list(st.session_state.custom_axis_labels):
                    _render_axis_management_row(
                        label,
                        "",
                        separated=True,
                        axis_counts=axis_counts,
                        compare_axis_counts=compare_axis_counts,
                        result=result,
                        compare_result=compare_result,
                        full_axes=full_axes,
                        other_threshold=other_threshold,
                        standardize_reference=standardize_reference,
                        copy_destination=copy_destination,
                        on_remove=lambda lbl=label: _remove_custom_axis(lbl),
                        key_suffix="custom",
                    )

    st.subheader("Scatter view")
    st.caption(
        "A complementary view: each point is one image, positioned so that "
        "similar images sit close together — position doesn't mean anything "
        "on its own, only which points cluster together does. Colored by "
        "dominant axis. Click a point to see the full image."
    )
    projection_method = st.radio(
        "Projection",
        options=["UMAP", "t-SNE"],
        horizontal=True,
        help=(
            "Both reduce the high-dimensional embeddings to 2D so similar images "
            "sit close together — position/orientation itself means nothing, only "
            "which points cluster matters. 't-SNE' tends to form tighter, more "
            "separated clusters; 'UMAP' usually runs faster and can preserve more "
            "of the global structure between clusters. Each is cached independently "
            "for this dataset, so switching between them is instant after the first "
            "computation."
        ),
    )

    # The projection + per-image dominant-axis/similarity computation is
    # expensive enough (thousands of images) that redoing it on every
    # unrelated rerun — e.g. closing a dialog anywhere else on the page —
    # made the whole app feel sluggish. Memoize it in session_state, same
    # pattern as the Visual similarity chain section below: only recompute
    # when ITS OWN parameters actually change — including whether a
    # comparison feed is loaded, and which one, since that changes what
    # gets plotted entirely (see below).
    scatter_cache_key = (
        result["path"],
        projection_method,
        other_threshold,
        tuple(axis.label for axis in full_axes),
        compare_result["path"] if compare_result is not None else None,
    )
    if st.session_state.get("scatter_data_cache_key") != scatter_cache_key:
        with st.spinner(
            "Computing projection — this can take a while the first time, "
            "then it's cached for this session."
        ):
            (
                embeddings_for_projection,
                dominant_labels,
                cluster_numbers,
                similarities,
                all_paths,
                dataset_origin,
                dataset_display_names,
            ) = _score_and_pool_for_scatter(result, compare_result, full_axes, other_threshold)

            if compare_result is None:
                # Single dataset — use the disk-cached projection tied to
                # this dataset's own path.
                if projection_method == "t-SNE":
                    coords = get_or_compute_tsne(result["path"], embeddings_for_projection)
                else:
                    coords = get_or_compute_umap(result["path"], embeddings_for_projection)
            else:
                # Two datasets — project them TOGETHER so their points are
                # spatially comparable in the same 2D space. t-SNE has no
                # way to place new points into an already-fit projection
                # (no .transform()), so both methods are computed fresh on
                # the pooled embeddings here — this deliberately does NOT
                # use the disk-cached get_or_compute_* wrappers (those are
                # keyed by the primary dataset's own path, and writing a
                # combined-dataset result under that same key could poison
                # the cache for future single-dataset runs). Session-state
                # memoization above (scatter_cache_key) still avoids
                # recomputing this on every unrelated rerun.
                if projection_method == "t-SNE":
                    coords = compute_tsne_projection(embeddings_for_projection)
                else:
                    coords = compute_umap_projection(embeddings_for_projection)

        st.session_state.scatter_data_cache_key = scatter_cache_key
        st.session_state.scatter_coords = coords
        st.session_state.scatter_dominant_labels = dominant_labels
        st.session_state.scatter_cluster_numbers = cluster_numbers
        st.session_state.scatter_similarities = similarities
        st.session_state.scatter_paths = all_paths
        st.session_state.scatter_dataset_origin = dataset_origin
        st.session_state.scatter_dataset_display_names = dataset_display_names

    coords = st.session_state.scatter_coords
    dominant_labels = st.session_state.scatter_dominant_labels
    cluster_numbers = st.session_state.scatter_cluster_numbers
    similarities = st.session_state.scatter_similarities
    scatter_paths = st.session_state.scatter_paths
    dataset_origin = st.session_state.scatter_dataset_origin
    dataset_display_names = st.session_state.scatter_dataset_display_names

    axis_filter = st.selectbox(
        "Filter by axis",
        options=["All axes"] + sorted(set(dominant_labels)),
        help="Show only the images whose dominant axis matches your selection.",
    )

    if axis_filter == "All axes":
        filtered_indices = list(range(len(scatter_paths)))
    else:
        filtered_indices = [i for i, lbl in enumerate(dominant_labels) if lbl == axis_filter]

    filtered_coords = coords[filtered_indices]
    filtered_paths = [scatter_paths[i] for i in filtered_indices]
    filtered_labels = [dominant_labels[i] for i in filtered_indices]
    filtered_similarities = [similarities[i] for i in filtered_indices]
    filtered_clusters = [cluster_numbers[i] for i in filtered_indices]
    filtered_origin = (
        [dataset_origin[i] for i in filtered_indices] if dataset_origin is not None else None
    )

    scatter_fig = build_scatter_figure(
        filtered_coords,
        filtered_labels,
        filtered_paths,
        filtered_similarities,
        filtered_clusters,
        dataset_origin=filtered_origin,
        dataset_display_names=dataset_display_names,
        title=f"{projection_method} — {result['dataset_name']}",
    )
    scatter_event = st.plotly_chart(
        scatter_fig,
        key="tsne_chart",
        width="stretch",
        on_select="rerun",
        selection_mode="points",
    )

    clicked_points = scatter_event["selection"]["points"] if scatter_event else []
    if clicked_points:
        cd = clicked_points[0]["customdata"]
        clicked_path = cd[1]
        # Ignore the click if it's the same point we just dismissed —
        # Plotly's selection is sticky and keeps reporting it as
        # "clicked" on every rerun until the user picks a different one.
        currently_selected_path = (
            st.session_state.scatter_selected_info["path"]
            if st.session_state.scatter_selected_info
            else None
        )
        if clicked_path != st.session_state.scatter_dismissed_path and (
            clicked_path != currently_selected_path
        ):
            st.session_state.scatter_selected_info = {
                "path": cd[1],
                "filename": cd[0],
                "axis": cd[2],
                "similarity": cd[3],
                "cluster": cd[4],
            }
            st.session_state.scatter_dismissed_path = None

    if st.session_state.scatter_selected_info:
        show_single_image_dialog(st.session_state.scatter_selected_info)

    st.subheader("Visual similarity chain")
    st.caption(
        "Pixel-level visual similarity (perceptual hashing) — deliberately NOT "
        "semantic, unlike the radar and scatter above."
    )

    chain_method = st.radio(
        "Ordering method",
        options=["Greedy chain", "Global order", "Grouped by cluster size"],
        index=2,
        horizontal=True,
        help=(
            "'Greedy chain' starts at the first image and repeatedly jumps to "
            "the closest not-yet-shown image — fast, but can jump abruptly once "
            "a tight cluster of near-duplicates is exhausted. 'Global order' "
            "considers all images at once (optimal leaf ordering) for a smoother "
            "overall sequence — can take noticeably longer the first time on "
            "large datasets, but is cached afterward. 'Grouped by cluster size' "
            "finds near-duplicate groups (e.g. burst-mode shots) and shows the "
            "biggest group first, down to images with no close match at all."
        ),
    )

    group_threshold_bits = DEFAULT_GROUP_THRESHOLD_BITS
    if chain_method == "Grouped by cluster size":
        group_threshold_bits = st.slider(
            "Near-duplicate threshold (bits, out of 64)",
            min_value=1,
            max_value=20,
            value=DEFAULT_GROUP_THRESHOLD_BITS,
            help="Lower = stricter (only near-identical images group together). "
            "Higher = more permissive (looser visual matches count as a group).",
        )

    # This whole section is expensive (O(n²) grouping + base64-encoding up
    # to 200 thumbnails) — memoize its rendered output in session_state so
    # it only recomputes when ITS OWN parameters change, not on every
    # unrelated rerun elsewhere in the app (e.g. opening/closing an image
    # dialog in the Scatter view above forces a full-page rerun, which
    # would otherwise redo all of this for nothing every single time).
    cache_key = (
        result["path"],
        chain_method,
        group_threshold_bits if chain_method == "Grouped by cluster size" else None,
        compare_result["path"] if compare_result is not None else None,
    )

    if st.session_state.get("similarity_chain_cache_key") != cache_key:
        phash_progress = st.progress(0, text="Computing perceptual hashes...")

        def _update_phash_progress(done: int, total: int) -> None:
            phash_progress.progress(
                done / total, text=f"Computing perceptual hashes... {done}/{total}"
            )

        phashes = get_or_compute_phashes(
            result["path"], result["paths"], progress_callback=_update_phash_progress
        )

        # When comparing, pool both datasets' images/hashes together so
        # the SAME chain/order/grouping algorithms below naturally surface
        # cross-dataset near-duplicates too, right alongside the
        # within-dataset ones — not a separate 4th mode, just images from
        # the comparison feed slotting into wherever they visually belong.
        # dataset_name_by_path stays None for the single-dataset case,
        # which keeps every caption exactly as it was before.
        dataset_name_by_path: dict[str, str] | None = None
        if compare_result is not None:
            compare_phashes = get_or_compute_phashes(
                compare_result["path"], compare_result["paths"], progress_callback=_update_phash_progress
            )
            chain_paths = list(result["paths"]) + list(compare_result["paths"])
            chain_hashes = {**phashes, **compare_phashes}
            dataset_name_by_path = {
                **{str(p): result["dataset_name"] for p in result["paths"]},
                **{str(p): compare_result["dataset_name"] for p in compare_result["paths"]},
            }
        else:
            chain_paths = result["paths"]
            chain_hashes = phashes
        phash_progress.empty()

        if chain_method == "Greedy chain":
            raw_chain = build_similarity_chain(
                chain_paths, chain_hashes, start_index=0, max_length=DEFAULT_CHAIN_MAX_LENGTH
            )
            chain = [(p, "start" if d is None else f"Δ {d}") for p, d in raw_chain]
        elif chain_method == "Global order":
            with st.spinner(
                "Computing global visual order — this can take a while the first "
                "time on large datasets, then it's cached for this dataset."
            ):
                if compare_result is None:
                    full_order = get_or_compute_global_order(result["path"], chain_paths, chain_hashes)
                else:
                    # Bypasses the disk-cache wrapper when comparing — same
                    # reasoning as the UMAP/scatter projection: that cache
                    # is keyed by the primary dataset's own path, and
                    # writing a pooled-dataset result under that key risks
                    # confusing a later single-dataset-only run. Session-
                    # state memoization above (cache_key) still avoids
                    # recomputing this on every unrelated rerun.
                    full_order = compute_global_order(chain_paths, chain_hashes)
            raw_chain = full_order[:DEFAULT_CHAIN_MAX_LENGTH]
            chain = [(p, "start" if d is None else f"Δ {d}") for p, d in raw_chain]
        else:
            with st.spinner("Grouping near-duplicate clusters..."):
                chain = build_grouped_chain(
                    chain_paths,
                    chain_hashes,
                    threshold_bits=group_threshold_bits,
                    max_length=DEFAULT_CHAIN_MAX_LENGTH,
                )

        if not chain:
            chain_html = None
        else:
            items_html = []
            for img_path, label in chain:
                b64 = _image_to_base64_thumb(img_path)
                if b64 is None:
                    continue
                extra_line = ""
                if dataset_name_by_path is not None:
                    ds_name = dataset_name_by_path.get(str(img_path), "")
                    # Same blue/amber convention as everywhere else a
                    # second dataset is shown (PDF radar/scatter, the
                    # Axes list) — kept as local literal hex values here
                    # since app.py otherwise has no reason to import
                    # pdf_report's matplotlib-oriented color constants.
                    # Must stay in sync with MPL_ACCENT/MPL_ACCENT_COMPARE
                    # in src/report/pdf_report.py if either ever changes.
                    ds_color = "#5980A6" if ds_name == result["dataset_name"] else "#D97B29"
                    extra_line = (
                        f'<div style="font-size: 10px; color: {ds_color}; margin-top: 1px;">'
                        f"{img_path.name}</div>"
                        f'<div style="font-size: 10px; color: {ds_color};">{ds_name}</div>'
                    )
                items_html.append(
                    f'<div style="flex: 0 0 auto; text-align: center; margin-right: 8px;">'
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="height: {SIMILARITY_CHAIN_THUMB_SIZE}px; border-radius: 4px; '
                    f'display: block;" title="{img_path.name}" />'
                    f'<div style="font-size: 11px; color: #888; margin-top: 2px;">{label}</div>'
                    f"{extra_line}"
                    f"</div>"
                )
            chain_html = (
                '<div style="display: flex; overflow-x: auto; padding: 8px 0;">'
                + "".join(items_html)
                + "</div>"
            )

        st.session_state.similarity_chain_cache_key = cache_key
        st.session_state.similarity_chain_html = chain_html

    if st.session_state.similarity_chain_html is None:
        st.caption("No images could be hashed.")
    else:
        st.markdown(st.session_state.similarity_chain_html, unsafe_allow_html=True)

    st.subheader("Dataset Report")
    st.caption(
        "Nothing here is computed until you click below — this section stays "
        "cheap on every page interaction, and the numbers/charts only exist "
        "inside the generated PDF."
    )

    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner(
            "Building PDF report... if UMAP hasn't been computed for this dataset yet, or "
            "you're comparing two datasets for the first time (wavelet texture analysis reads "
            "every image, cached afterward), this can take a while the first time."
        ):
            # get_or_compute_phashes has its own disk cache — this is a
            # cheap cache-hit in virtually all cases (the Visual similarity
            # chain section above already triggers the real computation
            # the first time), not a recompute.
            report_phashes = get_or_compute_phashes(result["path"], result["paths"])
            duplicate_stats = get_or_compute_duplicate_stats(
                result["path"], result["paths"], report_phashes
            )
            overview = compute_overview_metrics(
                axis_counts=axis_counts,
                total_images=len(result["paths"]),
                duplicate_image_count=duplicate_stats["n_duplicate_images"],
            )

            # Same overview metrics for the comparison feed, when loaded —
            # its own pHash/duplicate-detection pass, cached per-dataset
            # exactly like the primary's above. Used to show the KPI
            # cards/bar chart broken down by dataset instead of only the
            # primary feed's own numbers.
            overview_compare = None
            if compare_result is not None:
                compare_report_phashes = get_or_compute_phashes(
                    compare_result["path"], compare_result["paths"]
                )
                compare_duplicate_stats = get_or_compute_duplicate_stats(
                    compare_result["path"], compare_result["paths"], compare_report_phashes
                )
                overview_compare = compute_overview_metrics(
                    axis_counts=compare_axis_counts,
                    total_images=len(compare_result["paths"]),
                    duplicate_image_count=compare_duplicate_stats["n_duplicate_images"],
                )

            representative_pairs = get_representative_images_by_axis(full_axes)
            # Custom axes have no centroid image of their own — fall back to
            # a random image that actually dominates that axis, so every
            # axis (auto or custom) ends up represented in the report.
            # Candidates are pooled from BOTH datasets when comparing (the
            # representative could come from either one) — previously this
            # only ever considered the primary feed.
            covered_labels = {label for label, _ in representative_pairs}
            for axis in full_axes:
                if axis.label in covered_labels:
                    continue
                candidates = get_ranked_images_for_axis(
                    result["embeddings"],
                    full_axes,
                    result["paths"],
                    axis.label,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                )
                if compare_result is not None:
                    candidates = candidates + get_ranked_images_for_axis(
                        compare_result["embeddings"],
                        full_axes,
                        compare_result["paths"],
                        axis.label,
                        other_threshold=other_threshold,
                        standardize_reference=standardize_reference,
                    )
                if candidates:
                    random_path, _score = random.choice(candidates)
                    representative_pairs.append((axis.label, random_path))

            low_fit_samples = [
                p
                for p, _score in get_ranked_images_for_axis(
                    result["embeddings"],
                    full_axes,
                    result["paths"],
                    OTHER_LABEL,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                )
            ]
            # ALL near-duplicate groups — pooled across both datasets in a
            # SINGLE pass when comparing, so a group can end up entirely
            # from one dataset OR span both, with no separate "Near
            # duplicates" vs "Cross-dataset matches" split; the reader
            # sees which is which from the color-coded captions
            # (steel-blue/amber) instead of two sections to reconcile.
            # Every group is shown in the PDF (not just the biggest),
            # largest first — same "biggest evidence first" convention as
            # build_grouped_chain. Capping to a readable number of
            # thumbnails per group happens inside _flowing_thumbnail_grid.
            duplicate_groups = get_all_duplicate_groups_combined(
                result["paths"],
                report_phashes,
                result["dataset_name"],
                compare_result["paths"] if compare_result is not None else None,
                compare_report_phashes if compare_result is not None else None,
                compare_result["dataset_name"] if compare_result is not None else None,
                threshold_bits=DEFAULT_GROUP_THRESHOLD_BITS,
            )
            duplicate_match_count = sum(len(g) for g in duplicate_groups)
            low_fit_dataset_names = [result["dataset_name"]] * len(low_fit_samples)

            # Combined low-fit samples from BOTH datasets when comparing —
            # previously this (unlike the KPI cards, already fixed) only
            # ever reflected the primary feed. The combined COUNT for the
            # callout/section header is computed inside generate_pdf_report
            # itself, from overview/overview_compare (same as it already
            # does for the KPI cards) — no need to duplicate that here.
            if compare_result is not None:
                compare_low_fit = get_ranked_images_for_axis(
                    compare_result["embeddings"],
                    full_axes,
                    compare_result["paths"],
                    OTHER_LABEL,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                )
                low_fit_samples += [p for p, _score in compare_low_fit]
                low_fit_dataset_names += [compare_result["dataset_name"]] * len(compare_low_fit)

            # For each low-fit image, which near-duplicate group (if any)
            # it belongs to — lets the PDF's "Low semantic fit" sample
            # selection avoid picking 2+ images that are themselves
            # near-duplicates of each other, when a distinct alternative
            # is available (see _sample_thumbnail_row's dedupe_keys).
            duplicate_group_key_by_path: dict[str, int] = {}
            for gi, group in enumerate(duplicate_groups):
                for p, _name in group:
                    duplicate_group_key_by_path[str(p)] = gi
            low_fit_dupe_keys = [duplicate_group_key_by_path.get(str(p)) for p in low_fit_samples]

            # Both radar variants, over the same full_axes (auto + custom).
            radar_dominance_values = get_radar_values_by_dominance(
                result["embeddings"],
                full_axes,
                other_threshold=other_threshold,
                standardize_reference=standardize_reference,
            )
            radar_dominance_values = {
                label: v for label, v in radar_dominance_values.items() if label != OTHER_LABEL
            }
            radar_normalized_values = get_radar_values_normalized(
                result["embeddings"], full_axes
            )
            radar_normalized_values = {
                label: v for label, v in radar_normalized_values.items() if label != OTHER_LABEL
            }

            # Same two variants for the comparison feed, when loaded —
            # scored against the SAME full_axes as the primary feed (it
            # never gets its own clustering), same pattern as the live
            # radar's overlay above.
            radar_dominance_values_compare = None
            radar_normalized_values_compare = None
            if compare_result is not None:
                radar_dominance_values_compare = get_radar_values_by_dominance(
                    compare_result["embeddings"],
                    full_axes,
                    other_threshold=other_threshold,
                    standardize_reference=standardize_reference,
                )
                radar_dominance_values_compare = {
                    label: v
                    for label, v in radar_dominance_values_compare.items()
                    if label != OTHER_LABEL
                }
                radar_normalized_values_compare = get_radar_values_normalized(
                    compare_result["embeddings"], full_axes
                )
                radar_normalized_values_compare = {
                    label: v
                    for label, v in radar_normalized_values_compare.items()
                    if label != OTHER_LABEL
                }

            # UMAP — computed on the fly if not already cached (per your
            # preference); can take a while the first time on a large
            # dataset, but is cached afterward like everywhere else. When a
            # comparison feed is loaded, both feeds are pooled and
            # projected together (same shared logic as the live Scatter
            # view) so the PDF's scatter shows both, shaped by dataset —
            # this deliberately bypasses the disk cache in that case, same
            # reasoning as the live view (keyed by the primary dataset's
            # own path; a pooled result under that key could poison future
            # single-dataset runs).
            (
                umap_embeddings_for_projection,
                umap_dominant_labels,
                _umap_cluster_numbers,
                _umap_similarities,
                _umap_paths,
                umap_dataset_origin,
                umap_dataset_display_names,
            ) = _score_and_pool_for_scatter(result, compare_result, full_axes, other_threshold)
            if compare_result is None:
                umap_coords = get_or_compute_umap(result["path"], umap_embeddings_for_projection)
            else:
                umap_coords = compute_umap_projection(umap_embeddings_for_projection)

            # CLIP-MMD — a single "how similar overall" number between the
            # two feeds, only when a comparison feed is loaded. Reuses the
            # embeddings already computed for both feeds, no new embedding
            # pass needed. The self-split baseline gives a same-dataset
            # "noise floor" to read the cross-feed number against.
            clip_mmd_value = None
            clip_mmd_baseline_value = None
            wavelet_mmd_value = None
            wavelet_mmd_baseline_value = None
            compare_dataset_name = None
            if compare_result is not None:
                clip_mmd_value = compute_clip_mmd(
                    result["embeddings"], compare_result["embeddings"]
                )
                clip_mmd_baseline_value = compute_self_split_mmd(result["embeddings"])
                compare_dataset_name = compare_result["dataset_name"]

                # Wavelet-MMD — same idea, but over texture statistics
                # instead of CLIP's semantic space; complementary, catches
                # differences CLIP wouldn't be sensitive to (sensor noise,
                # compression, rendering engine). Unlike CLIP-MMD, this
                # needs its own pass reading every image the first time (no
                # existing embeddings to reuse) — cached to disk afterward
                # (get_or_compute_wavelet_features), same convention as
                # phash's own disk cache.
                _primary_wavelet_paths, primary_wavelet_features = get_or_compute_wavelet_features(
                    result["path"], result["paths"]
                )
                _compare_wavelet_paths, compare_wavelet_features = get_or_compute_wavelet_features(
                    compare_result["path"], compare_result["paths"]
                )
                if primary_wavelet_features.size and compare_wavelet_features.size:
                    wavelet_mmd_value = compute_wavelet_mmd(
                        primary_wavelet_features, compare_wavelet_features
                    )
                    wavelet_mmd_baseline_value = compute_wavelet_self_split_mmd(primary_wavelet_features)

            st.session_state.pdf_report_bytes = generate_pdf_report(
                dataset_name=result["dataset_name"],
                overview=overview,
                representative_images=representative_pairs,
                low_fit_samples=low_fit_samples,
                low_fit_dupe_keys=low_fit_dupe_keys,
                low_fit_dataset_names=low_fit_dataset_names,
                duplicate_groups=duplicate_groups,
                duplicate_match_count=duplicate_match_count,
                radar_dominance_values=radar_dominance_values,
                radar_normalized_values=radar_normalized_values,
                radar_dominance_values_compare=radar_dominance_values_compare,
                radar_normalized_values_compare=radar_normalized_values_compare,
                umap_coords=umap_coords,
                umap_labels=umap_dominant_labels,
                umap_dataset_origin=umap_dataset_origin,
                umap_dataset_display_names=umap_dataset_display_names,
                clip_mmd=clip_mmd_value,
                clip_mmd_baseline=clip_mmd_baseline_value,
                wavelet_mmd=wavelet_mmd_value,
                wavelet_mmd_baseline=wavelet_mmd_baseline_value,
                compare_dataset_name=compare_dataset_name,
                overview_compare=overview_compare,
            )

    if st.session_state.pdf_report_bytes:
        today_str = datetime.now().strftime("%Y%m%d")
        if compare_result is not None:
            pdf_file_name = (
                f"sneakyReport_{result['dataset_name']}_vs_{compare_result['dataset_name']}_{today_str}.pdf"
            )
        else:
            pdf_file_name = f"sneakyReport_{result['dataset_name']}_{today_str}.pdf"
        st.download_button(
            "Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=pdf_file_name,
            mime="application/pdf",
        )

    if path and Path(path).exists() and Path(path).is_dir():
        with st.expander("Cache management"):
            cache_info = cache.get_cache_info(path)
            if not cache_info:
                st.caption("Nothing cached for this folder yet.")
            else:
                total_size = sum(size for _, size in cache_info)
                st.caption(
                    f"{len(cache_info)} file(s) cached, {total_size / 1_048_576:.1f} MB total."
                )
                file_list_md = "\n".join(
                    f"- `{filename}` — {size / 1024:.0f} KB" for filename, size in cache_info
                )
                st.markdown(file_list_md)

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                if st.button(
                    "Clear clustering & projections",
                    help="Keeps embeddings — clears trees, axes, t-SNE, UMAP, global order.",
                ):
                    cache.invalidate_tree_and_axes(path)
                    st.session_state.last_result = None
                    st.success("Cleared. Re-analyze to recompute.")
            with cc2:
                if st.button(
                    "Clear visual similarity cache",
                    help="Clears perceptual hashes and global order only.",
                ):
                    cache.clear_similarity_cache(path)
                    st.success("Cleared.")
            with cc3:
                if st.button(
                    "Clear everything", help="Full reset for this folder, including embeddings."
                ):
                    cache.clear_all(path)
                    st.session_state.last_result = None
                    st.session_state.compare_result = None
                    st.success("Cleared. Re-analyze to start from scratch.")

