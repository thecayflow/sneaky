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
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import pillow_heif
import streamlit as st
from PIL import Image, ImageOps

# Registers HEIC/HEIF as an opener Pillow understands — must happen before
# any Image.open() call on such a file. Safe to call multiple times.
pillow_heif.register_heif_opener()

from src.axes.custom import create_custom_axis
from src.axes.hierarchical import MIN_AXES
from src.embeddings.clip_embedder import ClipEmbedder
from src.persistence import cache
from src.pipeline import get_embeddings_only, run_pipeline
from src.scoring.scoring import (
    OTHER_LABEL,
    get_axis_counts_by_dominance,
    get_dominant_labels,
    get_radar_values_by_dominance,
    get_radar_values_normalized,
    get_ranked_images_for_axis,
)
from src.viz.radar import build_radar_figure, build_stacked_radar_figure
from src.viz.scatter import build_scatter_figure
from src.viz.tsne_projection import get_or_compute_tsne
from src.similarity.phash import (
    DEFAULT_CHAIN_MAX_LENGTH,
    DEFAULT_GROUP_THRESHOLD_BITS,
    build_grouped_chain,
    build_similarity_chain,
    get_duplicate_sample_paths,
    get_or_compute_duplicate_stats,
    get_or_compute_global_order,
    get_or_compute_phashes,
)
from src.report.metrics import compute_overview_metrics, get_representative_images_by_axis
from src.report.pdf_report import generate_pdf_report
from src.scoring.dataset_similarity import compute_clip_mmd, compute_self_split_mmd
from src.viz.umap_projection import get_or_compute_umap

MAX_AXES = 25  # soft cap for the + button; raising axis count re-runs captioning
THUMB_BATCH_SIZE = 24  # how many more thumbnails "Load more" reveals each click
THUMB_COLUMNS = 4
SIMILARITY_CHAIN_THUMB_SIZE = 140  # px, for the horizontal-scroll chain

st.set_page_config(page_title="sneaky™ Semantic Report", layout="centered")

st.title("sneaky™ Semantic Report")
st.caption(
    "Point this at any local folder of images to see its semantic makeup as a radar chart. "
    "Use the 'View images' button next to any axis to browse the images behind it."
)


@st.cache_resource
def get_embedder() -> ClipEmbedder:
    """Loaded once per Streamlit session and reused for custom axis text embeddings."""
    return ClipEmbedder()


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
            axes = run_pipeline(path, k=k, linkage_method=linkage_method)
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


def _open_axis_dialog(label: str) -> None:
    st.session_state.viewing_axis = label
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


def _copy_axis_images_to(label: str, dest_folder: str, embeddings, axes, paths, other_threshold: float) -> str:
    """
    Copies every image currently assigned to `label` into dest_folder
    (created automatically if needed), skipping any source image already
    copied there earlier this session — so clicking "Copy images to..."
    again for the same axis/destination combo is always safe and never
    duplicates a file.

    Returns the summary message; the caller displays it (kept separate so
    the caller can decide exactly when/how to show it).
    """
    dest_dir = Path(dest_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ranked = get_ranked_images_for_axis(embeddings, axes, paths, label, other_threshold)
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


@st.dialog("Axis images", width="large", dismissible=False)
def show_axis_images_dialog(label: str, embeddings, axes, paths, other_threshold: float) -> None:
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

    ranked = get_ranked_images_for_axis(embeddings, axes, paths, label, other_threshold=other_threshold)
    total = len(ranked)
    shown = min(st.session_state.thumb_shown_count, total)

    with header_col:
        st.subheader(label)
        if label == OTHER_LABEL:
            st.caption(f"{total} images — ordered worst-fit first (clearest outliers)")
        else:
            st.caption(f"{total} images — ordered by similarity to the axis, closest first")

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

    visible = ranked[:shown]
    for row_start in range(0, len(visible), THUMB_COLUMNS):
        row_items = visible[row_start : row_start + THUMB_COLUMNS]
        cols = st.columns(THUMB_COLUMNS)
        for col, (img_path, score) in zip(cols, row_items):
            with col:
                try:
                    with Image.open(img_path) as im:
                        im = ImageOps.exif_transpose(im)
                        st.image(im, width="stretch", caption=f"similarity: {score:.3f}")
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
                    c_paths, c_embeddings, _ = get_embeddings_only(compare_path)
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
        result["embeddings"], full_axes, other_threshold=other_threshold
    )

    compare_result = st.session_state.compare_result

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
            result["embeddings"], full_axes, other_threshold=other_threshold
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
            compare_result["embeddings"], full_axes, other_threshold=other_threshold
        )
        if radar_mode == "Dominance (% of images)":
            compare_values = get_radar_values_by_dominance(
                compare_result["embeddings"], full_axes, other_threshold=other_threshold
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
        show_axis_images_dialog(
            st.session_state.viewing_axis,
            result["embeddings"],
            full_axes,
            result["paths"],
            other_threshold,
        )

    st.subheader("Axes")
    st.caption("Click 'View images' to browse the images behind any axis.")
    st.caption(
        "Tip: close any open 'View images' window before using "
        "'Copy images to...' — the two aren't meant to be used at the "
        "same time."
    )
    # Read the destination value already stored in session_state (set by
    # the text_input rendered at the END of this block, below) —
    # Streamlit widget values persist in session_state across reruns, so
    # this is safe to read here even though the widget itself appears
    # later on the page.
    copy_destination = st.session_state.get("copy_destination_input", "")
    for label, count in sorted(axis_counts.items(), key=lambda item: -item[1]):
        if label == OTHER_LABEL:
            tag = " _(unclassified — no clear match)_"
        elif label in st.session_state.custom_axis_labels:
            tag = " _(custom)_"
        else:
            tag = ""
        is_removable_auto_axis = label != OTHER_LABEL and label not in st.session_state.custom_axis_labels
        if is_removable_auto_axis:
            c1, c2, c3, c4 = st.columns([2.4, 1.3, 1.6, 1])
        else:
            c1, c2, c3 = st.columns([3.2, 1.3, 1.6])
        c1.write(f"- **{label}**{tag} — {count} images")
        c2.button("View images", key=f"view_{label}", on_click=_open_axis_dialog, args=(label,))
        if c3.button("Copy images to...", key=f"copy_{label}"):
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
                )
                st.toast(message, icon="✅")
        if is_removable_auto_axis:
            if c4.button("Remove", key=f"remove_auto_{label}"):
                st.session_state.excluded_axis_labels.add(label)
                st.rerun()

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
        if not label:
            st.warning("Type a word or short phrase first.")
        elif label in st.session_state.custom_axis_labels or any(
            label == a.label for a in result["base_axes"]
        ):
            st.warning(f"'{label}' is already an active axis.")
        else:
            st.session_state.custom_axis_labels.append(label)
            st.rerun()

    if st.session_state.custom_axis_labels:
        for label in list(st.session_state.custom_axis_labels):
            c1, c2, c3 = st.columns([3, 1.3, 1])
            c1.write(f"- {label}")
            c2.button(
                "View images",
                key=f"view_custom_{label}",
                on_click=_open_axis_dialog,
                args=(label,),
            )
            if c3.button("Remove", key=f"remove_{label}"):
                st.session_state.custom_axis_labels.remove(label)
                st.rerun()

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
    # when ITS OWN parameters actually change.
    scatter_cache_key = (
        result["path"],
        projection_method,
        other_threshold,
        tuple(axis.label for axis in full_axes),
    )
    if st.session_state.get("scatter_data_cache_key") != scatter_cache_key:
        with st.spinner(
            "Computing projection — this can take a while the first time, "
            "then it's cached for this dataset."
        ):
            if projection_method == "t-SNE":
                coords = get_or_compute_tsne(result["path"], result["embeddings"])
            else:
                coords = get_or_compute_umap(result["path"], result["embeddings"])
            dominant_labels, score_matrix, _ = get_dominant_labels(
                result["embeddings"], full_axes, other_threshold=other_threshold
            )

        # Cluster display number: 1-based position among the active axes (0
        # for "Other", which has no real axis/centroid of its own).
        # Similarity is each image's raw score against its OWN dominant axis.
        axis_index_by_label = {axis.label: i + 1 for i, axis in enumerate(full_axes)}
        cluster_numbers = []
        similarities = []
        for i, lbl in enumerate(dominant_labels):
            if lbl == OTHER_LABEL:
                cluster_numbers.append(0)
                similarities.append(float(score_matrix[i].max()))
            else:
                axis_idx = axis_index_by_label[lbl]
                cluster_numbers.append(axis_idx)
                similarities.append(float(score_matrix[i, axis_idx - 1]))

        st.session_state.scatter_data_cache_key = scatter_cache_key
        st.session_state.scatter_coords = coords
        st.session_state.scatter_dominant_labels = dominant_labels
        st.session_state.scatter_cluster_numbers = cluster_numbers
        st.session_state.scatter_similarities = similarities

    coords = st.session_state.scatter_coords
    dominant_labels = st.session_state.scatter_dominant_labels
    cluster_numbers = st.session_state.scatter_cluster_numbers
    similarities = st.session_state.scatter_similarities

    axis_filter = st.selectbox(
        "Filter by axis",
        options=["All axes"] + sorted(set(dominant_labels)),
        help="Show only the images whose dominant axis matches your selection.",
    )

    if axis_filter == "All axes":
        filtered_indices = list(range(len(result["paths"])))
    else:
        filtered_indices = [i for i, lbl in enumerate(dominant_labels) if lbl == axis_filter]

    filtered_coords = coords[filtered_indices]
    filtered_paths = [result["paths"][i] for i in filtered_indices]
    filtered_labels = [dominant_labels[i] for i in filtered_indices]
    filtered_similarities = [similarities[i] for i in filtered_indices]
    filtered_clusters = [cluster_numbers[i] for i in filtered_indices]

    scatter_fig = build_scatter_figure(
        filtered_coords,
        filtered_labels,
        filtered_paths,
        filtered_similarities,
        filtered_clusters,
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
        phash_progress.empty()

        if chain_method == "Greedy chain":
            raw_chain = build_similarity_chain(
                result["paths"], phashes, start_index=0, max_length=DEFAULT_CHAIN_MAX_LENGTH
            )
            chain = [(p, "start" if d is None else f"Δ {d}") for p, d in raw_chain]
        elif chain_method == "Global order":
            with st.spinner(
                "Computing global visual order — this can take a while the first "
                "time on large datasets, then it's cached for this dataset."
            ):
                full_order = get_or_compute_global_order(
                    result["path"], result["paths"], phashes
                )
            raw_chain = full_order[:DEFAULT_CHAIN_MAX_LENGTH]
            chain = [(p, "start" if d is None else f"Δ {d}") for p, d in raw_chain]
        else:
            with st.spinner("Grouping near-duplicate clusters..."):
                chain = build_grouped_chain(
                    result["paths"],
                    phashes,
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
                items_html.append(
                    f'<div style="flex: 0 0 auto; text-align: center; margin-right: 8px;">'
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="height: {SIMILARITY_CHAIN_THUMB_SIZE}px; border-radius: 4px; '
                    f'display: block;" title="{img_path.name}" />'
                    f'<div style="font-size: 11px; color: #888; margin-top: 2px;">{label}</div>'
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
            "Building PDF report... if UMAP hasn't been computed for this dataset yet, "
            "this can take a while the first time (cached afterward)."
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
            representative_pairs = get_representative_images_by_axis(full_axes)
            # Custom axes have no centroid image of their own — fall back to
            # a random image that actually dominates that axis, so every
            # axis (auto or custom) ends up represented in the report.
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
                )[:6]
            ]
            duplicate_samples = get_duplicate_sample_paths(
                result["paths"], report_phashes, threshold_bits=DEFAULT_GROUP_THRESHOLD_BITS
            )

            # Both radar variants, over the same full_axes (auto + custom).
            radar_dominance_values = get_radar_values_by_dominance(
                result["embeddings"], full_axes, other_threshold=other_threshold
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

            # UMAP — computed on the fly if not already cached (per your
            # preference); can take a while the first time on a large
            # dataset, but is cached afterward like everywhere else.
            umap_coords = get_or_compute_umap(result["path"], result["embeddings"])
            umap_dominant_labels, _, _ = get_dominant_labels(
                result["embeddings"], full_axes, other_threshold=other_threshold
            )

            # CLIP-MMD — a single "how similar overall" number between the
            # two feeds, only when a comparison feed is loaded. Reuses the
            # embeddings already computed for both feeds, no new embedding
            # pass needed. The self-split baseline gives a same-dataset
            # "noise floor" to read the cross-feed number against.
            clip_mmd_value = None
            clip_mmd_baseline_value = None
            compare_dataset_name = None
            if compare_result is not None:
                clip_mmd_value = compute_clip_mmd(
                    result["embeddings"], compare_result["embeddings"]
                )
                clip_mmd_baseline_value = compute_self_split_mmd(result["embeddings"])
                compare_dataset_name = compare_result["dataset_name"]

            st.session_state.pdf_report_bytes = generate_pdf_report(
                dataset_name=result["dataset_name"],
                overview=overview,
                representative_images=representative_pairs,
                low_fit_samples=low_fit_samples,
                duplicate_samples=duplicate_samples,
                radar_dominance_values=radar_dominance_values,
                radar_normalized_values=radar_normalized_values,
                umap_coords=umap_coords,
                umap_labels=umap_dominant_labels,
                clip_mmd=clip_mmd_value,
                clip_mmd_baseline=clip_mmd_baseline_value,
                compare_dataset_name=compare_dataset_name,
            )

    if st.session_state.pdf_report_bytes:
        st.download_button(
            "Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=f"dataset_report_{result['dataset_name']}.pdf",
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

