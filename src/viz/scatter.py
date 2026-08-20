"""
src/viz/scatter.py

Builds the t-SNE/UMAP scatter plot — colored by dominant axis, and (when
comparing two datasets) shaped by which dataset each point came from.

Real data points never get their own individual legend entries — with one
trace per (axis, dataset) combination, that would blow up into a large,
confusing legend. Instead this uses "proxy" legend entries — invisible,
off-chart points, one per axis and one per dataset — grouped into two
clearly titled sections via Plotly's legendgroup mechanism: a color
swatch per axis, and a shape swatch per dataset (only shown at all when
there's more than one dataset to distinguish).
"""

from __future__ import annotations

from pathlib import Path

import plotly.colors
import plotly.graph_objects as go

import numpy as np

# Marker shape assigned per dataset "slot", in a fixed order — currently
# only "primary" and "comparison" are ever used (the app supports
# comparing one feed against one other), but this list has headroom if a
# future version compares more datasets at once. "x-thin" (a line-only X,
# not a bold filled one) reads much more clearly next to a circle at
# small sizes — matches the treatment already used in the PDF's version
# of this same chart.
DATASET_SYMBOLS = ["circle", "x-thin", "diamond", "triangle-up", "square", "star"]

# Per-symbol marker size/line-width tweak — "x-thin" needs to run a
# little bigger and with an explicit stroke width to read as clearly as
# a filled "circle" of the nominal size; every other symbol just uses
# the nominal size with no outline.
_SYMBOL_SIZE_OVERRIDES = {"x-thin": 9}
_DEFAULT_MARKER_SIZE = 6
_SYMBOL_LINE_WIDTH = {"x-thin": 1.5}

# Neutral color for the dataset-shape legend proxies — those entries are
# about SHAPE, not color, so a fixed neutral gray avoids implying any
# color meaning for them.
_LEGEND_PROXY_COLOR = "#666666"


def _marker_kwargs(symbol: str, color: str) -> dict:
    """
    Full marker dict (size/color/line) tuned per symbol, shared by the
    real data traces and the legend proxies so the legend swatch always
    matches what's actually plotted.

    Line-only symbols (like "x-thin") are drawn via marker.line.color,
    NOT marker.color — there's no fill area for Plotly to color. Setting
    both `color` and `line.color` to the same value here means the right
    one takes effect regardless of which symbol needs it, instead of
    silently falling back to whatever gray Plotly defaults line.color to.
    """
    size = _SYMBOL_SIZE_OVERRIDES.get(symbol, _DEFAULT_MARKER_SIZE)
    line_width = _SYMBOL_LINE_WIDTH.get(symbol, 0)
    return {"size": size, "color": color, "line": dict(width=line_width, color=color)}


def _axis_color_map(unique_labels: list[str]) -> dict[str, str]:
    """One consistent color per axis label, shared across every trace for
    that axis regardless of which dataset a given point belongs to."""
    palette = plotly.colors.qualitative.Plotly
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}


def build_scatter_figure(
    coords: np.ndarray,
    labels: list[str],
    paths: list[Path],
    similarities: list[float],
    cluster_numbers: list[int],
    dataset_origin: list[str] | None = None,
    dataset_display_names: dict[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """
    coords: (n_images, 2) t-SNE/UMAP coordinates.
    labels: dominant axis label per image, same length/order as coords.
    paths: image path per point, same length/order.
    similarities: raw cosine similarity of each image to its OWN dominant
        axis's centroid, same length/order.
    cluster_numbers: 1-based display index of each image's dominant axis
        (e.g. 7 for "Cluster #07"), same length/order. 0 for "Other".
    dataset_origin: which dataset each point came from (e.g. "primary" /
        "comparison"), same length/order as coords. None (the default)
        means a single dataset — every point gets the same marker shape
        and no dataset legend is shown.
    dataset_display_names: origin key -> human-readable name (e.g.
        "primary" -> "sample_01"), used as the dataset legend's labels
        and in the hover tooltip. Falls back to the origin key itself for
        any key not present here.
    """
    fig = go.Figure()

    if dataset_origin is None:
        dataset_origin = ["primary"] * len(labels)
    dataset_display_names = dataset_display_names or {}

    unique_labels = sorted(set(labels))
    axis_colors = _axis_color_map(unique_labels)

    # Preserve first-appearance order (not sorted) so "primary" reliably
    # gets the first symbol (circle) rather than depending on string sort.
    unique_origins: list[str] = []
    for origin in dataset_origin:
        if origin not in unique_origins:
            unique_origins.append(origin)
    origin_symbols = {
        origin: DATASET_SYMBOLS[i % len(DATASET_SYMBOLS)] for i, origin in enumerate(unique_origins)
    }
    multi_dataset = len(unique_origins) > 1

    MARKER_OPACITY = 0.75
    if multi_dataset:
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Dataset: %{customdata[5]}<br>"
            "Dominant: %{customdata[2]}<br>"
            "Similarity: %{customdata[3]:.2f}<br>"
            "Cluster: #%{customdata[4]}"
            "<extra></extra>"
        )
    else:
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Dominant: %{customdata[2]}<br>"
            "Similarity: %{customdata[3]:.2f}<br>"
            "Cluster: #%{customdata[4]}"
            "<extra></extra>"
        )

    # Real data traces: one per (dataset, axis) combination that actually
    # has points — color by axis, shape by dataset. Never shown in the
    # legend individually (showlegend=False) — see the proxy traces below.
    for origin in unique_origins:
        origin_name = dataset_display_names.get(origin, origin)
        for lbl in unique_labels:
            indices = [
                i for i in range(len(labels)) if labels[i] == lbl and dataset_origin[i] == origin
            ]
            if not indices:
                continue
            customdata = [
                [
                    paths[i].name,  # 0: display filename
                    str(paths[i]),  # 1: full path (for click handling, not shown)
                    labels[i],  # 2: dominant axis
                    similarities[i],  # 3: similarity to that axis
                    cluster_numbers[i],  # 4: cluster display number
                    origin_name,  # 5: dataset display name (only used when multi_dataset)
                ]
                for i in indices
            ]
            fig.add_trace(
                go.Scatter(
                    x=coords[indices, 0],
                    y=coords[indices, 1],
                    mode="markers",
                    name=lbl,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    marker=dict(
                        opacity=MARKER_OPACITY,
                        symbol=origin_symbols[origin],
                        **_marker_kwargs(origin_symbols[origin], axis_colors[lbl]),
                    ),
                    # Explicitly keep selected/unselected points looking
                    # identical — otherwise Plotly dims unselected points
                    # once any point is clicked, and that dimmed state can
                    # get visually "stuck" across the app's forced full
                    # reruns (needed elsewhere to make the image dialog
                    # close reliably). We don't need the highlight-on-select
                    # look; the dialog itself already shows which point was
                    # clicked.
                    selected=dict(marker=dict(opacity=MARKER_OPACITY)),
                    unselected=dict(marker=dict(opacity=MARKER_OPACITY)),
                    showlegend=False,
                )
            )

    # Proxy legend entries — Axis: a color swatch per axis (always a
    # square, regardless of what shape the real points actually use), as
    # invisible off-chart points. This is what makes the axis legend show
    # color only, independent of the dataset-shape legend below.
    for i, lbl in enumerate(unique_labels):
        kwargs = {"legendgrouptitle_text": "Axis"} if i == 0 else {}
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, symbol="square", color=axis_colors[lbl]),
                name=lbl,
                legendgroup="axis",
                showlegend=True,
                hoverinfo="skip",
                **kwargs,
            )
        )

    # Proxy legend entries — Dataset: a shape swatch per dataset, in a
    # single neutral color (this legend is about shape, not color). Only
    # added at all when there's more than one dataset to distinguish.
    if multi_dataset:
        for i, origin in enumerate(unique_origins):
            origin_name = dataset_display_names.get(origin, origin)
            kwargs = {"legendgrouptitle_text": "Dataset"} if i == 0 else {}
            legend_symbol = origin_symbols[origin]
            legend_marker = _marker_kwargs(legend_symbol, _LEGEND_PROXY_COLOR)
            legend_marker["size"] = legend_marker["size"] + 4  # legend swatches read bigger than points
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(symbol=legend_symbol, **legend_marker),
                    name=origin_name,
                    legendgroup="dataset",
                    showlegend=True,
                    hoverinfo="skip",
                    **kwargs,
                )
            )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # Plotly's default height doesn't leave enough room for a legend
        # with two grouped sections (Axis + Dataset) once there are many
        # axes — it was getting cut off at the bottom. Taller figure +
        # tighter margins reclaims the empty space around the plot area
        # for the legend instead.
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(tracegroupgap=10),
    )

    return fig
