"""
src/viz/scatter.py

Builds the t-SNE scatter plot — the alternative view to the radar. Each
point is one image, positioned by its 2D t-SNE coordinates and colored by
its dominant axis (same "who does this image belong to" rule used
everywhere else: scoring.get_dominant_labels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def build_scatter_figure(
    coords: np.ndarray,
    labels: list[str],
    paths: list[Path],
    similarities: list[float],
    cluster_numbers: list[int],
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
    """
    fig = go.Figure()

    unique_labels = sorted(set(labels))
    coords_by_label: dict[str, list[int]] = {lbl: [] for lbl in unique_labels}
    for i, lbl in enumerate(labels):
        coords_by_label[lbl].append(i)

    MARKER_OPACITY = 0.75
    HOVERTEMPLATE = (
        "<b>%{customdata[0]}</b><br>"
        "Dominant: %{customdata[2]}<br>"
        "Similarity: %{customdata[3]:.2f}<br>"
        "Cluster: #%{customdata[4]}"
        "<extra></extra>"
    )

    for lbl in unique_labels:
        indices = coords_by_label[lbl]
        customdata = [
            [
                paths[i].name,  # 0: display filename
                str(paths[i]),  # 1: full path (for click handling, not shown)
                labels[i],  # 2: dominant axis
                similarities[i],  # 3: similarity to that axis
                cluster_numbers[i],  # 4: cluster display number
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
                hovertemplate=HOVERTEMPLATE,
                marker=dict(size=6, opacity=MARKER_OPACITY),
                # Explicitly keep selected/unselected points looking
                # identical — otherwise Plotly dims unselected points once
                # any point is clicked, and that dimmed state can get
                # visually "stuck" across the app's forced full reruns
                # (needed elsewhere to make the image dialog close
                # reliably). We don't need the highlight-on-select look;
                # the dialog itself already shows which point was clicked.
                selected=dict(marker=dict(opacity=MARKER_OPACITY)),
                unselected=dict(marker=dict(opacity=MARKER_OPACITY)),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend_title_text="Axis",
    )

    return fig
