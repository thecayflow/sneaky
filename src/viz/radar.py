"""
src/viz/radar.py

Builds the radar chart from per-axis aggregated scores (from scoring.py).
Uses Plotly so it renders interactively — both standalone (saved as HTML)
and later embedded directly in Streamlit.
"""

from __future__ import annotations

import plotly.graph_objects as go

# The same primary/comparison color pair used everywhere else a second
# dataset is shown (PDF radar and scatter — see MPL_ACCENT/
# MPL_ACCENT_COMPARE in src/report/pdf_report.py). Duplicated here rather
# than imported — this module is Plotly-based and shouldn't depend on the
# PDF module's matplotlib setup — but MUST be kept in sync with those two
# hex values if either ever changes, or the live app and the PDF will
# show different colors for the same comparison.
DATASET_COLORS = ["#5980A6", "#D97B29"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """hex color -> 'rgba(r, g, b, alpha)' string, for a fill that needs
    its own transparency independent of the line/marker opacity (Plotly's
    trace-level `opacity` would otherwise dim those too)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_radar_figure(
    datasets: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]] | None = None,
    title: str | None = None,
    value_label: str = "share of dataset",
    value_format: str = ".1%",
) -> go.Figure:
    """
    Build a radar chart from one or more series of axis->value scores.

    `datasets` maps a series name (e.g. a dataset/feed label) to its
    {axis_label: value} dict — either dominance fractions
    (scoring.get_radar_values_by_dominance) or raw mean similarity
    (scoring.get_radar_values), depending on what the caller wants to
    plot. A single entry draws one shape; multiple entries overlay them —
    this is what the future feed-comparison feature (Phase 2) will use,
    without needing any change to this function.

    `counts` (optional) maps the same series names to {axis_label: n_images}
    — when provided, the hover tooltip shows the image count for that axis
    alongside the plotted value.

    `value_label` / `value_format` control how the value is described and
    formatted in the hover tooltip and radial axis ticks (e.g. "share of
    dataset" / ".1%" for dominance fractions, or "mean similarity" / ".3f"
    for raw cosine similarity) — the two metrics live on different scales
    and mean different things, so the caller picks which is being shown.

    All series are expected to share the same set of axis labels; if they
    don't, the union of all labels is used and missing values are treated
    as 0.
    """
    # Union of axis labels across all series, in first-seen order.
    all_labels: list[str] = []
    for values in datasets.values():
        for label in values:
            if label not in all_labels:
                all_labels.append(label)

    fig = go.Figure()

    all_raw_values: list[float] = []
    for i, (series_name, values) in enumerate(datasets.items()):
        r = [values.get(label, 0.0) for label in all_labels]
        all_raw_values.extend(r)

        # Close the polygon by repeating the first point at the end.
        r_closed = r + [r[0]]
        theta_closed = all_labels + [all_labels[0]]

        series_counts = (counts or {}).get(series_name)
        if series_counts is not None:
            customdata = [series_counts.get(label, 0) for label in all_labels]
            customdata_closed = customdata + [customdata[0]]
            hovertemplate = (
                "<b>%{theta}</b><br>"
                f"{value_label}: %{{r:{value_format}}}<br>"
                "images: %{customdata}"
                "<extra>" + series_name + "</extra>"
            )
        else:
            customdata_closed = None
            hovertemplate = (
                "<b>%{theta}</b><br>"
                f"{value_label}: %{{r:{value_format}}}"
                "<extra>" + series_name + "</extra>"
            )

        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        multi = len(datasets) > 1
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="toself",
                mode="lines+markers",
                marker=dict(size=8, color=color),
                line=dict(color=color),
                # A lighter fill when overlaying two series — otherwise
                # the second polygon's fill can fully obscure the first's,
                # and the overlap itself (the interesting part of a
                # comparison) becomes unreadable. Same reasoning as the
                # PDF's matplotlib version of this same chart.
                fillcolor=_hex_to_rgba(color, 0.3 if multi else 0.45),
                name=series_name,
                customdata=customdata_closed,
                hovertemplate=hovertemplate,
            )
        )

    # Auto-scale the radial axis around the actual data range, with some
    # padding — raw CLIP cosine similarities live in a narrow band (e.g.
    # 0.35-0.55), so a fixed 0-1 range would make every dataset look nearly
    # flat. This is a display choice: it emphasizes RELATIVE differences
    # between axes, not absolute similarity magnitude.
    if all_raw_values:
        data_min, data_max = min(all_raw_values), max(all_raw_values)
        padding = max((data_max - data_min) * 0.15, 0.02)
        radial_range = [max(0.0, data_min - padding), data_max + padding]
    else:
        radial_range = [0, 1]

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=radial_range, tickformat=value_format)
        ),
        showlegend=len(datasets) > 1,
        title=title,
    )

    return fig


def build_stacked_radar_figure(
    datasets: dict[str, dict[str, float]],
    title: str | None = None,
    value_label: str = "images",
    value_format: str = ".0f",
) -> go.Figure:
    """
    Build a STACKED radar — each series' wedge starts where the previous
    one ends, per axis, so the total visually represents the sum across
    series. Uses Barpolar (Plotly's radial bar chart), which supports
    native stacking — a filled Scatterpolar polygon has no concept of
    "starting from" another series' value, so this is a genuinely
    different chart type, not a variant of build_radar_figure.

    Only really makes sense when `datasets` values are additive counts
    (e.g. raw image counts) — summing percentages or normalized
    similarities across series doesn't mean anything coherent. The caller
    is responsible for only offering this when that's the case.
    """
    all_labels: list[str] = []
    for values in datasets.values():
        for label in values:
            if label not in all_labels:
                all_labels.append(label)

    fig = go.Figure()
    for i, (series_name, values) in enumerate(datasets.items()):
        r = [values.get(label, 0.0) for label in all_labels]
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        fig.add_trace(
            go.Barpolar(
                r=r,
                theta=all_labels,
                name=series_name,
                marker=dict(color=color),
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    f"{value_label}: %{{r:{value_format}}}"
                    "<extra>" + series_name + "</extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        polar=dict(radialaxis=dict(visible=True, tickformat=value_format)),
        showlegend=True,
        title=title,
    )

    return fig


if __name__ == "__main__":
    # Quick manual test, chained with the full cached pipeline:
    #   python src\viz\radar.py "E:\dataset_unificado" 8
    import logging
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_PROJECT_ROOT))

    from src.persistence import cache
    from src.pipeline import run_pipeline
    from src.scoring.scoring import get_axis_counts_by_dominance, get_radar_values_by_dominance

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python radar.py <folder_path> [k]")
        sys.exit(1)

    root = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    axes = run_pipeline(root, k=k)
    cached = cache.load_scan_and_embeddings(root)
    if cached is None:
        print("No cached embeddings found — run pipeline.py first.")
        sys.exit(1)
    _, embeddings, _ = cached

    radar_values = get_radar_values_by_dominance(embeddings, axes)
    axis_counts = get_axis_counts_by_dominance(embeddings, axes)

    dataset_name = Path(root).name
    fig = build_radar_figure(
        {dataset_name: radar_values},
        counts={dataset_name: axis_counts},
        title=f"Semantic radar — {dataset_name}",
    )

    output_path = _PROJECT_ROOT / "radar_preview.html"
    fig.write_html(str(output_path))
    print(f"\nRadar saved to: {output_path}")
    print("Open it in your browser to view it.")
