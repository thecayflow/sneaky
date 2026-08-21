"""
src/report/pdf_report.py

Renders the Dataset Report as a designed PDF — same content as the
on-screen dashboard (src/report/metrics.py), laid out with reportlab and
matplotlib-rendered chart images for a clean, print-quality result.

Visual language: "Industry" — steel-blue on a light technical ground,
Barlow Condensed headings over Barlow body text, a modular grid, and
figures/cards framed as blueprint objects (square corners, hairline
borders, "+" registration marks at the corners). Photographs stay in
natural color (unlike the rest of the system's imagery) because this
report's whole job is letting a human visually verify real dataset
content — a duotone wash would work against that.

Public API is unchanged: `generate_pdf_report(...)` takes the same
arguments and returns the same `bytes` as before, so nothing calling this
module needs to change.

matplotlib is imported lazily (inside each function that uses it, not at
module level) — it's the heavier of this module's two big dependencies.
reportlab stays at module level: its own import cost is small, and its
color/font constants below are woven through nearly every function, so
deferring it would add much more restructuring risk for little benefit.
"""

from __future__ import annotations

import colorsys
import io
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

from src.report.metrics import DatasetOverviewMetrics

# ─── Industry design-system tokens ──────────────────────────────────────
# Mirrors theme.json / styles.css from the Industry design system: a mono
# steel-blue scheme on a light ground, with 100–900 tonal ramps.
COLOR_BG = colors.HexColor("#F2F2F3")
COLOR_TEXT = colors.HexColor("#1D1F20")
COLOR_NEUTRAL_300 = colors.HexColor("#D4D4D7")  # hairline borders
COLOR_NEUTRAL_600 = colors.HexColor("#7A7A7D")  # secondary text
COLOR_NEUTRAL_700 = colors.HexColor("#5D5D60")  # tertiary text on light fills
COLOR_ACCENT = colors.HexColor("#5980A6")
COLOR_ACCENT_100 = colors.HexColor("#EEF6FF")
COLOR_ACCENT_200 = colors.HexColor("#D6EBFF")
COLOR_ACCENT_700 = colors.HexColor("#416180")  # accent text on light fills
COLOR_ACCENT_800 = colors.HexColor("#2C455D")
COLOR_ACCENT_900 = colors.HexColor("#1D2D3D")
COLOR_REG_MARK = colors.HexColor("#5D5D60")  # corner registration marks

# matplotlib (hex) equivalents of the same tokens, for chart code below.
MPL_BG = "#F2F2F3"
MPL_TEXT = "#1D1F20"
MPL_NEUTRAL_300 = "#D4D4D7"
MPL_NEUTRAL_600 = "#7A7A7D"
MPL_ACCENT = "#5980A6"
MPL_ACCENT_200 = "#D6EBFF"
MPL_ACCENT_800 = "#2C455D"
# Second accent, used only where two datasets are plotted together (radar/
# scatter comparison pages) — needs to read clearly against the steel-blue
# MPL_ACCENT family above, not blend into it.
MPL_ACCENT_COMPARE = "#D97B29"
MPL_ACCENT_COMPARE_800 = "#8A4E15"

PAGE_MARGIN = 0.65 * inch
PORTRAIT_SIZE = letter
LANDSCAPE_SIZE = landscape(letter)

# Modular spacing scale (Industry's --space-* tokens, px→pt at 0.75).
SP_1, SP_2, SP_3, SP_4, SP_6, SP_8 = 2.5, 5, 7.5, 10, 15, 20

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _register_fonts() -> dict[str, str]:
    """Registers Barlow / Barlow Condensed if their .ttf files are present
    in src/report/fonts/ (download from Google Fonts: 'Barlow' and 'Barlow
    Condensed', weights 400/500/700 and 400/600). Falls back to Helvetica
    per-role so the report always renders even without the font files —
    this can never break the build."""
    names = {
        "cond_semibold": ("BarlowCondensed-SemiBold.ttf", "Helvetica-Bold"),
        "cond_regular": ("BarlowCondensed-Regular.ttf", "Helvetica"),
        "body_regular": ("Barlow-Regular.ttf", "Helvetica"),
        "body_medium": ("Barlow-Medium.ttf", "Helvetica-Bold"),
        "body_bold": ("Barlow-Bold.ttf", "Helvetica-Bold"),
    }
    resolved = {}
    for key, (filename, fallback) in names.items():
        path = _FONTS_DIR / filename
        font_name = path.stem
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(path)))
                resolved[key] = font_name
                continue
            except Exception:  # noqa: BLE001
                pass
        resolved[key] = fallback
    return resolved


FONT = _register_fonts()


def _configure_matplotlib_fonts() -> dict[str, str | None]:
    """
    Registers the same Barlow files with matplotlib and returns their
    resolved family names — mirroring FONT (the reportlab dict above) —
    so chart text uses the IDENTICAL typeface as the rest of the report.

    The previous approach only set rcParams['font.sans-serif'] to a name
    list and hoped matplotlib's family-name matching would pick the right
    weight for fontweight="bold" — that mapping is fragile (regular/medium
    /bold are separate font files with no guaranteed name-based link
    between them). This resolves each weight to its own registered font
    explicitly instead, same pattern as `_register_fonts()` for reportlab.
    Any role whose file isn't present falls back to None, meaning "let
    matplotlib use its own default weight" (see `_mpl_font` below).
    """
    resolved: dict[str, str | None] = {
        "regular": None,
        "medium": None,
        "bold": None,
        "cond_semibold": None,
    }
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        name_map = {
            "regular": "Barlow-Regular.ttf",
            "medium": "Barlow-Medium.ttf",
            "bold": "Barlow-Bold.ttf",
            "cond_semibold": "BarlowCondensed-SemiBold.ttf",
        }
        for key, filename in name_map.items():
            path = _FONTS_DIR / filename
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                resolved[key] = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        base = resolved["regular"] or "Helvetica"
        plt.rcParams["font.sans-serif"] = [base, "Helvetica", "Arial", "DejaVu Sans"]
    except Exception:  # noqa: BLE001
        pass
    return resolved


# Lazy singleton — was previously computed eagerly at module import time
# (`MPL_FONT = _configure_matplotlib_fonts()`), which forced matplotlib to
# be imported just by importing this module. Now computed on first actual
# use, via _get_mpl_font_map() below, same pattern as `_NLP` in labeling.py.
_MPL_FONT_CACHE: dict[str, str | None] | None = None


def _get_mpl_font_map() -> dict[str, str | None]:
    global _MPL_FONT_CACHE
    if _MPL_FONT_CACHE is None:
        _MPL_FONT_CACHE = _configure_matplotlib_fonts()
    return _MPL_FONT_CACHE


def _mpl_font(role: str) -> dict:
    """kwargs to spread into any matplotlib text call (title/label/legend/
    annotate/...) for the given role ('regular' | 'medium' | 'bold' |
    'cond_semibold') — resolves to the real registered Barlow font when its
    file is present, otherwise falls back to matplotlib's own bold/regular
    weight so text is never left unstyled."""
    name = _get_mpl_font_map().get(role)
    if name:
        return {"fontfamily": name}
    return {"fontweight": "bold"} if role in ("bold", "medium", "cond_semibold") else {}


def _categorical_palette(n: int) -> list[tuple[float, float, float]]:
    """`n` mutually distinguishable colors spanning the FULL hue spectrum.
    An earlier version constrained hue to a narrow band around the brand's
    steel-blue (~210° ±7°) for visual consistency, but with more than a
    handful of categories that made same-family colors practically
    impossible to tell apart against the legend — defeating the point of
    a legend at all. Readability wins over brand-color purity here.
    Hues are spread using the golden-ratio increment (0.618...), a
    standard technique for well-separated categorical colors that avoids
    clustering even as `n` grows."""
    golden_ratio_conjugate = 0.6180339887
    start_hue = 0.02  # avoid starting exactly on pure red
    out = []
    for i in range(max(n, 1)):
        hue = (start_hue + i * golden_ratio_conjugate) % 1.0
        light = 0.42 + 0.10 * ((i * 0.37) % 1.0)
        sat = 0.55 + 0.25 * ((i * 0.61) % 1.0)
        out.append(colorsys.hls_to_rgb(hue, light, sat))
    return out


class _Blueprint(Flowable):
    """Wraps a flowable in the Industry system's frame: a hairline square
    border with a small '+' registration mark at each corner. This is the
    one visual motif every figure and card in the system shares."""

    def __init__(self, inner, pad=6, margin=7):
        Flowable.__init__(self)
        self.inner = inner
        self.pad = pad
        self.margin = margin

    def wrap(self, availWidth, availHeight):
        inner_avail_w = max(availWidth - 2 * (self.pad + self.margin), 1)
        inner_avail_h = max(availHeight - 2 * (self.pad + self.margin), 1)
        iw, ih = self.inner.wrap(inner_avail_w, inner_avail_h)
        self.inner_w, self.inner_h = iw, ih
        self.width = iw + 2 * self.pad + 2 * self.margin
        self.height = ih + 2 * self.pad + 2 * self.margin
        return self.width, self.height

    def draw(self):
        c = self.canv
        m, p = self.margin, self.pad
        box_w = self.inner_w + 2 * p
        box_h = self.inner_h + 2 * p
        c.saveState()
        c.setStrokeColor(COLOR_NEUTRAL_300)
        c.setLineWidth(0.6)
        c.rect(m, m, box_w, box_h, stroke=1, fill=0)
        c.setStrokeColor(COLOR_REG_MARK)
        c.setLineWidth(0.7)
        t = 3.6
        for cx, cy in ((m, m), (m + box_w, m), (m, m + box_h), (m + box_w, m + box_h)):
            c.line(cx - t, cy, cx + t, cy)
            c.line(cx, cy - t, cx, cy + t)
        c.restoreState()
        self.inner.drawOn(c, m + p, m + p)


def _paint_page(canv, doc):
    """Fills the page with the system's ground color and adds a hairline
    footer rule + page number — a technical-sheet touch that also makes
    every page (portrait or landscape) unmistakably part of one document."""
    w, h = canv._pagesize
    canv.saveState()
    canv.setFillColor(COLOR_BG)
    canv.rect(0, 0, w, h, fill=1, stroke=0)
    canv.setStrokeColor(COLOR_NEUTRAL_300)
    canv.setLineWidth(0.6)
    canv.line(PAGE_MARGIN, PAGE_MARGIN * 0.62, w - PAGE_MARGIN, PAGE_MARGIN * 0.62)
    canv.setFont(FONT["body_regular"], 7.5)
    canv.setFillColor(COLOR_NEUTRAL_600)
    canv.drawString(PAGE_MARGIN, PAGE_MARGIN * 0.4, "SEMANTIC REPORT BY SNEAKY\u2122")
    canv.drawRightString(w - PAGE_MARGIN, PAGE_MARGIN * 0.4, str(canv.getPageNumber()))
    canv.restoreState()


def _rl_image_for_width(buf: io.BytesIO, target_width: float) -> RLImage:
    """RLImage sized to `target_width`, height computed from the PNG's real
    aspect ratio — avoids stretching/wasted space from guessed height ratios.
    Used for EVERY image in this report (photos and charts alike) so nothing
    is ever distorted."""
    with Image.open(buf) as im:
        w, h = im.size
    buf.seek(0)
    return RLImage(buf, width=target_width, height=target_width * h / w)


def _make_styles() -> dict[str, ParagraphStyle]:
    return {
        "kicker": ParagraphStyle(
            "kicker",
            fontName=FONT["body_medium"],
            fontSize=9,
            textColor=COLOR_ACCENT_700,
            leading=11,
        ),
        "title": ParagraphStyle(
            "title", fontName=FONT["cond_semibold"], fontSize=30, textColor=COLOR_TEXT, leading=32
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=FONT["body_regular"], fontSize=10.5, textColor=COLOR_NEUTRAL_600, leading=15
        ),
        "section": ParagraphStyle(
            "section",
            fontName=FONT["cond_semibold"],
            fontSize=13.5,
            textColor=COLOR_ACCENT_900,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "card_value": ParagraphStyle(
            "card_value",
            fontName=FONT["cond_semibold"],
            fontSize=21,
            textColor=COLOR_ACCENT_700,
            leading=24,
            alignment=1,
        ),
        "card_label": ParagraphStyle(
            "card_label",
            fontName=FONT["body_regular"],
            fontSize=8.5,
            textColor=COLOR_NEUTRAL_600,
            leading=11,
            alignment=1,
        ),
        "caption": ParagraphStyle(
            "caption", fontName=FONT["body_regular"], fontSize=7.5, textColor=COLOR_NEUTRAL_600, alignment=1
        ),
        "body": ParagraphStyle(
            "body", fontName=FONT["body_regular"], fontSize=9.5, textColor=COLOR_NEUTRAL_700, leading=13.5
        ),
        "callout_body": ParagraphStyle(
            "callout_body", fontName=FONT["body_regular"], fontSize=9.5, textColor=COLOR_TEXT, leading=14
        ),
    }


def _section_header(text: str, styles: dict, width: float) -> Table:
    """Section title with a small accent tick to the left and a hairline
    rule below — a modular-grid marker rather than a plain heading."""
    tick = Table([[""]], colWidths=[3], rowHeights=[14])
    tick.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), COLOR_ACCENT)]))
    head = Table(
        [[tick, Paragraph(text.upper(), styles["section"])]],
        colWidths=[10, width - 10],
    )
    head.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.6, COLOR_NEUTRAL_300),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return head


def _metric_card(
    value: str,
    label: str,
    styles: dict,
    breakdown: list[dict] | None = None,
    card_width: float = 1.5 * inch,
) -> _Blueprint:
    """
    One KPI 'card' — big number, small label underneath, framed as a
    blueprint object (transparent, hairline border, corner marks) rather
    than a filled gray box.

    `breakdown`, if given, is a list of {"value": str, "color": hex} (one
    entry per dataset being compared) — rendered as a row of smaller
    colored numbers beneath the label, e.g. a big "480"/"TOTAL IMAGES"
    with "400" (steel-blue) and "280" (amber) underneath it. None (the
    default) renders exactly as before — a single big value and label,
    no breakdown row — so every existing single-dataset call site is
    completely unaffected.

    card_width: the card's own INNER content width (before pad/margin are
    added) — defaults to the original fixed 1.5 inch, but the KPI row
    passes an explicitly computed width so the row of cards' own outer
    edges land exactly on portrait_w (matching Axis sizes/the callout/
    the donuts), instead of leaving a gap on the right where these
    fixed-width cards used to fall short of a wider shared cell.
    """
    n_cols = len(breakdown) if breakdown else 1
    col_width = card_width / n_cols

    rows = [
        [Paragraph(value, styles["card_value"])] + [""] * (n_cols - 1),
        [Paragraph(label.upper(), styles["card_label"])] + [""] * (n_cols - 1),
    ]
    style_commands = [
        ("SPAN", (0, 0), (n_cols - 1, 0)),
        ("SPAN", (0, 1), (n_cols - 1, 1)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8 if not breakdown else 2),
    ]

    if breakdown:
        breakdown_row = []
        for item in breakdown:
            item_style = ParagraphStyle(
                f"card_breakdown_{id(item)}",
                parent=styles["card_label"],
                fontSize=9,
                textColor=colors.HexColor(item["color"]),
            )
            breakdown_row.append(Paragraph(item["value"], item_style))
        rows.append(breakdown_row)
        style_commands += [
            ("TOPPADDING", (0, 2), (-1, 2), 0),
            ("BOTTOMPADDING", (0, 2), (-1, 2), 8),
        ]

    inner = Table(rows, colWidths=[col_width] * n_cols)
    inner.setStyle(TableStyle(style_commands))
    return _Blueprint(inner, pad=2, margin=7)


def _render_donut(
    labels: list[str], values: list[float], chart_colors: list[str], title: str
) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")  # headless — no GUI backend needed/available
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.6, 2.15), dpi=200)
    wedges, _ = ax.pie(
        values,
        colors=chart_colors,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor=MPL_BG, linewidth=2),
    )
    ax.legend(
        wedges,
        [f"{lbl} ({val:.1f}%)" for lbl, val in zip(labels, values)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        frameon=False,
        fontsize=7.5,
        labelcolor=MPL_NEUTRAL_600,
        handlelength=1.2,
        handletextpad=0.6,
    )
    ax.set_title(title, fontsize=10, color=MPL_TEXT, pad=6, **_mpl_font("bold"))
    fig.tight_layout()
    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    # No bbox_inches="tight" here deliberately — it crops to actual content,
    # and two donuts with different-length legend text (e.g. "Unclassified"
    # vs "Near-duplicate") would then crop to slightly different aspect
    # ratios, making them render at different sizes even with identical
    # figsize. A fixed canvas guarantees both donuts come out the same size.
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_donut_with_breakdown(
    title: str,
    combined_labels: list[str],
    combined_values: list[float],
    chart_colors: list[str],
    per_dataset: list[dict],
) -> io.BytesIO:
    """
    A big combined donut on top (same as _render_donut), with one smaller
    donut per dataset stacked below it — same two-slice composition
    (e.g. Classified/Unclassified), just computed for that ONE dataset —
    all rendered into a single image so they fit inside one blueprint
    frame together.

    Colors are the SAME steel-blue/gray scheme across every donut here
    (classified vs unclassified always means the same thing) — the
    dataset accent colors (blue/amber) are deliberately NOT reused for
    the mini donuts' own slices, since that would clash with what color
    already means in this chart. Only the dataset NAME below each mini
    donut identifies which is which.

    per_dataset: list of {"name": str, "values": [v1, v2]} — same slice
    order/meaning as combined_values, one entry per dataset.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_mini = len(per_dataset)
    fig = plt.figure(figsize=(2.6, 2.85), dpi=200)
    gs = fig.add_gridspec(2, n_mini, height_ratios=[1.7, 1.15], hspace=0.75)

    ax_big = fig.add_subplot(gs[0, :])
    wedges, _ = ax_big.pie(
        combined_values,
        colors=chart_colors,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor=MPL_BG, linewidth=2),
    )
    ax_big.legend(
        wedges,
        [f"{lbl} ({val:.1f}%)" for lbl, val in zip(combined_labels, combined_values)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        frameon=False,
        fontsize=7.5,
        labelcolor=MPL_NEUTRAL_600,
        handlelength=1.2,
        handletextpad=0.6,
    )
    ax_big.set_title(title, fontsize=10, color=MPL_TEXT, pad=6, **_mpl_font("bold"))

    for i, ds in enumerate(per_dataset):
        ax_mini = fig.add_subplot(gs[1, i])
        ax_mini.pie(
            ds["values"],
            colors=chart_colors,
            startangle=90,
            wedgeprops=dict(width=0.42, edgecolor=MPL_BG, linewidth=1.5),
        )
        ax_mini.set_title(ds["name"], fontsize=7.5, color=MPL_NEUTRAL_600, pad=3, **_mpl_font("regular"))

    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_bar_chart(
    axis_sizes: dict[str, int], axis_sizes_breakdown: dict[str, list[dict]] | None = None
) -> io.BytesIO:
    """
    axis_sizes_breakdown, if given: label -> list of {"value": int,
    "color": hex, "name": str} (one entry per dataset, in the same order
    for every label) — each bar becomes a horizontal STACKED segment (one
    color per dataset) instead of a single solid-color bar, with a small
    legend for which color is which dataset. None (the default) renders
    exactly as before — single solid-color bars, no legend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    items = sorted(axis_sizes.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    height = max(1.6, 0.20 * len(items) + 0.45)
    fig, ax = plt.subplots(figsize=(6.8, height), dpi=200)

    legend_handles = None
    if axis_sizes_breakdown:
        n_datasets = max((len(axis_sizes_breakdown.get(lbl, [])) for lbl in labels), default=0)
        left = [0.0] * len(labels)
        dataset_names_seen: list[tuple[str, str]] = []  # (name, color), for the legend
        for ds_idx in range(n_datasets):
            seg_values, seg_color, seg_name = [], MPL_ACCENT, None
            for lbl in labels:
                parts = axis_sizes_breakdown.get(lbl, [])
                if ds_idx < len(parts):
                    seg_values.append(parts[ds_idx]["value"])
                    seg_color = parts[ds_idx]["color"]
                    seg_name = parts[ds_idx]["name"]
                else:
                    seg_values.append(0)
            ax.barh(labels, seg_values, left=left, color=seg_color, height=0.62)
            left = [l + v for l, v in zip(left, seg_values)]
            if seg_name is not None:
                dataset_names_seen.append((seg_name, seg_color))
        legend_handles = [Patch(facecolor=c, label=n) for n, c in dataset_names_seen]
    else:
        ax.barh(labels, values, color=MPL_ACCENT, height=0.62)

    max_val = max(values) if values else 1
    for i, value in enumerate(values):
        ax.text(
            value + max_val * 0.015, i, f"{value:,}",
            va="center", fontsize=7.5, color=MPL_NEUTRAL_600, **_mpl_font("regular"),
        )
    ax.set_xlim(0, max_val * 1.12)
    ax.set_xlabel("images", fontsize=8.5, color=MPL_NEUTRAL_600)
    ax.set_title("Axis sizes", fontsize=10, color=MPL_TEXT, loc="left", **_mpl_font("bold"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, colors=MPL_NEUTRAL_600)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MPL_NEUTRAL_300)
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="lower right", fontsize=7.5, frameon=False,
            labelcolor=MPL_NEUTRAL_600,
        )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _squarify_layout(
    values: list[float], x: float, y: float, w: float, h: float
) -> list[tuple[float, float, float, float]]:
    """
    Classic "squarified treemap" layout algorithm (Bruls, Huizing, van Wijk
    1999) — implemented directly instead of depending on the `squarify`
    package, so no extra dependency is needed. `values` must already be
    scaled so that sum(values) == w * h. Returns (x, y, dx, dy) rectangles
    in the same order as the input values.
    """

    def worst(row: list[float], length: float) -> float:
        if not row:
            return float("inf")
        s = sum(row)
        row_max, row_min = max(row), min(row)
        return max((length * length * row_max) / (s * s), (s * s) / (length * length * row_min))

    def layoutrow(
        row: list[float], x: float, y: float, w: float, h: float, horizontal: bool
    ) -> list[tuple[float, float, float, float]]:
        s = sum(row)
        rects = []
        if horizontal:
            rh = s / w if w > 0 else 0
            rx = x
            for v in row:
                rw = v / rh if rh > 0 else 0
                rects.append((rx, y, rw, rh))
                rx += rw
        else:
            rw = s / h if h > 0 else 0
            ry = y
            for v in row:
                rh = v / rw if rw > 0 else 0
                rects.append((x, ry, rw, rh))
                ry += rh
        return rects

    remaining = list(values)
    rects: list[tuple[float, float, float, float]] = []
    row: list[float] = []
    cx, cy, cw, ch = x, y, w, h
    horizontal = cw >= ch

    while remaining:
        horizontal = cw >= ch
        length = cw if horizontal else ch
        c = remaining[0]
        if worst(row + [c], length) <= worst(row, length):
            row.append(c)
            remaining.pop(0)
        else:
            rects.extend(layoutrow(row, cx, cy, cw, ch, horizontal))
            s = sum(row)
            if horizontal:
                rh = s / cw if cw > 0 else 0
                cy += rh
                ch -= rh
            else:
                rw = s / ch if ch > 0 else 0
                cx += rw
                cw -= rw
            row = []
    if row:
        rects.extend(layoutrow(row, cx, cy, cw, ch, horizontal))

    return rects


def _lerp_hex(a: str, b: str, t: float) -> tuple[float, float, float]:
    ar, ag, ab = (int(a[i : i + 2], 16) / 255 for i in (1, 3, 5))
    br, bg, bb = (int(b[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def _render_treemap(
    axis_sizes: dict[str, int], axis_sizes_breakdown: dict[str, list[dict]] | None = None
) -> io.BytesIO:
    """
    "Cluster imbalance" as a treemap — visually distinct from the plain
    "Axis sizes" bar chart earlier in the report. Rectangles too small to
    hold a readable label are left blank rather than overflowing text
    onto neighboring cells. Its own landscape page in the report (see
    generate_pdf_report) — more room than portrait for each sub-box's
    own label when comparing two datasets.

    axis_sizes_breakdown, if given: label -> list of {"value": int,
    "color": hex, "name": str} (one entry per dataset, matching
    axis_sizes' totals) — each axis's rectangle is split into sub-boxes,
    one per dataset, colored by dataset (with a small legend) instead of
    shaded by relative size. The axis name sits near the TOP of the full
    rectangle (not centered) specifically so it doesn't visually compete
    with each sub-box's own percentage, shown centered within that
    sub-box when there's room — otherwise (with a dominant dataset) the
    smaller sub-box was reading as unlabeled, just a bare color with
    nothing of its own. None (the default) renders exactly as before —
    size-shaded rectangles, no split, no legend, label centered as one
    block.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    items = sorted(axis_sizes.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    W, H = 900.0, 380.0
    total = sum(values) if values else 1
    scaled = [v / total * (W * H) for v in values]
    rects = _squarify_layout(scaled, 0, 0, W, H)

    fig, ax = plt.subplots(figsize=(11.5, 4.9), dpi=200)
    max_val = max(values) if values else 1
    legend_handles = None

    for (rx, ry, rw, rh), label, value in zip(rects, labels, values):
        pct = value / total * 100
        breakdown = axis_sizes_breakdown.get(label) if axis_sizes_breakdown else None

        if breakdown:
            # Split along whichever dimension is larger, so each
            # sub-box stays reasonably proportioned regardless of the
            # parent rectangle's own shape (squarify doesn't guarantee
            # perfect squares, just near-square).
            part_total = sum(p["value"] for p in breakdown) or 1
            sub_boxes = []  # (cx, cy, sub_w, sub_h, sub_pct) per dataset, for labeling below
            if rw >= rh:
                cx = rx
                for p in breakdown:
                    sub_w = rw * (p["value"] / part_total)
                    # A thin sliver's white edge (2pt on each side) can eat
                    # a large share of its own already-small area — drop
                    # the edge there so the true proportion isn't visually
                    # eroded further. This doesn't make the sliver BIGGER
                    # than its real share (that would misrepresent the
                    # data) — it just stops shrinking it further with a
                    # border it can't really afford.
                    edge_width = 2 if sub_w > 15 else 0
                    ax.add_patch(
                        plt.Rectangle(
                            (cx, ry), sub_w, rh, facecolor=p["color"], edgecolor=MPL_BG, linewidth=edge_width
                        )
                    )
                    sub_boxes.append((cx, ry, sub_w, rh, p["value"] / total * 100))
                    cx += sub_w
            else:
                cy = ry
                for p in breakdown:
                    sub_h = rh * (p["value"] / part_total)
                    edge_width = 2 if sub_h > 15 else 0
                    ax.add_patch(
                        plt.Rectangle(
                            (rx, cy), rw, sub_h, facecolor=p["color"], edgecolor=MPL_BG, linewidth=edge_width
                        )
                    )
                    sub_boxes.append((rx, cy, rw, sub_h, p["value"] / total * 100))
                    cy += sub_h
            if legend_handles is None:
                legend_handles = [Patch(facecolor=p["color"], label=p["name"]) for p in breakdown]

            # Short/wide rows (e.g. the smallest axes, squeezed into a
            # thin band) don't have room for BOTH the top-positioned axis
            # label and a separately-positioned sub-box percentage
            # without the two colliding — fall back to a single centered
            # label for the whole box, same as the single-dataset case
            # below, and skip the per-sub-box percentages entirely there.
            if rw > 35 and rh > 45:
                ax.text(
                    rx + rw / 2, ry + rh * 0.12, label, ha="center", va="top",
                    fontsize=8, color="white", **_mpl_font("bold"),
                )
                # Each sub-box's own percentage, centered within ITS OWN
                # area (below where the axis label sits) — this is what
                # gives the smaller (often comparison-dataset) sub-box a
                # visible number of its own, instead of just an unlabeled
                # sliver of color.
                for sub_x, sub_y, sub_w, sub_h, sub_pct in sub_boxes:
                    if sub_w > 22 and sub_h > 20:
                        ax.text(
                            sub_x + sub_w / 2, sub_y + sub_h * 0.62, f"{sub_pct:.1f}%",
                            ha="center", va="center", fontsize=7, color="white", **_mpl_font("regular"),
                        )
            elif rw > 35 and rh > 20:
                ax.text(
                    rx + rw / 2, ry + rh / 2, label, ha="center", va="center",
                    fontsize=8, color="white", **_mpl_font("bold"),
                )
        else:
            norm = value / max_val
            color = _lerp_hex(MPL_ACCENT_200, MPL_ACCENT_800, 0.15 + 0.85 * norm)
            ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor=MPL_BG, linewidth=2))
            text_color = "white" if norm > 0.45 else MPL_TEXT

            if rw > 35 and rh > 20:
                if rh > 32:
                    ax.text(
                        rx + rw / 2, ry + rh / 2, label, ha="center", va="bottom",
                        fontsize=8, color=text_color, **_mpl_font("bold"),
                    )
                    ax.text(
                        rx + rw / 2, ry + rh / 2, f"({pct:.1f}%)", ha="center", va="top",
                        fontsize=7, color=text_color, **_mpl_font("regular"),
                    )
                else:
                    ax.text(
                        rx + rw / 2, ry + rh / 2, label, ha="center", va="center",
                        fontsize=8, color=text_color, **_mpl_font("bold"),
                    )
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        "Cluster imbalance",
        fontsize=10,
        color=MPL_TEXT,
        loc="left",
        **_mpl_font("bold"),
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(legend_handles),
            fontsize=8, frameon=False, labelcolor=MPL_NEUTRAL_600,
        )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_dual_radar(
    labels: list[str],
    dominance_values: dict[str, float],
    normalized_values: dict[str, float],
    dataset_name: str,
    dominance_values_compare: dict[str, float] | None = None,
    normalized_values_compare: dict[str, float] | None = None,
    compare_dataset_name: str | None = None,
) -> io.BytesIO:
    """
    Two polar radar charts side by side — Dominance % and Normalized
    similarity, same axes, for direct visual comparison. When a second
    (comparison) dataset's values are provided, both are OVERLAID as
    separate polygons on each panel — never stacked/summed, since these
    are percentage-like values where a sum wouldn't mean anything
    coherent (same reasoning the live app's Stacked mode already applies:
    it's only offered for additive raw counts).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    multi_dataset = dominance_values_compare is not None and normalized_values_compare is not None

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200, subplot_kw=dict(polar=True))

    panels = [
        (dominance_values, dominance_values_compare, "Dominance (% of images)", lambda v: f"{v * 100:.0f}%"),
        (normalized_values, normalized_values_compare, "Normalized similarity", lambda v: f"{v:.2f}"),
    ]
    for ax, (values_dict, values_dict_compare, subtitle, fmt) in zip(axes, panels):
        # One series per dataset — (values, display name, fill/line color,
        # marker/annotation color). Always at least the primary; a second
        # entry is appended only when comparing.
        series = [(values_dict, dataset_name, MPL_ACCENT, MPL_ACCENT_800)]
        if multi_dataset:
            series.append(
                (values_dict_compare, compare_dataset_name, MPL_ACCENT_COMPARE, MPL_ACCENT_COMPARE_800)
            )

        all_vals_for_scale: list[float] = []
        for values_dict_series, series_name, fill_color, marker_color in series:
            vals = [values_dict_series.get(lbl, 0.0) for lbl in labels]
            all_vals_for_scale.extend(vals)
            vals_closed = vals + vals[:1]
            ax.plot(
                angles_closed, vals_closed, color=fill_color, linewidth=1.6,
                label=series_name if multi_dataset else None,
            )
            # A lighter fill when overlaying two series — otherwise the
            # second polygon's fill can fully obscure the first's, and
            # the overlap itself (the interesting part of a comparison)
            # becomes unreadable.
            ax.fill(angles_closed, vals_closed, color=fill_color, alpha=0.3 if not multi_dataset else 0.18)
            ax.plot(angles, vals, "o", color=marker_color, markersize=2.5, zorder=5)
            # Per-vertex value annotations only make sense for a single
            # series — with two overlaid polygons they'd collide and
            # become unreadable; the legend + polygon shapes carry the
            # comparison instead.
            if not multi_dataset:
                for angle, val in zip(angles, vals):
                    ax.annotate(
                        fmt(val),
                        xy=(angle, val),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color=marker_color,
                        **_mpl_font("bold"),
                    )

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8, color=MPL_NEUTRAL_600)
        ax.set_title(subtitle, fontsize=11, color=MPL_TEXT, pad=18, **_mpl_font("bold"))
        ax.tick_params(axis="y", labelsize=7, colors=MPL_NEUTRAL_600)
        ax.spines["polar"].set_color(MPL_NEUTRAL_300)
        ax.grid(color=MPL_NEUTRAL_300)
        if all_vals_for_scale:
            ax.set_ylim(0, max(all_vals_for_scale) * 1.22)
        if multi_dataset:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                fontsize=8, frameon=False, labelcolor=MPL_NEUTRAL_600,
            )

    if multi_dataset:
        title_names = f"{dataset_name} vs {compare_dataset_name}"
    else:
        title_names = dataset_name
    fig.suptitle(f"Semantic radar — {title_names}", fontsize=12, color=MPL_TEXT, **_mpl_font("bold"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_umap_scatter(
    coords: np.ndarray,
    point_labels: list[str],
    dataset_name: str,
    dataset_origin: list[str] | None = None,
    dataset_display_names: dict[str, str] | None = None,
) -> io.BytesIO:
    """
    Scatter of the 2D projection, colored by dominant axis (on-brand
    steel-blue family, see `_categorical_palette`), and — when comparing
    two datasets — shaped by which dataset each point came from. Two
    separate legends: axis color (right of the plot) and dataset shape
    (below the plot, only shown when there's more than one dataset).
    Mirrors the same color=axis / shape=dataset treatment as the live
    Scatter view in the app (src/viz/scatter.py).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if dataset_origin is None:
        dataset_origin = ["primary"] * len(point_labels)
    dataset_display_names = dataset_display_names or {}

    unique_labels = sorted(set(point_labels))
    palette = _categorical_palette(len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    # Preserve first-appearance order so "primary" reliably gets the
    # first marker (circle) rather than depending on string sort.
    unique_origins: list[str] = []
    for origin in dataset_origin:
        if origin not in unique_origins:
            unique_origins.append(origin)
    # Mirrors DATASET_SYMBOLS in src/viz/scatter.py (circle, x, diamond,
    # triangle-up, square, star) using matplotlib's own marker codes.
    marker_shapes = ["o", "x", "D", "^", "s", "*"]
    origin_markers = {
        origin: marker_shapes[i % len(marker_shapes)] for i, origin in enumerate(unique_origins)
    }
    multi_dataset = len(unique_origins) > 1

    fig, ax = plt.subplots(figsize=(9.8, 5.3), dpi=200)
    for origin in unique_origins:
        marker = origin_markers[origin]
        for lbl in unique_labels:
            idx = [
                i for i in range(len(point_labels))
                if point_labels[i] == lbl and dataset_origin[i] == origin
            ]
            if not idx:
                continue
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                s=16 if marker != "o" else 7,
                alpha=0.8,
                color=color_map[lbl],
                marker=marker,
                linewidths=0.8 if marker in ("x", "+") else 0,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if multi_dataset:
        names_for_title = " vs ".join(dataset_display_names.get(o, o) for o in unique_origins)
    else:
        names_for_title = dataset_name
    ax.set_title(f"UMAP — {names_for_title}", fontsize=11, color=MPL_TEXT, loc="left", **_mpl_font("bold"))

    # Axis legend: color squares, independent of whatever marker shape
    # the actual points use — built from explicit proxy handles (not the
    # scatter collections themselves) so it always shows squares.
    axis_handles = [
        Line2D([0], [0], marker="s", linestyle="", color=color_map[lbl], markersize=8)
        for lbl in unique_labels
    ]
    axis_legend = ax.legend(
        axis_handles, unique_labels,
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False,
        title="Axis", title_fontsize=9, labelcolor=MPL_NEUTRAL_600,
    )
    # matplotlib only keeps the MOST RECENT legend on an axes by default —
    # add_artist() is what lets a second, independent legend coexist.
    ax.add_artist(axis_legend)

    # Dataset legend: neutral-colored shapes, placed below the plot (not
    # stacked under the axis legend) so its position never depends on how
    # tall the axis legend happens to be. Only shown when comparing.
    dataset_legend = None
    if multi_dataset:
        dataset_handles = [
            Line2D([0], [0], marker=origin_markers[o], linestyle="", color=MPL_NEUTRAL_600, markersize=8)
            for o in unique_origins
        ]
        dataset_labels = [dataset_display_names.get(o, o) for o in unique_origins]
        dataset_legend = ax.legend(
            dataset_handles, dataset_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(unique_origins),
            fontsize=8, frameon=False, title="Dataset", title_fontsize=9, labelcolor=MPL_NEUTRAL_600,
        )

    fig.tight_layout()
    buf = io.BytesIO()
    # bbox_extra_artists tells the tight-bbox calculation to explicitly
    # account for these two legends — without it, bbox_inches="tight" can
    # fail to detect a legend added via add_artist() and crop it out of
    # the saved image entirely (this is what was happening to the Axis
    # legend specifically).
    extra_artists = [axis_legend] + ([dataset_legend] if dataset_legend is not None else [])
    fig.savefig(
        buf, format="png", bbox_inches="tight", bbox_extra_artists=extra_artists, transparent=True
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def _thumbnail_bytes(path: Path, max_size: int = 400) -> io.BytesIO | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return buf
    except Exception:  # noqa: BLE001
        return None


def _thumb_with_caption(image_flowable, caption_flowable, col_width: float):
    """
    Combines one thumbnail and its caption into a single small block —
    used instead of building one shared "row of images" + "row of
    captions" across a whole row of thumbnails, because reportlab sizes
    each of THOSE rows to its tallest member: with images of different
    real aspect ratios (never distorted to a uniform size — a firm rule
    for this report), the shorter thumbnails would end up with a big,
    uneven gap before their caption, while the tallest one has none.
    Bundling image+caption into one block per thumbnail keeps that gap
    FIXED for every thumbnail — any leftover height instead lands at the
    bottom of the shorter columns, where it doesn't disturb the caption's
    position relative to its own image.
    """
    inner = Table([[image_flowable], [caption_flowable]], colWidths=[col_width])
    inner.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return inner


def _sample_thumbnail_row(
    doc_width: float,
    paths: list[Path],
    styles: dict,
    n_cols: int = 6,
    thumb_size: float = None,
    dataset_names: list[str] | None = None,
    dedupe_keys: list | None = None,
) -> Table | None:
    """
    A single row of small thumbnails with the filename underneath — used
    for the Dataset composition 'samples' rows (low-fit / near-duplicate /
    cross-dataset match groups). Each thumbnail is framed as a blueprint
    object, same as every other figure in the report.

    Fixed WIDTH per column, height computed from each image's own aspect
    ratio (never distorted/stretched to a square), top-aligned so images
    of different proportions all start flush at the top of the row.

    When there are more than n_cols images, the LAST column becomes a
    "N MORE..." placeholder instead of a 6th (or Nth) photo — showing
    N-1 real thumbnails plus a count of how many more belong to this same
    group, rather than silently truncating with no indication more exist.

    dataset_names, if given: one entry per path (same order/length) —
    colors each thumbnail's caption (filename + dataset name) by which
    dataset it came from, steel-blue/amber in first-seen order, same
    convention as everywhere else two datasets are distinguished. None
    (the default) keeps the original plain gray filename-only caption.

    dedupe_keys, if given: one entry per path (same order/length), or
    None for a path that isn't part of any group worth avoiding — when
    picking which show_count paths to actually display (out of a longer
    ranked list), skips a candidate if its key has ALREADY been used by
    an earlier pick, walking further down the ranked list instead, so the
    shown sample doesn't end up with 2+ images that are actually
    near-duplicates of each other (e.g. two photos from the same
    near-duplicate GROUP both showing up in "Low semantic fit" just
    because they happened to rank next to each other). Falls back to
    allowing repeats only if there aren't enough distinct-key candidates
    left to fill show_count. None (the default) just takes the first
    show_count, unchanged from before.
    """
    thumb_w = (thumb_size or (doc_width / n_cols - 6)) - 14  # leave room for the frame
    total = len(paths)
    show_count = n_cols if total <= n_cols else n_cols - 1

    if dedupe_keys:
        shown_indices: list[int] = []
        used_keys = set()
        for i in range(total):
            key = dedupe_keys[i]
            if key is None or key not in used_keys:
                shown_indices.append(i)
                if key is not None:
                    used_keys.add(key)
            if len(shown_indices) == show_count:
                break
        if len(shown_indices) < show_count:
            # Not enough distinct-key candidates to fill every slot —
            # fill what's left, in rank order, allowing repeats.
            shown_set = set(shown_indices)
            leftover = [i for i in range(total) if i not in shown_set]
            shown_indices += leftover[: show_count - len(shown_indices)]
        shown_indices.sort()
    else:
        shown_indices = list(range(min(show_count, total)))

    dataset_color_map: dict[str, str] = {}
    if dataset_names:
        seen_names: list[str] = []
        for name in dataset_names:
            if name not in seen_names:
                seen_names.append(name)
        palette = [MPL_ACCENT, MPL_ACCENT_COMPARE]
        dataset_color_map = {name: palette[i % len(palette)] for i, name in enumerate(seen_names)}

    cells = []
    for i in shown_indices:
        p = paths[i]
        buf = _thumbnail_bytes(p, max_size=300)
        if buf is None:
            continue
        framed = _Blueprint(_rl_image_for_width(buf, thumb_w), pad=3, margin=6)
        if dataset_names:
            ds_name = dataset_names[i]
            color = dataset_color_map.get(ds_name, MPL_NEUTRAL_600)
            caption = Paragraph(f'<font color="{color}">{p.name}<br/>{ds_name}</font>', styles["caption"])
        else:
            caption = Paragraph(p.name, styles["caption"])
        cells.append([framed, caption])

    if total > n_cols:
        remaining = total - show_count
        more_inner = Table([[Paragraph(f'<font size="13"><b>+{remaining}</b></font><br/>'
                                        f'<font size="8">MORE...</font>', styles["caption"])]],
                            colWidths=[thumb_w], rowHeights=[thumb_w * 0.72])
        more_inner.setStyle(
            TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])
        )
        framed_more = _Blueprint(more_inner, pad=3, margin=6)
        cells.append([framed_more, ""])

    if not cells:
        return None
    col_width = doc_width / n_cols
    blocks = [_thumb_with_caption(framed, caption, col_width) for framed, caption in cells]
    while len(blocks) < n_cols:
        blocks.append("")

    table = Table([blocks], colWidths=[col_width] * n_cols)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def _select_diverse_subset(
    group: list[tuple[Path, str]], show_count: int
) -> list[tuple[Path, str]]:
    """
    Picks which `show_count` members of a group to actually display as
    thumbnails, prioritizing representation from BOTH datasets when a
    group spans two. Groups are built by concatenating one dataset's
    matching paths then the other's, so a plain "first N" slice tends to
    grab only the majority dataset's images — showing a supposed "mix"
    that turns out to be all one color defeats the point of the
    blue/amber distinction. Guarantees up to 2 images from the minority
    dataset are included (or all of them, if it has fewer than 2), filling
    the remaining slots with the majority dataset's images. Falls back to
    the group's original order for single-dataset groups (e.g. every
    "Near duplicates" group, which never spans two datasets by
    construction) — a no-op there.
    """
    if show_count >= len(group):
        return list(group)

    by_dataset: dict[str, list[tuple[Path, str]]] = {}
    for item in group:
        by_dataset.setdefault(item[1], []).append(item)

    if len(by_dataset) < 2:
        return group[:show_count]

    datasets_by_size = sorted(by_dataset.items(), key=lambda kv: len(kv[1]))
    minority_items = datasets_by_size[0][1]
    guaranteed = minority_items[: min(2, len(minority_items), show_count)]
    guaranteed_set = set(guaranteed)

    remaining_slots = show_count - len(guaranteed)
    filler = [item for item in group if item not in guaranteed_set][:remaining_slots]

    selected = guaranteed + filler
    # Restore the group's original relative order for a natural-looking
    # row, rather than clumping the minority picks at the front.
    selected.sort(key=lambda item: group.index(item))
    return selected


def _flowing_thumbnail_grid(
    doc_width: float,
    groups: list[list[tuple[Path, str]]],
    styles: dict,
    n_cols: int = 6,
    thumb_size: float = None,
) -> list:
    """
    Packs images from MULTIPLE groups into a continuous flowing grid of
    n_cols-wide rows — unlike one row per group (padded with blank cells
    when a group doesn't fill a full row), a group's remaining images
    continue filling the SAME row as the next group's, with no forced
    row break at group boundaries. This is what avoids wasting paper when
    there are many small groups (e.g. a large dataset with dozens of
    near-duplicate pairs) — every row stays full until the very last one.

    groups: list of groups, EACH a list of (Path, dataset_name) tuples —
    already sorted largest-first by the caller (same convention as
    before: biggest evidence shown first). A group with more than n_cols
    images shows its first (n_cols - 1) thumbnails plus a single "N
    MORE..." placeholder cell, same as the previous one-row-per-group
    version. Each group's first cell gets a small "Group N" label
    prepended to its caption, since row boundaries no longer align with
    group boundaries, so groups still need SOME visual identification.

    Returns a list of Flowables (one Table per row) — append them all to
    the story, there's no wrapping frame around the whole grid.
    """
    thumb_w = (thumb_size or (doc_width / n_cols - 6)) - 14

    # Color per unique dataset name (first-seen order across ALL groups,
    # not per-row) — same steel-blue/amber convention as _sample_
    # thumbnail_row, computed once so it's consistent across the whole
    # grid regardless of which row a given dataset's image lands in.
    all_names: list[str] = []
    for group in groups:
        for _p, name in group:
            if name not in all_names:
                all_names.append(name)
    palette = [MPL_ACCENT, MPL_ACCENT_COMPARE]
    dataset_color_map = {name: palette[i % len(palette)] for i, name in enumerate(all_names)}

    # Flatten every group into one continuous sequence of cell
    # descriptors — this flat sequence is what actually gets chunked
    # into fixed-width rows below, irrespective of group boundaries.
    cells_flat: list[dict] = []
    for gi, group in enumerate(groups):
        total = len(group)
        show_count = n_cols if total <= n_cols else n_cols - 1
        shown_items = _select_diverse_subset(group, show_count)
        for idx, (path, ds_name) in enumerate(shown_items):
            cells_flat.append(
                {
                    "type": "image",
                    "path": path,
                    "dataset_name": ds_name,
                    "group_label": f"Group {gi + 1}" if idx == 0 else None,
                }
            )
        if total > n_cols:
            cells_flat.append({"type": "more", "count": total - show_count})

    rendered_cells = []
    for cell in cells_flat:
        if cell["type"] == "more":
            more_inner = Table(
                [[Paragraph(f'<font size="13"><b>+{cell["count"]}</b></font><br/>'
                            f'<font size="8">MORE...</font>', styles["caption"])]],
                colWidths=[thumb_w], rowHeights=[thumb_w * 0.72],
            )
            more_inner.setStyle(
                TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])
            )
            rendered_cells.append((_Blueprint(more_inner, pad=3, margin=6), ""))
            continue

        buf = _thumbnail_bytes(cell["path"], max_size=300)
        if buf is None:
            continue
        framed = _Blueprint(_rl_image_for_width(buf, thumb_w), pad=3, margin=6)
        color = dataset_color_map.get(cell["dataset_name"], MPL_NEUTRAL_600)
        caption_html = f'<font color="{color}">{cell["path"].name}<br/>{cell["dataset_name"]}</font>'
        if cell["group_label"]:
            caption_html = f'<font color="{MPL_NEUTRAL_600}"><b>{cell["group_label"]}</b></font><br/>' + caption_html
        rendered_cells.append((framed, Paragraph(caption_html, styles["caption"])))

    if not rendered_cells:
        return []

    col_width = doc_width / n_cols
    row_flowables = []
    for i in range(0, len(rendered_cells), n_cols):
        row_cells = list(rendered_cells[i : i + n_cols])
        blocks = [_thumb_with_caption(framed, caption, col_width) for framed, caption in row_cells]
        while len(blocks) < n_cols:
            blocks.append("")
        table = Table([blocks], colWidths=[col_width] * n_cols)
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        row_flowables.append(table)
    return row_flowables


def generate_pdf_report(
    dataset_name: str,
    overview: DatasetOverviewMetrics,
    representative_images: list[tuple[str, Path]],
    low_fit_samples: list[Path] | None = None,
    low_fit_dataset_names: list[str] | None = None,
    low_fit_dupe_keys: list | None = None,
    duplicate_groups: list[list[tuple[Path, str]]] | None = None,
    duplicate_match_count: int | None = None,
    radar_dominance_values: dict[str, float] | None = None,
    radar_normalized_values: dict[str, float] | None = None,
    radar_dominance_values_compare: dict[str, float] | None = None,
    radar_normalized_values_compare: dict[str, float] | None = None,
    umap_coords: np.ndarray | None = None,
    umap_labels: list[str] | None = None,
    umap_dataset_origin: list[str] | None = None,
    umap_dataset_display_names: dict[str, str] | None = None,
    clip_mmd: float | None = None,
    clip_mmd_baseline: float | None = None,
    compare_dataset_name: str | None = None,
    overview_compare: DatasetOverviewMetrics | None = None,
) -> bytes:
    """
    Build the full Dataset Report PDF and return its bytes (ready for a
    download button). The radar comparison and UMAP pages are landscape;
    everything else is portrait — reportlab handles the mixed orientation
    via two named page templates (see `doc.addPageTemplates` below).
    """
    styles = _make_styles()
    buffer = io.BytesIO()

    portrait_w = PORTRAIT_SIZE[0] - 2 * PAGE_MARGIN
    portrait_h = PORTRAIT_SIZE[1] - 2 * PAGE_MARGIN
    landscape_w = LANDSCAPE_SIZE[0] - 2 * PAGE_MARGIN
    landscape_h = LANDSCAPE_SIZE[1] - 2 * PAGE_MARGIN

    # Both dataset names when comparing, primary only otherwise — same
    # "X vs Y" convention already used for the radar/scatter/treemap
    # titles further down. Computed directly from the function's own
    # parameters here since it's needed before `comparing` gets computed
    # later (right before the "What should I look at?" callout).
    report_title_names = f"{dataset_name} vs {compare_dataset_name}" if overview_compare is not None else dataset_name

    doc = BaseDocTemplate(
        buffer,
        pagesize=PORTRAIT_SIZE,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=f"Semantic Report by sneaky\u2122 — {report_title_names}",
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="Portrait",
                pagesize=PORTRAIT_SIZE,
                frames=[Frame(PAGE_MARGIN, PAGE_MARGIN, portrait_w, portrait_h, id="portrait")],
                onPage=_paint_page,
            ),
            PageTemplate(
                id="Landscape",
                pagesize=LANDSCAPE_SIZE,
                frames=[Frame(PAGE_MARGIN, PAGE_MARGIN, landscape_w, landscape_h, id="landscape")],
                onPage=_paint_page,
            ),
        ]
    )

    story = []

    # --- Header — dataset name leads (it's the thing that changes report
    # to report); "Dataset Report" becomes a small kicker label above it,
    # like a title block on a technical drawing. -------------------------
    story.append(Paragraph("SEMANTIC REPORT BY SNEAKY\u2122", styles["kicker"]))
    story.append(Spacer(1, SP_1))
    story.append(Paragraph(report_title_names, styles["title"]))
    story.append(Spacer(1, SP_1))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            f"{overview.total_images + (overview_compare.total_images if overview_compare else 0):,} images",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, SP_3))
    rule = Table([[""]], colWidths=[portrait_w], rowHeights=[1.2])
    rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, COLOR_ACCENT)]))
    story.append(rule)
    story.append(Spacer(1, SP_6))

    # --- "What should I look at?" — the metric-to-evidence bridge, up
    # front, in plain language, before any chart or jargon. ---------------
    # `comparing` and the combined counts are computed here (moved earlier
    # than the KPI cards section below, which used to be the only place
    # this existed) so the callout AND the "Low semantic fit"/"Near
    # duplicates" section headers further down all agree with each other
    # and with the KPI cards — previously only the KPI cards reflected
    # both datasets; the callout and headers still only counted the
    # primary feed, which could show e.g. "0 near duplicates" here even
    # when the app's own duplicate view found matches living entirely in
    # the comparison feed.
    comparing = overview_compare is not None
    combined_unclassified_count = overview.unclassified_count + (
        overview_compare.unclassified_count if comparing else 0
    )
    # Still the WITHIN-dataset-only redundancy figure — this is what the
    # "Visual redundancy" KPI card/donut means (how bloated is each
    # dataset with its OWN near-duplicates), a per-dataset data-quality
    # signal that's conceptually different from — and shouldn't change
    # just because of — the unified duplicate_match_count below.
    combined_duplicate_count = overview.visual_duplicates_count + (
        overview_compare.visual_duplicates_count if comparing else 0
    )
    # The broader figure for the callout/section header: ALL near-duplicate
    # groups shown in the unified thumbnail section below, whether within
    # one dataset or spanning both (see get_all_duplicate_groups_combined)
    # — falls back to the narrower within-dataset count if the caller
    # didn't pass it, so this function still works with older call sites.
    shown_duplicate_count = (
        duplicate_match_count if duplicate_match_count is not None else combined_duplicate_count
    )

    callout_lines = []
    if combined_unclassified_count > 0:
        callout_lines.append(
            f"<b>{combined_unclassified_count:,} images have low semantic fit.</b> "
            "They don't strongly match any of the detected themes — worth a quick "
            "look to confirm they belong in this dataset."
        )
    if shown_duplicate_count > 0:
        callout_lines.append(
            f"<b>{shown_duplicate_count:,} images belong to near-duplicate groups.</b> "
            "This may be expected (e.g. burst-mode photos, or the same subject "
            "appearing in both datasets) or a sign of redundancy, depending on what "
            "this dataset is for."
        )
    if clip_mmd is not None and clip_mmd_baseline is not None and compare_dataset_name:
        callout_lines.append(
            f"<b>Compared with \u201c{compare_dataset_name}\u201d: distributional "
            f"distance {clip_mmd:.3f}.</b> For reference, splitting this dataset "
            f"into two random halves gives {clip_mmd_baseline:.3f} \u2014 that's "
            "roughly the gap you'd expect from sampling alone, even between two "
            "halves of the exact same dataset. A distance well above that suggests "
            "a genuinely different overall makeup; a distance close to it suggests "
            "the two are hard to tell apart, distribution-wise."
        )
    if callout_lines:
        callout_inner = Table(
            [[Paragraph("What should I look at?", styles["section"])]]
            + [[Paragraph(line, styles["callout_body"])] for line in callout_lines],
            colWidths=[portrait_w - 34],
        )
        callout_inner.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, 0), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 1), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 0),
                ]
            )
        )
        story.append(_Blueprint(callout_inner, pad=10, margin=7))
        story.append(Spacer(1, SP_6))

    # --- KPI cards ------------------------------------------------------
    story.append(_section_header("Overview", styles, portrait_w))
    story.append(Spacer(1, SP_4))

    # Computed once here (not inline at each chart) so the bar chart AND
    # the treemap both use the exact same combined totals/breakdown —
    # and so it's safe to reference at the treemap's call site even in
    # the edge case where the bar chart's own "if axis_sizes:" guard
    # further down might not fire.
    combined_axis_sizes: dict[str, int] = overview.axis_sizes
    axis_sizes_breakdown: dict[str, list[dict]] | None = None
    # Combined totals/percentages — computed from underlying COUNTS, not
    # by averaging the two datasets' own percentages (averaging would
    # misrepresent the combined figure whenever the two datasets are
    # different sizes: a 1000-image dataset at 90% and a 10-image dataset
    # at 10% is NOT meaningfully "50% combined" — it's ~89%, dominated by
    # the much larger dataset). Defaults to the single dataset's own
    # values when there's no comparison feed, so every one of these is
    # always safe to reference below regardless of `comparing`.
    combined_total = overview.total_images
    combined_unclassified_pct = overview.unclassified_pct
    combined_semantic_coverage_pct = overview.semantic_coverage_pct
    combined_duplicates_pct = overview.visual_duplicates_pct
    if comparing:
        all_axis_labels = set(overview.axis_sizes) | set(overview_compare.axis_sizes)
        combined_axis_sizes = {
            lbl: overview.axis_sizes.get(lbl, 0) + overview_compare.axis_sizes.get(lbl, 0)
            for lbl in all_axis_labels
        }
        axis_sizes_breakdown = {
            lbl: [
                {"value": overview.axis_sizes.get(lbl, 0), "color": MPL_ACCENT, "name": dataset_name},
                {
                    "value": overview_compare.axis_sizes.get(lbl, 0),
                    "color": MPL_ACCENT_COMPARE,
                    "name": compare_dataset_name,
                },
            ]
            for lbl in all_axis_labels
        }
        combined_total = overview.total_images + overview_compare.total_images
        combined_unclassified_pct = (
            (overview.unclassified_count + overview_compare.unclassified_count) / combined_total * 100
            if combined_total
            else 0.0
        )
        combined_semantic_coverage_pct = 100.0 - combined_unclassified_pct
        combined_duplicates_pct = (
            (overview.visual_duplicates_count + overview_compare.visual_duplicates_count)
            / combined_total
            * 100
            if combined_total
            else 0.0
        )
    if comparing:
        primary_swatch = {"color": MPL_ACCENT, "name": dataset_name}
        compare_swatch = {"color": MPL_ACCENT_COMPARE, "name": compare_dataset_name}

        def _bd(primary_value: str, compare_value: str) -> list[dict]:
            return [
                {**primary_swatch, "value": primary_value},
                {**compare_swatch, "value": compare_value},
            ]

        # A small fixed gap between cards (matching the donuts' own gap
        # convention below), with each card's width computed so 3 cards +
        # 2 gaps land exactly on portrait_w — the cards' own fixed 1.5in
        # default width otherwise falls well short of the row's outer
        # edges, unlike Axis sizes/the callout/the donuts. card_total_w is
        # each card's full rendered (Blueprint) width, used for the outer
        # table's column widths below; card_w is the INNER width to pass
        # into _metric_card, before its own pad/margin add 18pt back.
        kpi_gap = 16
        card_total_w = (portrait_w - 2 * kpi_gap) / 3
        card_w = card_total_w - 18

        cards = [
            _metric_card(
                f"{combined_total:,}",
                "Total images",
                styles,
                breakdown=_bd(f"{overview.total_images:,}", f"{overview_compare.total_images:,}"),
                card_width=card_w,
            ),
            _metric_card(
                f"{combined_semantic_coverage_pct:.0f}%",
                "Semantic coverage",
                styles,
                breakdown=_bd(
                    f"{overview.semantic_coverage_pct:.0f}%", f"{overview_compare.semantic_coverage_pct:.0f}%"
                ),
                card_width=card_w,
            ),
            _metric_card(
                f"{combined_unclassified_pct:.1f}%",
                "Low semantic fit",
                styles,
                breakdown=_bd(
                    f"{overview.unclassified_pct:.1f}%", f"{overview_compare.unclassified_pct:.1f}%"
                ),
                card_width=card_w,
            ),
            _metric_card(
                f"{combined_duplicates_pct:.1f}%",
                "Visual redundancy",
                styles,
                breakdown=_bd(
                    f"{overview.visual_duplicates_pct:.1f}%", f"{overview_compare.visual_duplicates_pct:.1f}%"
                ),
                card_width=card_w,
            ),
            # Semantic axes: both datasets are scored against the exact
            # SAME axis set, so a per-dataset breakdown would just show
            # two identical numbers — no breakdown here on purpose.
            _metric_card(str(overview.n_semantic_axes), "Semantic axes", styles, card_width=card_w),
            # Largest/smallest cluster stays as a single combined card —
            # already fairly dense at 2 numbers; a 4-number breakdown
            # here would be hard to read at this card size.
            _metric_card(
                f"{overview.largest_cluster_pct:.0f}% / {overview.smallest_cluster_pct:.0f}%",
                "Largest / smallest cluster",
                styles,
                card_width=card_w,
            ),
        ]
    else:
        kpi_gap = 16
        card_total_w = (portrait_w - 2 * kpi_gap) / 3
        card_w = card_total_w - 18
        cards = [
            _metric_card(f"{overview.total_images:,}", "Total images", styles, card_width=card_w),
            _metric_card(
                f"{overview.semantic_coverage_pct:.0f}%", "Semantic coverage", styles, card_width=card_w
            ),
            _metric_card(
                f"{overview.unclassified_pct:.1f}%", "Low semantic fit", styles, card_width=card_w
            ),
            _metric_card(
                f"{overview.visual_duplicates_pct:.1f}%", "Visual redundancy", styles, card_width=card_w
            ),
            _metric_card(str(overview.n_semantic_axes), "Semantic axes", styles, card_width=card_w),
            _metric_card(
                f"{overview.largest_cluster_pct:.0f}% / {overview.smallest_cluster_pct:.0f}%",
                "Largest / smallest cluster",
                styles,
                card_width=card_w,
            ),
        ]
    # 5 explicit columns (card, gap, card, gap, card) rather than 3
    # equal-width cells with padding — each card's own width is already
    # computed so this sums to exactly portrait_w, matching Axis sizes/
    # the callout/the donuts. Same explicit-column approach as the donut
    # table above, for the same reason: a fixed-width flowable inside a
    # WIDER equal-width cell doesn't reliably reach that cell's own outer
    # edge just by adjusting padding.
    col_widths = [card_total_w, kpi_gap, card_total_w, kpi_gap, card_total_w]
    rows = []
    for i in range(0, len(cards), 3):
        row_cards = cards[i : i + 3]
        while len(row_cards) < 3:
            row_cards.append("")
        rows.append([row_cards[0], "", row_cards[1], "", row_cards[2]])
    kpi_table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    kpi_table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (2, 0), (2, -1), "CENTER"),
                ("ALIGN", (4, 0), (4, -1), "RIGHT"),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, SP_4))
    # --- Axis size bar chart -----------------------------------------
    if overview.axis_sizes or (overview_compare and overview_compare.axis_sizes):
        if comparing:
            bar_buf = _render_bar_chart(combined_axis_sizes, axis_sizes_breakdown)
        else:
            bar_buf = _render_bar_chart(overview.axis_sizes)
        story.append(Spacer(1, SP_2))
        story.append(_Blueprint(_rl_image_for_width(bar_buf, portrait_w - 26), pad=6, margin=7))

    # --- Donut charts ---------------------------------------------------
    # Right under Axis sizes, on the SAME portrait page — a dedicated
    # landscape page for these (tried previously) left an awkward amount
    # of empty space below Axis sizes with a typical axis count, before
    # the donuts even appeared on the following page.
    if comparing:
        coverage_buf = _render_donut_with_breakdown(
            "Semantic coverage",
            ["Classified", "Unclassified"],
            [combined_semantic_coverage_pct, combined_unclassified_pct],
            [MPL_ACCENT, MPL_NEUTRAL_300],
            [
                {
                    "name": dataset_name,
                    "values": [overview.semantic_coverage_pct, overview.unclassified_pct],
                },
                {
                    "name": compare_dataset_name,
                    "values": [overview_compare.semantic_coverage_pct, overview_compare.unclassified_pct],
                },
            ],
        )
        duplicates_buf = _render_donut_with_breakdown(
            "Visual redundancy",
            ["Unique", "Near-duplicate"],
            [100 - combined_duplicates_pct, combined_duplicates_pct],
            [MPL_ACCENT, MPL_NEUTRAL_300],
            [
                {
                    "name": dataset_name,
                    "values": [100 - overview.visual_duplicates_pct, overview.visual_duplicates_pct],
                },
                {
                    "name": compare_dataset_name,
                    "values": [
                        100 - overview_compare.visual_duplicates_pct,
                        overview_compare.visual_duplicates_pct,
                    ],
                },
            ],
        )
    else:
        coverage_buf = _render_donut(
            ["Classified", "Unclassified"],
            [overview.semantic_coverage_pct, overview.unclassified_pct],
            [MPL_ACCENT, MPL_NEUTRAL_300],
            "Semantic coverage",
        )
        duplicates_buf = _render_donut(
            ["Unique", "Near-duplicate"],
            [100 - overview.visual_duplicates_pct, overview.visual_duplicates_pct],
            [MPL_ACCENT, MPL_NEUTRAL_300],
            "Visual redundancy",
        )

    # Two donut cards side by side, with a fixed gap between them, sized
    # so the COMBINED block's outer edges land exactly on portrait_w —
    # the same left/right edges as the Axis sizes blueprint above and
    # the KPI cards row below, rather than each card just centered
    # within half the page (which doesn't guarantee matching outer
    # edges). pad=4/margin=7 per card is unchanged from before.
    donut_gap = 20
    donut_total_w = (portrait_w - donut_gap) / 2
    chart_w = donut_total_w - 2 * 4 - 2 * 7
    donut_table = Table(
        [
            [
                _Blueprint(_rl_image_for_width(coverage_buf, chart_w), pad=4, margin=7),
                "",
                _Blueprint(_rl_image_for_width(duplicates_buf, chart_w), pad=4, margin=7),
            ]
        ],
        colWidths=[donut_total_w, donut_gap, donut_total_w],
        hAlign="LEFT",
    )
    donut_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(Spacer(1, SP_4))
    story.append(donut_table)

    # --- Representative images ----------------------------------------
    if representative_images:
        story.append(NextPageTemplate("Portrait"))
        story.append(PageBreak())
        story.append(
            _section_header(
                "What is in your datasets?" if comparing else "What is in your dataset?", styles, portrait_w
            )
        )
        story.append(Spacer(1, SP_3))
        story.append(
            Paragraph(
                "One representative image per semantic axis — for auto-detected axes, "
                "the image closest to the cluster centroid; for custom text axes (which "
                "have no centroid image of their own), a random image that dominates "
                "that axis.",
                styles["body"],
            )
        )
        story.append(Spacer(1, SP_4))

        n_cols = 4
        thumb_w = portrait_w / n_cols - 14
        cells = []
        for label, img_path in representative_images:
            thumb_buf = _thumbnail_bytes(img_path)
            if thumb_buf is None:
                continue
            framed = _Blueprint(_rl_image_for_width(thumb_buf, thumb_w), pad=3, margin=6)
            cell = Table(
                [[framed], [Paragraph(label, styles["caption"])]],
                colWidths=[thumb_w + 18],
            )
            cell.setStyle(
                TableStyle(
                    [
                        ("TOPPADDING", (0, 1), (-1, 1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]
                )
            )
            cells.append(cell)

        rows = [cells[i : i + n_cols] for i in range(0, len(cells), n_cols)]
        if rows and len(rows[-1]) < n_cols:
            rows[-1] += [""] * (n_cols - len(rows[-1]))

        img_table = Table(rows, colWidths=[portrait_w / n_cols] * n_cols)
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(img_table)

    # --- Dataset composition --------------------------------------------
    story.append(PageBreak())
    story.append(_section_header("Dataset composition", styles, portrait_w))
    story.append(Spacer(1, SP_3))
    story.append(
        Paragraph(
            "Two independent signals: whether an image fits a semantic theme at "
            "all, and whether it's visually a near-duplicate of another image.",
            styles["body"],
        )
    )
    story.append(Spacer(1, SP_4))

    story.append(
        Paragraph(
            f"What doesn't fit?: Low semantic fit — {combined_unclassified_count:,} images "
            f"({combined_unclassified_pct:.1f}%)",
            styles["section"],
        )
    )
    story.append(Spacer(1, SP_2))
    if low_fit_samples:
        row = _sample_thumbnail_row(
            portrait_w,
            low_fit_samples,
            styles,
            dataset_names=low_fit_dataset_names,
            dedupe_keys=low_fit_dupe_keys,
        )
        if row:
            story.append(row)
    story.append(Spacer(1, SP_4))

    shown_duplicate_pct = (shown_duplicate_count / combined_total * 100) if combined_total else 0.0
    story.append(
        Paragraph(
            f"How much visual redundancy is there?: Near duplicates — {shown_duplicate_count:,} images "
            f"({shown_duplicate_pct:.1f}%)",
            styles["section"],
        )
    )
    story.append(Spacer(1, SP_2))
    for row in _flowing_thumbnail_grid(portrait_w, duplicate_groups or [], styles):
        story.append(row)
    story.append(Spacer(1, SP_4))

    # Treemap — deliberately a different chart TYPE from "Axis sizes"
    # earlier in the report, so the two don't look like duplicates. Its
    # own landscape page (same pattern as the radar/scatter below) — the
    # portrait width didn't leave enough room for each sub-box's own
    # label when comparing two datasets.
    if overview.axis_sizes:
        story.append(NextPageTemplate("Landscape"))
        story.append(PageBreak())
        diversity_header = _section_header(
            "How diverse are they?" if comparing else "How diverse is it?", styles, landscape_w
        )
        _, header_h = diversity_header.wrap(landscape_w, landscape_h)
        story.append(diversity_header)
        story.append(Spacer(1, SP_4))
        treemap_buf = _render_treemap(combined_axis_sizes, axis_sizes_breakdown if comparing else None)
        treemap_frame = _Blueprint(_rl_image_for_width(treemap_buf, landscape_w - 26), pad=6, margin=7)
        remaining_h = landscape_h - header_h - SP_4
        treemap_frame.wrap(landscape_w, remaining_h)
        story.append(Spacer(1, max(0, (remaining_h - treemap_frame.height) / 2)))
        story.append(treemap_frame)

    # --- Landscape: dual radar comparison -----------------------------
    if radar_dominance_values and radar_normalized_values:
        story.append(NextPageTemplate("Landscape"))
        story.append(PageBreak())
        header_h = 0.0
        if comparing:
            compare_header = _section_header("How do the two datasets compare?", styles, landscape_w)
            _, header_h = compare_header.wrap(landscape_w, landscape_h)
            story.append(compare_header)
            story.append(Spacer(1, SP_4))
            header_h += SP_4
        radar_labels = sorted(set(radar_dominance_values) | set(radar_normalized_values))
        radar_buf = _render_dual_radar(
            radar_labels,
            radar_dominance_values,
            radar_normalized_values,
            dataset_name,
            radar_dominance_values_compare,
            radar_normalized_values_compare,
            compare_dataset_name,
        )
        radar_frame = _Blueprint(_rl_image_for_width(radar_buf, landscape_w - 26), pad=6, margin=7)
        remaining_h = landscape_h - header_h
        radar_frame.wrap(landscape_w, remaining_h)
        story.append(Spacer(1, max(0, (remaining_h - radar_frame.height) / 2)))
        story.append(radar_frame)

    # --- Landscape: UMAP scatter ---------------------------------------
    if umap_coords is not None and umap_labels:
        story.append(NextPageTemplate("Landscape"))
        story.append(PageBreak())
        umap_buf = _render_umap_scatter(
            umap_coords, umap_labels, dataset_name, umap_dataset_origin, umap_dataset_display_names
        )
        umap_frame = _Blueprint(_rl_image_for_width(umap_buf, landscape_w - 26), pad=6, margin=7)
        umap_frame.wrap(landscape_w, landscape_h)
        story.append(Spacer(1, max(0, (landscape_h - umap_frame.height) / 2)))
        story.append(umap_frame)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
