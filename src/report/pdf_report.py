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


def _metric_card(value: str, label: str, styles: dict) -> _Blueprint:
    """One KPI 'card' — big number, small label underneath, framed as a
    blueprint object (transparent, hairline border, corner marks) rather
    than a filled gray box."""
    inner = Table(
        [[Paragraph(value, styles["card_value"])], [Paragraph(label.upper(), styles["card_label"])]],
        colWidths=[1.5 * inch],
    )
    inner.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 1),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )
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


def _render_bar_chart(axis_sizes: dict[str, int]) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(axis_sizes.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    height = max(1.6, 0.20 * len(items) + 0.45)
    fig, ax = plt.subplots(figsize=(6.8, height), dpi=200)
    bars = ax.barh(labels, values, color=MPL_ACCENT, height=0.62)
    max_val = max(values) if values else 1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=7.5,
            color=MPL_NEUTRAL_600,
            **_mpl_font("regular"),
        )
    ax.set_xlim(0, max_val * 1.12)
    ax.set_xlabel("images", fontsize=8.5, color=MPL_NEUTRAL_600)
    ax.set_title("Axis sizes", fontsize=10, color=MPL_TEXT, loc="left", **_mpl_font("bold"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, colors=MPL_NEUTRAL_600)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MPL_NEUTRAL_300)
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


def _render_treemap(axis_sizes: dict[str, int]) -> io.BytesIO:
    """
    "Cluster imbalance" as a treemap — visually distinct from the plain
    "Axis sizes" bar chart earlier in the report. Darker/bigger = more
    images, shaded on the system's own accent ramp (light→dark steel)
    rather than a generic colormap. Rectangles too small to hold a
    readable label are left blank rather than overflowing text onto
    neighboring cells.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(axis_sizes.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    W, H = 700.0, 420.0
    total = sum(values) if values else 1
    scaled = [v / total * (W * H) for v in values]
    rects = _squarify_layout(scaled, 0, 0, W, H)

    fig, ax = plt.subplots(figsize=(6.8, 4.1), dpi=200)
    max_val = max(values) if values else 1
    for (rx, ry, rw, rh), label, value in zip(rects, labels, values):
        norm = value / max_val
        color = _lerp_hex(MPL_ACCENT_200, MPL_ACCENT_800, 0.15 + 0.85 * norm)
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor=MPL_BG, linewidth=2))
        if rw > 35 and rh > 20:
            text_color = "white" if norm > 0.45 else MPL_TEXT
            pct = value / total * 100
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
        "How diverse is it? — Cluster imbalance",
        fontsize=10,
        color=MPL_TEXT,
        loc="left",
        **_mpl_font("bold"),
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
) -> io.BytesIO:
    """Two polar radar charts side by side — Dominance % and Normalized
    similarity, same axes, for direct visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200, subplot_kw=dict(polar=True))

    panels = [
        (dominance_values, "Dominance (% of images)", lambda v: f"{v * 100:.0f}%"),
        (normalized_values, "Normalized similarity", lambda v: f"{v:.2f}"),
    ]
    for ax, (values_dict, subtitle, fmt) in zip(axes, panels):
        vals = [values_dict.get(lbl, 0.0) for lbl in labels]
        vals_closed = vals + vals[:1]
        ax.plot(angles_closed, vals_closed, color=MPL_ACCENT, linewidth=1.6)
        ax.fill(angles_closed, vals_closed, color=MPL_ACCENT, alpha=0.3)
        ax.plot(angles, vals, "o", color=MPL_ACCENT_800, markersize=2.5, zorder=5)
        for angle, val in zip(angles, vals):
            ax.annotate(
                fmt(val),
                xy=(angle, val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=MPL_ACCENT_800,
                **_mpl_font("bold"),
            )
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8, color=MPL_NEUTRAL_600)
        ax.set_title(subtitle, fontsize=11, color=MPL_TEXT, pad=18, **_mpl_font("bold"))
        ax.tick_params(axis="y", labelsize=7, colors=MPL_NEUTRAL_600)
        ax.spines["polar"].set_color(MPL_NEUTRAL_300)
        ax.grid(color=MPL_NEUTRAL_300)
        if vals:
            ax.set_ylim(0, max(vals) * 1.22)

    fig.suptitle(f"Semantic radar — {dataset_name}", fontsize=12, color=MPL_TEXT, **_mpl_font("bold"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_umap_scatter(
    coords: np.ndarray, point_labels: list[str], dataset_name: str
) -> io.BytesIO:
    """Scatter of the 2D projection, colored by dominant axis, with a
    legend — position/orientation means nothing on its own, only which
    points cluster together does. Colors are an on-brand steel-blue family
    (see `_categorical_palette`) rather than a generic rainbow palette."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(point_labels))
    palette = _categorical_palette(len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    fig, ax = plt.subplots(figsize=(9.8, 5.3), dpi=200)
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(point_labels) if l == lbl]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=7, alpha=0.8,
            color=color_map[lbl], label=lbl, linewidths=0,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"UMAP — {dataset_name}", fontsize=11, color=MPL_TEXT, loc="left", **_mpl_font("bold"))
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False,
        title="Axis", title_fontsize=9, markerscale=1.6, labelcolor=MPL_NEUTRAL_600,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
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


def _sample_thumbnail_row(
    doc_width: float, paths: list[Path], styles: dict, n_cols: int = 6, thumb_size: float = None
) -> Table | None:
    """
    A single row of small thumbnails with the filename underneath — used
    for the Dataset composition 'samples' rows (low-fit / near-duplicate).
    Each thumbnail is framed as a blueprint object, same as every other
    figure in the report.

    Fixed WIDTH per column, height computed from each image's own aspect
    ratio (never distorted/stretched to a square), top-aligned so images
    of different proportions all start flush at the top of the row.
    """
    thumb_w = (thumb_size or (doc_width / n_cols - 6)) - 14  # leave room for the frame
    cells = []
    for p in paths[:n_cols]:
        buf = _thumbnail_bytes(p, max_size=300)
        if buf is None:
            continue
        framed = _Blueprint(_rl_image_for_width(buf, thumb_w), pad=3, margin=6)
        cells.append([framed, Paragraph(p.name, styles["caption"])])
    if not cells:
        return None
    while len(cells) < n_cols:
        cells.append(["", ""])

    table = Table(
        [[c[0] for c in cells], [c[1] for c in cells]],
        colWidths=[doc_width / n_cols] * n_cols,
    )
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
            ]
        )
    )
    return table


def generate_pdf_report(
    dataset_name: str,
    overview: DatasetOverviewMetrics,
    representative_images: list[tuple[str, Path]],
    low_fit_samples: list[Path] | None = None,
    duplicate_samples: list[Path] | None = None,
    radar_dominance_values: dict[str, float] | None = None,
    radar_normalized_values: dict[str, float] | None = None,
    umap_coords: np.ndarray | None = None,
    umap_labels: list[str] | None = None,
    clip_mmd: float | None = None,
    clip_mmd_baseline: float | None = None,
    compare_dataset_name: str | None = None,
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

    doc = BaseDocTemplate(
        buffer,
        pagesize=PORTRAIT_SIZE,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=f"Semantic Report by sneaky\u2122 — {dataset_name}",
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
    story.append(Paragraph(dataset_name, styles["title"]))
    story.append(Spacer(1, SP_1))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            f"{overview.total_images:,} images",
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
    callout_lines = []
    if overview.unclassified_count > 0:
        callout_lines.append(
            f"<b>{overview.unclassified_count:,} images have low semantic fit.</b> "
            "They don't strongly match any of the detected themes — worth a quick "
            "look to confirm they belong in this dataset."
        )
    if overview.visual_duplicates_count > 0:
        callout_lines.append(
            f"<b>{overview.visual_duplicates_count:,} images belong to near-duplicate "
            "groups.</b> This may be expected (e.g. burst-mode photos) or a sign of "
            "redundancy, depending on what this dataset is for."
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
            colWidths=[portrait_w - 32],
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
    cards = [
        _metric_card(f"{overview.total_images:,}", "Total images", styles),
        _metric_card(f"{overview.semantic_coverage_pct:.0f}%", "Semantic coverage", styles),
        _metric_card(f"{overview.unclassified_pct:.1f}%", "Low semantic fit", styles),
        _metric_card(f"{overview.visual_duplicates_pct:.1f}%", "Visual redundancy", styles),
        _metric_card(str(overview.n_semantic_axes), "Semantic axes", styles),
        _metric_card(
            f"{overview.largest_cluster_pct:.0f}% / {overview.smallest_cluster_pct:.0f}%",
            "Largest / smallest cluster",
            styles,
        ),
    ]
    n_cols = 3
    rows = [cards[i : i + n_cols] for i in range(0, len(cards), n_cols)]
    kpi_table = Table(rows, colWidths=[portrait_w / n_cols] * n_cols, hAlign="LEFT")
    kpi_table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, SP_4))

    # --- Donut charts -----------------------------------------------
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
    chart_w = portrait_w * 0.30
    donut_table = Table(
        [
            [
                _Blueprint(_rl_image_for_width(coverage_buf, chart_w), pad=4, margin=7),
                _Blueprint(_rl_image_for_width(duplicates_buf, chart_w), pad=4, margin=7),
            ]
        ],
        colWidths=[portrait_w / 2, portrait_w / 2],
    )
    donut_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(donut_table)

    # --- Axis size bar chart -----------------------------------------
    if overview.axis_sizes:
        bar_buf = _render_bar_chart(overview.axis_sizes)
        story.append(Spacer(1, SP_2))
        story.append(_Blueprint(_rl_image_for_width(bar_buf, portrait_w - 26), pad=6, margin=7))

    # --- Representative images ----------------------------------------
    if representative_images:
        story.append(Spacer(1, SP_4))
        story.append(_section_header("What's in this dataset", styles, portrait_w))
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
            f"Low semantic fit — {overview.unclassified_count:,} images "
            f"({overview.unclassified_pct:.1f}%)",
            styles["section"],
        )
    )
    story.append(Spacer(1, SP_2))
    if low_fit_samples:
        row = _sample_thumbnail_row(portrait_w, low_fit_samples, styles)
        if row:
            story.append(row)
    story.append(Spacer(1, SP_4))

    story.append(
        Paragraph(
            f"Near duplicates — {overview.visual_duplicates_count:,} images "
            f"({overview.visual_duplicates_pct:.1f}%)",
            styles["section"],
        )
    )
    story.append(Spacer(1, SP_2))
    if duplicate_samples:
        row = _sample_thumbnail_row(portrait_w, duplicate_samples, styles)
        if row:
            story.append(row)
    story.append(Spacer(1, SP_4))

    # Treemap — deliberately a different chart TYPE from "Axis sizes"
    # earlier in the report, so the two don't look like duplicates.
    if overview.axis_sizes:
        treemap_buf = _render_treemap(overview.axis_sizes)
        story.append(_Blueprint(_rl_image_for_width(treemap_buf, portrait_w - 26), pad=6, margin=7))

    # --- Landscape: dual radar comparison -----------------------------
    if radar_dominance_values and radar_normalized_values:
        story.append(NextPageTemplate("Landscape"))
        story.append(PageBreak())
        radar_labels = sorted(set(radar_dominance_values) | set(radar_normalized_values))
        radar_buf = _render_dual_radar(
            radar_labels, radar_dominance_values, radar_normalized_values, dataset_name
        )
        radar_frame = _Blueprint(_rl_image_for_width(radar_buf, landscape_w - 26), pad=6, margin=7)
        radar_frame.wrap(landscape_w, landscape_h)
        story.append(Spacer(1, max(0, (landscape_h - radar_frame.height) / 2)))
        story.append(radar_frame)

    # --- Landscape: UMAP scatter ---------------------------------------
    if umap_coords is not None and umap_labels:
        story.append(NextPageTemplate("Landscape"))
        story.append(PageBreak())
        umap_buf = _render_umap_scatter(umap_coords, umap_labels, dataset_name)
        umap_frame = _Blueprint(_rl_image_for_width(umap_buf, landscape_w - 26), pad=6, margin=7)
        umap_frame.wrap(landscape_w, landscape_h)
        story.append(Spacer(1, max(0, (landscape_h - umap_frame.height) / 2)))
        story.append(umap_frame)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
