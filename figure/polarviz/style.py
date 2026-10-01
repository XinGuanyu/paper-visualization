"""Local Matplotlib styling helpers for reusable visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import patheffects

from .presets.palette import DARK_STROKES, DEFAULT_SERIES_COLORS


def paper_style():
    """Return a scoped publication-style Matplotlib context manager."""
    return mpl.rc_context(
        rc={
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


def series_color_map(
    series_order: Sequence[str],
    configured: Sequence[tuple[str, str]] = (),
) -> dict[str, str]:
    """Map semantic series to explicit colors, then palette defaults."""
    pairs: list[tuple[str, str]] = []
    for pair in configured:
        if len(pair) != 2:
            raise ValueError("series colors must contain (series, color) pairs")
        pairs.append((pair[0], pair[1]))
    names = [series for series, _ in pairs]
    duplicates = list(dict.fromkeys(
        series for index, series in enumerate(names) if series in names[:index]
    ))
    if duplicates:
        raise ValueError(f"duplicate explicit series color mappings: {duplicates}")
    unknown = [series for series in names if series not in series_order]
    if unknown:
        raise ValueError(f"unknown series color keys: {unknown}")
    explicit = dict(pairs)
    result: dict[str, str] = {}
    for index, series in enumerate(series_order):
        if series in explicit:
            result[series] = explicit[series]
        elif index < len(DEFAULT_SERIES_COLORS):
            result[series] = DEFAULT_SERIES_COLORS[index]
        else:
            raise ValueError(
                "series require explicit color mappings beyond the default palette"
            )
    return result


def _normal_color(color: str) -> str:
    return mcolors.to_hex(color, keep_alpha=False).upper()


def pale_path_effects(color: str, width: float) -> list[patheffects.AbstractPathEffect]:
    """Add contrast beneath the two pale approved series colors."""
    normalized = _normal_color(color)
    dark = DARK_STROKES.get(normalized)
    if dark is None:
        return []
    return [
        patheffects.Stroke(linewidth=width + 1.5, foreground=dark),
        patheffects.Normal(),
    ]


def marker_edge_color(color: str) -> str:
    """Return a dark edge for pale markers and the fill color otherwise."""
    normalized = _normal_color(color)
    return DARK_STROKES.get(normalized, color)
