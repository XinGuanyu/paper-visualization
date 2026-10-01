"""Shared bounded-layout primitives for Cartesian bar renderers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from textwrap import fill as wrap_text

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backend_bases import RendererBase


MAX_LEGEND_ATTEMPTS = 120
MAX_LAYOUT_DRAWS = 72


class LayoutDrawBudget:
    """Bound canvas-independent layout measurements across one renderer call."""

    def __init__(self, figure: plt.Figure, layout_name: str) -> None:
        self.figure = figure
        self.layout_name = layout_name
        self.draws = 0
        self.renderer = RendererAgg(
            figure.bbox.width,
            figure.bbox.height,
            figure.dpi,
        )

    def draw(self) -> None:
        if self.draws >= MAX_LAYOUT_DRAWS:
            raise ValueError(
                f"{self.layout_name} layout exceeded canvas draw budget "
                f"of {MAX_LAYOUT_DRAWS}"
            )
        self.draws += 1
        self.figure.draw(self.renderer)


def metadata(artist, role: str, key: object, gid: str) -> None:
    artist.set_gid(gid)
    artist._polarviz_role = role
    artist._polarviz_key = key


def wrapped(value: str, width: int) -> str:
    return wrap_text(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=True,
    )


def inside(
    artist,
    figure: plt.Figure,
    renderer: RendererBase,
    padding: float = 1.0,
) -> bool:
    box = artist.get_window_extent(renderer)
    outer = figure.bbox
    return (
        box.x0 >= outer.x0 + padding
        and box.y0 >= outer.y0 + padding
        and box.x1 <= outer.x1 - padding
        and box.y1 <= outer.y1 - padding
    )


def pairwise_clear(
    artists,
    figure: plt.Figure,
    renderer: RendererBase,
) -> bool:
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    return all(
        not left.overlaps(right)
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )


def legend_with_metadata(
    axis,
    handles: list,
    series_order: tuple[str, ...],
    labels: list[str],
    columns: int,
    font_size: float,
    legend_y: float,
):
    previous = axis.get_legend()
    if previous is not None:
        previous.remove()
    legend = axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=axis.figure.transFigure,
        ncol=columns,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
        fontsize=font_size,
    )
    for handle, series in zip(legend.legend_handles, series_order):
        metadata(handle, "legend-key", series, "polarviz-legend-key")
    return legend


def legend_column_candidates(
    series_count: int,
    requested_columns: int,
) -> tuple[int, ...]:
    if requested_columns <= 12:
        return tuple(range(requested_columns, 0, -1))
    half = ceil(series_count / 2)
    third = ceil(series_count / 3)
    candidates = {
        *range(1, 13),
        requested_columns,
        requested_columns - 1,
        half,
        half - 1,
        half + 1,
        third,
        third - 1,
        third + 1,
    }
    return tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if 1 <= candidate <= requested_columns
            ),
            reverse=True,
        )
    )


def legend_label_variants(
    series_order: Sequence[str], widths: Sequence[int]
) -> tuple[tuple[str, ...], ...]:
    """Return distinct wrapped-label variants in caller preference order."""
    return tuple(
        dict.fromkeys(
            tuple(wrapped(series, width) for series in series_order)
            for width in widths
        )
    )


def fit_top_legend(
    *,
    figure: plt.Figure,
    axis,
    handles: list,
    series_order: tuple[str, ...],
    label_variants: tuple[tuple[str, ...], ...],
    column_candidates: tuple[int, ...],
    title_sizes: tuple[float, ...],
    preset_anchor: float,
    left: float,
    right: float,
    bottom: float,
    semantic_layout_fits: Callable,
    legend_builder: Callable,
    top_decoration_height: Callable[[RendererBase], float],
    draw_budget: LayoutDrawBudget,
    layout_name: str,
):
    """Fit a top legend with a bounded two-stage geometry search."""
    font_sizes = (7.5, 7.0, 6.5, 6.0, 5.5)
    probe_font_size = font_sizes[-1]
    outer = figure.bbox
    renderer = draw_budget.renderer
    attempts = 0

    def build(labels: tuple[str, ...], columns: int, font_size: float):
        nonlocal attempts
        attempts += 1
        if attempts > MAX_LEGEND_ATTEMPTS:
            raise ValueError(
                f"{layout_name} layout exceeded bounded legend search"
            )
        return legend_builder(
            axis,
            handles,
            series_order,
            list(labels),
            columns,
            font_size,
            preset_anchor,
        )

    title_states: list[tuple[float, float]] = []
    for title_size in title_sizes:
        axis.title.set_fontsize(title_size)
        title_bottom = preset_anchor
        if axis.get_title():
            title_box = axis.title.get_window_extent(renderer)
            if not inside(axis.title, figure, renderer):
                continue
            title_bottom = title_box.y0 / outer.height - 0.015
        legend_anchor = min(preset_anchor, title_bottom)
        if legend_anchor <= bottom + 0.34:
            continue
        title_states.append((title_size, legend_anchor))
    if not title_states:
        raise ValueError(
            f"{layout_name} layout cannot fit title inside figure_size"
        )

    def place(legend) -> bool:
        first_anchor = title_states[0][1]
        legend.set_bbox_to_anchor(
            (0.5, first_anchor), transform=figure.transFigure
        )
        first_box = legend.get_window_extent(renderer)
        if first_box.x0 < outer.x0 + 1.0 or first_box.x1 > outer.x1 - 1.0:
            return False
        for title_size, legend_anchor in title_states:
            legend.set_bbox_to_anchor(
                (0.5, legend_anchor), transform=figure.transFigure
            )
            legend_box = legend.get_window_extent(renderer)
            if (
                legend_box.y0 < outer.y0 + 1.0
                or legend_box.y1 > outer.y1 - 1.0
            ):
                continue
            top = (
                legend_box.y0
                - outer.height * 0.025
                - max(0.0, top_decoration_height(renderer))
            ) / outer.height
            if top <= bottom + 0.30:
                continue
            axis.title.set_fontsize(title_size)
            figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            draw_budget.draw()
            if semantic_layout_fits(figure, axis, renderer):
                return True
        return False

    selected: tuple[tuple[str, ...], int] | None = None
    for columns in column_candidates:
        for labels in label_variants:
            legend = build(labels, columns, probe_font_size)
            if place(legend):
                selected = (labels, columns)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            f"{layout_name} layout cannot fit title, legend, and labels inside "
            "figure_size"
        )

    labels, columns = selected
    for font_size in font_sizes:
        legend = build(labels, columns, font_size)
        if place(legend):
            axis._polarviz_legend_attempts = attempts
            return legend
    raise ValueError(
        f"{layout_name} layout cannot fit title, legend, and labels inside "
        "figure_size"
    )
