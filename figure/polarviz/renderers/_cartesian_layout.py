"""Bounded, backend-independent layout helpers for Cartesian renderers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil, isfinite
from textwrap import fill as wrap_text

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib.pyplot as plt


MAX_LEGEND_ATTEMPTS = 120
MAX_LAYOUT_DRAWS = 72
MIN_LAYOUT_PIXELS = 2.0


class LayoutDrawBudget:
    """Measure a figure without replacing its backend-owned canvas."""

    def __init__(self, figure: plt.Figure, layout_name: str) -> None:
        self.figure = figure
        self.layout_name = layout_name
        self.draws = 0
        width = float(figure.bbox.width)
        height = float(figure.bbox.height)
        if (
            not isfinite(width)
            or not isfinite(height)
            or width < MIN_LAYOUT_PIXELS
            or height < MIN_LAYOUT_PIXELS
        ):
            raise ValueError(
                f"{layout_name} layout requires pixel width and height of at "
                f"least {MIN_LAYOUT_PIXELS:g}"
            )
        self.renderer = RendererAgg(
            int(round(width)),
            int(round(height)),
            figure.dpi,
        )

    def draw(self) -> None:
        if self.draws >= MAX_LAYOUT_DRAWS:
            raise ValueError(
                f"{self.layout_name} layout exceeded canvas draw budget of "
                f"{MAX_LAYOUT_DRAWS}"
            )
        self.draws += 1
        self.figure.draw(self.renderer)


def metadata(artist, role: str, key: object, gid: str) -> None:
    artist.set_gid(gid)
    artist._polarviz_role = role
    artist._polarviz_key = key


def wrapped(value: str, width: int) -> str:
    return wrap_text(value, width=width, break_long_words=True, break_on_hyphens=True)


def inside(artist, figure: plt.Figure, renderer: RendererBase, padding: float = 1.0) -> bool:
    box = artist.get_window_extent(renderer)
    outer = figure.bbox
    return (
        box.x0 >= outer.x0 + padding
        and box.y0 >= outer.y0 + padding
        and box.x1 <= outer.x1 - padding
        and box.y1 <= outer.y1 - padding
    )


def pairwise_clear(artists, figure: plt.Figure, renderer: RendererBase) -> bool:
    del figure
    boxes = [artist.get_window_extent(renderer) for artist in artists if artist.get_visible()]
    return all(
        not left.overlaps(right)
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )


def legend_column_candidates(series_count: int, requested_columns: int) -> tuple[int, ...]:
    requested = min(series_count, requested_columns)
    if requested <= 12:
        return tuple(range(requested, 0, -1))
    half = ceil(series_count / 2)
    third = ceil(series_count / 3)
    candidates = {
        *range(1, 13), requested, requested - 1,
        half, half - 1, half + 1, third, third - 1, third + 1,
    }
    return tuple(sorted((value for value in candidates if 1 <= value <= requested), reverse=True))


def legend_label_variants(series_order: Sequence[str], widths: Sequence[int]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        dict.fromkeys(
            tuple(wrapped(series, width) for series in series_order) for width in widths
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
    """Fit title and legend through a finite, preference-ordered search."""
    font_sizes = (7.5, 7.0, 6.5, 6.0, 5.5)
    renderer = draw_budget.renderer
    outer = figure.bbox
    attempts = 0

    def build(labels: tuple[str, ...], columns: int, font_size: float):
        nonlocal attempts
        attempts += 1
        if attempts > MAX_LEGEND_ATTEMPTS:
            raise ValueError(f"{layout_name} layout exceeded bounded legend search")
        return legend_builder(axis, handles, series_order, list(labels), columns, font_size, preset_anchor)

    title_states: list[tuple[float, float]] = []
    for title_size in title_sizes:
        axis.title.set_fontsize(title_size)
        title_bottom = preset_anchor
        if axis.get_title():
            box = axis.title.get_window_extent(renderer)
            if not inside(axis.title, figure, renderer):
                continue
            title_bottom = box.y0 / outer.height - 0.015
        anchor = min(preset_anchor, title_bottom)
        if anchor > bottom + 0.30:
            title_states.append((title_size, anchor))
    if not title_states:
        raise ValueError(f"{layout_name} layout cannot fit title inside figure_size")

    def place(legend) -> bool:
        for title_size, anchor in title_states:
            axis.title.set_fontsize(title_size)
            legend.set_bbox_to_anchor((0.5, anchor), transform=figure.transFigure)
            box = legend.get_window_extent(renderer)
            if box.x0 < outer.x0 + 1 or box.x1 > outer.x1 - 1:
                continue
            if box.y0 < outer.y0 + 1 or box.y1 > outer.y1 - 1:
                continue
            top = (box.y0 - outer.height * 0.025 - max(0.0, top_decoration_height(renderer))) / outer.height
            if top <= bottom + 0.24:
                continue
            figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            draw_budget.draw()
            if semantic_layout_fits(figure, axis, renderer):
                return True
        return False

    selected: tuple[tuple[str, ...], int] | None = None
    for columns in column_candidates:
        for labels in label_variants:
            legend = build(labels, columns, font_sizes[-1])
            if place(legend):
                selected = labels, columns
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            f"{layout_name} layout cannot fit title, legend, ticks, and axis labels inside figure_size"
        )
    labels, columns = selected
    for font_size in font_sizes:
        legend = build(labels, columns, font_size)
        if place(legend):
            axis._polarviz_legend_attempts = attempts
            return legend
    raise ValueError(f"{layout_name} layout cannot fit legend inside figure_size")


def semantic_obstacles(axis) -> list:
    """Return visible chart decorations that data labels must avoid."""
    artists = [
        *[tick for tick in axis.get_xticklabels() if tick.get_visible()],
        *[tick for tick in axis.get_yticklabels() if tick.get_visible()],
    ]
    legend = axis.get_legend()
    if legend is not None and legend.get_visible():
        artists.append(legend)
    for artist, value in (
        (axis.title, axis.get_title()),
        (axis.xaxis.label, axis.get_xlabel()),
        (axis.yaxis.label, axis.get_ylabel()),
    ):
        if value and artist.get_visible():
            artists.append(artist)
    offset = axis.yaxis.get_offset_text()
    if offset.get_visible() and offset.get_text().strip():
        artists.append(offset)
    return artists


def semantic_layout_fits(figure, axis, renderer) -> bool:
    artists = semantic_obstacles(axis)
    return all(inside(artist, figure, renderer) for artist in artists) and pairwise_clear(
        artists, figure, renderer
    )


def fit_category_semantic_layout(
    *,
    figure,
    axis,
    config,
    appearance,
    handles,
    display_labels,
    draw_budget,
    layout_name: str,
    legend_builder: Callable,
):
    """Fit shared title, legend, ticks, and labels for category charts."""
    width, _ = figure.get_size_inches()
    category_wrap = max(3, min(24, int(width * 12 / len(config.category_order))))
    axis.set_xticklabels([wrapped(label, category_wrap) for label in display_labels])
    axis.set_xlabel(
        wrapped(config.x_label, max(10, min(60, int(width * 9))))
        if config.x_label
        else ""
    )
    axis.set_ylabel(
        wrapped(config.y_label, max(10, min(54, int(width * 8))))
        if config.y_label
        else ""
    )
    axis.set_title(
        wrapped(config.title, max(14, min(68, int(width * 11))))
        if config.title
        else "",
        pad=0,
    )
    if config.title:
        axis._autotitlepos = False
        axis.title.set_transform(figure.transFigure)
        axis.title.set_position((0.5, 0.985))
        axis.title.set_ha("center")
        axis.title.set_va("top")
    ticks = axis.get_xticklabels()
    for index, tick in enumerate(ticks):
        tick.set_rotation(config.x_label_rotation)
        tick.set_rotation_mode("anchor")
        if len(ticks) == 1:
            tick.set_ha("center")
        elif index == 0:
            tick.set_ha("left")
        elif index == len(ticks) - 1:
            tick.set_ha("right")
        else:
            tick.set_ha("center")

    figure.subplots_adjust(left=0.14, right=0.96, bottom=0.17, top=0.74)
    draw_budget.draw()
    renderer = draw_budget.renderer
    outer = figure.bbox
    title_height = (
        axis.title.get_window_extent(renderer).height if axis.get_title() else 0
    )
    if title_height > outer.height * 0.35:
        raise ValueError(f"{layout_name} layout cannot fit title inside figure_size")
    x_boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
    x_height = max((box.height for box in x_boxes), default=0)
    xlabel_height = (
        axis.xaxis.label.get_window_extent(renderer).height if axis.get_xlabel() else 0
    )
    y_boxes = [tick.get_window_extent(renderer) for tick in axis.get_yticklabels()]
    y_width = max((box.width for box in y_boxes), default=0)
    ylabel_width = (
        axis.yaxis.label.get_window_extent(renderer).width if axis.get_ylabel() else 0
    )
    left_artists = [*axis.get_yticklabels()]
    if axis.get_ylabel():
        left_artists.append(axis.yaxis.label)
    left_depth = axis.bbox.x0 - min(
        (
            artist.get_window_extent(renderer).x0
            for artist in left_artists
        ),
        default=axis.bbox.x0,
    )
    left = max(
        0.08,
        (y_width + ylabel_width + 16) / outer.width,
        (left_depth + 8) / outer.width,
    )
    first = x_boxes[0].width / 2 if x_boxes else 0
    last = x_boxes[-1].width / 2 if x_boxes else 0
    left = max(left, (first + 5) / outer.width)
    right = min(0.98, 1 - (last + 5) / outer.width)
    bottom_artists = [*axis.get_xticklabels()]
    if axis.get_xlabel():
        bottom_artists.append(axis.xaxis.label)
    bottom_depth = axis.bbox.y0 - min(
        (
            artist.get_window_extent(renderer).y0
            for artist in bottom_artists
        ),
        default=axis.bbox.y0,
    )
    bottom = max(
        0.10,
        (x_height + xlabel_height + 18) / outer.height,
        (bottom_depth + 8) / outer.height,
    )
    if left + 0.22 >= right or bottom >= 0.60:
        raise ValueError(
            f"{layout_name} layout cannot fit ticks and axis labels inside figure_size"
        )

    requested = min(config.legend_columns, len(config.series_order))
    candidates = legend_column_candidates(len(config.series_order), requested)
    widths = tuple(
        dict.fromkeys((max(8, min(28, int(width * 5))), 18, 14, 10, 8))
    )
    variants = legend_label_variants(config.series_order, widths)
    preset_anchor = min(
        0.985,
        max(0.80, 0.98 + (appearance["legend_y"] - 1.16) * 0.25),
    )

    def top_decoration_height(renderer) -> float:
        offset = axis.yaxis.get_offset_text()
        if not offset.get_visible() or not offset.get_text().strip():
            return 0
        return max(0, offset.get_window_extent(renderer).y1 - axis.bbox.y1)

    return fit_top_legend(
        figure=figure,
        axis=axis,
        handles=handles,
        series_order=config.series_order,
        label_variants=variants,
        column_candidates=candidates,
        title_sizes=(10, 9, 8, 7) if config.title else (10,),
        preset_anchor=preset_anchor,
        left=left,
        right=right,
        bottom=bottom,
        semantic_layout_fits=semantic_layout_fits,
        legend_builder=legend_builder,
        top_decoration_height=top_decoration_height,
        draw_budget=draw_budget,
        layout_name=layout_name,
    )
