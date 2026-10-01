"""Reusable Cartesian scatter and bubble-chart renderer."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox, offset_copy
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from ..config import ScatterConfig
from ..data import validate_scatter_data
from ..presets.palette import COMMON_CHART_PALETTE, GRID, INK
from ..presets.scatter_standard import SCATTER_STANDARD
from ..style import paper_style
from ._cartesian_layout import (
    LayoutDrawBudget,
    fit_top_legend,
    inside,
    legend_column_candidates,
    legend_label_variants,
    metadata,
    pairwise_clear,
    semantic_obstacles,
    wrapped,
)


_PRESET_KEYS = frozenset({"grid_alpha", "legend_y"})
_DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def _validated_preset(preset: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(preset, Mapping):
        raise ValueError("preset must be a mapping")
    values = dict(preset)
    if any(not isinstance(key, str) for key in values):
        raise ValueError("preset keys must all be strings")
    if set(values) != _PRESET_KEYS:
        missing = sorted(_PRESET_KEYS - set(values))
        extra = sorted(set(values) - _PRESET_KEYS)
        raise ValueError(
            f"preset keys must match exactly; missing={missing}, extra={extra}"
        )
    for key, value in values.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not isfinite(value)
        ):
            raise ValueError(f"preset {key} must be a finite number")
    valid = {
        "grid_alpha": 0 <= values["grid_alpha"] <= 1,
        "legend_y": 1 <= values["legend_y"] <= 2,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _series_colors(config: ScatterConfig) -> dict[str, str]:
    explicit = dict(config.series_colors)
    missing = [
        series
        for index, series in enumerate(config.series_order)
        if index >= len(COMMON_CHART_PALETTE) and series not in explicit
    ]
    if missing:
        raise ValueError(
            "series require explicit color mappings beyond the eight-color common "
            f"palette: {missing}"
        )
    return {
        series: (
            explicit[series]
            if series in explicit
            else COMMON_CHART_PALETTE[index]
        )
        for index, series in enumerate(config.series_order)
    }


def _series_markers(config: ScatterConfig) -> dict[str, str]:
    explicit = dict(config.markers)
    return {
        series: explicit.get(series, _DEFAULT_MARKERS[index % len(_DEFAULT_MARKERS)])
        for index, series in enumerate(config.series_order)
    }


def _marker_style(marker: str) -> tuple[MarkerStyle, bool]:
    style = MarkerStyle(marker)
    path = style.get_path().transformed(style.get_transform())
    vertices = np.asarray(path.vertices, dtype=float)
    drawn = bool(vertices.size and np.isfinite(vertices).all())
    return style, drawn


def _effective_rgba(color: str, opacity: float) -> tuple[float, float, float, float]:
    red, green, blue, alpha = mcolors.to_rgba(color)
    return red, green, blue, alpha * opacity


def _edge_rgba(color: str, opacity: float) -> tuple[float, float, float, float]:
    red, green, blue, alpha = mcolors.to_rgba(color)
    return red * 0.55, green * 0.55, blue * 0.55, alpha * opacity


def _canonical_frame(frame: pd.DataFrame, series_order: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["__series_rank"] = result["series"].map(
        {series: index for index, series in enumerate(series_order)}
    )
    columns = ["__series_rank", "x", "y"]
    if "label" in result:
        columns.append("label")
    if "size" in result:
        columns.append("size")
    return (
        result.sort_values(columns, kind="stable")
        .drop(columns="__series_rank")
        .reset_index(drop=True)
    )


def _marker_areas(frame: pd.DataFrame, config: ScatterConfig) -> np.ndarray:
    if "size" not in frame:
        return np.full(len(frame), config.default_marker_area, dtype=float)
    values = frame["size"]
    raw_values = values.to_numpy(copy=False)
    exact_integers = is_integer_dtype(values.dtype) or all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in raw_values
    )
    if exact_integers:
        integers = tuple(int(value) for value in raw_values)
        minimum = min(integers)
        maximum = max(integers)
        if minimum == maximum:
            midpoint = config.bubble_area_min + (
                config.bubble_area_max - config.bubble_area_min
            ) / 2
            return np.full(len(frame), midpoint, dtype=float)
        span = maximum - minimum
        normalised = np.fromiter(
            ((value - minimum) / span for value in integers),
            dtype=float,
            count=len(integers),
        )
        return config.bubble_area_min + normalised * (
            config.bubble_area_max - config.bubble_area_min
        )

    raw = values.to_numpy(dtype=float)
    minimum = float(raw.min())
    maximum = float(raw.max())
    if minimum == maximum:
        midpoint = config.bubble_area_min + (
            config.bubble_area_max - config.bubble_area_min
        ) / 2
        return np.full(
            len(frame),
            midpoint,
            dtype=float,
        )
    normalised = (raw - minimum) / (maximum - minimum)
    return config.bubble_area_min + normalised * (
        config.bubble_area_max - config.bubble_area_min
    )


def _artist_has_visible_color(collection) -> bool:
    if not collection.get_visible():
        return False
    alpha = collection.get_alpha()
    if alpha is not None and not np.asarray(alpha, dtype=float).clip(min=0).any():
        return False
    return any(
        len(colors) and np.any(np.asarray(colors)[:, 3] > 0)
        for colors in (collection.get_facecolors(), collection.get_edgecolors())
    )


def _collection_point_boxes(collection, axis, figure: Figure) -> list[Bbox]:
    """Return conservative display-space boxes for actually visible markers."""
    if not _artist_has_visible_color(collection):
        return []
    paths = collection.get_paths()
    if not paths:
        return []
    path_box = paths[0].get_extents()
    if not np.isfinite(path_box.extents).all():
        return []
    offsets = axis.transData.transform(collection.get_offsets())
    sizes = collection.get_sizes()
    widths = collection.get_linewidths()
    boxes: list[Bbox] = []
    for index, ((x, y), area) in enumerate(zip(offsets, sizes)):
        scale = np.sqrt(float(area)) * figure.dpi / 72
        width = float(widths[index % len(widths)]) if len(widths) else 0.0
        padding = width * figure.dpi / 144 + 1.0
        boxes.append(
            Bbox.from_extents(
                x + path_box.x0 * scale - padding,
                y + path_box.y0 * scale - padding,
                x + path_box.x1 * scale + padding,
                y + path_box.y1 * scale + padding,
            )
        )
    return boxes


def _point_radius_points(marker: str, area: float, edge_width: float) -> float:
    style, drawn = _marker_style(marker)
    if not drawn:
        return 0.0
    box = style.get_path().transformed(style.get_transform()).get_extents()
    extent = max(abs(box.x0), abs(box.x1), abs(box.y0), abs(box.y1))
    return extent * np.sqrt(area) + edge_width / 2


def _number_label(value: float, decimals: int) -> str:
    magnitude = f"{abs(value):.{decimals}f}"
    return f"−{magnitude}" if value < 0 else magnitude


def _point_label(x: float, y: float, label: str, decimals: int) -> str:
    if label:
        return label
    return f"({_number_label(x, decimals)}, {_number_label(y, decimals)})"


def _fit_point_labels(
    figure: Figure,
    axis,
    texts: list,
    collections: list,
    radius_points: Mapping[tuple[object, ...], float],
    offset_points: float,
    draw_budget: LayoutDrawBudget,
) -> None:
    if not texts:
        return
    point_boxes = [
        box
        for collection in collections
        for box in _collection_point_boxes(collection, axis, figure)
    ]
    obstacles = semantic_obstacles(axis)
    x_offset = axis.xaxis.get_offset_text()
    if x_offset.get_visible() and x_offset.get_text().strip():
        obstacles.append(x_offset)
    for font_size in (7.0, 6.5, 6.0, 5.5):
        draw_budget.draw()
        renderer = draw_budget.renderer
        obstacle_boxes = [artist.get_window_extent(renderer) for artist in obstacles]
        placed: list[Bbox] = []
        success = True
        for text in texts:
            text.set_fontsize(font_size)
            distance = radius_points[text._polarviz_key] + offset_points + 2.0
            found = False
            for horizontal, vertical in (
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ):
                text.set_transform(
                    offset_copy(
                        axis.transData,
                        fig=figure,
                        x=horizontal * distance,
                        y=vertical * distance,
                        units="points",
                    )
                )
                text.set_ha(
                    "left" if horizontal > 0 else "right" if horizontal < 0 else "center"
                )
                text.set_va(
                    "bottom" if vertical > 0 else "top" if vertical < 0 else "center"
                )
                box = text.get_window_extent(renderer)
                if not inside(text, figure, renderer):
                    continue
                if (
                    box.x0 < axis.bbox.x0
                    or box.x1 > axis.bbox.x1
                    or box.y0 < axis.bbox.y0
                    or box.y1 > axis.bbox.y1
                ):
                    continue
                if any(box.overlaps(other) for other in placed):
                    continue
                if any(box.overlaps(point) for point in point_boxes):
                    continue
                if any(box.overlaps(obstacle) for obstacle in obstacle_boxes):
                    continue
                placed.append(box)
                found = True
                break
            if not found:
                success = False
                break
        if success:
            draw_budget.draw()
            renderer = draw_budget.renderer
            boxes = [text.get_window_extent(renderer) for text in texts]
            obstacle_boxes = [artist.get_window_extent(renderer) for artist in obstacles]
            if (
                all(inside(text, figure, renderer) for text in texts)
                and all(
                    box.x0 >= axis.bbox.x0
                    and box.x1 <= axis.bbox.x1
                    and box.y0 >= axis.bbox.y0
                    and box.y1 <= axis.bbox.y1
                    for box in boxes
                )
                and all(
                    not left.overlaps(right)
                    for index, left in enumerate(boxes)
                    for right in boxes[index + 1 :]
                )
                and not any(box.overlaps(point) for box in boxes for point in point_boxes)
                and not any(
                    box.overlaps(obstacle)
                    for box in boxes
                    for obstacle in obstacle_boxes
                )
            ):
                return
    raise ValueError(
        "scatter chart layout cannot fit point labels clear of bubbles and "
        "semantic decorations inside figure_size"
    )


def _legend(axis, handles, series_order, labels, columns, font_size, legend_y):
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
        handletextpad=0.45,
        columnspacing=1.1,
        fontsize=font_size,
    )
    for handle, series in zip(legend.legend_handles, series_order):
        metadata(handle, "legend-key", (series,), "polarviz-scatter-legend-key")
    return legend


def _fit_semantic_layout(
    figure: Figure,
    axis,
    config: ScatterConfig,
    appearance: Mapping[str, float],
    handles: list,
    draw_budget: LayoutDrawBudget,
) -> None:
    width, _ = figure.get_size_inches()
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

    figure.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.74)
    draw_budget.draw()
    renderer = draw_budget.renderer
    outer = figure.bbox
    x_ticks = [tick for tick in axis.get_xticklabels() if tick.get_visible()]
    y_ticks = [tick for tick in axis.get_yticklabels() if tick.get_visible()]
    left_artists = [*y_ticks]
    if axis.get_ylabel():
        left_artists.append(axis.yaxis.label)
    left_depth = axis.bbox.x0 - min(
        (artist.get_window_extent(renderer).x0 for artist in left_artists),
        default=axis.bbox.x0,
    )
    bottom_artists = [*x_ticks]
    if axis.get_xlabel():
        bottom_artists.append(axis.xaxis.label)
    x_offset = axis.xaxis.get_offset_text()
    if x_offset.get_visible() and x_offset.get_text().strip():
        bottom_artists.append(x_offset)
    bottom_depth = axis.bbox.y0 - min(
        (artist.get_window_extent(renderer).y0 for artist in bottom_artists),
        default=axis.bbox.y0,
    )
    left = max(0.08, (left_depth + 8) / outer.width)
    right_depth = max(
        (
            artist.get_window_extent(renderer).x1 - axis.bbox.x1
            for artist in bottom_artists
        ),
        default=0,
    )
    right = min(0.98, 1 - (right_depth + 8) / outer.width)
    bottom = max(0.10, (bottom_depth + 8) / outer.height)
    if left + 0.22 >= right or bottom >= 0.60:
        raise ValueError(
            "scatter chart layout cannot fit ticks and axis labels inside figure_size"
        )

    candidates = legend_column_candidates(
        len(config.series_order), min(config.legend_columns, len(config.series_order))
    )
    variants = legend_label_variants(
        config.series_order,
        tuple(dict.fromkeys((max(8, min(28, int(width * 5))), 18, 14, 10, 8))),
    )

    def top_decoration_height(renderer) -> float:
        offset = axis.yaxis.get_offset_text()
        if not offset.get_visible() or not offset.get_text().strip():
            return 0
        return max(0, offset.get_window_extent(renderer).y1 - axis.bbox.y1)

    def scatter_semantic_layout_fits(figure, axis, renderer) -> bool:
        artists = semantic_obstacles(axis)
        x_offset = axis.xaxis.get_offset_text()
        if x_offset.get_visible() and x_offset.get_text().strip():
            artists.append(x_offset)
        return all(inside(artist, figure, renderer) for artist in artists) and pairwise_clear(
            artists, figure, renderer
        )

    fit_top_legend(
        figure=figure,
        axis=axis,
        handles=handles,
        series_order=config.series_order,
        label_variants=variants,
        column_candidates=candidates,
        title_sizes=(10, 9, 8, 7) if config.title else (10,),
        preset_anchor=min(
            0.985, max(0.80, 0.98 + (appearance["legend_y"] - 1.16) * 0.25)
        ),
        left=left,
        right=right,
        bottom=bottom,
        semantic_layout_fits=scatter_semantic_layout_fits,
        legend_builder=_legend,
        top_decoration_height=top_decoration_height,
        draw_budget=draw_budget,
        layout_name="scatter chart",
    )


def plot_scatter_chart(
    data: pd.DataFrame,
    config: ScatterConfig,
    preset: Mapping[str, float] = SCATTER_STANDARD,
) -> Figure:
    """Plot grouped scatter data, optionally mapping a positive size column.

    Non-empty point labels are rendered verbatim.  An empty label cell is
    rendered as an ``(x, y)`` coordinate pair using ``point_label_decimals``;
    omitting the label column still disables point labels entirely.
    """
    if not isinstance(config, ScatterConfig):
        raise ValueError("config must be a ScatterConfig")
    frame = _canonical_frame(
        validate_scatter_data(data, config.series_order), config.series_order
    )
    if (
        (frame["x"] < config.x_min)
        | (frame["x"] > config.x_max)
        | (frame["y"] < config.y_min)
        | (frame["y"] > config.y_max)
    ).any():
        raise ValueError(
            f"x and y must lie within [{config.x_min}, {config.x_max}] and "
            f"[{config.y_min}, {config.y_max}]"
        )
    appearance = _validated_preset(preset)
    colors = _series_colors(config)
    markers = _series_markers(config)
    areas = _marker_areas(frame, config)

    active_figure = plt.gcf() if plt.get_fignums() else None
    figure: Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(figsize=config.figure_size)
            draw_budget = LayoutDrawBudget(figure, "scatter chart")
            collections = []
            point_texts = []
            point_radii: dict[tuple[object, ...], float] = {}
            for series in config.series_order:
                mask = frame["series"] == series
                rows = frame.loc[mask]
                point_areas = areas[np.asarray(mask)]
                marker_style, marker_drawn = _marker_style(markers[series])
                scatter_colors = (
                    {"color": "none"}
                    if not marker_drawn
                    else {
                        "color": _effective_rgba(
                            colors[series], config.marker_alpha
                        )
                    }
                    if not marker_style.is_filled()
                    else {
                        "facecolors": [
                            _effective_rgba(colors[series], config.marker_alpha)
                        ],
                        "edgecolors": [
                            _edge_rgba(colors[series], config.marker_alpha)
                        ],
                    }
                )
                collection = axis.scatter(
                    rows["x"].to_numpy(dtype=float),
                    rows["y"].to_numpy(dtype=float),
                    s=point_areas,
                    marker=markers[series],
                    linewidths=[config.edge_width],
                    zorder=3,
                    **scatter_colors,
                )
                metadata(
                    collection,
                    "scatter-series",
                    (series,),
                    "polarviz-scatter-series",
                )
                collections.append(collection)
                if config.show_point_labels and "label" in rows:
                    for row, area in zip(rows.itertuples(index=False), point_areas):
                        explicit_label = str(row.label)
                        label = _point_label(
                            float(row.x),
                            float(row.y),
                            explicit_label,
                            config.point_label_decimals,
                        )
                        key = (series, float(row.x), float(row.y), explicit_label)
                        text = axis.text(
                            float(row.x),
                            float(row.y),
                            label,
                            color=INK,
                            fontsize=7,
                            clip_on=False,
                            zorder=4,
                        )
                        metadata(
                            text,
                            "scatter-point-label",
                            key,
                            "polarviz-scatter-point-label",
                        )
                        point_texts.append(text)
                        point_radii[key] = (
                            _point_radius_points(
                                markers[series], float(area), config.edge_width
                            )
                            if _artist_has_visible_color(collection)
                            else 0.0
                        )

            axis.set_xlim(config.x_min, config.x_max)
            axis.set_ylim(config.y_min, config.y_max)
            axis.set_xticks(config.x_ticks)
            axis.set_yticks(config.y_ticks)
            axis.set_axisbelow(True)
            axis.grid(
                True,
                color=GRID,
                linewidth=0.65,
                alpha=appearance["grid_alpha"],
            )
            for name in ("right", "top"):
                axis.spines[name].set_visible(False)
            for name in ("left", "bottom"):
                axis.spines[name].set_color(GRID)
                axis.spines[name].set_linewidth(0.65)
            axis.tick_params(colors=INK)
            _fit_semantic_layout(
                figure, axis, config, appearance, collections, draw_budget
            )
            _fit_point_labels(
                figure,
                axis,
                point_texts,
                collections,
                point_radii,
                config.point_label_offset_points,
                draw_budget,
            )
            axis._polarviz_layout_draws = draw_budget.draws
            axis._polarviz_series_colors = tuple(
                colors[series] for series in config.series_order
            )
            return figure
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    finally:
        if active_figure is not None and plt.fignum_exists(active_figure.number):
            plt.figure(active_figure.number)
