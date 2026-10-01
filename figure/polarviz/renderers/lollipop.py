"""Reusable grouped Cartesian lollipop-chart renderer."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, offset_copy
import numpy as np
import pandas as pd

from ..config import LollipopConfig
from ..data import validate_lollipop_data
from ..presets.lollipop_standard import LOLLIPOP_STANDARD
from ..presets.palette import COMMON_CHART_PALETTE, GRID, INK
from ..style import paper_style
from ._cartesian_layout import (
    LayoutDrawBudget,
    fit_category_semantic_layout,
    inside,
    metadata,
    semantic_obstacles,
)


_PRESET_KEYS = frozenset(
    {
        "grid_alpha",
        "stem_alpha",
        "marker_alpha",
        "edge_width",
        "label_offset_points",
        "legend_y",
        "group_fill_fraction",
    }
)


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
        "stem_alpha": 0 <= values["stem_alpha"] <= 1,
        "marker_alpha": 0 <= values["marker_alpha"] <= 1,
        "edge_width": values["edge_width"] > 0,
        "label_offset_points": values["label_offset_points"] >= 0,
        "legend_y": 1 <= values["legend_y"] <= 2,
        "group_fill_fraction": 0 < values["group_fill_fraction"] <= 0.9,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _series_colors(config: LollipopConfig) -> dict[str, str]:
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


def _effective_rgba(color: str, opacity: float) -> tuple[float, float, float, float]:
    red, green, blue, alpha = mcolors.to_rgba(color)
    return red, green, blue, alpha * opacity


def _effective_edge_rgba(
    color: str, opacity: float
) -> tuple[float, float, float, float]:
    red, green, blue, alpha = mcolors.to_rgba(color)
    return red * 0.55, green * 0.55, blue * 0.55, alpha * opacity


def _value_label(value: float, decimals: int) -> str:
    magnitude = f"{abs(value):.{decimals}f}"
    return f"−{magnitude}" if value < 0 else magnitude


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
        handlelength=1.4,
        columnspacing=1.1,
        fontsize=font_size,
    )
    for handle, series in zip(legend.legend_handles, series_order):
        metadata(handle, "legend-key", (series,), "polarviz-lollipop-legend-key")
    return legend


def _marker_boxes(markers, axis, figure) -> list[Bbox]:
    boxes: list[Bbox] = []
    for marker in markers:
        if not _artist_has_visible_color(
            marker, (marker.get_facecolors(), marker.get_edgecolors())
        ):
            continue
        center = axis.transData.transform(marker.get_offsets()[0])
        edge = marker.get_linewidths()[0] if len(marker.get_linewidths()) else 0.0
        radius = (np.sqrt(float(marker.get_sizes()[0])) / 2 + edge / 2) * figure.dpi / 72
        radius += 1.0
        boxes.append(
            Bbox.from_extents(
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            )
        )
    return boxes


def _stem_boxes(stems, axis, figure) -> list[Bbox]:
    boxes: list[Bbox] = []
    for stem in stems:
        if not _artist_has_visible_color(stem, (stem.get_color(),)):
            continue
        points = axis.transData.transform(
            np.column_stack((stem.get_xdata(), stem.get_ydata()))
        )
        radius = stem.get_linewidth() * figure.dpi / 144 + 1.0
        boxes.append(
            Bbox.from_extents(
                points[:, 0].min() - radius,
                points[:, 1].min() - radius,
                points[:, 0].max() + radius,
                points[:, 1].max() + radius,
            )
        )
    return boxes


def _artist_has_visible_color(artist, colors) -> bool:
    if not artist.get_visible():
        return False
    alpha = artist.get_alpha()
    if alpha is not None and not np.asarray(alpha, dtype=float).clip(min=0).any():
        return False
    for color in colors:
        rgba = mcolors.to_rgba_array(color)
        if len(rgba) and np.any(rgba[:, 3] > 0):
            return True
    return False


def _fit_value_labels(
    figure: Figure,
    axis,
    texts,
    markers,
    stems,
    offset_points: float,
    baseline: float,
    draw_budget: LayoutDrawBudget,
) -> None:
    if not texts:
        return
    marker_boxes = _marker_boxes(markers, axis, figure)
    stem_boxes = _stem_boxes(stems, axis, figure)
    obstacles = semantic_obstacles(axis)
    visible_markers = [
        marker
        for marker in markers
        if _artist_has_visible_color(
            marker, (marker.get_facecolors(), marker.get_edgecolors())
        )
    ]
    marker_radius_points = max(
        (np.sqrt(float(marker.get_sizes()[0])) / 2 for marker in visible_markers),
        default=0.0,
    )
    distance = marker_radius_points + offset_points + 2.0
    for font_size in (7.0, 6.5, 6.0, 5.5):
        draw_budget.draw()
        renderer = draw_budget.renderer
        obstacle_boxes = [artist.get_window_extent(renderer) for artist in obstacles]
        placed: list[Bbox] = []
        success = True
        for text in texts:
            text.set_fontsize(font_size)
            value = float(text.get_position()[1])
            outward = 1 if value >= baseline else -1
            inward = -outward
            preferred = (
                (0, outward),
                (1, outward),
                (-1, outward),
                (1, 0),
                (-1, 0),
                (1, inward),
                (-1, inward),
                (0, inward),
            )
            found = False
            for horizontal, vertical in preferred:
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
                    "bottom"
                    if vertical > 0
                    else "top"
                    if vertical < 0
                    else "bottom"
                    if outward < 0
                    else "top"
                )
                box = text.get_window_extent(renderer)
                if not inside(text, figure, renderer):
                    continue
                if box.y0 < axis.bbox.y0 or box.y1 > axis.bbox.y1:
                    continue
                if any(box.overlaps(other) for other in placed):
                    continue
                if any(box.overlaps(marker) for marker in marker_boxes):
                    continue
                if any(
                    box.overlaps(stem)
                    for stem in stem_boxes
                ):
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
            obstacle_boxes = [
                artist.get_window_extent(renderer) for artist in obstacles
            ]
            if (
                all(inside(text, figure, renderer) for text in texts)
                and all(
                    not left.overlaps(right)
                    for index, left in enumerate(boxes)
                    for right in boxes[index + 1 :]
                )
                and not any(
                    box.overlaps(marker)
                    for box in boxes
                    for marker in marker_boxes
                )
                and not any(
                    box.overlaps(stem)
                    for box in boxes
                    for stem in stem_boxes
                )
                and not any(
                    box.overlaps(obstacle)
                    for box in boxes
                    for obstacle in obstacle_boxes
                )
            ):
                return
    raise ValueError(
        "lollipop chart layout cannot fit value labels clear of markers, stems, "
        "and semantic decorations inside figure_size"
    )


def plot_lollipop_chart(
    data: pd.DataFrame,
    config: LollipopConfig,
    preset: Mapping[str, float] = LOLLIPOP_STANDARD,
) -> Figure:
    """Plot complete rectangular category-by-series data as grouped lollipops."""
    if not isinstance(config, LollipopConfig):
        raise ValueError("config must be a LollipopConfig")
    frame = validate_lollipop_data(
        data, config.category_order, config.series_order
    )
    if ((frame["value"] < config.y_min) | (frame["value"] > config.y_max)).any():
        raise ValueError(f"value must lie within [{config.y_min}, {config.y_max}]")
    appearance = _validated_preset(preset)
    colors = _series_colors(config)
    display_map = dict(config.category_labels)
    display_labels = [
        display_map.get(category, category) for category in config.category_order
    ]
    value_lookup = {
        key: float(value)
        for key, value in zip(
            frame[["category", "series"]].itertuples(index=False, name=None),
            frame["value"],
        )
    }

    active_figure = plt.gcf() if plt.get_fignums() else None
    figure: Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(figsize=config.figure_size)
            draw_budget = LayoutDrawBudget(figure, "lollipop chart")
            centers = np.arange(len(config.category_order), dtype=float)
            if len(config.series_order) == 1:
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(
                    -appearance["group_fill_fraction"] / 2,
                    appearance["group_fill_fraction"] / 2,
                    len(config.series_order),
                )
            stems = []
            markers = []
            value_texts = []
            handles = []
            for category_index, category in enumerate(config.category_order):
                for series_index, series in enumerate(config.series_order):
                    value = value_lookup[(category, series)]
                    x_value = float(centers[category_index] + offsets[series_index])
                    color = colors[series]
                    stem = axis.plot(
                        [x_value, x_value],
                        [config.baseline, value],
                        color=_effective_rgba(color, appearance["stem_alpha"]),
                        linewidth=config.stem_width,
                        solid_capstyle="round",
                        zorder=2,
                    )[0]
                    metadata(
                        stem,
                        "lollipop-stem",
                        (category, series),
                        "polarviz-lollipop-stem",
                    )
                    stems.append(stem)
                    marker = axis.scatter(
                        [x_value],
                        [value],
                        s=[config.marker_size],
                        facecolors=[
                            _effective_rgba(color, appearance["marker_alpha"])
                        ],
                        edgecolors=[
                            _effective_edge_rgba(
                                color, appearance["marker_alpha"]
                            )
                        ],
                        linewidths=[appearance["edge_width"]],
                        zorder=3,
                    )
                    metadata(
                        marker,
                        "lollipop-marker",
                        (category, series),
                        "polarviz-lollipop-marker",
                    )
                    markers.append(marker)
                    if config.show_value_labels:
                        text = axis.text(
                            x_value,
                            value,
                            _value_label(value, config.value_decimals),
                            color=INK,
                            fontsize=7,
                            clip_on=False,
                            zorder=4,
                        )
                        metadata(
                            text,
                            "lollipop-value-label",
                            (category, series),
                            "polarviz-lollipop-value-label",
                        )
                        value_texts.append(text)

            for series in config.series_order:
                color = colors[series]
                handle = Line2D(
                    [],
                    [],
                    color=_effective_rgba(color, appearance["stem_alpha"]),
                    linewidth=config.stem_width,
                    marker="o",
                    markersize=np.sqrt(config.marker_size),
                    markerfacecolor=_effective_rgba(
                        color, appearance["marker_alpha"]
                    ),
                    markeredgecolor=_effective_edge_rgba(
                        color, appearance["marker_alpha"]
                    ),
                    markeredgewidth=appearance["edge_width"],
                )
                metadata(
                    handle,
                    "legend-key",
                    (series,),
                    "polarviz-lollipop-legend-key",
                )
                handles.append(handle)

            side_margin = max(0.24, appearance["group_fill_fraction"] / 2 + 0.12)
            axis.set_xlim(-side_margin, len(config.category_order) - 1 + side_margin)
            axis.set_ylim(config.y_min, config.y_max)
            axis.set_yticks(config.y_ticks)
            axis.set_xticks(centers, labels=display_labels)
            axis.set_axisbelow(True)
            axis.yaxis.grid(
                True,
                color=GRID,
                linewidth=0.65,
                alpha=appearance["grid_alpha"],
            )
            axis.xaxis.grid(False)
            for name in ("right", "top"):
                axis.spines[name].set_visible(False)
            for name in ("left", "bottom"):
                axis.spines[name].set_color(GRID)
                axis.spines[name].set_linewidth(0.65)
            axis.tick_params(axis="x", length=0, colors=INK)
            axis.tick_params(axis="y", colors=INK)

            fit_category_semantic_layout(
                figure=figure,
                axis=axis,
                config=config,
                appearance=appearance,
                handles=handles,
                display_labels=display_labels,
                draw_budget=draw_budget,
                layout_name="lollipop chart",
                legend_builder=_legend,
            )
            _fit_value_labels(
                figure,
                axis,
                value_texts,
                markers,
                stems,
                appearance["label_offset_points"],
                config.baseline,
                draw_budget,
            )
            axis._polarviz_category_count = len(config.category_order)
            axis._polarviz_series_count = len(config.series_order)
            axis._polarviz_layout_draws = draw_budget.draws
            axis._polarviz_series_colors = tuple(
                colors[series] for series in config.series_order
            )
            axis._polarviz_group_offsets = tuple(float(value) for value in offsets)
            return figure
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    finally:
        if active_figure is not None and plt.fignum_exists(active_figure.number):
            plt.figure(active_figure.number)
