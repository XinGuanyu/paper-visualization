"""Grouped vertical bars for Cartesian comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd

from ..config import VerticalBarConfig
from ..data import validate_bar_data
from ..presets.palette import GRID, INK
from ..presets.vertical_grouped_bars import VERTICAL_GROUPED_BARS
from ..style import marker_edge_color, paper_style, series_color_map
from ._bar_layout import (
    LayoutDrawBudget,
    fit_top_legend as _fit_top_legend,
    inside as _inside,
    legend_column_candidates as _legend_column_candidates,
    legend_label_variants as _legend_label_variants,
    legend_with_metadata as _legend_with_metadata,
    metadata as _metadata,
    pairwise_clear as _pairwise_clear,
    wrapped as _wrapped,
)


_PRESET_KEYS = frozenset(
    {
        "grid_alpha",
        "edge_width",
        "bar_alpha",
        "band_alpha_scale",
        "value_label_offset_points",
        "legend_y",
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
        "grid_alpha": 0.0 <= values["grid_alpha"] <= 1.0,
        "edge_width": values["edge_width"] > 0.0,
        "bar_alpha": 0.0 <= values["bar_alpha"] <= 1.0,
        "band_alpha_scale": 0.0 <= values["band_alpha_scale"] <= 1.0,
        "value_label_offset_points": values["value_label_offset_points"] >= 0.0,
        "legend_y": 1.0 <= values["legend_y"] <= 2.0,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _value_label(value: float, decimals: int) -> str:
    magnitude = f"{abs(value):.{decimals}f}"
    return f"−{magnitude}" if value < 0.0 else magnitude


def _semantic_obstacles(axis) -> list:
    artists = [
        *[tick for tick in axis.get_xticklabels() if tick.get_visible()],
        *[tick for tick in axis.get_yticklabels() if tick.get_visible()],
    ]
    legend = axis.get_legend()
    if legend is not None and legend.get_visible():
        artists.append(legend)
    if axis.get_ylabel() and axis.yaxis.label.get_visible():
        artists.append(axis.yaxis.label)
    if axis.get_title() and axis.title.get_visible():
        artists.append(axis.title)
    offset_text = axis.yaxis.get_offset_text()
    if offset_text.get_visible() and offset_text.get_text().strip():
        artists.append(offset_text)
    return artists


def _semantic_layout_fits(figure: plt.Figure, axis, renderer) -> bool:
    artists = _semantic_obstacles(axis)
    legend = axis.get_legend()
    if not all(_inside(artist, figure, renderer) for artist in artists):
        return False
    if not _pairwise_clear(axis.get_xticklabels(), figure, renderer):
        return False
    if not _pairwise_clear(axis.get_yticklabels(), figure, renderer):
        return False
    if not _pairwise_clear(artists, figure, renderer):
        return False
    if axis.get_title() and legend is not None and axis.title.get_window_extent(
        renderer
    ).overlaps(legend.get_window_extent(renderer)):
        return False
    return True


def _fit_semantic_layout(
    figure: plt.Figure,
    axis,
    config: VerticalBarConfig,
    appearance: Mapping[str, float],
    handles: list[Patch],
    display_labels: list[str],
    draw_budget: LayoutDrawBudget,
):
    figure_width, _ = figure.get_size_inches()
    category_wrap = max(
        3,
        min(24, int(figure_width * 12.0 / len(config.category_order))),
    )
    title_wrap = max(14, min(68, int(figure_width * 11.0)))
    axis_wrap = max(12, min(54, int(figure_width * 9.0)))
    legend_wrap = max(8, min(28, int(figure_width * 5.0)))
    axis.set_xticklabels([_wrapped(label, category_wrap) for label in display_labels])
    axis.set_ylabel(_wrapped(config.axis_label, axis_wrap) if config.axis_label else "")
    axis.set_title(_wrapped(config.title, title_wrap) if config.title else "", pad=0.0)
    if config.title:
        axis._autotitlepos = False
        axis.title.set_transform(figure.transFigure)
        axis.title.set_position((0.5, 0.985))
        axis.title.set_horizontalalignment("center")
        axis.title.set_verticalalignment("top")

    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.14, top=0.75)
    draw_budget.draw()
    renderer = draw_budget.renderer
    outer = figure.bbox
    if axis.get_title() and axis.title.get_window_extent(renderer).height > outer.height * 0.38:
        raise ValueError("vertical bar layout cannot fit title inside figure_size")
    x_height = max(
        (tick.get_window_extent(renderer).height for tick in axis.get_xticklabels()),
        default=0.0,
    )
    if x_height > outer.height * 0.48:
        raise ValueError(
            "vertical bar layout cannot fit category labels inside figure_size"
        )
    y_width = max(
        (tick.get_window_extent(renderer).width for tick in axis.get_yticklabels()),
        default=0.0,
    )
    ylabel_width = (
        axis.yaxis.label.get_window_extent(renderer).width
        if axis.get_ylabel()
        else 0.0
    )
    first_half_width = max(
        (
            tick.get_window_extent(renderer).width / 2.0
            for tick in axis.get_xticklabels()[:1]
        ),
        default=0.0,
    )
    last_half_width = max(
        (
            tick.get_window_extent(renderer).width / 2.0
            for tick in axis.get_xticklabels()[-1:]
        ),
        default=0.0,
    )
    left = max(
        0.12,
        (y_width + ylabel_width + 18.0) / outer.width,
        (first_half_width + 8.0) / outer.width,
    )
    right = min(0.97, 1.0 - (last_half_width + 8.0) / outer.width)
    offset_text = axis.yaxis.get_offset_text()
    left_artists = [*axis.get_yticklabels()]
    if axis.get_ylabel():
        left_artists.append(axis.yaxis.label)
    if offset_text.get_visible() and offset_text.get_text().strip():
        left_artists.append(offset_text)
    if left_artists:
        decoration_width = axis.bbox.x0 - min(
            artist.get_window_extent(renderer).x0 for artist in left_artists
        )
        left = max(left, (decoration_width + 2.0) / outer.width)
    bottom = max(0.10, (x_height + 12.0) / outer.height)
    if left + 0.24 >= right or bottom >= 0.56:
        raise ValueError("vertical bar layout cannot fit labels inside figure_size")

    requested_columns = min(config.legend_columns, len(config.series_order))
    column_candidates = _legend_column_candidates(
        len(config.series_order), requested_columns
    )
    legend_widths = tuple(dict.fromkeys((legend_wrap, 18, 14, 10, 8)))
    label_variants = _legend_label_variants(
        config.series_order, legend_widths
    )
    title_sizes = (10.0, 9.0, 8.0, 7.0) if config.title else (10.0,)
    preset_anchor = min(
        0.985,
        max(0.80, 0.98 + (float(appearance["legend_y"]) - 1.16) * 0.25),
    )
    def top_decoration_height(renderer) -> float:
        offset = axis.yaxis.get_offset_text()
        if not offset.get_visible() or not offset.get_text().strip():
            return 0.0
        return max(
            0.0,
            offset.get_window_extent(renderer).y1
            - axis.bbox.y1,
        )

    return _fit_top_legend(
        figure=figure,
        axis=axis,
        handles=handles,
        series_order=config.series_order,
        label_variants=label_variants,
        column_candidates=column_candidates,
        title_sizes=title_sizes,
        preset_anchor=preset_anchor,
        left=left,
        right=right,
        bottom=bottom,
        semantic_layout_fits=_semantic_layout_fits,
        legend_builder=_legend_with_metadata,
        top_decoration_height=top_decoration_height,
        draw_budget=draw_budget,
        layout_name="vertical bar",
    )


def _position_value_label(
    text,
    axis,
    figure: plt.Figure,
    offset_points: float,
    *,
    inside: bool,
) -> None:
    center, value = text.get_position()
    direction = text._polarviz_endpoint_direction
    if inside:
        direction *= -1.0
    text.set_transform(
        offset_copy(
            axis.transData,
            fig=figure,
            y=direction * offset_points,
            units="points",
        )
    )
    text.set_va("bottom" if direction > 0.0 else "top")
    text.set_position((center, value))


def _value_layout_fits(
    figure: plt.Figure, axis, value_texts: list, renderer
) -> bool:
    obstacles = _semantic_obstacles(axis)
    return (
        all(_inside(text, figure, renderer) for text in value_texts)
        and _pairwise_clear(value_texts, figure, renderer)
        and not any(
            text.get_window_extent(renderer).overlaps(
                obstacle.get_window_extent(renderer)
            )
            for text in value_texts
            for obstacle in obstacles
        )
    )


def _fit_value_labels(
    figure: plt.Figure,
    axis,
    value_texts: list,
    offset_points: float,
    draw_budget: LayoutDrawBudget,
) -> None:
    if not value_texts:
        return
    obstacles = _semantic_obstacles(axis)
    for font_size, rotation in ((7.0, 0.0), (6.5, 0.0), (6.0, 90.0), (5.5, 90.0)):
        for text in value_texts:
            text.set_fontsize(font_size)
            text.set_rotation(rotation)
            _position_value_label(text, axis, figure, offset_points, inside=False)
        draw_budget.draw()
        renderer = draw_budget.renderer
        for text in value_texts:
            text_box = text.get_window_extent(renderer)
            outside_axis_y = (
                text_box.y0 < axis.bbox.y0 or text_box.y1 > axis.bbox.y1
            )
            if outside_axis_y or not _inside(text, figure, renderer) or any(
                text_box.overlaps(obstacle.get_window_extent(renderer))
                for obstacle in obstacles
            ):
                _position_value_label(text, axis, figure, offset_points, inside=True)
        draw_budget.draw()
        if _value_layout_fits(figure, axis, value_texts, renderer):
            return
    raise ValueError(
        "vertical bar layout cannot fit or separate endpoint value labels inside "
        "figure_size"
    )


def plot_grouped_vertical_bars(
    data: pd.DataFrame,
    config: VerticalBarConfig,
    preset: Mapping[str, float] = VERTICAL_GROUPED_BARS,
) -> plt.Figure:
    """Render series side-by-side within left-to-right categories."""
    if not isinstance(config, VerticalBarConfig):
        raise ValueError("config must be a VerticalBarConfig")
    frame = validate_bar_data(data, config.category_order, config.series_order)
    if ((frame["value"] < config.y_min) | (frame["value"] > config.y_max)).any():
        raise ValueError(f"value must lie within [{config.y_min}, {config.y_max}]")
    appearance = _validated_preset(preset)
    colors = series_color_map(config.series_order, config.series_colors)
    if any(not mcolors.is_color_like(color) for color in colors.values()):
        raise ValueError("configured series colors must be valid Matplotlib colors")

    category_stride = 1.0 + float(config.group_gap_fraction)
    group_width = float(config.bar_fill_fraction)
    bar_width = group_width / len(config.series_order)
    if not all(
        isfinite(value) and value > 0.0
        for value in (category_stride, group_width, bar_width)
    ):
        raise ValueError("derived vertical bar geometry must be finite and positive")
    category_centers = (
        np.arange(len(config.category_order), dtype=float) * category_stride
    )
    series_offsets = (
        np.arange(len(config.series_order), dtype=float)
        - (len(config.series_order) - 1.0) / 2.0
    ) * bar_width

    labels = dict(config.category_labels)
    display_labels = [
        labels.get(category, category) for category in config.category_order
    ]
    effective_band_alpha = (
        float(config.group_band_alpha) * appearance["band_alpha_scale"]
    )
    active_figure = plt.gcf() if plt.get_fignums() else None
    figure: plt.Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(figsize=config.figure_size)
            draw_budget = LayoutDrawBudget(figure, "vertical bar")
            axis.set_axisbelow(True)

            if config.group_band_colors:
                band_half_width = category_stride / 2.0
                for center, category, color in zip(
                    category_centers,
                    config.category_order,
                    config.group_band_colors,
                ):
                    band = axis.axvspan(
                        center - band_half_width,
                        center + band_half_width,
                        color=color,
                        alpha=effective_band_alpha,
                        linewidth=0.0,
                        zorder=0,
                    )
                    _metadata(
                        band,
                        "category-band",
                        category,
                        "polarviz-category-band",
                    )

            row = 0
            value_texts = []
            for category_index, category in enumerate(config.category_order):
                for series_index, series in enumerate(config.series_order):
                    value = float(frame.iloc[row]["value"])
                    center = (
                        category_centers[category_index]
                        + series_offsets[series_index]
                    )
                    bar = axis.bar(
                        center,
                        value - config.baseline,
                        width=bar_width,
                        bottom=config.baseline,
                        align="center",
                        color=colors[series],
                        edgecolor=marker_edge_color(colors[series]),
                        linewidth=appearance["edge_width"],
                        alpha=appearance["bar_alpha"],
                        zorder=2,
                    )[0]
                    key = (category, series)
                    _metadata(bar, "bar", key, "polarviz-vertical-bar")

                    if config.show_value_labels:
                        endpoint_direction = (
                            -1.0 if value < config.baseline else 1.0
                        )
                        transform = offset_copy(
                            axis.transData,
                            fig=figure,
                            y=endpoint_direction
                            * appearance["value_label_offset_points"],
                            units="points",
                        )
                        text = axis.text(
                            center,
                            value,
                            _value_label(value, config.value_decimals),
                            transform=transform,
                            ha="center",
                            va="top" if endpoint_direction < 0.0 else "bottom",
                            color=INK,
                            fontsize=7.0,
                            clip_on=False,
                            zorder=4,
                        )
                        text._polarviz_endpoint_direction = endpoint_direction
                        _metadata(text, "value-label", key, "polarviz-value-label")
                        value_texts.append(text)
                    row += 1

            axis.set_ylim(config.y_min, config.y_max)
            axis.set_yticks(config.y_ticks)
            axis.set_xticks(category_centers, labels=display_labels)
            lower = category_centers[0] - category_stride / 2.0
            upper = category_centers[-1] + category_stride / 2.0
            axis.set_xlim(lower, upper)
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

            handles = []
            for series in config.series_order:
                handle = Patch(
                    facecolor=colors[series],
                    edgecolor=marker_edge_color(colors[series]),
                    linewidth=appearance["edge_width"],
                    alpha=appearance["bar_alpha"],
                )
                _metadata(handle, "legend-key", series, "polarviz-legend-key")
                handles.append(handle)
            _fit_semantic_layout(
                figure,
                axis,
                config,
                appearance,
                handles,
                display_labels,
                draw_budget,
            )
            _fit_value_labels(
                figure,
                axis,
                value_texts,
                appearance["value_label_offset_points"],
                draw_budget,
            )

            axis._polarviz_category_count = len(config.category_order)
            axis._polarviz_series_count = len(config.series_order)
            axis._polarviz_bar_count = len(frame)
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
