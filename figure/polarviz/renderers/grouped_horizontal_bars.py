"""Grouped horizontal bars for signed Cartesian comparisons."""

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

from ..config import HorizontalBarConfig
from ..data import validate_bar_data
from ..presets.horizontal_delta_bars import HORIZONTAL_DELTA_BARS
from ..presets.palette import GRID, INK
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
        "zero_line_width",
        "zero_line_alpha",
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
        "zero_line_width": values["zero_line_width"] > 0.0,
        "zero_line_alpha": 0.0 <= values["zero_line_alpha"] <= 1.0,
        "value_label_offset_points": values["value_label_offset_points"] >= 0.0,
        "legend_y": 1.0 <= values["legend_y"] <= 2.0,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _value_label(value: float, decimals: int) -> str:
    magnitude = f"{abs(value):.{decimals}f}"
    if value > 0:
        return f"+{magnitude}"
    if value < 0:
        return f"−{magnitude}"
    return magnitude


def _semantic_obstacles(axis) -> list:
    artists = [
        *[tick for tick in axis.get_xticklabels() if tick.get_visible()],
        *[tick for tick in axis.get_yticklabels() if tick.get_visible()],
    ]
    legend = axis.get_legend()
    if legend is not None and legend.get_visible():
        artists.append(legend)
    if axis.get_xlabel() and axis.xaxis.label.get_visible():
        artists.append(axis.xaxis.label)
    if axis.get_title() and axis.title.get_visible():
        artists.append(axis.title)
    offset_text = axis.xaxis.get_offset_text()
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
    config: HorizontalBarConfig,
    appearance: Mapping[str, float],
    handles: list[Patch],
    display_labels: list[str],
    draw_budget: LayoutDrawBudget,
):
    figure_width, _ = figure.get_size_inches()
    category_wrap = max(8, min(28, int(figure_width * 5.0)))
    title_wrap = max(14, min(68, int(figure_width * 11.0)))
    axis_wrap = max(14, min(68, int(figure_width * 11.0)))
    legend_wrap = max(8, min(28, int(figure_width * 5.0)))
    axis.set_yticklabels([_wrapped(label, category_wrap) for label in display_labels])
    axis.set_xlabel(_wrapped(config.axis_label, axis_wrap) if config.axis_label else "")
    axis.set_title(_wrapped(config.title, title_wrap) if config.title else "", pad=0.0)
    if config.title:
        axis._autotitlepos = False
        axis.title.set_transform(figure.transFigure)
        axis.title.set_position((0.5, 0.985))
        axis.title.set_horizontalalignment("center")
        axis.title.set_verticalalignment("top")

    bottom = 0.17 if config.axis_label else 0.12
    figure.subplots_adjust(left=0.10, right=0.95, bottom=bottom, top=0.75)
    draw_budget.draw()
    renderer = draw_budget.renderer
    outer = figure.bbox
    if axis.get_title() and axis.title.get_window_extent(renderer).height > outer.height * 0.38:
        raise ValueError(
            "horizontal bar layout cannot fit title inside figure_size"
        )
    if any(
        tick.get_window_extent(renderer).height > outer.height * 0.70
        for tick in axis.get_yticklabels()
    ):
        raise ValueError(
            "horizontal bar layout cannot fit category labels inside figure_size"
        )

    y_width = max(
        (tick.get_window_extent(renderer).width for tick in axis.get_yticklabels()),
        default=0.0,
    )
    x_half_width = max(
        (
            tick.get_window_extent(renderer).width / 2.0
            for tick in axis.get_xticklabels()
        ),
        default=0.0,
    )
    left = max(
        0.08,
        (y_width + 12.0) / outer.width,
        (x_half_width + 8.0) / outer.width,
    )
    right = min(0.97, 1.0 - (x_half_width + 8.0) / outer.width)
    offset_text = axis.xaxis.get_offset_text()
    bottom_artists = [*axis.get_xticklabels()]
    if axis.get_xlabel():
        bottom_artists.append(axis.xaxis.label)
    if offset_text.get_visible() and offset_text.get_text().strip():
        bottom_artists.append(offset_text)
    if bottom_artists:
        decoration_depth = axis.bbox.y0 - min(
            artist.get_window_extent(renderer).y0 for artist in bottom_artists
        )
        bottom = max(bottom, (decoration_depth + 2.0) / outer.height)
    if offset_text.get_visible() and offset_text.get_text().strip():
        right_extension = max(
            0.0,
            offset_text.get_window_extent(renderer).x1 - axis.bbox.x1,
        )
        right = min(right, 1.0 - (right_extension + 2.0) / outer.width)
    if left + 0.24 >= right:
        raise ValueError(
            "horizontal bar layout cannot fit labels inside figure_size"
        )

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
        max(
            0.80,
            0.98 + (float(appearance["legend_y"]) - 1.16) * 0.25,
        ),
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
        top_decoration_height=lambda renderer: 0.0,
        draw_budget=draw_budget,
        layout_name="horizontal bar",
    )


def _fit_value_labels(
    figure: plt.Figure,
    axis,
    value_texts: list,
    offset_points: float,
    draw_budget: LayoutDrawBudget,
) -> None:
    draw_budget.draw()
    obstacles = _semantic_obstacles(axis)
    renderer = draw_budget.renderer
    for text in value_texts:
        text_box = text.get_window_extent(renderer)
        overlaps_obstacle = any(
            text_box.overlaps(obstacle.get_window_extent(renderer))
            for obstacle in obstacles
        )
        if _inside(text, figure, renderer) and not overlaps_obstacle:
            continue
        value, center = text.get_position()
        if value > 0.0:
            direction, alignment = -1.0, "right"
        elif value < 0.0:
            direction, alignment = 1.0, "left"
        else:
            direction, alignment = 1.0, "left"
        text.set_transform(
            offset_copy(
                axis.transData,
                fig=figure,
                x=direction * offset_points,
                units="points",
            )
        )
        text.set_ha(alignment)
        text.set_position((value, center))
    draw_budget.draw()
    if not all(_inside(text, figure, renderer) for text in value_texts):
        raise ValueError(
            "horizontal bar layout cannot fit endpoint value labels inside "
            "figure_size"
        )
    if not _pairwise_clear(value_texts, figure, renderer):
        raise ValueError(
            "horizontal bar layout cannot separate value labels inside figure_size"
        )
    if any(
        text.get_window_extent(renderer).overlaps(
            obstacle.get_window_extent(renderer)
        )
        for text in value_texts
        for obstacle in obstacles
    ):
        raise ValueError(
            "horizontal bar layout cannot separate value labels from semantic "
            "labels inside figure_size"
        )


def plot_grouped_horizontal_bars(
    data: pd.DataFrame,
    config: HorizontalBarConfig,
    preset: Mapping[str, float] = HORIZONTAL_DELTA_BARS,
) -> plt.Figure:
    """Render signed series side-by-side within top-to-bottom categories."""
    if not isinstance(config, HorizontalBarConfig):
        raise ValueError("config must be a HorizontalBarConfig")
    frame = validate_bar_data(data, config.category_order, config.series_order)
    if ((frame["value"] < config.x_min) | (frame["value"] > config.x_max)).any():
        raise ValueError(f"value must lie within [{config.x_min}, {config.x_max}]")
    appearance = _validated_preset(preset)
    colors = series_color_map(config.series_order, config.series_colors)
    if any(not mcolors.is_color_like(color) for color in colors.values()):
        raise ValueError("configured series colors must be valid Matplotlib colors")

    category_stride = 1.0 + float(config.group_gap_fraction)
    group_height = float(config.bar_fill_fraction)
    bar_height = group_height / len(config.series_order)
    if not all(
        isfinite(value) and value > 0.0
        for value in (category_stride, group_height, bar_height)
    ):
        raise ValueError("derived horizontal bar geometry must be finite and positive")
    category_centers = (
        np.arange(len(config.category_order), dtype=float) * category_stride
    )
    series_offsets = (
        np.arange(len(config.series_order), dtype=float)
        - (len(config.series_order) - 1.0) / 2.0
    ) * bar_height

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
            draw_budget = LayoutDrawBudget(figure, "horizontal bar")
            axis.set_axisbelow(True)

            if config.group_band_colors:
                band_half_height = category_stride / 2.0
                for center, category, color in zip(
                    category_centers,
                    config.category_order,
                    config.group_band_colors,
                ):
                    band = axis.axhspan(
                        center - band_half_height,
                        center + band_half_height,
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
                    bar = axis.barh(
                        center,
                        value,
                        height=bar_height,
                        left=0.0,
                        align="center",
                        color=colors[series],
                        edgecolor=marker_edge_color(colors[series]),
                        linewidth=appearance["edge_width"],
                        alpha=appearance["bar_alpha"],
                        zorder=2,
                    )[0]
                    key = (category, series)
                    _metadata(bar, "bar", key, "polarviz-horizontal-bar")

                    if config.show_value_labels:
                        direction = -1.0 if value < 0.0 else 1.0
                        transform = offset_copy(
                            axis.transData,
                            fig=figure,
                            x=direction * appearance["value_label_offset_points"],
                            units="points",
                        )
                        text = axis.text(
                            value,
                            center,
                            _value_label(value, config.value_decimals),
                            transform=transform,
                            ha="right" if value < 0.0 else "left",
                            va="center",
                            color=INK,
                            fontsize=7.0,
                            clip_on=False,
                            zorder=4,
                        )
                        _metadata(text, "value-label", key, "polarviz-value-label")
                        value_texts.append(text)
                    row += 1

            zero_line = axis.axvline(
                0.0,
                color=config.zero_line_color,
                linewidth=appearance["zero_line_width"],
                alpha=appearance["zero_line_alpha"],
                zorder=3,
            )
            _metadata(zero_line, "zero-line", 0.0, "polarviz-zero-line")

            axis.set_xlim(config.x_min, config.x_max)
            axis.set_xticks(config.x_ticks)
            axis.set_yticks(category_centers, labels=display_labels)
            lower = category_centers[0] - category_stride / 2.0
            upper = category_centers[-1] + category_stride / 2.0
            axis.set_ylim(upper, lower)
            axis.xaxis.grid(
                True,
                color=GRID,
                linewidth=0.65,
                alpha=appearance["grid_alpha"],
            )
            axis.yaxis.grid(False)
            for name in ("left", "right", "top"):
                axis.spines[name].set_visible(False)
            axis.spines["bottom"].set_color(GRID)
            axis.spines["bottom"].set_linewidth(0.65)
            axis.tick_params(axis="y", length=0, colors=INK)
            axis.tick_params(axis="x", colors=INK)

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
