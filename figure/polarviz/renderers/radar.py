"""Segmented-ring radar renderer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite, log10, pi
from numbers import Integral, Real
from textwrap import fill as wrap_text

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd

from ..config import RadarConfig
from ..data import validate_radar_data
from ..presets import radar_segmented_ring
from ..presets.palette import DARK_STROKES, GRID, INK, OUTER_RING_COLORS
from ..style import paper_style, series_color_map


_REQUIRED_PRESET_KEYS = frozenset(
    {
        "outer_ring_width_fraction",
        "outer_ring_gap_fraction",
        "grid_alpha",
        "line_width",
        "marker_size",
        "legend_y",
    }
)
_OPTIONAL_PRESET_KEYS = frozenset(
    {
        "secondary_line_width",
        "primary_understroke_extra",
        "secondary_understroke_extra",
        "axis_label_pad_fraction",
    }
)
_PRESET_KEYS = _REQUIRED_PRESET_KEYS | _OPTIONAL_PRESET_KEYS
_DEFAULT_LINE_STYLES = ("-",)
_DEFAULT_MARKERS = ("o", "s", "D", "^")
# The public radial-gap range reaches 1.0; cap its angular reuse so segments
# remain visible and separators retain the intended narrow visual grammar.
_MAX_ANGULAR_GAP_FRACTION = 0.12
_TITLE_PAD_POINTS = 18.0


def radar_angles(count: Integral) -> np.ndarray:
    """Return equally spaced radar-axis angles in ``[0, 2π)``."""
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, Integral)
        or count < 3
    ):
        raise ValueError("count must be an integer of at least 3")
    return np.linspace(0.0, 2.0 * pi, int(count), endpoint=False)


def _finite_vector(values: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional real array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be a finite one-dimensional real array")
    for value in array:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain finite real numbers")
        try:
            finite = isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(array, dtype=float)


def close_path(theta: object, radius: object) -> tuple[np.ndarray, np.ndarray]:
    """Return validated radar coordinates closed exactly once."""
    theta_values = _finite_vector(theta, "theta")
    radius_values = _finite_vector(radius, "radius")
    if len(theta_values) != len(radius_values) or len(theta_values) < 3:
        raise ValueError("theta and radius must have equal lengths of at least 3")
    if theta_values[0] == theta_values[-1] and radius_values[0] == radius_values[-1]:
        raise ValueError("path is already closed")
    return (
        np.concatenate((theta_values, theta_values[:1])),
        np.concatenate((radius_values, radius_values[:1])),
    )


def _validated_preset(preset: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(preset, Mapping):
        raise ValueError("preset must be a mapping")
    supplied = dict(preset)
    keys = set(supplied)
    missing = sorted(_REQUIRED_PRESET_KEYS - keys)
    extra = sorted(keys - _PRESET_KEYS)
    if missing:
        raise ValueError(f"preset is missing required keys: {missing}")
    if extra:
        raise ValueError(f"preset contains unknown keys: {extra}")
    values = {**radar_segmented_ring, **supplied}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"preset {key} must be a finite number")
    ranges = {
        "outer_ring_width_fraction": 0.0 < values["outer_ring_width_fraction"] <= 1.0,
        "outer_ring_gap_fraction": 0.0 <= values["outer_ring_gap_fraction"] <= 1.0,
        "grid_alpha": 0.0 <= values["grid_alpha"] <= 1.0,
        "line_width": 0.0 < values["line_width"],
        "marker_size": 0.0 < values["marker_size"],
        "legend_y": -1.0 <= values["legend_y"] <= 0.0,
        "secondary_line_width": 0.0 < values["secondary_line_width"],
        "primary_understroke_extra": 0.0 <= values["primary_understroke_extra"],
        "secondary_understroke_extra": 0.0 <= values["secondary_understroke_extra"],
        "axis_label_pad_fraction": 0.0 < values["axis_label_pad_fraction"],
    }
    invalid = [key for key, valid in ranges.items() if not valid]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _configured_styles(
    series_order: Sequence[str],
    configured: Sequence[tuple[str, str]],
    defaults: Sequence[str],
    name: str,
) -> dict[str, str]:
    explicit: dict[str, str] = {}
    for pair in configured:
        if len(pair) != 2:
            raise ValueError(f"{name} must contain (series, style) pairs")
        series, value = pair
        if (
            series not in series_order
            or series in explicit
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError(f"{name} contains an invalid series or style")
        explicit[series] = value
    result = {
        series: explicit.get(series, defaults[index % len(defaults)])
        for index, series in enumerate(series_order)
    }
    try:
        if name == "line_styles":
            for value in result.values():
                Line2D([], [], linestyle=value)
        else:
            for value in result.values():
                MarkerStyle(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains an invalid style") from error
    return result


def _axis_label_alignment(
    angle: float,
    offset_degrees: float,
    direction: int,
) -> str:
    """Align an external horizontal label by its final displayed side."""
    horizontal = np.cos(np.deg2rad(offset_degrees) + direction * angle)
    if abs(horizontal) < 1e-7:
        return "center"
    return "left" if horizontal > 0 else "right"


def _contrast_color(color: object) -> tuple[float, float, float, float]:
    """Return a dark outline while preserving the source color's alpha."""
    red, green, blue, alpha = mcolors.to_rgba(color)
    normalized = mcolors.to_hex((red, green, blue)).upper()
    if normalized in DARK_STROKES:
        dark = mcolors.to_rgb(DARK_STROKES[normalized])
        return (*dark, alpha)
    rgb = np.asarray((red, green, blue))
    ink = np.asarray(mcolors.to_rgb(INK))
    darkened = 0.58 * rgb + 0.42 * ink
    return (*map(float, darkened), alpha)


def _contrast_path_effects(
    color: object,
    width: float,
    extra_width: float,
) -> list[patheffects.AbstractPathEffect]:
    if isinstance(color, str) and color.strip().casefold() == "none":
        return []
    return [
        patheffects.Stroke(
            linewidth=width + extra_width,
            foreground=_contrast_color(color),
        ),
        patheffects.Normal(),
    ]


def _radial_tick_labels(ticks: Sequence[Real], span: float) -> list[str]:
    """Format ticks positionally with enough precision to preserve identity."""
    values = [float(tick) for tick in ticks]
    if len(values) == 1:
        return [_format_tick_value(values[0], precision=6)]
    gaps = [right - left for left, right in zip(values, values[1:])]
    scale = max([abs(span), *(abs(value) for value in values), 1e-300])
    relative_gap = min(gaps) / scale
    initial_precision = 17
    if relative_gap > 0 and isfinite(relative_gap):
        initial_precision = max(6, min(17, ceil(-log10(relative_gap)) + 1))
    for precision in range(initial_precision, 18):
        labels = [_format_tick_value(value, precision) for value in values]
        if len(set(labels)) == len(labels):
            return labels
    labels = [_format_tick_value(value, precision=17) for value in values]
    if len(set(labels)) != len(labels):
        raise ValueError("radial ticks cannot be formatted as distinct finite labels")
    return labels


def _format_tick_value(value: float, precision: int) -> str:
    magnitude = abs(value)
    if value != 0 and (magnitude < 1e-4 or magnitude >= 1e6):
        mantissa, exponent = f"{value:.{precision}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent):+d}"
    return np.format_float_positional(
        value,
        precision=precision,
        unique=False,
        fractional=False,
        trim="-",
    )


def _radar_geometry(
    config: RadarConfig,
    appearance: Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    """Compute finite ordered radial geometry before creating a figure."""
    r_min = float(config.r_min)
    r_max = float(config.r_max)
    span = r_max - r_min
    ring_gap = span * appearance["outer_ring_gap_fraction"]
    ring_width = span * appearance["outer_ring_width_fraction"]
    ring_bottom = r_max + ring_gap
    ring_outer = ring_bottom + ring_width
    radial_limit = ring_outer + span * 0.025
    values = (span, ring_gap, ring_width, ring_bottom, radial_limit)
    gap_ordered = ring_bottom >= r_max
    if appearance["outer_ring_gap_fraction"] > 0:
        gap_ordered = ring_bottom > r_max
    if (
        not all(isfinite(value) for value in values)
        or span <= 0
        or ring_gap < 0
        or ring_width <= 0
        or not gap_ordered
        or ring_outer <= ring_bottom
        or radial_limit <= ring_outer
    ):
        raise ValueError("derived radar geometry must be finite and strictly ordered")
    return span, ring_gap, ring_width, ring_bottom, radial_limit


def _axis_label_font_size(axis_count: int) -> float:
    return max(6.0, 8.0 - 0.25 * max(0, axis_count - 8))


def _bbox_is_contained(inner, outer, padding: float) -> bool:
    return (
        inner.x0 >= outer.x0 + padding
        and inner.y0 >= outer.y0 + padding
        and inner.x1 <= outer.x1 - padding
        and inner.y1 <= outer.y1 - padding
    )


def _legend_labels(labels: Sequence[str], width: int | None) -> list[str]:
    if width is None:
        return list(labels)
    return [
        wrap_text(label, width=width, break_long_words=True, break_on_hyphens=True)
        for label in labels
    ]


def _fit_legend(
    figure: plt.Figure,
    axis,
    config: RadarConfig,
    legend_y: float,
) -> None:
    """Measure and adapt the legend until it is contained by the canvas."""
    labels = list(config.series_order)
    by_series = {
        line._polarviz_series: line
        for line in axis.lines
        if line.get_gid() == "polarviz-series"
    }
    handles = [by_series[label] for label in labels]
    requested_columns = min(config.legend_columns, len(labels))
    wrap_widths: tuple[int | None, ...] = (None, 32, 24, 18, 14, 10, 8)
    font_sizes = (7.5, 7.0, 6.5, 6.0, 5.5)
    top = 0.90
    base_bottom = 0.14
    minimum_axis_height = 0.32

    for columns in range(requested_columns, 0, -1):
        for font_size in font_sizes:
            for wrap_width in wrap_widths:
                figure.subplots_adjust(bottom=base_bottom, top=top)
                legend = axis.legend(
                    handles,
                    _legend_labels(labels, wrap_width),
                    loc="upper center",
                    bbox_to_anchor=(0.5, legend_y),
                    ncol=columns,
                    frameon=False,
                    handlelength=2.3,
                    columnspacing=1.25,
                    fontsize=font_size,
                )
                bottom = base_bottom
                for _ in range(12):
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    legend_box = legend.get_window_extent(renderer)
                    figure_box = figure.bbox
                    padding = max(
                        2.0,
                        min(figure_box.width, figure_box.height) * 0.01,
                    )
                    horizontal_fit = (
                        legend_box.x0 >= figure_box.x0 + padding
                        and legend_box.x1 <= figure_box.x1 - padding
                    )
                    if not horizontal_fit:
                        break
                    if _bbox_is_contained(legend_box, figure_box, padding):
                        return
                    if legend_box.y1 > figure_box.y1 - padding:
                        break
                    deficit = figure_box.y0 + padding - legend_box.y0
                    if deficit <= 0:
                        break
                    new_bottom = (
                        bottom
                        + deficit / figure_box.height
                        + padding / figure_box.height
                    )
                    if new_bottom >= top - minimum_axis_height:
                        break
                    bottom = new_bottom
                    figure.subplots_adjust(bottom=bottom, top=top)

    raise ValueError(
        "radar legend layout cannot fit inside figure_size; increase figure_size "
        "or shorten series labels"
    )


def _fit_dense_axis_labels(
    figure: plt.Figure,
    axis,
    texts: Sequence,
    labels: Sequence[str],
    initial_font_size: float,
) -> None:
    """Fit external labels while keeping the title separate and contained."""
    wrap_widths: tuple[int | None, ...] = (None, 20, 16, 12, 10, 8, 6)
    font_sizes = np.arange(initial_font_size, 3.9, -0.5)
    initial_top = figure.subplotpars.top
    bottom = figure.subplotpars.bottom
    minimum_axis_height = 0.30
    title_value = axis.get_title()
    title = axis.title if title_value else None

    for wrap_width in wrap_widths:
        displayed = _legend_labels(labels, wrap_width)
        for font_size in font_sizes:
            figure.subplots_adjust(top=initial_top)
            title_pad = _TITLE_PAD_POINTS
            if title is not None:
                title = axis.set_title(title_value, pad=title_pad)
            for text, label in zip(texts, displayed):
                text.set_text(label)
                text.set_fontsize(float(font_size))
                text.set_multialignment("center")
            current_top = initial_top
            for _ in range(8):
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                boxes = [text.get_window_extent(renderer) for text in texts]
                contained = all(
                    _bbox_is_contained(box, figure.bbox, padding=1.0)
                    for box in boxes
                )
                separate = all(
                    not left.overlaps(right)
                    for index, left in enumerate(boxes)
                    for right in boxes[index + 1:]
                )
                if not contained or not separate:
                    break
                if title is None:
                    return

                title_box = title.get_window_extent(renderer)
                padding = max(
                    1.0,
                    min(figure.bbox.width, figure.bbox.height) * 0.005,
                )
                horizontal_fit = (
                    title_box.x0 >= figure.bbox.x0 + padding
                    and title_box.x1 <= figure.bbox.x1 - padding
                )
                if not horizontal_fit:
                    break
                overlaps = [box for box in boxes if title_box.overlaps(box)]
                if overlaps:
                    upward_shift = max(
                        box.y1 - title_box.y0 + padding for box in overlaps
                    )
                    title_pad += upward_shift * 72.0 / figure.dpi
                    title = axis.set_title(title_value, pad=title_pad)
                    continue
                if _bbox_is_contained(title_box, figure.bbox, padding):
                    return
                if title_box.y1 <= figure.bbox.y1 - padding:
                    break
                deficit = title_box.y1 - (figure.bbox.y1 - padding)
                new_top = current_top - (
                    deficit + padding
                ) / figure.bbox.height
                if new_top <= bottom + minimum_axis_height:
                    break
                current_top = new_top
                figure.subplots_adjust(top=current_top)
    raise ValueError(
        "radar title and axis label layout cannot fit inside figure_size; "
        "increase figure_size or shorten title/axis labels"
        if title is not None
        else "radar axis label layout cannot fit inside figure_size; increase "
        "figure_size or shorten axis labels"
    )


def plot_radar(
    frame: pd.DataFrame,
    config: RadarConfig,
    *,
    preset: Mapping[str, float] = radar_segmented_ring,
) -> plt.Figure:
    """Render a validated radar chart with a segmented navigation ring."""
    data = validate_radar_data(frame, config)
    appearance = _validated_preset(preset)
    colors = series_color_map(config.series_order, config.series_colors)
    for color in colors.values():
        if not mcolors.is_color_like(color):
            raise ValueError(f"invalid configured series color: {color!r}")
    line_styles = _configured_styles(
        config.series_order, config.line_styles, _DEFAULT_LINE_STYLES, "line_styles"
    )
    markers = _configured_styles(
        config.series_order, config.markers, _DEFAULT_MARKERS, "markers"
    )
    ring_colors = config.outer_ring_colors or tuple(
        OUTER_RING_COLORS[index % len(OUTER_RING_COLORS)]
        for index in range(len(config.axis_order))
    )
    for color in ring_colors:
        if not mcolors.is_color_like(color):
            raise ValueError(f"invalid configured outer ring color: {color!r}")

    angles = radar_angles(len(config.axis_order))
    direction = -1 if config.clockwise else 1
    span, _, ring_width, ring_bottom, radial_limit = _radar_geometry(
        config, appearance
    )
    tick_labels = _radial_tick_labels(config.r_ticks, span)
    sector_width = 2.0 * pi / len(angles)
    angular_gap_fraction = min(
        appearance["outer_ring_gap_fraction"],
        _MAX_ANGULAR_GAP_FRACTION,
    )
    segment_width = sector_width * (1.0 - angular_gap_fraction)
    label_font_size = _axis_label_font_size(len(config.axis_order))
    primary_series = config.primary_series or config.series_order[0]
    drawing_order = tuple(
        series for series in config.series_order if series != primary_series
    ) + (primary_series,)

    figure: plt.Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(
                figsize=config.figure_size,
                subplot_kw={"projection": "polar"},
            )
            axis.set_theta_offset(np.deg2rad(config.start_angle_degrees))
            axis.set_theta_direction(direction)
            axis.set_ylim(config.r_min, radial_limit)
            axis.set_xticks([])
            axis.set_yticks(config.r_ticks)
            axis.set_yticklabels(tick_labels)
            axis.tick_params(
                axis="y",
                labelleft=config.show_radial_tick_labels,
            )
            axis.set_rlabel_position(90.0)
            axis.set_axisbelow(True)
            axis.yaxis.grid(
                True,
                color=GRID,
                linewidth=0.65,
                alpha=appearance["grid_alpha"],
            )
            axis.xaxis.grid(False)
            axis.spines["polar"].set_visible(False)

            sectors = axis.bar(
                angles,
                np.full(len(angles), span),
                width=sector_width,
                bottom=config.r_min,
                align="center",
                color=ring_colors,
                edgecolor="none",
                linewidth=0.0,
                alpha=config.sector_fill_alpha,
                zorder=0.1,
            )
            for sector in sectors:
                sector.set_gid("polarviz-sector-wash")

            for angle in angles:
                spoke, = axis.plot(
                    [angle, angle],
                    [config.r_min, config.r_max],
                    color=GRID,
                    linewidth=0.65,
                    alpha=appearance["grid_alpha"],
                    zorder=0.5,
                )
                spoke.set_gid("polarviz-spoke")

            segments = axis.bar(
                angles,
                np.full(len(angles), ring_width),
                width=segment_width,
                bottom=ring_bottom,
                align="center",
                color=ring_colors,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
            for segment in segments:
                segment.set_gid("polarviz-outer-segment")

            label_radius = (
                ring_bottom
                + ring_width
                + span * appearance["axis_label_pad_fraction"]
            )
            axis_label_texts = []
            for angle, label in zip(angles, config.axis_order):
                text = axis.text(
                    angle,
                    label_radius,
                    label,
                    ha=_axis_label_alignment(
                        angle, config.start_angle_degrees, direction
                    ),
                    va="center",
                    rotation=0.0,
                    rotation_mode="anchor",
                    color=INK,
                    fontsize=label_font_size,
                    fontweight="semibold",
                    clip_on=False,
                    zorder=6,
                )
                text.set_gid("polarviz-axis-label")
                axis_label_texts.append(text)

            for series in drawing_order:
                values = data.loc[
                    data["series"] == series, "value"
                ].to_numpy(dtype=float)
                closed_angles, closed_values = close_path(angles, values)
                color = colors[series]
                is_primary = series == primary_series
                line_width = appearance[
                    "line_width" if is_primary else "secondary_line_width"
                ]
                marker = markers[series] if config.show_markers else None
                line, = axis.plot(
                    closed_angles,
                    closed_values,
                    color=color,
                    linewidth=line_width,
                    linestyle=line_styles[series],
                    marker=marker,
                    markersize=appearance["marker_size"],
                    markerfacecolor=color,
                    markeredgecolor=_contrast_color(color),
                    markeredgewidth=0.8,
                    label=series,
                    zorder=4 if is_primary else 3,
                )
                line.set_gid("polarviz-series")
                line._polarviz_series = series
                line.set_path_effects(
                    _contrast_path_effects(
                        color,
                        line_width,
                        appearance[
                            "primary_understroke_extra"
                            if is_primary
                            else "secondary_understroke_extra"
                        ],
                    )
                )
                if is_primary and config.fill_alpha > 0:
                    fill, = axis.fill(
                        closed_angles,
                        closed_values,
                        facecolor=color,
                        edgecolor="none",
                        alpha=config.fill_alpha,
                        zorder=2,
                    )
                    fill.set_gid("polarviz-series-fill")
                    fill._polarviz_series = series

            if config.title is not None:
                axis.set_title(config.title, pad=_TITLE_PAD_POINTS)
            _fit_legend(figure, axis, config, appearance["legend_y"])
            _fit_dense_axis_labels(
                figure,
                axis,
                axis_label_texts,
                config.axis_order,
                label_font_size,
            )

            semantic_colors = tuple(colors[series] for series in config.series_order)
            axis._polarviz_axis_count = len(config.axis_order)
            axis._polarviz_series_count = len(config.series_order)
            axis._polarviz_outer_segment_count = len(config.axis_order)
            axis._polarviz_sector_wash_count = len(config.axis_order)
            axis._polarviz_spoke_count = len(config.axis_order)
            axis._polarviz_series_fill_count = int(config.fill_alpha > 0)
            axis._polarviz_series_colors = semantic_colors
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    return figure
