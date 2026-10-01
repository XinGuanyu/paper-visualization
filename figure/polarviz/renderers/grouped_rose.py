"""Grouped multi-series Nightingale rose renderer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite, log10, pi
from numbers import Integral, Real
from textwrap import fill as wrap_text

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..config import RoseConfig
from ..data import validate_rose_data
from ..presets import rose_grouped
from ..presets.palette import DARK_STROKES, GRID, INK
from ..style import paper_style, series_color_map


_PRESET_KEYS = frozenset({"grid_alpha", "edge_width", "bar_alpha", "legend_y"})


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        finite = isfinite(value)
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not finite:
        raise ValueError(f"{name} must be a finite real number")
    return result


def _finite_vector(values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be a finite one-dimensional real array") from error
    if array.ndim != 1:
        raise ValueError("values must be a finite one-dimensional real array")
    result = np.empty(len(array), dtype=float)
    for index, value in enumerate(array):
        result[index] = _finite_scalar(value, "values")
    return result


def radial_transform(
    values: object,
    r_min: Real,
    r_max: Real,
    encoding: str,
) -> np.ndarray:
    """Transform rose values to radii using radius or annular-area encoding."""
    lower = _finite_scalar(r_min, "r_min")
    upper = _finite_scalar(r_max, "r_max")
    if lower < 0.0 or lower >= upper:
        raise ValueError("radial bounds must satisfy 0 <= r_min < r_max")
    if encoding not in {"area", "radius"}:
        raise ValueError("encoding must be 'area' or 'radius'")
    result = _finite_vector(values)
    if np.any(result < lower) or np.any(result > upper):
        raise ValueError("values must lie within [r_min, r_max]")
    if encoding == "radius":
        return result.copy()

    fraction = (result - lower) / (upper - lower)
    baseline_ratio = lower / upper
    transformed = upper * np.sqrt(
        baseline_ratio * baseline_ratio
        + fraction * (1.0 - baseline_ratio * baseline_ratio)
    )
    transformed[result == lower] = lower
    transformed[result == upper] = upper
    return transformed


def category_angles(
    group_sizes: Sequence[Integral],
    category_gap: Real,
    group_gap: Real,
) -> tuple[np.ndarray, float]:
    """Return category centers and their common width with exact edge gaps."""
    if isinstance(group_sizes, (str, bytes)):
        raise ValueError("group_sizes must contain positive integers")
    try:
        sizes = tuple(group_sizes)
    except TypeError as error:
        raise ValueError("group_sizes must contain positive integers") from error
    if not sizes or any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, Integral)
        or size < 1
        for size in sizes
    ):
        raise ValueError("group_sizes must contain positive integers")
    within_gap = _finite_scalar(category_gap, "category_gap")
    between_gap = _finite_scalar(group_gap, "group_gap")
    if within_gap < 0.0 or between_gap < 0.0:
        raise ValueError("category_gap and group_gap must be non-negative")
    category_count = sum(int(size) for size in sizes)
    group_count = len(sizes)
    try:
        total_gap = (
            within_gap * (category_count - group_count)
            + between_gap * group_count
        )
    except OverflowError as error:
        raise ValueError("total angular gaps must be finite and less than 2*pi") from error
    if not isfinite(total_gap) or total_gap >= 2.0 * pi:
        raise ValueError("total angular gaps must be finite and less than 2*pi")
    width = (2.0 * pi - total_gap) / category_count
    if not isfinite(width) or width <= 0.0:
        raise ValueError("derived category width must be finite and positive")

    group_ends = set(np.cumsum(sizes).tolist())
    angles = np.empty(category_count, dtype=float)
    angles[0] = width / 2.0
    for index in range(1, category_count):
        gap = between_gap if index in group_ends else within_gap
        angles[index] = angles[index - 1] + width + gap
    if (
        not np.isfinite(angles).all()
        or np.any(angles < 0.0)
        or np.any(angles >= 2.0 * pi)
        or len(np.unique(angles)) != category_count
    ):
        raise ValueError("derived category angles must be finite and unique")
    return angles, float(width)


def _validated_preset(preset: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(preset, Mapping):
        raise ValueError("preset must be a mapping")
    values = dict(preset)
    keys = set(values)
    if keys != _PRESET_KEYS:
        missing = sorted(_PRESET_KEYS - keys)
        extra = sorted(keys - _PRESET_KEYS)
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
        "legend_y": -1.0 <= values["legend_y"] <= 0.0,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _group_geometry(
    data: pd.DataFrame,
    config: RoseConfig,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not config.group_order:
        return (len(config.category_order),), ()
    category_groups = tuple(
        data.loc[data["category"] == category, "group"].iloc[0]
        for category in config.category_order
    )
    block_names: list[str] = []
    block_sizes: list[int] = []
    for group in category_groups:
        if not block_names or group != block_names[-1]:
            block_names.append(group)
            block_sizes.append(1)
        else:
            block_sizes[-1] += 1
    if len(set(block_names)) != len(block_names):
        raise ValueError(
            "each group must form one contiguous block in category_order"
        )
    if tuple(block_names) != tuple(config.group_order):
        raise ValueError(
            "group_order must match the contiguous group blocks in category_order"
        )
    return tuple(block_sizes), tuple(block_names)


def _edge_color(color: str) -> str:
    normalized = mcolors.to_hex(color, keep_alpha=False).upper()
    return DARK_STROKES.get(normalized, INK)


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


def _radial_tick_labels(ticks: Sequence[Real], span: float) -> list[str]:
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


def _upright_tangent_rotation(
    angle: float,
    offset_degrees: float,
    direction: int,
) -> float:
    displayed = offset_degrees + direction * np.degrees(angle)
    rotation = (displayed - 90.0 + 180.0) % 360.0 - 180.0
    if rotation > 90.0:
        rotation -= 180.0
    elif rotation < -90.0:
        rotation += 180.0
    return float(rotation)


def _bbox_is_contained(inner, outer, padding: float) -> bool:
    return (
        inner.x0 >= outer.x0 + padding
        and inner.y0 >= outer.y0 + padding
        and inner.x1 <= outer.x1 - padding
        and inner.y1 <= outer.y1 - padding
    )


def _wrapped(labels: Sequence[str], width: int | None) -> list[str]:
    if width is None:
        return list(labels)
    return [
        wrap_text(label, width=width, break_long_words=True, break_on_hyphens=True)
        for label in labels
    ]


def _wrapped_title(title: str, width: int | None) -> str:
    if width is None:
        return title
    return "\n".join(
        wrap_text(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
        ) if line else ""
        for line in title.split("\n")
    )


def _fit_legend(
    figure: plt.Figure,
    axis,
    handles: Sequence[Patch],
    labels: Sequence[str],
    columns: int,
    legend_y: float,
) -> None:
    requested_columns = min(columns, len(labels))
    wrap_widths: tuple[int | None, ...] = (None, 32, 24, 18, 14, 10, 8)
    font_sizes = (7.5, 7.0, 6.5, 6.0, 5.5)
    top = 0.89
    base_bottom = 0.15
    minimum_axis_height = 0.30
    for column_count in range(requested_columns, 0, -1):
        for font_size in font_sizes:
            for wrap_width in wrap_widths:
                figure.subplots_adjust(bottom=base_bottom, top=top)
                legend = axis.legend(
                    handles,
                    _wrapped(labels, wrap_width),
                    loc="upper center",
                    bbox_to_anchor=(0.5, legend_y),
                    ncol=column_count,
                    frameon=False,
                    handlelength=1.8,
                    columnspacing=1.1,
                    fontsize=font_size,
                )
                renderer = figure.canvas.get_renderer()
                legend_box = legend.get_window_extent(renderer)
                figure_box = figure.bbox
                padding = max(
                    2.0, min(figure_box.width, figure_box.height) * 0.01
                )
                horizontal_fit = (
                    legend_box.x0 >= figure_box.x0 + padding
                    and legend_box.x1 <= figure_box.x1 - padding
                )
                if not horizontal_fit:
                    continue
                if _bbox_is_contained(legend_box, figure_box, padding):
                    return
                if legend_box.y1 > figure_box.y1 - padding:
                    continue
                deficit = figure_box.y0 + padding - legend_box.y0
                if deficit <= 0:
                    continue
                new_bottom = (
                    base_bottom
                    + deficit / figure_box.height
                    + padding / figure_box.height
                )
                if new_bottom >= top - minimum_axis_height:
                    continue
                figure.subplots_adjust(bottom=new_bottom, top=top)
                adjusted_box = legend.get_window_extent(renderer)
                if _bbox_is_contained(adjusted_box, figure_box, padding):
                    return
    raise ValueError(
        "rose legend layout cannot fit inside figure_size; increase figure_size "
        "or shorten series labels"
    )


def _fit_title(figure: plt.Figure, axis, title: str) -> None:
    """Wrap and position a title until its measured bbox fits the canvas."""
    if not title:
        return
    wrap_widths: tuple[int | None, ...] = (
        None, 100, 84, 72, 60, 52, 44, 36, 30, 24, 18, 14, 10,
    )
    font_sizes = (10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0)
    base_top = 0.89
    minimum_axis_height = 0.30
    for wrap_width in wrap_widths:
        displayed = _wrapped_title(title, wrap_width)
        for font_size in font_sizes:
            top = base_top
            figure.subplots_adjust(top=top)
            axis.set_title(displayed, pad=18.0, fontsize=font_size)
            renderer = figure.canvas.get_renderer()
            title_box = axis.title.get_window_extent(renderer)
            figure_box = figure.bbox
            padding = max(
                2.0, min(figure_box.width, figure_box.height) * 0.01
            )
            horizontal_fit = (
                title_box.x0 >= figure_box.x0 + padding
                and title_box.x1 <= figure_box.x1 - padding
            )
            if not horizontal_fit:
                continue
            if _bbox_is_contained(title_box, figure_box, padding):
                return
            if title_box.y0 < figure_box.y0 + padding:
                continue
            deficit = title_box.y1 - (figure_box.y1 - padding)
            if deficit <= 0.0:
                continue
            new_top = (
                top
                - deficit / figure_box.height
                - padding / figure_box.height
            )
            if new_top <= figure.subplotpars.bottom + minimum_axis_height:
                continue
            figure.subplots_adjust(top=new_top)
            adjusted_box = axis.title.get_window_extent(renderer)
            if _bbox_is_contained(adjusted_box, figure_box, padding):
                return
    raise ValueError(
        "rose title layout cannot fit inside figure_size; increase figure_size "
        "or shorten the title"
    )


def _artist_box(artist, renderer):
    bbox_patch_getter = getattr(artist, "get_bbox_patch", None)
    bbox_patch = bbox_patch_getter() if bbox_patch_getter is not None else None
    if bbox_patch is not None:
        artist.update_bbox_position_size(renderer)
        return bbox_patch.get_window_extent(renderer)
    return artist.get_window_extent(renderer)


def _artist_boxes(artists: Sequence, renderer) -> list:
    return [
        _artist_box(artist, renderer)
        for artist in artists
        if artist is not None and artist.get_visible()
    ]


def _fit_labels(
    figure: plt.Figure,
    category_texts: Sequence,
    category_labels: Sequence[str],
    group_texts: Sequence,
    group_labels: Sequence[str],
    initial_font_size: float,
    obstacles: Sequence = (),
) -> None:
    wrap_widths: tuple[int | None, ...] = (None, 20, 16, 12, 10, 8, 6, 4)
    font_sizes = np.arange(initial_font_size, 3.9, -0.5)
    for wrap_width in wrap_widths:
        displayed_categories = _wrapped(category_labels, wrap_width)
        displayed_groups = _wrapped(group_labels, wrap_width)
        for font_size in font_sizes:
            for text, label in zip(category_texts, displayed_categories):
                text.set_text(label)
                text.set_fontsize(float(font_size))
                text.set_multialignment("center")
            for text, label in zip(group_texts, displayed_groups):
                text.set_text(label)
                text.set_fontsize(float(max(4.0, font_size - 0.75)))
                text.set_multialignment("center")
            renderer = figure.canvas.get_renderer()
            boxes = _artist_boxes((*category_texts, *group_texts), renderer)
            obstacle_boxes = _artist_boxes(obstacles, renderer)
            contained = all(
                _bbox_is_contained(box, figure.bbox, padding=1.0)
                for box in boxes
            )
            separate = all(
                not left.overlaps(right)
                for index, left in enumerate(boxes)
                for right in boxes[index + 1:]
            )
            clear = all(
                not box.overlaps(obstacle)
                for box in boxes
                for obstacle in obstacle_boxes
            )
            if contained and separate and clear:
                return
    raise ValueError(
        "rose label layout cannot fit inside figure_size; increase figure_size "
        "or shorten category/group labels"
    )


def _fit_radial_ticks(
    figure: plt.Figure,
    axis,
    offset_degrees: float,
    direction: int,
    obstacles: Sequence,
) -> None:
    """Choose a measured radial-label position and font size."""
    coarse_positions = tuple(float(angle) for angle in range(0, 360, 30))
    font_sizes = (7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0)
    tick_texts = axis.get_yticklabels()

    def current_collisions() -> int | None:
        renderer = figure.canvas.get_renderer()
        boxes = _artist_boxes(tick_texts, renderer)
        obstacle_boxes = _artist_boxes(obstacles, renderer)
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
            return None
        return sum(
            box.overlaps(obstacle)
            for box in boxes
            for obstacle in obstacle_boxes
        )

    def evaluate(physical_degrees: float | None, font_size: float) -> int | None:
        if physical_degrees is None:
            logical_degrees = 90.0
        else:
            logical_degrees = (
                (physical_degrees - offset_degrees) / direction
            ) % 360.0
        axis.set_rlabel_position(logical_degrees)
        for text in tick_texts:
            text.set_fontsize(font_size)
            text.set_color(INK)
            text.set_zorder(5)
        return current_collisions()

    def confirmed(result: int | None) -> bool:
        if result != 0:
            return False
        figure.canvas.draw()
        return current_collisions() == 0

    for font_size in font_sizes:
        coarse_results: list[tuple[int, float]] = []
        preferred_collisions = evaluate(None, font_size)
        if confirmed(preferred_collisions):
            return
        for physical_degrees in coarse_positions:
            collisions = evaluate(physical_degrees, font_size)
            if confirmed(collisions):
                return
            if collisions is not None:
                coarse_results.append((collisions, physical_degrees))
        refinement_centers = [
            angle for _, angle in sorted(coarse_results)[:4]
        ]
        refined_positions = tuple(dict.fromkeys(
            float((center + offset) % 360)
            for center in refinement_centers
            for offset in range(-25, 30, 5)
            if offset % 30 != 0
        ))
        for physical_degrees in refined_positions:
            if confirmed(evaluate(physical_degrees, font_size)):
                return
    raise ValueError(
        "rose radial tick layout cannot fit inside figure_size; increase "
        "figure_size or use shorter radial tick values"
    )


def _verify_combined_layout(
    figure: plt.Figure,
    axis,
    semantic_texts: Sequence,
) -> None:
    """Reject any final layout whose measured public artists are cropped."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    artists = [*axis.get_yticklabels(), *semantic_texts]
    if axis.get_title():
        artists.append(axis.title)
    legend = axis.get_legend()
    if legend is not None:
        artists.append(legend)
    boxes = _artist_boxes(artists, renderer)
    if not all(_bbox_is_contained(box, figure.bbox, padding=1.0) for box in boxes):
        raise ValueError(
            "rose combined layout cannot fit inside figure_size; increase "
            "figure_size or shorten labels"
        )
    tick_boxes = _artist_boxes(axis.get_yticklabels(), renderer)
    semantic_boxes = _artist_boxes(semantic_texts, renderer)
    other_boxes = []
    if axis.get_title():
        other_boxes.extend(_artist_boxes((axis.title,), renderer))
    if legend is not None:
        other_boxes.extend(_artist_boxes((legend,), renderer))
    ticks_are_separate = all(
        not left.overlaps(right)
        for index, left in enumerate(tick_boxes)
        for right in tick_boxes[index + 1:]
    )
    ticks_are_clear = all(
        not tick.overlaps(obstacle)
        for tick in tick_boxes
        for obstacle in (*semantic_boxes, *other_boxes)
    )
    if not ticks_are_separate or not ticks_are_clear:
        raise ValueError(
            "rose radial tick layout overlaps labels; increase figure_size "
            "or use shorter labels"
        )


def _radial_geometry(
    config: RoseConfig,
    show_group_labels: bool,
) -> tuple[float, float, float]:
    span = float(config.r_max) - float(config.r_min)
    category_radius = float(config.r_max) + span * 0.07
    group_radius = float(config.r_max) + span * 0.27
    radial_limit = float(config.r_max) + span * (0.40 if show_group_labels else 0.20)
    values = (span, category_radius, group_radius, radial_limit)
    if (
        not all(isfinite(value) for value in values)
        or span <= 0.0
        or category_radius <= config.r_max
        or radial_limit <= category_radius
        or (show_group_labels and not category_radius < group_radius < radial_limit)
    ):
        raise ValueError("derived rose geometry must be finite and strictly ordered")
    return category_radius, group_radius, radial_limit


def plot_grouped_rose(
    frame: pd.DataFrame,
    config: RoseConfig,
    *,
    preset: Mapping[str, float] = rose_grouped,
) -> plt.Figure:
    """Render side-by-side petals on a common polar scale."""
    if not isinstance(config, RoseConfig):
        raise ValueError("config must be a RoseConfig")
    data = validate_rose_data(frame, config)
    appearance = _validated_preset(preset)
    colors = series_color_map(config.series_order, config.series_colors)
    for color in colors.values():
        if not mcolors.is_color_like(color):
            raise ValueError(f"invalid configured series color: {color!r}")
    group_sizes, group_names = _group_geometry(data, config)
    effective_group_gap = (
        config.group_gap_radians
        if config.group_order
        else config.category_gap_radians
    )
    angles, category_width = category_angles(
        group_sizes,
        config.category_gap_radians,
        effective_group_gap,
    )
    petal_width = (
        category_width * config.petal_fill_fraction / len(config.series_order)
    )
    if not isfinite(petal_width) or petal_width <= 0.0:
        raise ValueError("derived rose petal width must be finite and positive")
    transformed_values = radial_transform(
        data["value"].to_numpy(dtype=object),
        config.r_min,
        config.r_max,
        config.encoding,
    )
    transformed_ticks = radial_transform(
        config.r_ticks,
        config.r_min,
        config.r_max,
        config.encoding,
    )
    tick_labels = _radial_tick_labels(
        config.r_ticks, float(config.r_max) - float(config.r_min)
    )
    category_radius, group_radius, radial_limit = _radial_geometry(
        config, bool(group_names)
    )
    edge_colors = {series: _edge_color(colors[series]) for series in config.series_order}
    direction = -1 if config.clockwise else 1

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
            axis.set_yticks(transformed_ticks)
            axis.set_yticklabels(tick_labels)
            for tick_text in axis.get_yticklabels():
                tick_text.set_bbox({
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": GRID,
                    "linewidth": 0.3,
                    "alpha": 0.96,
                })
            axis.set_rlabel_position(90.0)
            axis.set_axisbelow(True)
            axis.yaxis.grid(
                True,
                color=GRID,
                linewidth=0.65,
                alpha=appearance["grid_alpha"],
            )
            axis.xaxis.grid(False)
            axis.spines["polar"].set_color(GRID)
            axis.spines["polar"].set_linewidth(0.65)

            value_index = 0
            for category_index, angle in enumerate(angles):
                for series_index, series in enumerate(config.series_order):
                    center = angle + (
                        series_index - (len(config.series_order) - 1) / 2.0
                    ) * petal_width
                    transformed = transformed_values[value_index]
                    petal = axis.bar(
                        center,
                        transformed - config.r_min,
                        width=petal_width,
                        bottom=config.r_min,
                        align="center",
                        color=colors[series],
                        edgecolor=edge_colors[series],
                        linewidth=appearance["edge_width"],
                        alpha=appearance["bar_alpha"],
                        zorder=2,
                    )[0]
                    petal.set_gid("polarviz-petal")
                    value_index += 1

            label_font_size = max(
                5.5, 8.0 - 0.20 * max(0, len(config.category_order) - 8)
            )
            category_texts = []
            for angle, label in zip(angles, config.category_order):
                text = axis.text(
                    angle,
                    category_radius,
                    label,
                    ha="center",
                    va="center",
                    rotation=_upright_tangent_rotation(
                        angle, config.start_angle_degrees, direction
                    ),
                    rotation_mode="anchor",
                    color=INK,
                    fontsize=label_font_size,
                    clip_on=False,
                    zorder=4,
                )
                text.set_gid("polarviz-category-label")
                category_texts.append(text)

            group_texts = []
            start = 0
            for group, size in zip(group_names, group_sizes):
                group_angle = float(np.mean(angles[start:start + size]))
                text = axis.text(
                    group_angle,
                    group_radius,
                    group,
                    ha="center",
                    va="center",
                    rotation=_upright_tangent_rotation(
                        group_angle, config.start_angle_degrees, direction
                    ),
                    rotation_mode="anchor",
                    color=INK,
                    alpha=0.72,
                    fontsize=max(4.5, label_font_size - 0.75),
                    fontweight="medium",
                    clip_on=False,
                    zorder=4,
                )
                text.set_gid("polarviz-group-label")
                group_texts.append(text)
                start += size

            handles = [
                Patch(
                    facecolor=colors[series],
                    edgecolor=edge_colors[series],
                    linewidth=appearance["edge_width"],
                    alpha=appearance["bar_alpha"],
                )
                for series in config.series_order
            ]
            if config.title is not None:
                axis.set_title(config.title, pad=18.0)
            figure.canvas.draw()
            _fit_legend(
                figure,
                axis,
                handles,
                config.series_order,
                config.legend_columns,
                appearance["legend_y"],
            )
            if config.title is not None:
                _fit_title(figure, axis, config.title)
            semantic_obstacles = [axis.get_legend()]
            if axis.get_title():
                semantic_obstacles.append(axis.title)
            _fit_labels(
                figure,
                category_texts,
                config.category_order,
                group_texts,
                group_names,
                label_font_size,
                semantic_obstacles,
            )
            tick_obstacles = [
                *category_texts,
                *group_texts,
                axis.get_legend(),
            ]
            if axis.get_title():
                tick_obstacles.append(axis.title)
            _fit_radial_ticks(
                figure,
                axis,
                config.start_angle_degrees,
                direction,
                tick_obstacles,
            )
            _verify_combined_layout(
                figure,
                axis,
                (*category_texts, *group_texts),
            )

            axis._polarviz_category_count = len(config.category_order)
            axis._polarviz_series_count = len(config.series_order)
            axis._polarviz_petal_count = len(config.category_order) * len(config.series_order)
            axis._polarviz_group_count = len(group_names) if group_names else 1
            axis._polarviz_series_colors = tuple(
                colors[series] for series in config.series_order
            )
            axis._polarviz_encoding = config.encoding
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    return figure
