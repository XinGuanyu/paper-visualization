"""Reusable Cartesian line-chart renderer."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox, IdentityTransform, offset_copy
import numpy as np
import pandas as pd

from ..config import LineConfig
from ..data import validate_line_data
from ..presets.line_standard import LINE_STANDARD
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
    {"grid_alpha", "line_alpha", "edge_width", "label_offset_points", "legend_y"}
)
_DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
_MARKER_PHASE_PADDING_PIXELS = 1.0


def _validated_preset(preset: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(preset, Mapping):
        raise ValueError("preset must be a mapping")
    values = dict(preset)
    if any(not isinstance(key, str) for key in values):
        raise ValueError("preset keys must all be strings")
    if set(values) != _PRESET_KEYS:
        missing = sorted(_PRESET_KEYS - set(values))
        extra = sorted(set(values) - _PRESET_KEYS)
        raise ValueError(f"preset keys must match exactly; missing={missing}, extra={extra}")
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"preset {key} must be a finite number")
    valid = {
        "grid_alpha": 0 <= values["grid_alpha"] <= 1,
        "line_alpha": 0 <= values["line_alpha"] <= 1,
        "edge_width": values["edge_width"] > 0,
        "label_offset_points": values["label_offset_points"] >= 0,
        "legend_y": 1 <= values["legend_y"] <= 2,
    }
    invalid = [key for key, accepted in valid.items() if not accepted]
    if invalid:
        raise ValueError(f"preset values outside valid ranges: {invalid}")
    return {key: float(value) for key, value in values.items()}


def _series_colors(config: LineConfig) -> dict[str, str]:
    explicit = dict(config.series_colors)
    missing = [
        series
        for index, series in enumerate(config.series_order)
        if index >= len(COMMON_CHART_PALETTE) and series not in explicit
    ]
    if missing:
        raise ValueError(
            "series require explicit color mappings beyond the eight-color common palette: "
            f"{missing}"
        )
    return {
        series: (
            explicit[series]
            if series in explicit
            else COMMON_CHART_PALETTE[index]
        )
        for index, series in enumerate(config.series_order)
    }


def _dark_edge(color: str) -> str:
    red, green, blue = mcolors.to_rgb(color)
    return mcolors.to_hex((red * 0.55, green * 0.55, blue * 0.55))


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
        handlelength=2.2,
        columnspacing=1.1,
        fontsize=font_size,
    )
    for handle, series in zip(legend.legend_handles, series_order):
        metadata(handle, "legend-key", (series,), "polarviz-line-legend-key")
    return legend


def _rendered_marker_extents(artist, figure) -> tuple[float, float, float, float]:
    """Measure one marker's real antialiased pixel footprint offscreen."""
    dpi = float(figure.dpi)
    nominal_pixels = max(
        float(artist.get_markersize()),
        float(artist.get_markeredgewidth()),
        1.0,
    ) * dpi / 72.0
    maximum_side = max(
        64,
        int(np.ceil(max(figure.bbox.width, figure.bbox.height) * 2.0 + 32.0)),
    )
    side = min(maximum_side, max(64, int(np.ceil(nominal_pixels * 8.0 + 16.0))))
    while True:
        center = side // 2
        renderer = RendererAgg(side, side, dpi)
        proxy = Line2D(
            [center],
            [center],
            linestyle="None",
            marker=artist.get_marker(),
            markersize=artist.get_markersize(),
            markeredgewidth=artist.get_markeredgewidth(),
            markerfacecolor="black",
            markeredgecolor="black",
            fillstyle=artist.get_fillstyle(),
            alpha=1.0,
            transform=IdentityTransform(),
        )
        proxy.set_figure(Figure(dpi=dpi))
        proxy.draw(renderer)
        alpha = np.asarray(renderer.buffer_rgba())[:, :, 3]
        rows, columns = np.nonzero(alpha)
        if not len(rows):
            return (0.0, 0.0, 0.0, 0.0)
        touches_edge = (
            columns.min() <= 1
            or rows.min() <= 1
            or columns.max() >= side - 2
            or rows.max() >= side - 2
        )
        if touches_edge:
            if side >= maximum_side:
                raise ValueError(
                    "line chart layout cannot fit value labels clear of the "
                    "rendered marker footprint inside figure_size"
                )
            side = min(maximum_side, side * 2)
            continue
        return (
            float(center - columns.min()) + _MARKER_PHASE_PADDING_PIXELS,
            float(columns.max() + 1 - center) + _MARKER_PHASE_PADDING_PIXELS,
            float(center - side + rows.max() + 1) + _MARKER_PHASE_PADDING_PIXELS,
            float(side - rows.min() - center) + _MARKER_PHASE_PADDING_PIXELS,
        )


def _artist_has_visible_color(artist, colors) -> bool:
    if not artist.get_visible():
        return False
    alpha = artist.get_alpha()
    if alpha is not None and alpha <= 0.0:
        return False
    return any(
        mcolors.to_rgba(color, alpha=alpha)[3] > 0.0
        for color in colors
    )


def _marker_geometry(marker_artists, axis, figure):
    boxes = []
    extents_by_series = {}
    extent_cache = {}
    for artist in marker_artists:
        if not _artist_has_visible_color(
            artist,
            (artist.get_markerfacecolor(), artist.get_markeredgecolor()),
        ):
            continue
        cache_key = (
            repr(artist.get_marker()),
            float(artist.get_markersize()),
            float(artist.get_markeredgewidth()),
            artist.get_fillstyle(),
            float(figure.dpi),
        )
        extents = extent_cache.get(cache_key)
        if extents is None:
            extents = _rendered_marker_extents(artist, figure)
            extent_cache[cache_key] = extents
        left, right, bottom, top = extents
        extents_by_series[artist._polarviz_key[0]] = extents
        points = axis.transData.transform(
            np.column_stack((artist.get_xdata(), artist.get_ydata()))
        )
        boxes.extend(
            Bbox.from_extents(x - left, y - bottom, x + right, y + top)
            for x, y in points
        )
    return boxes, extents_by_series


def _segment_polygon(start, end, radius):
    difference = end - start
    length = float(np.hypot(*difference))
    if length <= 0.0:
        return None
    direction = difference / length
    normal = np.array((-direction[1], direction[0]))
    vertices = np.vstack(
        (
            start + normal * radius,
            end + normal * radius,
            end - normal * radius,
            start - normal * radius,
            start + normal * radius,
        )
    )
    return MplPath(vertices, closed=True)


def _backend_miter_limit(figure) -> float:
    module = type(figure.canvas).__module__
    if "backend_svg" in module:
        return 4.0
    return 10.0


def _line_segment_corridors(line_artists, axis, figure):
    corridors = []
    for artist in line_artists:
        if (
            not _artist_has_visible_color(artist, (artist.get_color(),))
            or artist.get_linestyle() in {"None", "none", "", " "}
            or artist.get_linewidth() <= 0.0
        ):
            continue
        points = axis.transData.transform(
            np.column_stack((artist.get_xdata(), artist.get_ydata()))
        )
        antialias_padding = 1.0
        half_width = artist.get_linewidth() * figure.dpi / 144.0
        radius = half_width + antialias_padding
        capstyle = (
            artist.get_dash_capstyle()
            if artist.is_dashed()
            else artist.get_solid_capstyle()
        )
        joinstyle = (
            artist.get_dash_joinstyle()
            if artist.is_dashed()
            else artist.get_solid_joinstyle()
        )
        segment_count = max(0, len(points) - 1)
        for index, (raw_start, raw_end) in enumerate(
            zip(points[:-1], points[1:])
        ):
            difference = raw_end - raw_start
            length = float(np.hypot(*difference))
            if length <= 0.0:
                continue
            direction = difference / length
            start = raw_start.astype(float, copy=True)
            end = raw_end.astype(float, copy=True)
            if index == 0:
                if capstyle == "projecting":
                    start -= direction * radius
                elif capstyle == "butt":
                    start -= direction * antialias_padding
                elif capstyle == "round":
                    corridors.append(("circle", raw_start, radius))
            if index == segment_count - 1:
                if capstyle == "projecting":
                    end += direction * radius
                elif capstyle == "butt":
                    end += direction * antialias_padding
                elif capstyle == "round":
                    corridors.append(("circle", raw_end, radius))
            polygon = _segment_polygon(start, end, radius)
            if polygon is not None:
                corridors.append(("polygon", polygon))
        for previous, vertex, following in zip(
            points[:-2], points[1:-1], points[2:]
        ):
            incoming = (vertex - previous).astype(float)
            outgoing = (following - vertex).astype(float)
            incoming_length = float(np.hypot(*incoming))
            outgoing_length = float(np.hypot(*outgoing))
            if incoming_length <= 0.0 or outgoing_length <= 0.0:
                continue
            incoming /= incoming_length
            outgoing /= outgoing_length
            if joinstyle == "round":
                corridors.append(("circle", vertex, radius))
                continue
            turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if abs(turn) <= 1e-12:
                continue
            side = -1.0 if turn > 0.0 else 1.0
            incoming_normal = np.array((-incoming[1], incoming[0]))
            outgoing_normal = np.array((-outgoing[1], outgoing[0]))
            first_outer = vertex + side * incoming_normal * radius
            second_outer = vertex + side * outgoing_normal * radius
            join_vertices = [vertex, first_outer]
            if joinstyle == "miter":
                base_first_outer = (
                    vertex + side * incoming_normal * half_width
                )
                base_second_outer = (
                    vertex + side * outgoing_normal * half_width
                )
                coefficients = np.column_stack((incoming, -outgoing))
                try:
                    distance, _ = np.linalg.solve(
                        coefficients,
                        base_second_outer - base_first_outer,
                    )
                except np.linalg.LinAlgError:
                    continue
                miter = base_first_outer + incoming * distance
                miter_ratio = (
                    float(np.hypot(*(miter - vertex)))
                    / (2.0 * half_width)
                )
                if miter_ratio <= _backend_miter_limit(figure):
                    join_vertices = [
                        vertex,
                        base_first_outer,
                        miter,
                        base_second_outer,
                        vertex,
                    ]
                    corridors.append(
                        (
                            "polygon",
                            MplPath(
                                np.asarray(join_vertices), closed=True
                            ),
                        )
                    )
                    for edge_start, edge_end in (
                        (base_first_outer, miter),
                        (miter, base_second_outer),
                    ):
                        edge = _segment_polygon(
                            edge_start, edge_end, antialias_padding
                        )
                        if edge is not None:
                            corridors.append(("polygon", edge))
                    corridors.append(
                        ("circle", miter, antialias_padding)
                    )
                    continue
            join_vertices.extend((second_outer, vertex))
            corridors.append(
                (
                    "polygon",
                    MplPath(np.asarray(join_vertices), closed=True),
                )
            )
    return corridors


def _hits_line_corridor(box: Bbox, corridors) -> bool:
    for kind, geometry, *values in corridors:
        if kind == "polygon":
            if geometry.intersects_bbox(box, filled=True):
                return True
            continue
        radius = values[0]
        center = geometry
        closest_x = min(max(center[0], box.x0), box.x1)
        closest_y = min(max(center[1], box.y0), box.y1)
        if float(np.hypot(center[0] - closest_x, center[1] - closest_y)) <= radius:
            return True
    return False


def _fit_value_labels(
    figure,
    axis,
    texts,
    offset_points,
    marker_artists,
    line_artists,
    draw_budget,
) -> None:
    if not texts:
        return
    obstacles = semantic_obstacles(axis)
    placements = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    )
    marker_boxes, marker_extents = _marker_geometry(marker_artists, axis, figure)
    extra_offset = 2.0 + offset_points
    for font_size in (7.0, 6.5, 6.0, 5.5):
        draw_budget.draw()
        renderer = draw_budget.renderer
        obstacle_boxes = [obstacle.get_window_extent(renderer) for obstacle in obstacles]
        corridors = _line_segment_corridors(line_artists, axis, figure)
        placed_boxes = []
        success = True
        for index, text in enumerate(texts):
            text.set_fontsize(font_size)
            found = False
            preferred = placements[index % len(placements):] + placements[:index % len(placements)]
            series = text._polarviz_key[1]
            left, right, bottom, top = marker_extents.get(
                series, (0.0, 0.0, 0.0, 0.0)
            )
            for horizontal, vertical in preferred:
                horizontal_extent = right if horizontal > 0 else left
                vertical_extent = top if vertical > 0 else bottom
                x_offset = (
                    horizontal
                    * (horizontal_extent * 72.0 / figure.dpi + extra_offset)
                    if horizontal
                    else 0.0
                )
                y_offset = (
                    vertical
                    * (vertical_extent * 72.0 / figure.dpi + extra_offset)
                    if vertical
                    else 0.0
                )
                text.set_transform(offset_copy(axis.transData, fig=figure, x=x_offset, y=y_offset, units="points"))
                text.set_ha("left" if horizontal > 0 else "right" if horizontal < 0 else "center")
                text.set_va("bottom" if vertical > 0 else "top" if vertical < 0 else "center")
                box = text.get_window_extent(renderer)
                if not inside(text, figure, renderer):
                    continue
                if box.y0 < axis.bbox.y0 or box.y1 > axis.bbox.y1:
                    continue
                if any(box.overlaps(other) for other in placed_boxes):
                    continue
                if any(box.overlaps(marker_box) for marker_box in marker_boxes):
                    continue
                if _hits_line_corridor(box, corridors):
                    continue
                if any(box.overlaps(obstacle_box) for obstacle_box in obstacle_boxes):
                    continue
                placed_boxes.append(box)
                found = True
                break
            if not found:
                success = False
                break
        if success:
            draw_budget.draw()
            renderer = draw_budget.renderer
            boxes = [text.get_window_extent(renderer) for text in texts]
            final_corridors = _line_segment_corridors(line_artists, axis, figure)
            if (
                all(inside(text, figure, renderer) for text in texts)
                and all(
                    not left.overlaps(right)
                    for index, left in enumerate(boxes)
                    for right in boxes[index + 1 :]
                )
                and not any(
                    box.overlaps(obstacle.get_window_extent(renderer))
                    for box in boxes
                    for obstacle in obstacles
                )
                and not any(
                    box.overlaps(marker_box)
                    for box in boxes
                    for marker_box in marker_boxes
                )
                and not any(
                    _hits_line_corridor(box, final_corridors)
                    for box in boxes
                )
            ):
                return
    raise ValueError(
        "line chart layout cannot fit value labels clear of markers, line "
        "segments, and semantic decorations inside figure_size"
    )


def plot_line_chart(
    data: pd.DataFrame,
    config: LineConfig,
    preset: Mapping[str, float] = LINE_STANDARD,
) -> plt.Figure:
    """Plot complete rectangular category-by-series data as ordered paths."""
    if not isinstance(config, LineConfig):
        raise ValueError("config must be a LineConfig")
    frame = validate_line_data(data, config.category_order, config.series_order)
    if ((frame["value"] < config.y_min) | (frame["value"] > config.y_max)).any():
        raise ValueError(f"value must lie within [{config.y_min}, {config.y_max}]")
    appearance = _validated_preset(preset)
    colors = _series_colors(config)
    styles = dict(config.line_styles)
    markers = dict(config.markers)
    display_map = dict(config.category_labels)
    display_labels = [display_map.get(category, category) for category in config.category_order]

    active_figure = plt.gcf() if plt.get_fignums() else None
    figure: plt.Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(figsize=config.figure_size)
            draw_budget = LayoutDrawBudget(figure, "line chart")
            x = np.arange(len(config.category_order), dtype=float)
            value_texts = []
            line_artists = []
            marker_artists = []
            handles = []
            for series_index, series in enumerate(config.series_order):
                values = frame.loc[frame["series"] == series, "value"].astype(float).to_numpy()
                color = colors[series]
                style = styles.get(series, "-")
                marker = markers.get(series, _DEFAULT_MARKERS[series_index % len(_DEFAULT_MARKERS)])
                line = axis.plot(
                    x, values, color=color, linestyle=style, linewidth=config.line_width,
                    alpha=appearance["line_alpha"], marker="None", zorder=2,
                )[0]
                metadata(line, "line-series", (series,), "polarviz-line-series")
                line_artists.append(line)
                if config.show_markers:
                    marker_line = axis.plot(
                        x, values, linestyle="None", marker=marker, markersize=config.marker_size,
                        markerfacecolor=color, markeredgecolor=_dark_edge(color),
                        markeredgewidth=appearance["edge_width"], alpha=appearance["line_alpha"], zorder=3,
                    )[0]
                    metadata(marker_line, "line-markers", (series,), "polarviz-line-markers")
                    marker_artists.append(marker_line)
                for category, x_value, value in zip(config.category_order, x, values):
                    if config.show_value_labels:
                        text = axis.text(
                            x_value, value, _value_label(float(value), config.value_decimals),
                            color=INK, fontsize=7, clip_on=False, zorder=4,
                        )
                        metadata(text, "line-value-label", (category, series), "polarviz-line-value-label")
                        value_texts.append(text)
                handle = Line2D(
                    [], [], color=color, linestyle=style, linewidth=config.line_width,
                    marker=marker if config.show_markers else "None",
                    markerfacecolor=color, markeredgecolor=_dark_edge(color),
                    markeredgewidth=appearance["edge_width"], alpha=appearance["line_alpha"],
                )
                metadata(handle, "legend-key", (series,), "polarviz-line-legend-key")
                handles.append(handle)

            axis.set_xlim(-0.24, max(len(config.category_order) - 0.76, 0.24))
            axis.set_ylim(config.y_min, config.y_max)
            axis.set_yticks(config.y_ticks)
            axis.set_xticks(x, labels=display_labels)
            axis.set_axisbelow(True)
            axis.yaxis.grid(True, color=GRID, linewidth=0.65, alpha=appearance["grid_alpha"])
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
                layout_name="line chart",
                legend_builder=_legend,
            )
            _fit_value_labels(
                figure,
                axis,
                value_texts,
                appearance["label_offset_points"],
                marker_artists,
                line_artists,
                draw_budget,
            )
            axis._polarviz_category_count = len(config.category_order)
            axis._polarviz_series_count = len(config.series_order)
            axis._polarviz_layout_draws = draw_budget.draws
            axis._polarviz_series_colors = tuple(colors[series] for series in config.series_order)
            return figure
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    finally:
        if active_figure is not None and plt.fignum_exists(active_figure.number):
            plt.figure(active_figure.number)
