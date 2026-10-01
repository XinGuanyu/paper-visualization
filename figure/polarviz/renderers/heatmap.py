"""Reusable annotated heatmap renderer."""

from __future__ import annotations

from collections.abc import Mapping
from math import floor, isfinite, log10
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import pandas as pd

from ..config import HeatmapConfig
from ..data import validate_heatmap_data
from ..presets.heatmap_standard import HEATMAP_STANDARD
from ..style import paper_style
from ._cartesian_layout import LayoutDrawBudget, inside, metadata, wrapped


_PRESET_KEYS = frozenset(
    {"grid_color", "grid_alpha", "annotation_dark", "annotation_light"}
)
_MIN_COLORBAR_POSITION_GAP = 0.02


class _StableNormalize(mcolors.Normalize):
    """Linear normalization that does not overflow at finite float limits."""

    def __init__(self, vmin: float, vmax: float) -> None:
        super().__init__(vmin=vmin, vmax=vmax, clip=False)
        self._polarviz_scale = max(abs(float(vmin)), abs(float(vmax)))
        if self._polarviz_scale == 0:
            self._polarviz_scale = 1.0
        self._scaled_min = float(vmin) / self._polarviz_scale
        self._scaled_max = float(vmax) / self._polarviz_scale
        self._scaled_span = self._scaled_max - self._scaled_min

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result, scalar = self.process_value(value)
        mask = np.ma.getmaskarray(result)
        scaled = np.asarray(result.data, dtype=float) / self._polarviz_scale
        normalized = (scaled - self._scaled_min) / self._scaled_span
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        output = np.ma.array(normalized, mask=mask, copy=False)
        return output[0] if scalar else output

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")
        result = np.asarray(value, dtype=float)
        return (
            result * self._scaled_span + self._scaled_min
        ) * self._polarviz_scale


class _ColorbarFormatter(Formatter):
    """Format a normalized colorbar using the heatmap's actual data domain."""

    def __init__(
        self,
        norm: _StableNormalize,
        tick_positions: tuple[float, ...],
        tick_values: tuple[float, ...],
    ) -> None:
        self.norm = norm
        self._offset, self._factor = _colorbar_decoration(norm)
        self._tick_positions = tick_positions
        labels: list[str] = []
        for value in tick_values:
            for precision in (6, 9, 12, 15, 17):
                self._precision = precision
                label = self._format_actual(value)
                if self._reconstruct(label) == value:
                    labels.append(label)
                    break
            else:
                labels = []
                break
        if not labels or len(labels) != len(set(labels)):
            # If an offset or scale cannot express every tick exactly even at
            # binary64 round-trip precision, prefer full actual values over a
            # compact but misleading decoration.
            self._offset = None
            self._factor = None
            self._precision = 17
            labels = [self._format_actual(value) for value in tick_values]
        self._tick_labels = tuple(labels)

    def _plain(self, value: float) -> str:
        if value == 0:
            return "0"
        return f"{value:.{self._precision}g}".replace("-", "−")

    def _format_actual(self, actual: float) -> str:
        if self._factor is not None:
            return self._plain(actual / self._factor)
        if self._offset is not None:
            # Both operands are actual representable tick floats.  Their
            # direct difference preserves the narrow-range ULP semantics that
            # would be lost by reconstructing a large value from 0..1.
            actual = actual - self._offset
        return self._plain(actual)

    def _reconstruct(self, label: str) -> float:
        displayed = float(label.replace("−", "-"))
        if self._factor is not None:
            return displayed * self._factor
        if self._offset is not None:
            return self._offset + displayed
        return displayed

    def __call__(self, value, pos=None) -> str:
        del pos
        normalized = float(value)
        if self._tick_positions:
            index = min(
                range(len(self._tick_positions)),
                key=lambda item: abs(self._tick_positions[item] - normalized),
            )
            if abs(self._tick_positions[index] - normalized) <= 1e-12:
                return self._tick_labels[index]
        scaled = normalized * self.norm._scaled_span + self.norm._scaled_min
        return self._format_actual(scaled * self.norm._polarviz_scale)

    def get_offset(self) -> str:
        if self._offset is not None:
            sign = "+" if self._offset >= 0 else "−"
            magnitude = _readable_float(abs(self._offset))
            return f"{sign}{magnitude}"
        if self._factor is not None:
            return f"×{_readable_float(self._factor)}"
        return ""


def _readable_float(value: float) -> str:
    text = repr(float(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent):+d}"
    return text.replace("-", "−")


def _colorbar_decoration(
    norm: _StableNormalize,
) -> tuple[float | None, float | None]:
    span = norm._scaled_span * norm._polarviz_scale
    if isfinite(span) and span > 0 and abs(float(norm.vmin)) > span * 10_000:
        return float(norm.vmin), None
    magnitude = max(abs(float(norm.vmin)), abs(float(norm.vmax)))
    if magnitude >= 1e5 or (0 < magnitude < 1e-3):
        exponent = floor(log10(magnitude))
        factor = 10.0**exponent
        if not isfinite(factor) or factor == 0:
            factor = magnitude
        return None, factor
    return None, None


def _display_value(
    actual: float,
    offset: float | None,
    factor: float | None,
) -> float:
    if offset is not None:
        return actual - offset
    if factor is not None:
        return actual / factor
    return actual


def _actual_value(
    displayed: float,
    offset: float | None,
    factor: float | None,
) -> float:
    if offset is not None:
        return offset + displayed
    if factor is not None:
        return factor * displayed
    return displayed


def _colorbar_ticks(
    norm: _StableNormalize,
    maximum: int = 6,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    offset, factor = _colorbar_decoration(norm)
    v_min = float(norm.vmin)
    v_max = float(norm.vmax)
    display_min = _display_value(v_min, offset, factor)
    display_max = _display_value(v_max, offset, factor)
    locator = MaxNLocator(
        nbins=max(1, maximum - 1),
        steps=(1, 2, 2.5, 5, 10),
        min_n_ticks=2,
    )
    nice = locator.tick_values(display_min, display_max)
    values = [v_min]
    for candidate in nice:
        if not display_min < candidate < display_max:
            continue
        displayed = float(f"{float(candidate):.12g}")
        actual = _actual_value(displayed, offset, factor)
        if v_min < actual < v_max and actual > values[-1]:
            values.append(actual)
    values.append(v_max)
    values = list(dict.fromkeys(values))
    endpoint_position = float(norm(v_max))
    kept_values = [v_min]
    kept_positions = [float(norm(v_min))]
    for value in values[1:-1]:
        position = float(norm(value))
        if (
            position > kept_positions[-1] + _MIN_COLORBAR_POSITION_GAP
            and position < endpoint_position - _MIN_COLORBAR_POSITION_GAP
        ):
            kept_values.append(value)
            kept_positions.append(position)
    kept_values.append(v_max)
    kept_positions.append(endpoint_position)
    if len(kept_values) > maximum:
        indices = np.linspace(0, len(kept_values) - 1, maximum).round().astype(int)
        selected = tuple(dict.fromkeys(int(index) for index in indices))
        kept_values = [kept_values[index] for index in selected]
        kept_positions = [kept_positions[index] for index in selected]
    return tuple(kept_positions), tuple(kept_values)


def _validated_preset(preset: Mapping[str, object]) -> dict[str, object]:
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
    for key in ("grid_color", "annotation_dark", "annotation_light"):
        value = values[key]
        if not isinstance(value, str) or not mcolors.is_color_like(value):
            raise ValueError(f"preset {key} must be a valid Matplotlib color")
    alpha = values["grid_alpha"]
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, Real)
        or not isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("preset grid_alpha must be between 0 and 1")
    values["grid_alpha"] = float(alpha)
    return values


def _composite(
    foreground: tuple[float, float, float, float],
    background: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    alpha = foreground[3] + background[3] * (1 - foreground[3])
    if alpha <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        *tuple(
            (foreground[index] * foreground[3]
             + background[index] * background[3] * (1 - foreground[3]))
            / alpha
            for index in range(3)
        ),
        alpha,
    )


def _effective_cell_rgba(axis, figure: Figure, rgba) -> tuple[float, float, float, float]:
    figure_background = mcolors.to_rgba(figure.get_facecolor())
    axis_background = _composite(mcolors.to_rgba(axis.get_facecolor()), figure_background)
    return _composite(tuple(float(value) for value in rgba), axis_background)


def _luminance(rgba: tuple[float, float, float, float]) -> float:
    red, green, blue = rgba[:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _number_label(value: float, decimals: int) -> str:
    absolute = abs(value)
    scientific = absolute >= 10 ** max(7, decimals + 5) or (
        absolute > 0 and absolute < 10 ** -(decimals + 2)
    )
    magnitude = (
        f"{absolute:.{decimals}e}" if scientific else f"{absolute:.{decimals}f}"
    ).replace("e-", "e−")
    return f"−{magnitude}" if value < 0 else magnitude


def _matplotlib_float_candidate(value: Real, context: str) -> tuple[float, bool]:
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"heatmap chart {context} cannot be represented as a finite "
            "Matplotlib float"
        ) from error
    if not np.isfinite(converted):
        raise ValueError(
            f"heatmap chart {context} cannot be represented as a finite "
            "Matplotlib float"
        )
    return converted, converted == 0.0 and value != 0


def _float_matrix(
    frame: pd.DataFrame,
    config: HeatmapConfig,
) -> tuple[np.ndarray, float, float]:
    v_min, minimum_underflow = _matplotlib_float_candidate(
        config.v_min, "range minimum"
    )
    v_max, maximum_underflow = _matplotlib_float_candidate(
        config.v_max, "range maximum"
    )
    if not v_min < v_max:
        raise ValueError(
            "heatmap chart range cannot be represented as strictly increasing "
            "Matplotlib floats"
        )
    if minimum_underflow or maximum_underflow:
        raise ValueError(
            "heatmap chart range cannot be represented as finite Matplotlib floats"
        )

    converted: list[float] = []
    for row in frame.itertuples(index=False):
        value, underflow = _matplotlib_float_candidate(
            row.value, f"value at ({row.row!r}, {row.column!r})"
        )
        if underflow:
            raise ValueError(
                f"heatmap chart value at ({row.row!r}, {row.column!r}) cannot "
                "be represented as a finite Matplotlib float"
            )
        converted.append(value)
    return (
        np.asarray(converted, dtype=float).reshape(
            len(config.row_order), len(config.column_order)
        ),
        v_min,
        v_max,
    )


def _semantic_artists(axis, colorbar) -> list:
    artists = [
        *[tick for tick in axis.get_xticklabels() if tick.get_visible()],
        *[tick for tick in axis.get_yticklabels() if tick.get_visible()],
    ]
    for artist, text in (
        (axis.title, axis.get_title()),
        (axis.xaxis.label, axis.get_xlabel()),
        (axis.yaxis.label, axis.get_ylabel()),
    ):
        if text and artist.get_visible():
            artists.append(artist)
    for offset in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
        if offset.get_visible() and offset.get_text().strip():
            artists.append(offset)
    if colorbar is not None:
        color_axis = colorbar.ax
        artists.extend(
            tick for tick in color_axis.get_yticklabels() if tick.get_visible()
        )
        if color_axis.get_ylabel():
            artists.append(color_axis.yaxis.label)
        offset = color_axis.yaxis.get_offset_text()
        if offset.get_visible() and offset.get_text().strip():
            artists.append(offset)
    return artists


def _boxes_clear(artists, renderer) -> bool:
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    return all(
        not left.overlaps(right)
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
    )


def _fit_layout(
    figure: Figure,
    axis,
    colorbar,
    config: HeatmapConfig,
    draw_budget: LayoutDrawBudget,
) -> None:
    width, _ = figure.get_size_inches()
    row_labels = dict(config.row_labels)
    column_labels = dict(config.column_labels)
    columns = [column_labels.get(value, value) for value in config.column_order]
    rows = [row_labels.get(value, value) for value in config.row_order]
    column_wrap = max(3, min(24, int(width * 11 / len(columns))))
    row_wrap = max(4, min(30, int(width * 5)))
    axis.set_xticklabels([wrapped(value, column_wrap) for value in columns])
    axis.set_yticklabels([wrapped(value, row_wrap) for value in rows])
    axis.set_xlabel(wrapped(config.x_label, max(10, int(width * 9))) if config.x_label else "")
    axis.set_ylabel(wrapped(config.y_label, max(10, int(width * 8))) if config.y_label else "")
    axis.set_title(wrapped(config.title, max(14, int(width * 11))) if config.title else "", pad=8)

    right_edge = 0.96
    cbar_width = 0.035
    cbar_gap = 0.045
    for font_size in (7.5, 7.0, 6.5, 6.0, 5.5):
        for rotation in (0, 30, 45, 60, 90):
            for tick in axis.get_xticklabels():
                tick.set_fontsize(font_size)
                tick.set_rotation(rotation)
                tick.set_rotation_mode("anchor")
                tick.set_ha("right" if rotation else "center")
            for tick in axis.get_yticklabels():
                tick.set_fontsize(font_size)
            if colorbar is not None:
                for tick in colorbar.ax.get_yticklabels():
                    tick.set_fontsize(font_size)

            left, bottom, top = 0.16, 0.15, 0.90
            outer_right = right_edge
            for _ in range(6):
                main_right = (
                    outer_right - cbar_width - cbar_gap
                    if colorbar is not None
                    else outer_right
                )
                if left + 0.20 >= main_right or bottom + 0.20 >= top:
                    break
                axis.set_position((left, bottom, main_right - left, top - bottom))
                if colorbar is not None:
                    colorbar.ax.set_position(
                        (main_right + cbar_gap, bottom, cbar_width, top - bottom)
                    )
                draw_budget.draw()
                renderer = draw_budget.renderer
                artists = _semantic_artists(axis, colorbar)
                boxes = [artist.get_window_extent(renderer) for artist in artists]
                outer = figure.bbox
                overflow_left = max((outer.x0 + 1 - box.x0 for box in boxes), default=0)
                overflow_right = max((box.x1 - outer.x1 + 1 for box in boxes), default=0)
                overflow_bottom = max((outer.y0 + 1 - box.y0 for box in boxes), default=0)
                overflow_top = max((box.y1 - outer.y1 + 1 for box in boxes), default=0)
                if max(overflow_left, overflow_right, overflow_bottom, overflow_top) <= 0:
                    if all(inside(artist, figure, renderer) for artist in artists) and _boxes_clear(artists, renderer):
                        axis._polarviz_x_label_rotation = rotation
                        return
                    break
                left += max(0, overflow_left + 4) / outer.width
                outer_right -= max(0, overflow_right + 4) / outer.width
                bottom += max(0, overflow_bottom + 4) / outer.height
                top -= max(0, overflow_top + 4) / outer.height
    raise ValueError(
        "heatmap chart layout cannot fit title, labels, ticks, and colorbar "
        "inside figure_size"
    )


def _fit_annotations(
    figure: Figure,
    axis,
    texts: list,
    draw_budget: LayoutDrawBudget,
) -> None:
    if not texts:
        return
    for size in (8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0):
        for text in texts:
            text.set_fontsize(size)
        draw_budget.draw()
        renderer = draw_budget.renderer
        boxes = [text.get_window_extent(renderer) for text in texts]
        cell_boxes = []
        for text in texts:
            x, y = text.get_position()
            corners = axis.transData.transform(
                ((x - 0.5, y - 0.5), (x + 0.5, y + 0.5))
            )
            cell_boxes.append(
                (
                    min(corners[:, 0]),
                    min(corners[:, 1]),
                    max(corners[:, 0]),
                    max(corners[:, 1]),
                )
            )
        if (
            all(inside(text, figure, renderer) for text in texts)
            and all(
                box.x0 >= axis.bbox.x0 + 1
                and box.x1 <= axis.bbox.x1 - 1
                and box.y0 >= axis.bbox.y0 + 1
                and box.y1 <= axis.bbox.y1 - 1
                for box in boxes
            )
            and all(
                box.x0 >= cell_x0 + 1
                and box.x1 <= cell_x1 - 1
                and box.y0 >= cell_y0 + 1
                and box.y1 <= cell_y1 - 1
                for box, (cell_x0, cell_y0, cell_x1, cell_y1)
                in zip(boxes, cell_boxes)
            )
        ):
            return
    raise ValueError(
        "heatmap chart layout cannot fit cell annotations inside figure_size"
    )


def plot_heatmap(
    data: pd.DataFrame,
    config: HeatmapConfig,
    preset: Mapping[str, object] = HEATMAP_STANDARD,
) -> Figure:
    """Render an exact rectangular heatmap in configured semantic order."""
    if not isinstance(config, HeatmapConfig):
        raise ValueError("config must be a HeatmapConfig")
    frame = validate_heatmap_data(data, config.row_order, config.column_order)
    matrix, v_min, v_max = _float_matrix(frame, config)
    values = frame["value"]
    if ((values < config.v_min) | (values > config.v_max)).any():
        raise ValueError(f"value must lie within [{config.v_min}, {config.v_max}]")
    appearance = _validated_preset(preset)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "polarviz-common-chart",
        config.color_scale,
        N=128 * (len(config.color_scale) - 1) + 1,
    )
    norm = _StableNormalize(v_min, v_max)

    active_figure = plt.gcf() if plt.get_fignums() else None
    figure: Figure | None = None
    try:
        with paper_style():
            figure, axis = plt.subplots(figsize=config.figure_size)
            draw_budget = LayoutDrawBudget(figure, "heatmap chart")
            image = axis.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                interpolation_stage="rgba",
            )
            metadata(
                image,
                "heatmap-image",
                (config.row_order, config.column_order),
                "polarviz-heatmap-image",
            )
            image._polarviz_color_scale = tuple(config.color_scale)
            axis.set_xticks(np.arange(len(config.column_order)))
            axis.set_yticks(np.arange(len(config.row_order)))
            axis.set_xticklabels(config.column_order)
            axis.set_yticklabels(config.row_order)
            axis.set_xticks(np.arange(-0.5, len(config.column_order), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(config.row_order), 1), minor=True)
            axis.grid(
                which="minor",
                color=appearance["grid_color"],
                linewidth=config.cell_gap_width,
                alpha=appearance["grid_alpha"],
            )
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(which="major", length=0)
            axis._polarviz_grid_color = appearance["grid_color"]
            axis._polarviz_grid_alpha = appearance["grid_alpha"]

            annotation_texts = []
            if config.show_annotations:
                for row_index, row in enumerate(config.row_order):
                    for column_index, column in enumerate(config.column_order):
                        value = float(matrix[row_index, column_index])
                        rgba = _effective_cell_rgba(axis, figure, cmap(norm(value)))
                        color = (
                            appearance["annotation_dark"]
                            if _luminance(rgba) >= config.text_contrast_threshold
                            else appearance["annotation_light"]
                        )
                        text = axis.text(
                            column_index,
                            row_index,
                            _number_label(value, config.annotation_decimals),
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                            clip_on=False,
                            zorder=3,
                        )
                        metadata(
                            text,
                            "heatmap-cell-label",
                            (row, column),
                            "polarviz-heatmap-cell-label",
                        )
                        annotation_texts.append(text)

            colorbar = None
            if config.show_colorbar:
                colorbar_mappable = ScalarMappable(
                    norm=mcolors.Normalize(0.0, 1.0), cmap=cmap
                )
                colorbar_mappable.set_array(np.asarray([0.0, 1.0]))
                colorbar = figure.colorbar(colorbar_mappable, ax=axis)
                tick_positions, tick_values = _colorbar_ticks(norm)
                colorbar.set_ticks(tick_positions)
                colorbar.formatter = _ColorbarFormatter(
                    norm, tick_positions, tick_values
                )
                colorbar.update_ticks()
                colorbar._polarviz_data_norm = norm
                colorbar.ax._polarviz_tick_values = tick_values
                colorbar.set_label(config.colorbar_label or "")
                metadata(
                    colorbar.outline,
                    "heatmap-colorbar",
                    ("value",),
                    "polarviz-heatmap-colorbar",
                )
                if config.colorbar_label:
                    metadata(
                        colorbar.ax.yaxis.label,
                        "heatmap-colorbar-label",
                        (config.colorbar_label,),
                        "polarviz-heatmap-colorbar-label",
                    )

            _fit_layout(figure, axis, colorbar, config, draw_budget)
            _fit_annotations(figure, axis, annotation_texts, draw_budget)
            if colorbar is not None:
                draw_budget.draw()
                for index, tick in enumerate(colorbar.ax.get_yticklabels()):
                    metadata(
                        tick,
                        "heatmap-colorbar-tick",
                        (index,),
                        "polarviz-heatmap-colorbar-tick",
                    )
                offset = colorbar.ax.yaxis.get_offset_text()
                if offset.get_visible() and offset.get_text().strip():
                    metadata(
                        offset,
                        "heatmap-colorbar-offset",
                        ("value",),
                        "polarviz-heatmap-colorbar-offset",
                    )
            axis._polarviz_layout_draws = draw_budget.draws
            return figure
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise
    finally:
        if active_figure is not None and plt.fignum_exists(active_figure.number):
            plt.figure(active_figure.number)
