"""Immutable public configuration for reusable visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from math import isfinite
from numbers import Rational, Real
from typing import Literal

from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.colors import is_color_like

from .presets.palette import COMMON_CHART_PALETTE


def _tuple(values: object, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error


def _ordered_tuple(values: object, name: str) -> tuple[object, ...]:
    """Normalize a bar field only when its input has deterministic iteration order."""
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be an ordered iterable")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an ordered iterable") from error


def _pairs(values: object, name: str) -> tuple[tuple[object, ...], ...]:
    if isinstance(values, (Mapping, Set)):
        raise ValueError(f"{name} must contain non-string two-item sequences")
    pairs: list[tuple[object, ...]] = []
    for item in _tuple(values, name):
        if isinstance(item, (str, bytes, Mapping, Set)):
            raise ValueError(f"{name} must contain non-string two-item sequences")
        try:
            pair = tuple(item)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                f"{name} must contain non-string two-item sequences"
            ) from error
        if len(pair) != 2:
            raise ValueError(f"{name} must contain non-string two-item sequences")
        pairs.append(pair)
    return tuple(pairs)


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _finite_heatmap(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = isinstance(value, Rational)
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def _labels(values: tuple[object, ...], name: str, minimum: int = 1) -> None:
    if len(values) < minimum or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must contain at least {minimum} non-empty labels")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique non-empty labels")


def _semantic_labels(values: tuple[object, ...], name: str) -> None:
    _labels(values, name)
    if any(value != value.strip() for value in values):  # type: ignore[union-attr]
        raise ValueError(f"{name} labels must not have leading or trailing whitespace")


def _radial(r_min: object, r_max: object, ticks: tuple[object, ...]) -> None:
    _finite(r_min, "r_min")
    _finite(r_max, "r_max")
    if any(isinstance(tick, bool) or not isinstance(tick, Real) or not isfinite(tick) for tick in ticks):
        raise ValueError("r_ticks must contain only finite numbers")
    if not r_min < r_max:  # type: ignore[operator]
        raise ValueError("r_min must be less than r_max")
    if not ticks or tuple(sorted(ticks)) != ticks or len(set(ticks)) != len(ticks):
        raise ValueError("r_ticks must be unique and strictly increasing")
    if any(tick < r_min or tick > r_max for tick in ticks):  # type: ignore[operator]
        raise ValueError("r_ticks must lie within [r_min, r_max]")


def _figure_size(figure_size: tuple[object, ...]) -> None:
    if len(figure_size) != 2 or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, Real)
        or not isfinite(dimension)
        or dimension <= 0
        for dimension in figure_size
    ):
        raise ValueError("figure_size dimensions must be finite positive numbers")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _axis_range(
    minimum: object,
    maximum: object,
    ticks: tuple[object, ...],
    minimum_name: str,
    maximum_name: str,
    ticks_name: str,
) -> None:
    _finite(minimum, minimum_name)
    _finite(maximum, maximum_name)
    if any(
        isinstance(tick, bool) or not isinstance(tick, Real) or not isfinite(tick)
        for tick in ticks
    ):
        raise ValueError(f"{ticks_name} must contain only finite numbers")
    if not minimum < maximum:  # type: ignore[operator]
        raise ValueError(f"{minimum_name} must be less than {maximum_name}")
    if not ticks or tuple(sorted(ticks)) != ticks or len(set(ticks)) != len(ticks):
        raise ValueError(f"{ticks_name} must be unique and strictly increasing")
    if any(tick < minimum or tick > maximum for tick in ticks):  # type: ignore[operator]
        raise ValueError(
            f"{ticks_name} must lie within [{minimum_name}, {maximum_name}]"
        )


def _fraction(
    value: object,
    name: str,
    *,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> None:
    interval = (
        ("[" if lower_inclusive else "(")
        + "0, 1"
        + ("]" if upper_inclusive else ")")
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and lie in {interval}")
    lower_valid = value >= 0 if lower_inclusive else value > 0
    upper_valid = value <= 1 if upper_inclusive else value < 1
    if not lower_valid or not upper_valid:
        raise ValueError(f"{name} must be finite and lie in {interval}")


def _category_labels(
    values: tuple[tuple[object, ...], ...], category_order: tuple[object, ...]
) -> None:
    if not values:
        return
    keys = tuple(pair[0] for pair in values)
    displays = tuple(pair[1] for pair in values)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (*keys, *displays)
    ) or len(set(keys)) != len(keys):
        raise ValueError(
            "category_labels must contain unique non-empty string keys and "
            "non-empty string displays"
        )
    if set(keys) != set(category_order):
        raise ValueError("category_labels must exactly cover category_order")


def _series_colors(
    values: tuple[tuple[object, ...], ...], series_order: tuple[object, ...]
) -> None:
    keys = tuple(pair[0] for pair in values)
    colors = tuple(pair[1] for pair in values)
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(color, str)
        or not color.strip()
        for key, color in zip(keys, colors)
    ):
        raise ValueError("series_colors must contain non-empty string keys and colors")
    if len(set(keys)) != len(keys) or any(key not in series_order for key in keys):
        raise ValueError("series_colors keys must be unique and occur in series_order")
    if any(not is_color_like(color) for color in colors):
        raise ValueError("series_colors must contain valid Matplotlib colors")


def _group_band_colors(
    values: tuple[object, ...], category_order: tuple[object, ...]
) -> None:
    if values and len(values) != len(category_order):
        raise ValueError("group_band_colors must be empty or match category_order length")
    if any(not isinstance(color, str) or not color.strip() for color in values):
        raise ValueError("group_band_colors must contain non-empty color strings")
    if any(not is_color_like(color) for color in values):
        raise ValueError("group_band_colors must contain valid Matplotlib colors")


def _optional_text(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")


def _display_labels(
    values: tuple[tuple[object, ...], ...],
    order: tuple[object, ...],
    name: str,
) -> None:
    if not values:
        return
    keys = tuple(pair[0] for pair in values)
    displays = tuple(pair[1] for pair in values)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (*keys, *displays)
    ) or len(set(keys)) != len(keys):
        raise ValueError(
            f"{name} must contain unique non-empty string keys and "
            "non-empty string displays"
        )
    if set(keys) != set(order):
        raise ValueError(f"{name} must exactly cover its semantic order")


def _positive_finite(value: object, name: str) -> None:
    _finite(value, name)
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be a positive finite number")


def _boolean(value: object, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _decimals(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _series_string_mapping(
    values: tuple[tuple[object, ...], ...],
    series_order: tuple[object, ...],
    name: str,
    kind: str,
) -> None:
    keys = tuple(pair[0] for pair in values)
    styles = tuple(pair[1] for pair in values)
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(style, str)
        or not style.strip()
        for key, style in zip(keys, styles)
    ):
        raise ValueError(f"{name} must contain non-empty string keys and {kind}")
    if len(set(keys)) != len(keys) or any(key not in series_order for key in keys):
        raise ValueError(f"{name} keys must be unique and occur in series_order")
    for style in styles:
        try:
            if name == "line_styles":
                Line2D([], [], linestyle=style)
            else:
                MarkerStyle(style)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain valid Matplotlib {kind}") from error


def _common_series_options(
    series_order: tuple[object, ...],
    figure_size: tuple[object, ...],
    legend_columns: object,
    series_colors: tuple[tuple[object, ...], ...],
    title: object,
    x_label: object,
    y_label: object,
) -> None:
    _labels(series_order, "series_order")
    _figure_size(figure_size)
    _positive_integer(legend_columns, "legend_columns")
    _series_colors(series_colors, series_order)
    _optional_text(title, "title")
    _optional_text(x_label, "x_label")
    _optional_text(y_label, "y_label")


def _bar_options(
    category_order: tuple[object, ...],
    series_order: tuple[object, ...],
    category_labels: tuple[tuple[object, ...], ...],
    figure_size: tuple[object, ...],
    legend_columns: object,
    bar_fill_fraction: object,
    group_gap_fraction: object,
    series_colors: tuple[tuple[object, ...], ...],
    group_band_colors: tuple[object, ...],
    group_band_alpha: object,
    show_value_labels: object,
    value_decimals: object,
    axis_label: object,
    title: object,
) -> None:
    _labels(category_order, "category_order")
    _labels(series_order, "series_order")
    _category_labels(category_labels, category_order)
    _figure_size(figure_size)
    _positive_integer(legend_columns, "legend_columns")
    _fraction(
        bar_fill_fraction,
        "bar_fill_fraction",
        lower_inclusive=False,
        upper_inclusive=True,
    )
    _fraction(
        group_gap_fraction,
        "group_gap_fraction",
        lower_inclusive=True,
        upper_inclusive=False,
    )
    _series_colors(series_colors, series_order)
    _group_band_colors(group_band_colors, category_order)
    _fraction(
        group_band_alpha,
        "group_band_alpha",
        lower_inclusive=True,
        upper_inclusive=True,
    )
    if type(show_value_labels) is not bool:
        raise ValueError("show_value_labels must be a boolean")
    if (
        isinstance(value_decimals, bool)
        or not isinstance(value_decimals, int)
        or value_decimals < 0
    ):
        raise ValueError("value_decimals must be a non-negative integer")
    _optional_text(axis_label, "axis_label")
    _optional_text(title, "title")


@dataclass(frozen=True)
class RadarConfig:
    axis_order: tuple[str, ...]
    series_order: tuple[str, ...]
    r_min: float = 0.0
    r_max: float = 1.0
    r_ticks: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    start_angle_degrees: float = 90.0
    clockwise: bool = True
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    fill_alpha: float = 0.14
    series_colors: tuple[tuple[str, str], ...] = ()
    line_styles: tuple[tuple[str, str], ...] = ()
    markers: tuple[tuple[str, str], ...] = ()
    outer_ring_colors: tuple[str, ...] = ()
    show_radial_tick_labels: bool = False
    primary_series: str | None = None
    show_markers: bool = False
    sector_fill_alpha: float = 0.055

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_order", _tuple(self.axis_order, "axis_order"))
        object.__setattr__(self, "series_order", _tuple(self.series_order, "series_order"))
        object.__setattr__(self, "r_ticks", _tuple(self.r_ticks, "r_ticks"))
        object.__setattr__(self, "figure_size", _tuple(self.figure_size, "figure_size"))
        object.__setattr__(self, "series_colors", _pairs(self.series_colors, "series_colors"))
        object.__setattr__(self, "line_styles", _pairs(self.line_styles, "line_styles"))
        object.__setattr__(self, "markers", _pairs(self.markers, "markers"))
        object.__setattr__(self, "outer_ring_colors", _tuple(self.outer_ring_colors, "outer_ring_colors"))
        _labels(self.axis_order, "axis_order", minimum=3)
        _labels(self.series_order, "series_order")
        _radial(self.r_min, self.r_max, self.r_ticks)
        _finite(self.start_angle_degrees, "start_angle_degrees")
        _figure_size(self.figure_size)
        _positive_integer(self.legend_columns, "legend_columns")
        if isinstance(self.fill_alpha, bool) or not isinstance(self.fill_alpha, Real) or not isfinite(self.fill_alpha) or not 0 <= self.fill_alpha <= 1:
            raise ValueError("fill_alpha must be a finite number in [0, 1]")
        for value, name in (
            (self.show_radial_tick_labels, "show_radial_tick_labels"),
            (self.show_markers, "show_markers"),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.primary_series is not None and (
            not isinstance(self.primary_series, str)
            or self.primary_series not in self.series_order
        ):
            raise ValueError("primary_series must occur in series_order")
        _finite(self.sector_fill_alpha, "sector_fill_alpha")
        if not 0 <= self.sector_fill_alpha <= 1:
            raise ValueError("sector_fill_alpha must lie in [0, 1]")
        if self.outer_ring_colors and len(self.outer_ring_colors) != len(self.axis_order):
            raise ValueError("outer_ring_colors must match axis_order length")


@dataclass(frozen=True)
class RoseConfig:
    category_order: tuple[str, ...]
    series_order: tuple[str, ...]
    group_order: tuple[str, ...] = ()
    r_min: float = 0.0
    r_max: float = 100.0
    r_ticks: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0)
    encoding: Literal["area", "radius"] = "area"
    category_gap_radians: float = 0.035
    group_gap_radians: float = 0.16
    petal_fill_fraction: float = 0.82
    start_angle_degrees: float = 90.0
    clockwise: bool = True
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    series_colors: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "category_order", _tuple(self.category_order, "category_order"))
        object.__setattr__(self, "series_order", _tuple(self.series_order, "series_order"))
        object.__setattr__(self, "group_order", _tuple(self.group_order, "group_order"))
        object.__setattr__(self, "r_ticks", _tuple(self.r_ticks, "r_ticks"))
        object.__setattr__(self, "figure_size", _tuple(self.figure_size, "figure_size"))
        object.__setattr__(self, "series_colors", _pairs(self.series_colors, "series_colors"))
        _labels(self.category_order, "category_order", minimum=2)
        _labels(self.series_order, "series_order")
        if self.group_order:
            _labels(self.group_order, "group_order")
        _radial(self.r_min, self.r_max, self.r_ticks)
        if self.r_min < 0:
            raise ValueError("rose r_min must be non-negative")
        _finite(self.start_angle_degrees, "start_angle_degrees")
        _figure_size(self.figure_size)
        if self.encoding not in {"area", "radius"}:
            raise ValueError("encoding must be 'area' or 'radius'")
        for value, name in ((self.category_gap_radians, "category_gap_radians"), (self.group_gap_radians, "group_gap_radians")):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if isinstance(self.petal_fill_fraction, bool) or not isinstance(self.petal_fill_fraction, Real) or not isfinite(self.petal_fill_fraction) or not 0 < self.petal_fill_fraction <= 1:
            raise ValueError("petal_fill_fraction must be finite and lie in (0, 1]")
        _positive_integer(self.legend_columns, "legend_columns")


@dataclass(frozen=True)
class HorizontalBarConfig:
    category_order: tuple[str, ...]
    series_order: tuple[str, ...]
    category_labels: tuple[tuple[str, str], ...] = ()
    x_min: float = 0.0
    x_max: float = 100.0
    x_ticks: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
    axis_label: str | None = None
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    bar_fill_fraction: float = 0.8
    group_gap_fraction: float = 0.2
    series_colors: tuple[tuple[str, str], ...] = ()
    group_band_colors: tuple[str, ...] = ()
    group_band_alpha: float = 0.08
    show_value_labels: bool = True
    value_decimals: int = 0
    zero_line_color: str = "#273238"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category_order",
            _ordered_tuple(self.category_order, "category_order"),
        )
        object.__setattr__(
            self,
            "series_order",
            _ordered_tuple(self.series_order, "series_order"),
        )
        object.__setattr__(
            self, "category_labels", _pairs(self.category_labels, "category_labels")
        )
        object.__setattr__(
            self, "x_ticks", _ordered_tuple(self.x_ticks, "x_ticks")
        )
        object.__setattr__(
            self,
            "figure_size",
            _ordered_tuple(self.figure_size, "figure_size"),
        )
        object.__setattr__(
            self, "series_colors", _pairs(self.series_colors, "series_colors")
        )
        object.__setattr__(
            self,
            "group_band_colors",
            _ordered_tuple(self.group_band_colors, "group_band_colors"),
        )
        _axis_range(
            self.x_min, self.x_max, self.x_ticks, "x_min", "x_max", "x_ticks"
        )
        if not self.x_min <= 0 <= self.x_max:
            raise ValueError("x_min and x_max must include zero")
        _bar_options(
            self.category_order,
            self.series_order,
            self.category_labels,
            self.figure_size,
            self.legend_columns,
            self.bar_fill_fraction,
            self.group_gap_fraction,
            self.series_colors,
            self.group_band_colors,
            self.group_band_alpha,
            self.show_value_labels,
            self.value_decimals,
            self.axis_label,
            self.title,
        )
        if not isinstance(self.zero_line_color, str) or not is_color_like(
            self.zero_line_color
        ):
            raise ValueError("zero_line_color must be a valid Matplotlib color")


@dataclass(frozen=True)
class VerticalBarConfig:
    category_order: tuple[str, ...]
    series_order: tuple[str, ...]
    category_labels: tuple[tuple[str, str], ...] = ()
    y_min: float = 0.0
    y_max: float = 100.0
    y_ticks: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
    baseline: float = 0.0
    axis_label: str | None = None
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    bar_fill_fraction: float = 0.8
    group_gap_fraction: float = 0.2
    series_colors: tuple[tuple[str, str], ...] = ()
    group_band_colors: tuple[str, ...] = ()
    group_band_alpha: float = 0.08
    show_value_labels: bool = True
    value_decimals: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category_order",
            _ordered_tuple(self.category_order, "category_order"),
        )
        object.__setattr__(
            self,
            "series_order",
            _ordered_tuple(self.series_order, "series_order"),
        )
        object.__setattr__(
            self, "category_labels", _pairs(self.category_labels, "category_labels")
        )
        object.__setattr__(
            self, "y_ticks", _ordered_tuple(self.y_ticks, "y_ticks")
        )
        object.__setattr__(
            self,
            "figure_size",
            _ordered_tuple(self.figure_size, "figure_size"),
        )
        object.__setattr__(
            self, "series_colors", _pairs(self.series_colors, "series_colors")
        )
        object.__setattr__(
            self,
            "group_band_colors",
            _ordered_tuple(self.group_band_colors, "group_band_colors"),
        )
        _axis_range(
            self.y_min, self.y_max, self.y_ticks, "y_min", "y_max", "y_ticks"
        )
        _finite(self.baseline, "baseline")
        if not self.y_min <= self.baseline <= self.y_max:
            raise ValueError("baseline must lie within [y_min, y_max]")
        _bar_options(
            self.category_order,
            self.series_order,
            self.category_labels,
            self.figure_size,
            self.legend_columns,
            self.bar_fill_fraction,
            self.group_gap_fraction,
            self.series_colors,
            self.group_band_colors,
            self.group_band_alpha,
            self.show_value_labels,
            self.value_decimals,
            self.axis_label,
            self.title,
        )


@dataclass(frozen=True)
class LineConfig:
    category_order: tuple[str, ...]
    series_order: tuple[str, ...]
    category_labels: tuple[tuple[str, str], ...] = ()
    y_min: float = 0
    y_max: float = 1
    y_ticks: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1)
    x_label: str | None = None
    y_label: str | None = None
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    series_colors: tuple[tuple[str, str], ...] = ()
    line_styles: tuple[tuple[str, str], ...] = ()
    markers: tuple[tuple[str, str], ...] = ()
    line_width: float = 1.8
    show_markers: bool = True
    marker_size: float = 5.5
    show_value_labels: bool = False
    value_decimals: int = 2
    x_label_rotation: float = 0

    def __post_init__(self) -> None:
        for name in ("category_order", "series_order", "y_ticks", "figure_size"):
            object.__setattr__(self, name, _ordered_tuple(getattr(self, name), name))
        for name in ("category_labels", "series_colors", "line_styles", "markers"):
            object.__setattr__(self, name, _pairs(getattr(self, name), name))
        _semantic_labels(self.category_order, "category_order")
        _display_labels(self.category_labels, self.category_order, "category_labels")
        _axis_range(self.y_min, self.y_max, self.y_ticks, "y_min", "y_max", "y_ticks")
        _semantic_labels(self.series_order, "series_order")
        _common_series_options(
            self.series_order, self.figure_size, self.legend_columns,
            self.series_colors, self.title, self.x_label, self.y_label,
        )
        _series_string_mapping(self.line_styles, self.series_order, "line_styles", "line styles")
        _series_string_mapping(self.markers, self.series_order, "markers", "markers")
        _positive_finite(self.line_width, "line_width")
        _positive_finite(self.marker_size, "marker_size")
        _boolean(self.show_markers, "show_markers")
        _boolean(self.show_value_labels, "show_value_labels")
        _decimals(self.value_decimals, "value_decimals")
        _finite(self.x_label_rotation, "x_label_rotation")


@dataclass(frozen=True)
class LollipopConfig:
    category_order: tuple[str, ...]
    series_order: tuple[str, ...]
    category_labels: tuple[tuple[str, str], ...] = ()
    y_min: float = 0
    y_max: float = 100
    y_ticks: tuple[float, ...] = (0, 25, 50, 75, 100)
    baseline: float = 0
    x_label: str | None = None
    y_label: str | None = None
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    series_colors: tuple[tuple[str, str], ...] = ()
    stem_width: float = 1.4
    marker_size: float = 42.0
    show_value_labels: bool = True
    value_decimals: int = 1
    x_label_rotation: float = 0

    def __post_init__(self) -> None:
        for name in ("category_order", "series_order", "y_ticks", "figure_size"):
            object.__setattr__(self, name, _ordered_tuple(getattr(self, name), name))
        for name in ("category_labels", "series_colors"):
            object.__setattr__(self, name, _pairs(getattr(self, name), name))
        _semantic_labels(self.category_order, "category_order")
        _display_labels(self.category_labels, self.category_order, "category_labels")
        _axis_range(self.y_min, self.y_max, self.y_ticks, "y_min", "y_max", "y_ticks")
        _finite(self.baseline, "baseline")
        if not self.y_min <= self.baseline <= self.y_max:
            raise ValueError("baseline must lie within [y_min, y_max]")
        _semantic_labels(self.series_order, "series_order")
        _common_series_options(
            self.series_order, self.figure_size, self.legend_columns,
            self.series_colors, self.title, self.x_label, self.y_label,
        )
        _positive_finite(self.stem_width, "stem_width")
        _positive_finite(self.marker_size, "marker_size")
        _boolean(self.show_value_labels, "show_value_labels")
        _decimals(self.value_decimals, "value_decimals")
        _finite(self.x_label_rotation, "x_label_rotation")


@dataclass(frozen=True)
class ScatterConfig:
    series_order: tuple[str, ...]
    x_min: float = 0
    x_max: float = 1
    x_ticks: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1)
    y_min: float = 0
    y_max: float = 1
    y_ticks: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1)
    x_label: str | None = None
    y_label: str | None = None
    title: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    legend_columns: int = 4
    series_colors: tuple[tuple[str, str], ...] = ()
    markers: tuple[tuple[str, str], ...] = ()
    default_marker_area: float = 42
    bubble_area_min: float = 28
    bubble_area_max: float = 180
    marker_alpha: float = 0.86
    edge_width: float = 0.65
    show_point_labels: bool = False
    point_label_decimals: int = 2
    point_label_offset_points: float = 5

    def __post_init__(self) -> None:
        for name in ("series_order", "x_ticks", "y_ticks", "figure_size"):
            object.__setattr__(self, name, _ordered_tuple(getattr(self, name), name))
        for name in ("series_colors", "markers"):
            object.__setattr__(self, name, _pairs(getattr(self, name), name))
        _semantic_labels(self.series_order, "series_order")
        _axis_range(self.x_min, self.x_max, self.x_ticks, "x_min", "x_max", "x_ticks")
        _axis_range(self.y_min, self.y_max, self.y_ticks, "y_min", "y_max", "y_ticks")
        _common_series_options(
            self.series_order, self.figure_size, self.legend_columns,
            self.series_colors, self.title, self.x_label, self.y_label,
        )
        _series_string_mapping(self.markers, self.series_order, "markers", "markers")
        for name in (
            "default_marker_area", "bubble_area_min", "bubble_area_max", "edge_width"
        ):
            _positive_finite(getattr(self, name), name)
        if self.bubble_area_min > self.bubble_area_max:
            raise ValueError("bubble_area_min must not exceed bubble_area_max")
        _fraction(
            self.marker_alpha, "marker_alpha", lower_inclusive=True, upper_inclusive=True
        )
        _boolean(self.show_point_labels, "show_point_labels")
        _decimals(self.point_label_decimals, "point_label_decimals")
        _finite(self.point_label_offset_points, "point_label_offset_points")


@dataclass(frozen=True)
class HeatmapConfig:
    row_order: tuple[str, ...]
    column_order: tuple[str, ...]
    row_labels: tuple[tuple[str, str], ...] = ()
    column_labels: tuple[tuple[str, str], ...] = ()
    v_min: float = 0
    v_max: float = 1
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    colorbar_label: str | None = None
    figure_size: tuple[float, float] = (6.6, 6.6)
    show_colorbar: bool = True
    show_annotations: bool = True
    annotation_decimals: int = 2
    cell_gap_width: float = 0.8
    text_contrast_threshold: float = 0.52
    color_scale: tuple[str, ...] = COMMON_CHART_PALETTE

    def __post_init__(self) -> None:
        for name in ("row_order", "column_order", "figure_size", "color_scale"):
            object.__setattr__(self, name, _ordered_tuple(getattr(self, name), name))
        for name in ("row_labels", "column_labels"):
            object.__setattr__(self, name, _pairs(getattr(self, name), name))
        _semantic_labels(self.row_order, "row_order")
        _semantic_labels(self.column_order, "column_order")
        _display_labels(self.row_labels, self.row_order, "row_labels")
        _display_labels(self.column_labels, self.column_order, "column_labels")
        _finite_heatmap(self.v_min, "v_min")
        _finite_heatmap(self.v_max, "v_max")
        if not self.v_min < self.v_max:
            raise ValueError("v_min must be less than v_max")
        _figure_size(self.figure_size)
        for value, name in (
            (self.title, "title"), (self.x_label, "x_label"),
            (self.y_label, "y_label"), (self.colorbar_label, "colorbar_label"),
        ):
            _optional_text(value, name)
        _boolean(self.show_colorbar, "show_colorbar")
        _boolean(self.show_annotations, "show_annotations")
        _decimals(self.annotation_decimals, "annotation_decimals")
        _positive_finite(self.cell_gap_width, "cell_gap_width")
        _fraction(
            self.text_contrast_threshold,
            "text_contrast_threshold",
            lower_inclusive=True,
            upper_inclusive=True,
        )
        if len(self.color_scale) < 2:
            raise ValueError("color_scale must contain at least 2 colors")
        if any(
            not isinstance(color, str) or not color.strip() or not is_color_like(color)
            for color in self.color_scale
        ):
            raise ValueError("color_scale must contain valid Matplotlib colors")


@dataclass(frozen=True)
class ExportConfig:
    formats: tuple[Literal["png", "pdf"], ...] = ("png", "pdf")
    dpi: int = 300
    transparent: bool = False
    create_directory: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", _tuple(self.formats, "formats"))
        if not self.formats or any(fmt not in {"png", "pdf"} for fmt in self.formats):
            raise ValueError("formats may contain only 'png' and 'pdf'")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must not contain duplicates")
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive finite integer")
