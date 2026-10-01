"""Tidy CSV readers and validation for reusable visualizations."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence, Set
from copy import deepcopy
from itertools import product
from math import isfinite
from numbers import Rational, Real
from os import PathLike

import numpy as np
import pandas as pd

from .config import RadarConfig, RoseConfig
from .presets.palette import DEFAULT_SERIES_COLORS


def _required_frame(frame: object, required: Sequence[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    duplicates = list(frame.columns[frame.columns.duplicated()])
    if duplicates:
        raise ValueError(f"duplicate column names: {duplicates}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return frame.copy(deep=True)


def _labels(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = frame[column]
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{column} must contain nonblank string labels")
        if isinstance(values.dtype, pd.CategoricalDtype):
            categories = tuple(values.cat.categories)
            if any(not isinstance(value, str) for value in categories):
                raise ValueError(f"{column} must contain nonblank string labels")
            stripped = tuple(value.strip() for value in categories)
            if any(not value for value in stripped):
                raise ValueError(
                    f"{column} categorical labels must be nonblank after stripping"
                )
            if len(set(stripped)) != len(stripped):
                raise ValueError(
                    f"{column} categorical labels must be unique after stripping"
                )
            frame[column] = values.cat.rename_categories(stripped)
            continue
        frame[column] = values.str.strip()
        if any(value == "" for value in frame[column]):
            raise ValueError(f"{column} must contain nonblank string labels")


def _finite_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError):
        return False


def _finite_heatmap_real(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return isinstance(value, Rational)
    except TypeError:
        return False


def _values(frame: pd.DataFrame, coerce_text: bool) -> None:
    if not coerce_text:
        if not frame["value"].map(_finite_real).all():
            raise ValueError("value must contain finite real numbers")
        return
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    values = frame["value"]
    valid = values.map(_finite_real)
    if values.isna().any() or not valid.all():
        raise ValueError("value must contain finite real numbers")


def _duplicates(frame: pd.DataFrame, keys: Sequence[str]) -> None:
    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if duplicated.any():
        values = list(frame.loc[duplicated, list(keys)].itertuples(index=False, name=None))
        raise ValueError(f"duplicate semantic keys: {values}")


def _normalise(
    frame: object,
    required: Sequence[str],
    keys: Sequence[str],
    coerce_text: bool = False,
) -> pd.DataFrame:
    result = _required_frame(frame, required)
    _labels(result, keys)
    _values(result, coerce_text)
    _duplicates(result, keys)
    return result


def _read_csv(
    path: str | PathLike[str],
    string_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Read CSV input after rejecting duplicate source header names."""
    with open(path, newline="") as source:
        header = next(csv.reader(source), None)
    if header is None:
        raise ValueError("CSV is empty and has no header")
    duplicates = [name for index, name in enumerate(header) if name in header[:index]]
    if duplicates:
        raise ValueError(f"duplicate CSV column names: {duplicates}")
    try:
        return pd.read_csv(
            path,
            dtype={
                name: "string"
                for name in ("axis", "series", "category", "group", *string_columns)
            },
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV is empty and has no header") from error


def _read(
    path: str | PathLike[str],
    required: Sequence[str],
    keys: Sequence[str],
    string_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Read and validate a tidy CSV without discarding unrelated columns."""
    return _normalise(
        _read_csv(path, string_columns), required, keys, coerce_text=True
    )


def read_radar_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a radar CSV containing ``axis``, ``series``, and ``value``."""
    return _read(path, ("axis", "series", "value"), ("axis", "series"))


def read_rose_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a rose CSV containing ``category``, ``series``, and ``value``."""
    frame = _read_csv(path)
    result = _normalise(frame, ("category", "series", "value"), ("category", "series"), coerce_text=True)
    if "group" in result.columns:
        _labels(result, ("group",))
    return result


def read_bar_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read bar-chart CSV data while preserving row and column order."""
    result = _read(path, ("category", "series", "value"), ("category", "series"))
    result["value"] = result["value"].astype(float)
    return _deepcopy_cells(result)


def read_line_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read line-chart CSV data using the complete tidy bar schema."""
    return read_bar_csv(path)


def read_lollipop_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read lollipop-chart CSV data using the complete tidy bar schema."""
    return read_bar_csv(path)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    coerce_text: bool,
    positive: bool = False,
) -> None:
    if coerce_text:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[column]
    valid = values.map(_finite_real)
    description = "positive finite real numbers" if positive else "finite real numbers"
    if values.isna().any() or not valid.all():
        raise ValueError(f"{column} must contain {description}")
    if positive and (values <= 0).any():
        raise ValueError(f"{column} must contain {description}")


def _normalise_scatter(frame: object, *, coerce_text: bool) -> pd.DataFrame:
    result = _required_frame(frame, ("x", "y", "series"))
    _labels(result, ("series",))
    _numeric_column(result, "x", coerce_text=coerce_text)
    _numeric_column(result, "y", coerce_text=coerce_text)
    if "size" in result.columns:
        _numeric_column(result, "size", coerce_text=coerce_text, positive=True)
    duplicate_keys = ["series", "x", "y"]
    if "label" in result.columns:
        if not all(isinstance(value, str) for value in result["label"]):
            raise ValueError("label must contain strings")
        duplicate_keys.append("label")
    _duplicates(result, duplicate_keys)
    return result


def read_scatter_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read scatter CSV data while preserving optional and unrelated columns."""
    return _deepcopy_cells(
        _normalise_scatter(_read_csv(path, ("label",)), coerce_text=True)
    )


def read_heatmap_csv(path: str | PathLike[str]) -> pd.DataFrame:
    """Read heatmap CSV data containing ``row``, ``column``, and ``value``."""
    result = _read(
        path,
        ("row", "column", "value"),
        ("row", "column"),
        string_columns=("row", "column"),
    )
    result["value"] = result["value"].astype(float)
    return _deepcopy_cells(result)


def _complete(left: Sequence[str], series: Sequence[str], actual: Iterable[tuple[str, str]]) -> None:
    expected = list(product(left, series))
    actual_set = set(actual)
    expected_set = set(expected)
    missing = [key for key in expected if key not in actual_set]
    extra = sorted(actual_set - expected_set)
    errors: list[str] = []
    if missing:
        errors.append(f"missing combinations: {missing}")
    if extra:
        errors.append(f"extra combinations: {extra}")
    if errors:
        raise ValueError("; ".join(errors))


def _colors(series_order: Sequence[str], configured: Sequence[tuple[object, ...]]) -> None:
    pairs: list[tuple[str, str]] = []
    for mapping in configured:
        if len(mapping) != 2 or not all(isinstance(value, str) and value.strip() for value in mapping):
            raise ValueError("series_colors must contain (series, color) nonblank string pairs")
        pairs.append((mapping[0], mapping[1]))
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate explicit series color mappings")
    unknown = sorted(set(names) - set(series_order))
    if unknown:
        raise ValueError(f"unknown series color keys: {unknown}")
    explicit = dict(pairs)
    missing = [name for index, name in enumerate(series_order) if index >= len(DEFAULT_SERIES_COLORS) and name not in explicit]
    if missing:
        raise ValueError(f"series require explicit color mappings beyond the default palette: {missing}")


def _bounds(frame: pd.DataFrame, r_min: float, r_max: float, rose: bool = False) -> None:
    if rose and (frame["value"] < 0).any():
        raise ValueError("rose value must be non-negative")
    if ((frame["value"] < r_min) | (frame["value"] > r_max)).any():
        raise ValueError(f"value must lie within [{r_min}, {r_max}]")


def _order(frame: pd.DataFrame, columns: Sequence[str], orders: Sequence[Sequence[str]]) -> pd.DataFrame:
    for column, order in zip(columns, orders):
        frame[column] = pd.Categorical(frame[column], categories=order, ordered=True)
    return frame.sort_values(list(columns), kind="stable").reset_index(drop=True)


def _order_preserving_dtypes(
    frame: pd.DataFrame,
    columns: Sequence[str],
    orders: Sequence[Sequence[str]],
) -> pd.DataFrame:
    sort_columns: list[str] = []
    for index, (column, order) in enumerate(zip(columns, orders)):
        sort_column = f"__polarviz_sort_order_{index}"
        while sort_column in frame.columns:
            sort_column = f"_{sort_column}"
        frame[sort_column] = frame[column].map(
            {value: rank for rank, value in enumerate(order)}
        )
        sort_columns.append(sort_column)
    return (
        frame.sort_values(sort_columns, kind="stable")
        .drop(columns=sort_columns)
        .reset_index(drop=True)
    )


def _deepcopy_cells(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    object_columns = [
        column for column, dtype in enumerate(result.dtypes) if dtype == object
    ]
    for column in object_columns:
        for row in range(len(result.index)):
            result.iat[row, column] = deepcopy(result.iat[row, column])
    return result


def _bar_order(order: object, name: str) -> tuple[str, ...]:
    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
        raise ValueError(f"{name} must be a sequence")
    try:
        values = tuple(order)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error
    if (
        not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonblank string labels")
    if any(value != value.strip() for value in values):
        raise ValueError(f"{name} labels must not have leading or trailing whitespace")
    return values


def validate_bar_data(
    frame: object,
    category_order: Sequence[str],
    series_order: Sequence[str],
) -> pd.DataFrame:
    """Validate and canonically order a complete Cartesian bar-data frame."""
    categories = _bar_order(category_order, "category_order")
    series = _bar_order(series_order, "series_order")
    result = _normalise(frame, ("category", "series", "value"), ("category", "series"))
    _complete(categories, series, result[["category", "series"]].itertuples(index=False, name=None))
    ordered = _order(result, ("category", "series"), (categories, series))
    return _deepcopy_cells(ordered)


def validate_line_data(
    frame: object,
    category_order: Sequence[str],
    series_order: Sequence[str],
) -> pd.DataFrame:
    """Validate complete line-chart data and return category-major rows."""
    return _validate_common_category_data(frame, category_order, series_order)


def validate_lollipop_data(
    frame: object,
    category_order: Sequence[str],
    series_order: Sequence[str],
) -> pd.DataFrame:
    """Validate complete lollipop data and return category-major rows."""
    return _validate_common_category_data(frame, category_order, series_order)


def _semantic_order(order: object, name: str) -> tuple[str, ...]:
    if isinstance(order, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be an ordered iterable")
    try:
        values = tuple(order)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an ordered iterable") from error
    if (
        not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonblank string labels")
    if any(value != value.strip() for value in values):
        raise ValueError(f"{name} labels must not have leading or trailing whitespace")
    return values


def _validate_common_category_data(
    frame: object,
    category_order: object,
    series_order: object,
) -> pd.DataFrame:
    categories = _semantic_order(category_order, "category_order")
    series = _semantic_order(series_order, "series_order")
    result = _normalise(
        frame, ("category", "series", "value"), ("category", "series")
    )
    _complete(
        categories,
        series,
        result[["category", "series"]].itertuples(index=False, name=None),
    )
    return _deepcopy_cells(
        _order_preserving_dtypes(
            result, ("category", "series"), (categories, series)
        )
    )


def validate_scatter_data(frame: object, series_order: object) -> pd.DataFrame:
    """Validate scatter points and stably group them by configured series."""
    series = _semantic_order(series_order, "series_order")
    result = _normalise_scatter(frame, coerce_text=False)
    actual = set(result["series"])
    expected = set(series)
    missing = [name for name in series if name not in actual]
    extra = sorted(actual - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing series: {missing}")
        if extra:
            parts.append(f"extra series: {extra}")
        raise ValueError("; ".join(parts))
    ordered = _order_preserving_dtypes(result, ("series",), (series,))
    return _deepcopy_cells(ordered)


def validate_heatmap_data(
    frame: object,
    row_order: object,
    column_order: object,
) -> pd.DataFrame:
    """Validate an exact rectangular heatmap and return row-major cells."""
    rows = _semantic_order(row_order, "row_order")
    columns = _semantic_order(column_order, "column_order")
    result = _required_frame(frame, ("row", "column", "value"))
    _labels(result, ("row", "column"))
    if not result["value"].map(_finite_heatmap_real).all():
        raise ValueError("value must contain finite real numbers")
    _duplicates(result, ("row", "column"))
    _complete(
        rows,
        columns,
        result[["row", "column"]].itertuples(index=False, name=None),
    )
    return _deepcopy_cells(
        _order_preserving_dtypes(
            result, ("row", "column"), (rows, columns)
        )
    )


def validate_radar_data(frame: object, config: RadarConfig) -> pd.DataFrame:
    """Validate and canonically order a tidy radar frame."""
    result = _normalise(frame, ("axis", "series", "value"), ("axis", "series"))
    _complete(config.axis_order, config.series_order, result[["axis", "series"]].itertuples(index=False, name=None))
    _colors(config.series_order, config.series_colors)
    _bounds(result, config.r_min, config.r_max)
    return _order(result, ("axis", "series"), (config.axis_order, config.series_order))


def validate_rose_data(frame: object, config: RoseConfig) -> pd.DataFrame:
    """Validate and canonically order a tidy rose frame."""
    raw = _required_frame(frame, ("category", "series", "value"))
    has_group = "group" in raw.columns
    result = _normalise(raw, ("category", "series", "value"), ("category", "series"))
    if has_group:
        _labels(result, ("group",))
    if config.group_order and not has_group:
        raise ValueError("group_order cannot be used without a group column")
    if has_group:
        groups = list(dict.fromkeys(result["group"]))
        if len(groups) > 1 and not config.group_order:
            raise ValueError("group data with multiple groups requires group_order")
        split = result.groupby("category", sort=False)["group"].nunique()
        if (split > 1).any():
            categories = list(split[split > 1].index)
            raise ValueError(f"categories may not be split across groups: {categories}")
        if config.group_order:
            actual = set(groups)
            expected = set(config.group_order)
            if actual != expected:
                missing = [group for group in config.group_order if group not in actual]
                extra = sorted(actual - expected)
                raise ValueError(f"group labels must exactly match group_order; missing={missing}, extra={extra}")
    _complete(config.category_order, config.series_order, result[["category", "series"]].itertuples(index=False, name=None))
    _colors(config.series_order, config.series_colors)
    _bounds(result, config.r_min, config.r_max, rose=True)
    return _order(result, ("category", "series"), (config.category_order, config.series_order))
