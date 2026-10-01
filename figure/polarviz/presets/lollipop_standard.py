"""Appearance values for standard Cartesian lollipop charts."""

from types import MappingProxyType


LOLLIPOP_STANDARD = MappingProxyType(
    {
        "grid_alpha": 0.28,
        "stem_alpha": 0.74,
        "marker_alpha": 0.92,
        "edge_width": 0.75,
        "label_offset_points": 4.0,
        "legend_y": 1.16,
        "group_fill_fraction": 0.58,
    }
)
