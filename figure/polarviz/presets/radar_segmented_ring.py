"""Appearance values for the segmented-ring radar chart."""

from types import MappingProxyType

RADAR_SEGMENTED_RING = MappingProxyType({
    "outer_ring_width_fraction": 0.085,
    "outer_ring_gap_fraction": 0.018,
    "grid_alpha": 0.22,
    "line_width": 2.0,
    "marker_size": 4.8,
    "legend_y": -0.12,
    "secondary_line_width": 0.9,
    "primary_understroke_extra": 1.25,
    "secondary_understroke_extra": 0.65,
    "axis_label_pad_fraction": 0.075,
})
