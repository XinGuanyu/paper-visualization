"""Build the simulated-data scatter and bubble-chart example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated scatter example subprocess did not select Agg")
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment[_ISOLATED_RUN] = "1"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        check=True,
        env=environment,
    )


def main() -> None:
    import matplotlib

    pyplot_is_loaded = "matplotlib.pyplot" in sys.modules
    if pyplot_is_loaded and str(matplotlib.get_backend()).lower() != "agg":
        _run_with_agg_in_subprocess()
        return
    if not pyplot_is_loaded:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from polarviz import (
        COMMON_CHART_PALETTE,
        ExportConfig,
        ScatterConfig,
        export_figure,
        plot_scatter_chart,
        read_scatter_csv,
        scatter_standard,
    )

    series = tuple(f"Family {letter}" for letter in "ABCD")
    data = read_scatter_csv(HERE / "data" / "scatter.csv")
    config = ScatterConfig(
        series_order=series,
        x_min=0.0,
        x_max=100.0,
        x_ticks=(0.0, 25.0, 50.0, 75.0, 100.0),
        y_min=0.0,
        y_max=100.0,
        y_ticks=(0.0, 25.0, 50.0, 75.0, 100.0),
        x_label="Normalized Cost",
        y_label="Normalized Quality",
        title="Simulated Efficiency Frontier",
        legend_columns=4,
        series_colors=tuple(zip(series, COMMON_CHART_PALETTE[:4])),
        bubble_area_min=32.0,
        bubble_area_max=180.0,
        show_point_labels=True,
        point_label_offset_points=5.0,
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_scatter_chart(data, config, preset=scatter_standard)
    try:
        paths = export_figure(
            figure,
            output_dir / "scatter_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
