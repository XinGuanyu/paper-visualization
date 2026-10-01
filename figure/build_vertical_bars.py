"""Build the simulated-data grouped vertical bar example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated vertical bar example subprocess did not select Agg")
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
        ExportConfig,
        VerticalBarConfig,
        export_figure,
        plot_grouped_vertical_bars,
        read_bar_csv,
    )
    from polarviz.presets import vertical_grouped_bars

    data = read_bar_csv(HERE / "data" / "vertical_bars.csv")
    config = VerticalBarConfig(
        category_order=("Metric A", "Metric B", "Metric C"),
        series_order=("Series A", "Series B", "Series C"),
        y_min=0.0,
        y_max=1.0,
        y_ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        axis_label="Simulated score",
        title="Simulated Metric Scores",
        legend_columns=3,
        series_colors=(
            ("Series A", "#F38181"),
            ("Series B", "#FCE38A"),
            ("Series C", "#EAFFD0"),
        ),
        group_band_colors=("#F38181", "#FCE38A", "#EAFFD0"),
        group_band_alpha=0.05,
        value_decimals=2,
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_grouped_vertical_bars(
        data,
        config,
        preset=vertical_grouped_bars,
    )
    try:
        paths = export_figure(
            figure,
            output_dir / "vertical_bars_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
