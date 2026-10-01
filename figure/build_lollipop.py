"""Build the simulated-data grouped lollipop-chart example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated lollipop example subprocess did not select Agg")
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
        LollipopConfig,
        export_figure,
        lollipop_standard,
        plot_lollipop_chart,
        read_lollipop_csv,
    )

    series = tuple(f"System {letter}" for letter in "ABCD")
    data = read_lollipop_csv(HERE / "data" / "lollipop.csv")
    config = LollipopConfig(
        category_order=("accuracy", "latency", "throughput", "memory"),
        series_order=series,
        category_labels=(
            ("accuracy", "Accuracy"),
            ("latency", "Latency Efficiency"),
            ("throughput", "Throughput"),
            ("memory", "Memory Efficiency"),
        ),
        y_min=25.0,
        y_max=100.0,
        y_ticks=(25.0, 50.0, 75.0, 100.0),
        baseline=50.0,
        x_label="Evaluation Metric",
        y_label="Normalized Index",
        title="Simulated System Index",
        legend_columns=4,
        series_colors=tuple(zip(series, COMMON_CHART_PALETTE[:4])),
        show_value_labels=True,
        value_decimals=0,
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_lollipop_chart(data, config, preset=lollipop_standard)
    try:
        paths = export_figure(
            figure,
            output_dir / "lollipop_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
