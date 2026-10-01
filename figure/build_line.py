"""Build the simulated-data multi-series line-chart example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated line example subprocess did not select Agg")
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
        LineConfig,
        export_figure,
        line_standard,
        plot_line_chart,
        read_line_csv,
    )

    series = tuple(f"Method {letter}" for letter in "ABCDEFGH")
    data = read_line_csv(HERE / "data" / "line.csv")
    config = LineConfig(
        category_order=("setup", "ingest", "prefill", "decode", "serve"),
        series_order=series,
        category_labels=(
            ("setup", "Setup"),
            ("ingest", "Data Ingest"),
            ("prefill", "Prompt Prefill"),
            ("decode", "Decode Stage"),
            ("serve", "Serving"),
        ),
        y_min=60.0,
        y_max=100.0,
        y_ticks=(60.0, 70.0, 80.0, 90.0, 100.0),
        x_label="Workflow Stage",
        y_label="Normalized Score",
        title="Simulated Performance Trend",
        legend_columns=4,
        series_colors=tuple(zip(series, COMMON_CHART_PALETTE)),
        show_markers=True,
        show_value_labels=False,
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_line_chart(data, config, preset=line_standard)
    try:
        paths = export_figure(
            figure,
            output_dir / "line_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
