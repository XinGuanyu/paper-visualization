"""Build the simulated-data annotated heatmap example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated heatmap example subprocess did not select Agg")
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
        HeatmapConfig,
        export_figure,
        heatmap_standard,
        plot_heatmap,
        read_heatmap_csv,
    )

    rows = tuple(f"method_{letter}" for letter in "abcde")
    columns = tuple(f"dataset_{letter}" for letter in "abcde")
    data = read_heatmap_csv(HERE / "data" / "heatmap.csv")
    config = HeatmapConfig(
        row_order=rows,
        column_order=columns,
        row_labels=tuple(
            (row, f"Method {letter.upper()}")
            for row, letter in zip(rows, "abcde")
        ),
        column_labels=tuple(
            (column, f"Dataset {letter.upper()}")
            for column, letter in zip(columns, "abcde")
        ),
        v_min=0.70,
        v_max=1.0,
        x_label="Evaluation Dataset",
        y_label="Candidate Method",
        title="Simulated Benchmark Matrix",
        colorbar_label="Normalized Score",
        show_colorbar=True,
        show_annotations=True,
        annotation_decimals=2,
        color_scale=COMMON_CHART_PALETTE,
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_heatmap(data, config, preset=heatmap_standard)
    try:
        paths = export_figure(
            figure,
            output_dir / "heatmap_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
