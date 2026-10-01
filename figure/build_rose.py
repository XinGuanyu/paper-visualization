"""Build the simulated-data grouped Nightingale rose example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated rose example subprocess did not select Agg")
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
        RoseConfig,
        export_figure,
        plot_grouped_rose,
        read_rose_csv,
    )
    from polarviz.presets import rose_grouped

    data = read_rose_csv(HERE / "data" / "rose.csv")
    config = RoseConfig(
        category_order=(
            "Metric A",
            "Metric B",
            "Metric C",
            "Metric D",
            "Metric E",
            "Metric F",
        ),
        series_order=("Series A", "Series B", "Series C"),
        group_order=("Quality", "Efficiency"),
        r_min=0.0,
        r_max=100.0,
        r_ticks=(25.0, 50.0, 75.0, 100.0),
        encoding="area",
        title="Simulated Grouped Rose Comparison",
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_grouped_rose(data, config, preset=rose_grouped)
    try:
        paths = export_figure(
            figure,
            output_dir / "rose_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
