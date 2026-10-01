"""Build the simulated-data segmented-ring radar example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
_ISOLATED_RUN = "POLARVIZ_ISOLATED_EXAMPLE"


def _run_with_agg_in_subprocess() -> None:
    if os.environ.get(_ISOLATED_RUN) == "1":
        raise RuntimeError("isolated radar example subprocess did not select Agg")
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
        RadarConfig,
        export_figure,
        plot_radar,
        read_radar_csv,
    )
    from polarviz.presets import radar_segmented_ring

    data = read_radar_csv(HERE / "data" / "radar.csv")
    config = RadarConfig(
        axis_order=("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
        series_order=("Series A", "Series B", "Series C", "Series D"),
        r_min=0.0,
        r_max=1.0,
        r_ticks=(0.25, 0.5, 0.75, 1.0),
        primary_series="Series A",
        title="Simulated Radar Comparison",
    )
    output_dir = Path(os.environ.get("POLARVIZ_OUTPUT_DIR", HERE / "output"))
    figure = plot_radar(data, config, preset=radar_segmented_ring)
    try:
        paths = export_figure(
            figure,
            output_dir / "radar_example",
            ExportConfig(formats=("png",)),
        )
        for path in paths:
            print(f"Generated: {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
