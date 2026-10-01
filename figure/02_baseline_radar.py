"""Teaching demo: simulated scores, all axes normalized to [0, 1]."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from polarviz import ExportConfig, RadarConfig, plot_radar, export_figure

ROOT = Path(__file__).resolve().parents[1]
axes = ("Accuracy", "Robustness", "Efficiency", "Generalization")
data = pd.DataFrame({
    "axis": list(axes) * 2,
    "series": ["Ours"] * 4 + ["Baseline"] * 4,
    "value": [0.92, 0.84, 0.76, 0.88, 0.86, 0.72, 0.81, 0.79],
})
config = RadarConfig(
    axis_order=axes, series_order=("Ours", "Baseline"),
    r_min=0, r_max=1, r_ticks=(0.25, 0.5, 0.75, 1),
    title="Normalized capability profile",
    show_markers=True, show_radial_tick_labels=True,
    legend_columns=2,
)
figure = plot_radar(data, config)
for path in export_figure(figure, ROOT / "figure/output/02_baseline_radar", ExportConfig(formats=("png",))):
    print(path)
plt.close(figure)
