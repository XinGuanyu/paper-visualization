"""Teaching demo: simulated observed versus predicted scores."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from polarviz import ExportConfig, ScatterConfig, plot_scatter_chart, export_figure

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
observed = np.linspace(2, 28, 30)
data = pd.DataFrame({
    "x": observed,
    "y": np.clip(observed + rng.normal(0, 2.2, 30), 0, 30),
    "series": ["Held-out samples"] * 30,
})
config = ScatterConfig(
    series_order=("Held-out samples",),
    x_min=0, x_max=30, x_ticks=(0, 10, 20, 30),
    y_min=0, y_max=30, y_ticks=(0, 10, 20, 30),
    x_label="Observed score", y_label="Predicted score",
    title="Observed vs predicted (simulated)",
    figure_size=(6.6, 4.8), show_point_labels=False,
)
figure = plot_scatter_chart(data, config)
axis = figure.axes[0]
axis.plot([0, 30], [0, 30], "--", color="#68727E", linewidth=1)
for path in export_figure(figure, ROOT / "figure/output/03_mechanism_scatter", ExportConfig(formats=("png",))):
    print(path)
plt.close(figure)
