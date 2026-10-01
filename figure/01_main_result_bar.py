"""Teaching demo: simulated method comparison."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from polarviz import ExportConfig, VerticalBarConfig, plot_grouped_vertical_bars, export_figure

ROOT = Path(__file__).resolve().parents[1]
data = pd.DataFrame({
    "category": ["Accuracy"] * 3,
    "series": ["Ours", "Baseline A", "Baseline B"],
    "value": [92.4, 89.1, 87.8],
})
config = VerticalBarConfig(
    category_order=("Accuracy",),
    series_order=("Ours", "Baseline A", "Baseline B"),
    y_min=0, y_max=100, y_ticks=(0, 25, 50, 75, 100),
    axis_label="Accuracy (%)", title="One headline metric",
    figure_size=(6.6, 3.8), legend_columns=3, value_decimals=1,
)
figure = plot_grouped_vertical_bars(data, config)
for path in export_figure(figure, ROOT / "figure/output/01_main_result_bar", ExportConfig(formats=("png",))):
    print(path)
plt.close(figure)
