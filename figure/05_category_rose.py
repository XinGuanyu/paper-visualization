"""Teaching demo: simulated category scores encoded by petal area."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from polarviz import ExportConfig, RoseConfig, plot_grouped_rose, export_figure

ROOT = Path(__file__).resolve().parents[1]
categories = ("Task A", "Task B", "Task C", "Task D", "Task E", "Task F")
data = pd.DataFrame({
    "category": list(categories) * 2,
    "series": ["Ours"] * 6 + ["Baseline"] * 6,
    "value": [82, 65, 78, 90, 72, 85, 70, 68, 60, 76, 64, 80],
    "group": ["Group 1"] * 3 + ["Group 2"] * 3
             + ["Group 1"] * 3 + ["Group 2"] * 3,
})
config = RoseConfig(
    category_order=categories, series_order=("Ours", "Baseline"),
    group_order=("Group 1", "Group 2"), r_min=0, r_max=100,
    r_ticks=(25, 50, 75, 100), encoding="area",
    title="Category pattern (simulated)", legend_columns=2,
)
figure = plot_grouped_rose(data, config)
for path in export_figure(figure, ROOT / "figure/output/05_category_rose", ExportConfig(formats=("png",))):
    print(path)
plt.close(figure)
