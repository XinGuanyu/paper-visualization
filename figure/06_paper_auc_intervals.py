"""Redraw published AUC and 95% CIs from Mishra et al. (2022), Table 4.
Source: https://doi.org/10.3389/fdgth.2022.869812
The intervals are reported values, not recalculated from patient data.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "figure/data/mishra2022_auc.csv")
y = np.arange(len(data))
errors = np.vstack([data.auc - data.ci_low, data.ci_high - data.auc])
fig, ax = plt.subplots(figsize=(6.6, 3.6), layout="constrained")
ax.errorbar(data.auc, y, xerr=errors, fmt="o",
            color="#157F8C", capsize=4, markersize=6)
ax.set_yticks(y, labels=data.model)
ax.invert_yaxis()
ax.set(xlim=(0.5, 0.9), xlabel="AUC (reported 95% CI)",
       title="Published values: Mishra et al., Table 4")
ax.grid(axis="x", alpha=0.2)
output = ROOT / "figure/output"
output.mkdir(parents=True, exist_ok=True)
for ext in ("png",):
    fig.savefig(output / f"06_paper_auc_intervals.{ext}", dpi=300)
plt.close(fig)
