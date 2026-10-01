"""Teaching demo: pointwise normal-approximation CIs from simulated data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "figure/output"
output.mkdir(parents=True, exist_ok=True)
data_dir = ROOT / "figure/data"
data_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
time = np.arange(5)
# 40 independent simulated subjects, each observed at five stages.
values = (np.array([0.42, 0.51, 0.63, 0.68, 0.74])
          + rng.normal(0, 0.07, (40, 1))
          + rng.normal(0, 0.05, (40, 5)))
mean = values.mean(axis=0)
se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
low, high = mean - 1.96 * se, mean + 1.96 * se
pd.DataFrame({
    "time": time, "mean": mean, "ci_low": low, "ci_high": high,
}).to_csv(data_dir / "04_trajectory_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
ax.plot(time, mean, "o-", color="#2397FA", label="Mean")
ax.fill_between(time, low, high, color="#2397FA",
                alpha=0.20, label="Pointwise 95% CI (approx.)")
ax.set(xlabel="Stage", ylabel="Score", ylim=(0, 1),
       title="Mean trajectory (simulated, n=40)")
ax.set_xticks(time)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
for ext in ("png",):
    fig.savefig(output / f"04_case_study_line_with_ci.{ext}", dpi=300)
plt.close(fig)
