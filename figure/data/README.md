# 教程中的数据

`mishra2022_auc.csv` 抄录 Mishra et al. (2022), Table 4 的 AUC 及 95% CI：
https://doi.org/10.3389/fdgth.2022.869812
行序沿用原表，没有重算区间，也没有原始患者数据。

`heatmap.csv`、`horizontal_bars.csv`、`line.csv`、`lollipop.csv`、`radar.csv`、`rose.csv`、`scatter.csv` 和 `vertical_bars.csv` 是仓库原有 `build_*.py` 使用的教学数据。它们是可读的模拟 CSV，用于展示字段组织、分组顺序和绘图 API；对应图已经放在 `../output/`。

其余编号示例的数据在脚本内明确构造，也都是教学模拟数据。`04_case_study_line_with_ci.py` 使用 40 个独立模拟个体在 5 个阶段的观测，计算逐时间点的正态近似 95% CI。`04_trajectory_summary.csv` 是替换为自己数据时可以参考的汇总格式。
