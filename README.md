# 论文数据可视化入门包

这是论文数据可视化入门教程的配套代码仓库，包含图表源码、表格模板、输入数据和 PNG 预览。教程正文单独发布，使用者可以从本仓库的代码开始复现实验图表。

## 运行示例

需要 Python 3.10 或更新版本。在本目录打开终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python figure/01_main_result_bar.py
```

将最后一行替换为其他示例文件即可运行。生成的图片与数据保存在 `figure/output/`。

## 教程配套示例

| 源码 | 已生成的 demo | 内容 |
| --- | --- | --- |
| [01_main_result_bar.py](figure/01_main_result_bar.py) | [PNG](figure/output/01_main_result_bar.png) | 主结果分组柱状图 |
| [02_baseline_radar.py](figure/02_baseline_radar.py) | [PNG](figure/output/02_baseline_radar.png) | 多指标雷达图 |
| [03_mechanism_scatter.py](figure/03_mechanism_scatter.py) | [PNG](figure/output/03_mechanism_scatter.png) | 观测值与预测值散点图 |
| [04_case_study_line_with_ci.py](figure/04_case_study_line_with_ci.py) | [PNG](figure/output/04_case_study_line_with_ci.png) | 折线与置信区间带 |
| [05_category_rose.py](figure/05_category_rose.py) | [PNG](figure/output/05_category_rose.png) | 分组玫瑰图 |
| [06_paper_auc_intervals.py](figure/06_paper_auc_intervals.py) | [PNG](figure/output/06_paper_auc_intervals.png) | Mishra 论文中的 AUC 与 95% CI |
| [07_minimal_table.tex](table/templates/07_minimal_table.tex) | [PNG](table/examples/07_minimal_table.png) | 可复制的主结果表源码 |
| [08_feature_engineering_diagnostics.py](figure/08_feature_engineering_diagnostics.py) | [PCA](figure/output/08_feature_engineering_pca.png) · [相关热图](figure/output/08_feature_correlation.png) | 特征空间、相关结构与缺失模式 |

## 原有 `figure/build_*.py` demo

下面这组是仓库原有的 `polarviz` 示例代码，源码、输入 CSV 和已经跑出的 PNG 预览都保留在 `figure/` 下：

| 源码 | 已生成的 demo | 适合说明 |
| --- | --- | --- |
| [build_heatmap.py](figure/build_heatmap.py) | [PNG](figure/output/heatmap_example.png) | 多数据集与方法矩阵 |
| [build_horizontal_bars.py](figure/build_horizontal_bars.py) | [PNG](figure/output/horizontal_bars_example.png) | 正负差值与 baseline 对比 |
| [build_line.py](figure/build_line.py) | [PNG](figure/output/line_example.png) | 多阶段性能趋势 |
| [build_lollipop.py](figure/build_lollipop.py) | [PNG](figure/output/lollipop_example.png) | 多指标点线比较 |
| [build_radar.py](figure/build_radar.py) | [PNG](figure/output/radar_example.png) | 多维能力轮廓 |
| [build_rose.py](figure/build_rose.py) | [PNG](figure/output/rose_example.png) | 分组类别模式 |
| [build_scatter.py](figure/build_scatter.py) | [PNG](figure/output/scatter_example.png) | 成本—质量关系与气泡大小 |
| [build_vertical_bars.py](figure/build_vertical_bars.py) | [PNG](figure/output/vertical_bars_example.png) | 多指标分组柱状图 |

重新生成这一组图：

```bash
for script in figure/build_*.py; do
  MPLBACKEND=Agg POLARVIZ_OUTPUT_DIR=figure/output python "$script"
done
```

数据来源与字段见 [数据说明](figure/data/README.md)。

## LaTeX 表格 demo

`table/templates/` 保存表格源码，`table/examples/` 只保存对应的 PNG 预览；`table/catalog.tex` 仍然保留为全部模板的汇总入口。完整清单见 [LaTeX 表格 demo README](table/examples/README.md)：

- [最小主结果表 PNG](table/examples/07_minimal_table.png)
- [目录第 1 页 PNG](table/examples/latex_table_catalog_page1.png) · [第 2 页 PNG](table/examples/latex_table_catalog_page2.png)
- [全部模板 PNG 预览清单](table/examples/README.md)

修改模板后，先在临时目录编译 `table/catalog.tex`，再用 `pdftoppm` 转换页面；`table/examples/` 只保留最终 PNG 预览。

## 目录

- `figure/`：示例源码、输入数据和已经生成的 demo。
- `figure/polarviz/`：绘图 API、样式配置与导出代码。
- `table/`：供读者复制到论文中的表格示例代码，使用方法见 [表格说明](table/README.md)。
- `pyproject.toml`：Python 包和依赖配置。
