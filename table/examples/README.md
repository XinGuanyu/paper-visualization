# LaTeX 表格 PNG 预览

这里是表格模板的编译预览。源码统一放在上一级的 `../templates/`，本目录只保留 PNG，PDF 只是生成 PNG 时的中间文件。

| 模板源码 | PNG 预览 | 适合场景 |
| --- | --- | --- |
| [`../templates/07_minimal_table.tex`](../templates/07_minimal_table.tex) | [07_minimal_table.png](07_minimal_table.png) | 最小主结果表 |
| [`../templates/compact_results.tex`](../templates/compact_results.tex) | [compact_results_demo.png](compact_results_demo.png) | 多指标主结果表 |
| [`../templates/grouped_benchmark.tex`](../templates/grouped_benchmark.tex) | [grouped_benchmark_demo.png](grouped_benchmark_demo.png) | 多数据集与分组表头 |
| [`../templates/paired_metrics.tex`](../templates/paired_metrics.tex) | [paired_metrics_demo.png](paired_metrics_demo.png) | 质量与成本成对指标 |
| [`../templates/direction_blocks.tex`](../templates/direction_blocks.tex) | [direction_blocks_demo.png](direction_blocks_demo.png) | 按方向或条件分块 |
| [`../templates/ablation_sections.tex`](../templates/ablation_sections.tex) | [ablation_sections_demo.png](ablation_sections_demo.png) | 消融实验分块表 |
| [`../templates/confidence_intervals.tex`](../templates/confidence_intervals.tex) | [confidence_intervals_demo.png](confidence_intervals_demo.png) | 均值与置信区间 |
| [`../templates/in_cell_bars.tex`](../templates/in_cell_bars.tex) | [in_cell_bars_demo.png](in_cell_bars_demo.png) | 单元格内差值条形图 |
| [`../templates/side_by_side.tex`](../templates/side_by_side.tex) | [side_by_side_demo.png](side_by_side_demo.png) | 并排小表 |

`latex_table_catalog_page1.png` 和 `latex_table_catalog_page2.png` 是 8 个通用模板的汇总目录页面预览。

如果修改了 `../templates/` 中的源码，可用 `../catalog.tex` 编译生成中间 PDF，再用 `pdftoppm` 转成 PNG。交付目录不保留中间 PDF。
