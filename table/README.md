# Reusable LaTeX table templates

This directory contains paper-ready table layouts extracted from the TACIT
paper style. The templates are ordinary .tex files, so they can be copied
into another LaTeX project or included with \input{...}. Each file contains
an illustrative table with placeholder values that are meant to be replaced.

## Included layouts

| Template | Use it for |
| --- | --- |
| compact_results.tex | Compact method and metric comparisons |
| grouped_benchmark.tex | Multi-level benchmark headers with \multicolumn and \cmidrule |
| paired_metrics.tex | Accuracy/latency or quality/cost pairs with \multirow |
| direction_blocks.tex | Source-to-target or condition blocks with wrapped labels |
| ablation_sections.tex | Several named ablation blocks in one table |
| confidence_intervals.tex | Scores with bootstrap or other confidence intervals |
| in_cell_bars.tex | Small horizontal bars inside numeric cells |
| side_by_side.tex | Two aligned tables inside one float |

The layouts reflect patterns already used in the paper: booktabs rules,
compact spacing, tabularx/tabular*, multirow, multicolumn, rowcolor,
light summary washes, and blue emphasis for a best result. Captions and labels
remain in the examples because each file is a standalone copyable table.
When a table is inserted into a larger manuscript, edit the caption and label
in that file so they are unique.

## Quick start

The shared macros are in table_style.tex. Load them after the document class
and before any table template:

    \usepackage{booktabs}
    \usepackage{array}
    \usepackage{caption} % needed by side_by_side.tex
    \input{table/table_style.tex}
    \input{table/templates/compact_results.tex}

For a complete compile of all layouts from the repository root, use a temporary
output directory for the intermediate PDF:

    mkdir -p /tmp/visual_intro_tables
    pdflatex -interaction=nonstopmode -halt-on-error \
      -output-directory=/tmp/visual_intro_tables table/catalog.tex

The package keeps only the PNG previews in `table/examples/`; the
intermediate PDF is not part of the deliverable. The repository's chart templates use
the same eight-color palette, while the table emphasis colors mirror the TACIT
paper:

    #2397FA  #F76AAE  #FCBA70  #7DD277
    #B597D5  #EF8B8B  #D84A58  #ED9239

## Macro reference

table_style.tex provides:

- facttable for local compact table spacing;
- facttableTextCol{width} and facttableNumCol{width} for fixed-width
  text/numeric columns;
- facttableTextX, facttableNumX, and facttableCenterX for tabularx;
- facttableHeaderCell{...}, facttableBestCell{...},
  facttableBestText{...}, facttableFactRow, and facttableSummaryRow for
  semantic emphasis;
- facttableMetricLabel{...} and facttableMetricBlock{...} for secondary
  metrics;
- facttableBarCell{color}{cell width}{fraction} for in-cell bars.

When a table lives inside a narrow minipage, use \linewidth in the table
environment and keep text columns flexible. Use \textwidth only when the
table intentionally spans the full document width. Keep \caption and \label
next to the table float or \captionof{table} that owns the table counter.

## Adapting a template

1. Copy a template or include it with \input.
2. Replace the caption, label, column headings, and example rows.
3. Keep the column count consistent with every row and update each
   \cmidrule{...} range after changing grouped headers.
4. Use \facttableBestCell{...} only for the intended best value; use
   \facttableFactRow for a highlighted method row.
5. Compile the paper and inspect for overfull boxes after changing long model
   names or confidence intervals.

The catalog.tex file is a regression fixture. It compiles all eight layouts
with one document, so it is a quick check after editing the shared style.

For visual inspection, `examples/` contains PNG previews for every template. The source remains in `templates/`; the PDF used to create each PNG is an intermediate file and is not kept in the package. See `examples/README.md` for the full mapping.
