"""Integration checks for the reusable LaTeX table gallery."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "table"
NAMES = (
    "compact_results",
    "grouped_benchmark",
    "paired_metrics",
    "direction_blocks",
    "ablation_sections",
    "confidence_intervals",
    "in_cell_bars",
    "side_by_side",
)


def test_all_table_patterns_have_standalone_snippets() -> None:
    for name in NAMES:
        source = TABLES / "templates" / f"{name}.tex"
        assert source.is_file(), name
        text = source.read_text(encoding="utf-8")
        assert r"\begin{table}" in text or r"\begin{table}[" in text
        assert r"\end{table}" in text
        assert r"\caption{" in text or r"\captionof{table}" in text
        assert r"\label{" in text
        assert r"\toprule" in text
        assert r"\bottomrule" in text
        assert "sections/" not in text


def test_gallery_compiles_without_table_errors(tmp_path: Path) -> None:
    compiler = shutil.which("pdflatex")
    if compiler is None:
        pytest.skip("pdflatex is not installed")
    gallery = TABLES / "catalog.tex"
    assert gallery.is_file()
    result = subprocess.run(
        [
            compiler,
            "-draftmode",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tmp_path}",
            str(gallery),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-1000:]
    log = (tmp_path / "catalog.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull \\hbox" not in log
    assert "Undefined control sequence" not in log
