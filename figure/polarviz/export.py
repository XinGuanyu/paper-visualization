"""Deterministic, transactional export for Matplotlib figures."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp, TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .config import ExportConfig


_FIXED_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)
_METADATA = {
    "png": {
        "Software": "polarviz",
        "Creation Time": "2000-01-01T00:00:00Z",
    },
    "pdf": {
        "Creator": "polarviz",
        "Producer": "polarviz",
        "CreationDate": _FIXED_DATE,
        "ModDate": _FIXED_DATE,
    },
}


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _rollback_publish(
    backups: dict[Path, Path],
    attempted: list[Path],
    backup_directory: Path,
) -> list[tuple[str, BaseException]]:
    """Try every restoration step and retain any backup that cannot be restored."""
    failures: list[tuple[str, BaseException]] = []
    for destination, backup in backups.items():
        if not _lexists(backup):
            continue
        try:
            os.replace(backup, destination)
        except BaseException as error:
            if _lexists(backup):
                failures.append((f"restore {destination}", error))
    for destination in attempted:
        if destination in backups or not _lexists(destination):
            continue
        try:
            destination.unlink()
        except BaseException as error:
            if _lexists(destination):
                failures.append((f"remove new output {destination}", error))
    if not failures:
        try:
            shutil.rmtree(backup_directory)
        except BaseException as error:
            if _lexists(backup_directory):
                failures.append((f"remove backup directory {backup_directory}", error))
    return failures


def _publish(staged: tuple[Path, ...], destinations: tuple[Path, ...]) -> None:
    """Publish staged files and restore every prior destination on failure."""
    backup_directory = Path(
        mkdtemp(
            prefix=f".{destinations[0].stem}-backup-",
            dir=destinations[0].parent,
        )
    )
    backups: dict[Path, Path] = {}
    attempted: list[Path] = []
    try:
        for index, destination in enumerate(destinations):
            if _lexists(destination):
                backup = backup_directory / f"backup-{index}{destination.suffix}"
                backups[destination] = backup
                os.replace(destination, backup)
        for source, destination in zip(staged, destinations):
            attempted.append(destination)
            os.replace(source, destination)
    except BaseException as error:
        rollback_failures = _rollback_publish(
            backups, attempted, backup_directory
        )
        if rollback_failures:
            details = "; ".join(
                f"{action}: {type(failure).__name__}: {failure}"
                for action, failure in rollback_failures
            )
            retained = [
                str(backup) for backup in backups.values() if _lexists(backup)
            ]
            recovery = retained or [str(backup_directory)]
            raise RuntimeError(
                f"figure publish failed: {type(error).__name__}: {error}; "
                f"rollback incomplete: {details}; recoverable backups retained: "
                f"{recovery}"
            ) from error
        if isinstance(error, OSError):
            raise OSError(f"could not publish exported figure files: {error}") from error
        raise
    try:
        shutil.rmtree(backup_directory)
    except BaseException as error:
        raise OSError(
            f"figure files were published but backup cleanup failed at "
            f"{backup_directory}: {error}"
        ) from error


class _FigureState:
    """State mutated by Matplotlib's draw and layout passes during savefig."""

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.axes = [
            (
                axis,
                axis.get_position().frozen(),
                axis.get_position(original=True).frozen(),
                axis.get_in_layout(),
            )
            for axis in figure.axes
        ]
        self.subplotpars = figure.subplotpars
        self.subplot_values = {
            name: getattr(self.subplotpars, name)
            for name in ("left", "right", "bottom", "top", "wspace", "hspace")
        }
        self.layout_engine = figure.get_layout_engine()
        self.layout_config = (
            None if self.layout_engine is None else deepcopy(self.layout_engine.get())
        )
        artists = [figure, *figure.findobj()]
        seen: set[int] = set()
        self.stale = []
        for artist in artists:
            if id(artist) not in seen:
                seen.add(id(artist))
                self.stale.append((artist, artist.stale))

    def restore(self) -> None:
        if self.figure.get_layout_engine() is not self.layout_engine:
            self.figure.set_layout_engine(self.layout_engine)
        if self.layout_engine is not None and self.layout_config is not None:
            self.layout_engine.set(**self.layout_config)
        self.figure.subplotpars = self.subplotpars
        self.subplotpars.update(**self.subplot_values)
        for axis, active, original, in_layout in self.axes:
            axis._set_position(original, which="original")
            axis._set_position(active, which="active")
            axis.set_in_layout(in_layout)
        for artist, stale in self.stale:
            artist._stale = stale


def export_figure(
    figure: Figure,
    stem: str | os.PathLike[str],
    config: ExportConfig = ExportConfig(),
) -> list[Path]:
    """Write a figure in configured formats and return paths in that order.

    Every format is rendered into a temporary directory first. Destinations are
    updated only after all renders succeed, so a render failure cannot partially
    replace a previous export.
    """
    if not isinstance(figure, Figure):
        raise TypeError("figure must be a Matplotlib Figure")
    output_stem = Path(stem)
    if output_stem.suffix:
        raise ValueError("stem must not have a suffix")

    parent = output_stem.parent
    if config.create_directory:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(f"could not create export parent directory {parent}: {error}") from error
    elif not parent.is_dir():
        raise FileNotFoundError(
            f"export parent directory does not exist: {parent}"
        )

    destinations = tuple(output_stem.with_suffix(f".{fmt}") for fmt in config.formats)
    directory_destinations = [path for path in destinations if path.is_dir()]
    if directory_destinations:
        raise IsADirectoryError(
            f"export destination is a directory: {directory_destinations[0]}"
        )
    source_state = _FigureState(figure)
    isolated = True
    render_figure: Figure | None = None
    try:
        try:
            render_figure = deepcopy(figure)
        except Exception:
            render_figure = figure
            isolated = False
        try:
            temporary = TemporaryDirectory(prefix=f".{output_stem.name}-", dir=parent)
        except OSError as error:
            raise OSError(
                f"could not create export staging directory in {parent}: {error}"
            ) from error
        with temporary as temporary_name:
            staging = Path(temporary_name)
            staged: list[Path] = []
            for index, (fmt, destination) in enumerate(zip(config.formats, destinations)):
                staged_path = staging / f"{index}-{output_stem.name}.{fmt}"
                save_options = {
                    "format": fmt,
                    "dpi": config.dpi,
                    "transparent": config.transparent,
                    "metadata": _METADATA[fmt],
                }
                if not config.transparent:
                    save_options["facecolor"] = "white"
                try:
                    with mpl.rc_context(rc={"pdf.fonttype": 42, "ps.fonttype": 42}):
                        render_figure.savefig(staged_path, **save_options)
                except Exception as error:
                    raise OSError(
                        f"could not export {fmt} figure to {destination}: {error}"
                    ) from error
                if not staged_path.is_file() or staged_path.stat().st_size == 0:
                    raise OSError(
                        f"could not export {fmt} figure to {destination}: output is empty"
                    )
                staged.append(staged_path)
            _publish(tuple(staged), destinations)
    finally:
        if isolated and render_figure is not None:
            plt.close(render_figure)
        else:
            source_state.restore()

    missing = [path for path in destinations if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise OSError(f"exported figure output is missing or empty: {missing}")
    return list(destinations)
