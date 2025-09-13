from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TextColumn

_console = Console()


def console() -> Console:
    return _console


def progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=True,
    )


def print_config(cfg: dict) -> None:
    table = Table(title="Config")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    def _walk(prefix: str, d: dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _walk(key, v)
            else:
                table.add_row(key, str(v))
    _walk("", cfg)
    _console.print(table)
