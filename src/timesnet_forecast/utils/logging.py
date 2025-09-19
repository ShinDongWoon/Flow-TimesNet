from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TextColumn,
    TaskID,
)

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


@dataclass
class _TaskState:
    key: str
    description: str
    total: float
    completed: float = 0.0
    task_id: Optional[TaskID] = None


class TrainingProgressTracker:
    """Manage Rich progress bars for the training pipeline.

    The tracker gracefully degrades to simple console logging when the
    output is not connected to an interactive terminal or when progress
    reporting is disabled via configuration.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._console = _console
        self._requested_enabled = bool(enabled)
        self._use_progress = self._requested_enabled and self._console.is_terminal
        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, _TaskState] = {}

    def __enter__(self) -> "TrainingProgressTracker":
        if self._use_progress:
            self._progress = progress()
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._use_progress and self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._tasks.clear()

    def add_task(self, key: str, description: str, total: float = 1.0) -> str:
        if not self._requested_enabled:
            return key
        state = _TaskState(key=key, description=description, total=float(total) if total else 0.0)
        if self._use_progress and self._progress is not None:
            state.task_id = self._progress.add_task(description, total=total)
        else:
            self._log_simple(f"[bold blue]{description}[/bold blue] (total={int(total) if float(total).is_integer() else total})")
        self._tasks[key] = state
        return key

    def advance(self, key: str, advance: float = 1.0, description: Optional[str] = None) -> None:
        if not self._requested_enabled:
            return
        state = self._tasks.get(key)
        if state is None:
            raise KeyError(f"Unknown progress task '{key}'")
        if description is not None:
            state.description = description
        if self._use_progress and self._progress is not None and state.task_id is not None:
            if description is not None:
                self._progress.update(state.task_id, description=description)
            if advance:
                self._progress.advance(state.task_id, advance)
            task = self._progress.tasks[state.task_id]
            state.completed = float(task.completed)
            state.total = float(task.total)
        else:
            if advance:
                state.completed += float(advance)
            self._emit_simple_status(state)

    def complete(self, key: str, description: Optional[str] = None) -> None:
        if not self._requested_enabled:
            return
        state = self._tasks.get(key)
        if state is None:
            raise KeyError(f"Unknown progress task '{key}'")
        if description is not None:
            state.description = description
        if self._use_progress and self._progress is not None and state.task_id is not None:
            task = self._progress.tasks[state.task_id]
            total = task.total if task.total is not None else state.total
            self._progress.update(
                state.task_id,
                description=state.description,
                completed=total,
            )
            state.completed = float(total)
            state.total = float(total)
            self._progress.refresh()
        else:
            state.completed = state.total
            self._emit_simple_status(state, completed=True)

    def _emit_simple_status(self, state: _TaskState, completed: bool = False) -> None:
        total = state.total if state.total else 0.0
        completed_val = state.completed
        if total > 0:
            ratio = completed_val / total if total else 0.0
            msg = f"{state.description} [{completed_val:.0f}/{total:.0f} - {ratio * 100:.1f}%]"
        else:
            msg = f"{state.description}"
        if completed or (total > 0 and completed_val >= total):
            self._log_simple(f"[green]{msg}[/green]")
        else:
            self._log_simple(msg)

    def _log_simple(self, message: str) -> None:
        if self._requested_enabled:
            self._console.print(message)
