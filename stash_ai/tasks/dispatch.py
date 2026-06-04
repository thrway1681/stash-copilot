"""Generic task-dispatch seam.

One deep seam owns the per-task lifecycle: a standard :class:`TaskContext` is
built once (Stash client + log/progress callbacks + resolved plugin settings +
request id), the task is constructed from it, run, and any failure is logged
uniformly. The entry-point handlers collapse to a thin "build this task"
closure; per-task knowledge lives on the task, not in the entry point.

This module is the pilot (commit 1 of #4). Later commits add a ``ResultStore``
(result persistence) and self-describing construction (``Task.from_context``)
so ``dispatch`` becomes fully generic across every task and the per-handler
``try/except`` duplication disappears.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from ..stash_client import StashClient

LogCallback = Callable[[str, str], None]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class TaskContext:
    """Standard wiring handed to every task built through :func:`dispatch`.

    Bundles the dependencies each handler used to resolve by hand: the Stash
    client, the log/progress callbacks, the resolved plugin settings, the raw
    task ``args``, and the frontend ``request_id`` used for result-file naming.
    """

    stash: StashClient
    log: LogCallback
    progress: ProgressCallback
    plugin_settings: dict[str, Any]
    args: dict[str, Any]
    request_id: str = ""


class RunnableTask(Protocol):
    """Minimal contract a task must satisfy to be run through the seam.

    ``run()`` returns the task's result (or ``None`` for log-only tasks).
    """

    def run(self) -> Any: ...


@runtime_checkable
class SelfBuildingTask(RunnableTask, Protocol):
    """A task that declares how to construct itself from a :class:`TaskContext`.

    The self-describing construction hook (commit 3 of #4): a task implements
    ``from_context`` as a classmethod that resolves its own settings from the
    standard context, so a handler points :func:`dispatch` at the task instead
    of hand-wiring the constructor — the bound ``Task.from_context`` is exactly
    the ``Callable[[TaskContext], RunnableTask]`` that ``build_task`` expects.
    Per-task construction knowledge lives on the task, keeping ``dispatch``
    generic across every task.
    """

    @classmethod
    def from_context(cls, ctx: TaskContext) -> RunnableTask: ...


TaskT = TypeVar("TaskT", bound=RunnableTask)


def _attribution(task_label: str) -> str:
    """Error-message prefix naming the failing task, or empty if it never built.

    A build-phase failure (bad import, missing connection) happens before the
    task object exists, so there is no class name to attribute — those stay
    unprefixed. Once construction succeeds, every later failure is tagged with
    ``[TaskClassName]`` so the uniform log line still says *which* task failed.
    """
    return f"[{task_label}] " if task_label else ""


def dispatch(
    *,
    log: LogCallback,
    build_context: Callable[[], TaskContext],
    build_task: Callable[[TaskContext], TaskT],
    on_result: Callable[[TaskT, Any], None] | None = None,
) -> None:
    """Run one task through the seam with uniform error handling.

    Builds the context, constructs the task from it, runs it, and routes the
    result to ``on_result`` if given. Every error branch the per-handler
    ``try/except`` used to repeat (import, connection, runtime, last-resort)
    lives here once. Failures are logged via ``log`` and never re-raised, so a
    failing task never crashes the plugin's dispatch loop. Once the task is
    constructed, error lines are prefixed with its class name (``[TaskName]``)
    so the uniform handler still attributes the failure to a specific task.

    Args:
        log: Logger used for error reporting; always available even if building
            the context itself fails.
        build_context: Resolves settings and standard wiring into a
            :class:`TaskContext`. Called inside the error boundary.
        build_task: Constructs the task from the context (the future
            ``Task.from_context`` hook).
        on_result: Optional sink for the task and its return value (e.g. logging
            the output). Runs inside the same error boundary.
    """
    # Class name of the constructed task, for error attribution. Empty until
    # build_task succeeds, so build-phase failures (import/connection) stay
    # unattributed — there is no task object to name yet.
    task_label = ""
    try:
        ctx = build_context()
        task = build_task(ctx)
        task_label = type(task).__name__
        result = task.run()
        if on_result is not None:
            on_result(task, result)
    except ImportError as e:
        log(f"Failed to import Stash AI modules: {e}", "error")
        log("Make sure the stash_ai package is properly installed.", "error")
    except ConnectionError as e:
        log(f"{_attribution(task_label)}Connection error: {e}", "error")
    except RuntimeError as e:
        log(f"{_attribution(task_label)}Task failed: {e}", "error")
    except Exception as e:
        # Uniform last-resort handler: any task failure is logged, never raised.
        log(f"{_attribution(task_label)}Unexpected error: {e}", "error")
