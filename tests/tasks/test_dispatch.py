"""Tests for the generic task-dispatch seam (issue #4, commit 1).

Exercise :func:`dispatch` and :class:`TaskContext` through their public surface
with the in-memory fake Stash client: a task is built from the context, run, and
its result routed to ``on_result``; every failure mode is logged uniformly and
never re-raised.
"""

from __future__ import annotations

from typing import Any

from stash_ai.tasks.dispatch import TaskContext, dispatch
from tests.fakes.fake_stash_client import FakeStashClient


def _make_context(log: Any) -> TaskContext:
    return TaskContext(
        stash=FakeStashClient(),
        log=log,
        progress=lambda cur, total: None,
        plugin_settings={"excluded_tags": "a, b"},
        args={"mode": "stats_summary"},
        request_id="req-1",
    )


class _RecordingLog:
    """Captures ``(message, level)`` log calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, message: str, level: str) -> None:
        self.calls.append((message, level))

    def messages(self) -> list[str]:
        return [m for m, _ in self.calls]


class _FakeTask:
    """Minimal :class:`RunnableTask` returning a fixed result."""

    def __init__(self, result: Any = "ok") -> None:
        self.result = result
        self.ran = False

    def run(self) -> Any:
        self.ran = True
        return self.result


def test_dispatch_builds_runs_and_routes_result() -> None:
    """Task is constructed from the context, run, and its result handed to on_result."""
    log = _RecordingLog()
    task = _FakeTask(result="summary text")
    received: list[tuple[Any, Any]] = []

    dispatch(
        log=log,
        build_context=lambda: _make_context(log),
        build_task=lambda ctx: task,
        on_result=lambda t, result: received.append((t, result)),
    )

    assert task.ran is True
    assert received == [(task, "summary text")]


def test_dispatch_passes_built_context_to_build_task() -> None:
    """The context from build_context is exactly what build_task receives."""
    log = _RecordingLog()
    ctx = _make_context(log)
    seen: list[TaskContext] = []

    dispatch(
        log=log,
        build_context=lambda: ctx,
        build_task=lambda c: seen.append(c) or _FakeTask(),
        on_result=None,
    )

    assert seen == [ctx]
    assert seen[0].request_id == "req-1"
    assert seen[0].plugin_settings == {"excluded_tags": "a, b"}


def test_dispatch_without_on_result_is_fine() -> None:
    """A log-only task (no on_result) runs without error."""
    log = _RecordingLog()
    task = _FakeTask(result=None)

    dispatch(
        log=log,
        build_context=lambda: _make_context(log),
        build_task=lambda ctx: task,
    )

    assert task.ran is True
    assert log.calls == []  # nothing logged on the happy path


def test_dispatch_logs_runtime_error_uniformly() -> None:
    """A RuntimeError from the task is logged as 'Task failed' and not re-raised."""
    log = _RecordingLog()

    def build_task(ctx: TaskContext) -> Any:
        raise RuntimeError("boom")

    dispatch(log=log, build_context=lambda: _make_context(log), build_task=build_task)

    assert log.calls == [("Task failed: boom", "error")]


def test_dispatch_logs_connection_error_uniformly() -> None:
    """A ConnectionError is reported through the connection branch."""
    log = _RecordingLog()

    class _Failing(_FakeTask):
        def run(self) -> Any:
            raise ConnectionError("no llm")

    dispatch(
        log=log,
        build_context=lambda: _make_context(log),
        build_task=lambda ctx: _Failing(),
    )

    assert log.calls == [("Connection error: no llm", "error")]


def test_dispatch_logs_import_error_uniformly() -> None:
    """An ImportError yields the two-line install hint."""
    log = _RecordingLog()

    def build_task(ctx: TaskContext) -> Any:
        raise ImportError("no module")

    dispatch(log=log, build_context=lambda: _make_context(log), build_task=build_task)

    assert log.calls == [
        ("Failed to import Stash AI modules: no module", "error"),
        ("Make sure the stash_ai package is properly installed.", "error"),
    ]


def test_dispatch_logs_unexpected_error_uniformly() -> None:
    """Any other exception falls through to the last-resort handler."""
    log = _RecordingLog()

    def build_task(ctx: TaskContext) -> Any:
        raise ValueError("weird")

    dispatch(log=log, build_context=lambda: _make_context(log), build_task=build_task)

    assert log.calls == [("Unexpected error: weird", "error")]


def test_dispatch_handles_context_build_failure() -> None:
    """A failure while building the context is caught (log is always available)."""
    log = _RecordingLog()

    def build_context() -> TaskContext:
        raise RuntimeError("no stash connection")

    dispatch(
        log=log,
        build_context=build_context,
        build_task=lambda ctx: _FakeTask(),
    )

    assert log.calls == [("Task failed: no stash connection", "error")]
