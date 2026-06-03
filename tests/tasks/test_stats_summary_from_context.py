"""Tests for the self-describing construction hook (issue #4, commit 3).

``StatsSummaryTask.from_context`` is the pilot for the seam's self-construction
mechanism: a task builds itself from a standard :class:`TaskContext`, resolving
its own settings, so the entry-point handler only points :func:`dispatch` at the
class. These assert the externally observable construction result — the right
config and excluded-tags land on the task — using the in-memory fake client.
"""

from __future__ import annotations

from typing import Any

from stash_ai.tasks.dispatch import SelfBuildingTask, TaskContext, dispatch
from stash_ai.tasks.stats_summary import StatsSummaryTask
from tests.fakes.fake_stash_client import FakeStashClient


def _context(
    plugin_settings: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
) -> TaskContext:
    return TaskContext(
        stash=FakeStashClient(),
        log=lambda msg, level: None,
        progress=lambda cur, total: None,
        plugin_settings=plugin_settings or {},
        args=args or {},
        request_id="req-1",
    )


def test_from_context_resolves_llm_settings_from_plugin_settings() -> None:
    """LLM provider/model are resolved from plugin settings onto the task config."""
    ctx = _context(
        plugin_settings={
            "text_llm_provider": "openrouter",
            "text_llm_model": "some-model",
            "text_llm_base_url": "https://example.test",
        }
    )

    task = StatsSummaryTask.from_context(ctx)

    assert isinstance(task, StatsSummaryTask)
    assert task.llm_config.provider == "openrouter"
    assert task.llm_config.model == "some-model"
    assert task.llm_config.base_url == "https://example.test"


def test_from_context_wires_stash_and_callbacks() -> None:
    """The context's stash client and callbacks are wired onto the task."""
    ctx = _context()

    task = StatsSummaryTask.from_context(ctx)

    assert task.stash is ctx.stash
    assert task.log is ctx.log
    assert task.progress is ctx.progress


def test_from_context_parses_excluded_tags() -> None:
    """A comma-separated excluded_tags string is split and stripped into a list."""
    ctx = _context(plugin_settings={"excluded_tags": "alpha, beta ,, gamma"})

    task = StatsSummaryTask.from_context(ctx)

    assert task.excluded_tags == ["alpha", "beta", "gamma"]


def test_from_context_empty_excluded_tags_is_empty_list() -> None:
    """No excluded_tags setting yields an empty list, not a [''] singleton."""
    ctx = _context(plugin_settings={})

    task = StatsSummaryTask.from_context(ctx)

    assert task.excluded_tags == []


def test_from_context_args_override_plugin_settings() -> None:
    """Task args take precedence over plugin settings when resolving the provider."""
    ctx = _context(
        plugin_settings={"text_llm_provider": "ollama"},
        args={"text_llm_provider": "anthropic"},
    )

    task = StatsSummaryTask.from_context(ctx)

    assert task.llm_config.provider == "anthropic"


def test_pilot_task_satisfies_self_building_contract() -> None:
    """The pilot task conforms to the seam's :class:`SelfBuildingTask` contract.

    Documents that the construction hook + ``run`` together are the interface
    the dispatch seam relies on (the commit-6 guard will assert this for every
    registered task).
    """
    task = StatsSummaryTask.from_context(_context())

    assert isinstance(task, SelfBuildingTask)


def test_dispatch_drives_the_construction_hook() -> None:
    """dispatch builds the task straight from ``from_context`` and runs it.

    Confirms the hook is exactly the ``Callable[[TaskContext], RunnableTask]``
    the seam expects: a handler can hand ``dispatch`` the construction hook with
    no per-task wiring. ``run`` is stubbed so the test stays offline.
    """
    ctx = _context(plugin_settings={"text_llm_provider": "ollama"})
    built: list[StatsSummaryTask] = []

    def build_task(c: TaskContext) -> StatsSummaryTask:
        task = StatsSummaryTask.from_context(c)
        # Keep the test offline: the construction hook is what's under test here,
        # not the LLM round-trip in run().
        task.run = lambda: "stub-summary"  # type: ignore[method-assign]
        built.append(task)
        return task

    received: list[Any] = []
    dispatch(
        log=lambda msg, level: None,
        build_context=lambda: ctx,
        build_task=build_task,
        on_result=lambda t, result: received.append(result),
    )

    assert len(built) == 1
    assert received == ["stub-summary"]
