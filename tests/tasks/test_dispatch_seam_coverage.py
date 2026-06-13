"""Cross-stack guard for the task-dispatch seam (issue #4, commit 6).

The seam (``stash_ai/tasks/dispatch.py``) is now the universal path for task
execution: every mode the plugin dispatches is handled by collapsing its
``run_<mode>`` entry point to a ``build_task`` closure that returns
``SomeTask.from_context(ctx)``. Each such task is exactly one of:

* **result-producing** — it persists a result the frontend polls and declares a
  ``result_key`` (``assets/{result_key}_{request_id}.json``). Some route through
  :class:`~stash_ai.tasks.result_store.ResultStore`; others (recommendations,
  taste map, eroscripts) write their own file. Both are valid — the
  guard only requires the ``result_key`` marker.
* **log-only** — it logs, prints to stdout, mutates the DB, or writes a
  non-polled artifact, and declares no ``result_key``.

Two demo/hook modes (``process_all``, ``process_scene``) deliberately bypass the
seam: they use ``StashPlugin`` helper methods directly rather than a task module
(see CLAUDE.md / the migration notes), so they are explicitly exempt.

This test is the backstop that keeps the seam total: register a new mode without
classifying it here (and wiring it onto the seam) and ``test_every_registered_
mode_is_classified`` fails. Mark a result-writer as log-only (or vice versa) and
the ``result_key`` assertions fail.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from stash_ai.tasks.ask import AskTask
from stash_ai.tasks.build_frame_index import BuildFrameIndexTask
from stash_ai.tasks.chat import ChatTask
from stash_ai.tasks.cleanup_orphaned import CleanupOrphanedTask
from stash_ai.tasks.clear_chat import ClearChatTask
from stash_ai.tasks.describe_performer import DescribePerformerTask
from stash_ai.tasks.embed_cached_frames import EmbedCachedFramesTask
from stash_ai.tasks.embed_o_moments import EmbedOMomentsTask
from stash_ai.tasks.embed_performers import EmbedPerformersTask
from stash_ai.tasks.embed_scenes import EmbedScenesTask
from stash_ai.tasks.embedding_models import GetEmbeddingModelsTask
from stash_ai.tasks.eroscripts_actions import (
    EroscriptsDownloadTask,
    EroscriptsSearchTask,
    EroscriptsStatusTask,
    EroscriptsValidateAuthTask,
)
from stash_ai.tasks.find_similar import FindSimilarTask
from stash_ai.tasks.find_similar_by_frame import FindSimilarByFrameTask
from stash_ai.tasks.find_similar_performers import FindSimilarPerformersTask
from stash_ai.tasks.frame_analysis import (
    CheckFrameAnalysisTask,
    FrameAnalysisTask,
    StartFrameAnalysisTask,
)
from stash_ai.tasks.recommendations import RecommendationsTask
from stash_ai.tasks.scene_vision import SceneVisionTask
from stash_ai.tasks.search_by_text import SearchByTextTask
from stash_ai.tasks.stats_summary import StatsSummaryTask
from stash_ai.tasks.tag_gap_detection import (
    PreviewTagImpactTask,
    SceneTagGapsTask,
    TagGapDetectionTask,
)
from stash_ai.tasks.tag_suggestion_actions import (
    ApplySuggestedTagTask,
    ClearDismissedTagsTask,
    DismissSuggestedTagTask,
)
from stash_ai.tasks.tag_suggestions import TagSuggestionsTask
from stash_ai.tasks.taste_map import TasteMapTask
from tests.fakes.fake_stash_client import FakeStashClient

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_FILE = REPO_ROOT / "stash-copilot.py"

# Modes that bypass the seam by design: demo/hook handlers that mutate Stash via
# StashPlugin helper methods (get_scene/update_scene/find_scenes) rather than a
# task module. Reimplementing those helpers inside a task would distort demo
# code, so they stay as-is. Documented in CLAUDE.md.
EXEMPT = {"process_all", "process_scene"}

# Result-producing modes → their task class. Each declares a ``result_key`` and
# persists a result the frontend polls.
RESULT_PRODUCING: dict[str, type] = {
    "ask": AskTask,
    "recommendations": RecommendationsTask,
    "build_taste_map": TasteMapTask,
    "detect_tag_gaps": TagGapDetectionTask,
    "get_scene_tag_gaps": SceneTagGapsTask,
    "preview_tag_impact": PreviewTagImpactTask,
    "get_tag_suggestions": TagSuggestionsTask,
    "find_similar": FindSimilarTask,
    "find_similar_by_frame": FindSimilarByFrameTask,
    "find_similar_performers": FindSimilarPerformersTask,
    "search_by_text": SearchByTextTask,
    "get_embedding_models": GetEmbeddingModelsTask,
    "eroscripts_validate_auth": EroscriptsValidateAuthTask,
    "eroscripts_search": EroscriptsSearchTask,
    "eroscripts_download": EroscriptsDownloadTask,
    "eroscripts_status": EroscriptsStatusTask,
}

# Log-only modes → their task class. These declare NO ``result_key``: they log,
# print to stdout (frame-analysis status / JSON_RESULT), mutate the DB, or write
# non-polled artifacts (FAISS index, embeddings, conversation history).
LOG_ONLY: dict[str, type] = {
    "stats_summary": StatsSummaryTask,
    "chat": ChatTask,
    "clear_chat": ClearChatTask,
    "scene_vision": SceneVisionTask,
    "embed_scenes": EmbedScenesTask,
    "frame_analysis": FrameAnalysisTask,
    "check_frame_analysis": CheckFrameAnalysisTask,
    "run_frame_analysis": StartFrameAnalysisTask,
    "embed_o_moments": EmbedOMomentsTask,
    "embed_cached_frames": EmbedCachedFramesTask,
    "build_frame_index": BuildFrameIndexTask,
    "cleanup_orphaned": CleanupOrphanedTask,
    "embed_performers": EmbedPerformersTask,
    "describe_performers": DescribePerformerTask,
    "apply_suggested_tag": ApplySuggestedTagTask,
    "dismiss_suggested_tag": DismissSuggestedTagTask,
    "clear_dismissed_tags": ClearDismissedTagsTask,
}

SEAM_TASKS: dict[str, type] = {**RESULT_PRODUCING, **LOG_ONLY}


def _registered_modes() -> set[str]:
    """Load the hyphenated entry-point module and read its live mode registry.

    Uses the injection path (a fake client) so construction touches neither
    stdin nor a real Stash connection — ``_task_handlers()`` just returns the
    bound handlers, whose keys are the dispatchable modes.
    """
    spec = importlib.util.spec_from_file_location("stash_copilot_entry_guard", PLUGIN_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = module.MyPlugin(stash_client=FakeStashClient())
    return set(plugin._task_handlers())


def test_every_registered_mode_is_classified() -> None:
    """Each dispatchable mode is exactly one of: result-producing, log-only, exempt.

    The completeness backstop: a new mode added to ``_task_handlers()`` without
    being wired onto the seam (and listed here) trips the first assertion; a
    removed mode trips the second.
    """
    registered = _registered_modes()
    classified = set(SEAM_TASKS) | EXEMPT

    unclassified = registered - classified
    assert not unclassified, (
        f"these modes are registered but not classified — wire them onto the dispatch "
        f"seam and add them to RESULT_PRODUCING / LOG_ONLY / EXEMPT: {sorted(unclassified)}"
    )

    stale = classified - registered
    assert not stale, (
        f"these modes are classified here but no longer registered — remove them: {sorted(stale)}"
    )


def test_classification_buckets_are_disjoint() -> None:
    """No mode is double-classified across the three buckets."""
    assert set(RESULT_PRODUCING) & set(LOG_ONLY) == set()
    assert set(RESULT_PRODUCING) & EXEMPT == set()
    assert set(LOG_ONLY) & EXEMPT == set()


@pytest.mark.parametrize("mode,task_cls", sorted(SEAM_TASKS.items()))
def test_seam_task_is_self_building(mode: str, task_cls: type) -> None:
    """Every seam task exposes a ``from_context`` classmethod (the construction hook)."""
    from_context = inspect.getattr_static(task_cls, "from_context", None)
    assert isinstance(from_context, classmethod), (
        f"{mode}: {task_cls.__name__}.from_context must be a classmethod (the seam's "
        f"self-describing construction hook)"
    )


@pytest.mark.parametrize("mode,task_cls", sorted(RESULT_PRODUCING.items()))
def test_result_producing_task_declares_result_key(mode: str, task_cls: type) -> None:
    """A result-producing task declares a non-empty ``result_key`` string."""
    result_key = getattr(task_cls, "result_key", None)
    assert isinstance(result_key, str) and result_key, (
        f"{mode}: {task_cls.__name__} is classified result-producing but declares no "
        f"result_key (got {result_key!r})"
    )


@pytest.mark.parametrize("mode,task_cls", sorted(LOG_ONLY.items()))
def test_log_only_task_declares_no_result_key(mode: str, task_cls: type) -> None:
    """A log-only task declares no ``result_key`` (guards against mis-marking a writer)."""
    result_key = getattr(task_cls, "result_key", None)
    assert result_key is None, (
        f"{mode}: {task_cls.__name__} is classified log-only but declares "
        f"result_key={result_key!r} — if it now produces a polled result, move it to "
        f"RESULT_PRODUCING"
    )
