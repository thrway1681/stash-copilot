"""Tests for the embeddings tools' engagement scoring (issue #1, commit 5).

The three ``_enrich_results`` paths in ``stash_ai.tools.embeddings``
(``QuerySimilarScenesTool``, ``SearchByTextTool``,
``FilterScenesByVisualContentTool``) were migrated off their inline engagement
query + formula onto the single ``EngagementCalculator`` (ADR-0004 canonical
formula: o_count*20 + replays*2 + stars*1.5, with stars = rating100 / 20).

The migration adds the rating term these paths previously omitted, so the
canonical scores differ from the old ``(o_count*20) + (replays*2)`` values.
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from stash_ai.embeddings.storage import SimilarityResult
from stash_ai.tools.embeddings import (
    FilterScenesByVisualContentTool,
    QuerySimilarScenesTool,
    SearchByTextTool,
)
from tests.conftest import NonClosingConnection


@pytest.fixture
def patched_enrich_db(mock_db: sqlite3.Connection, tmp_path: Path) -> Generator[None, None, None]:
    """Point both the embeddings and engagement DB accessors at the fixture DB.

    ``_enrich_results`` reads scene metadata through ``stash_ai.tools.embeddings``
    and engagement counts through ``EngagementCalculator`` (bound in
    ``stash_ai.recommendations.engagement``); both must hit the same in-memory DB.
    """
    mock_db_path = tmp_path / "stash-go.sqlite"
    mock_db_path.touch()
    wrapped = NonClosingConnection(mock_db)

    with (
        patch("stash_ai.tools.embeddings.get_stash_db_path", return_value=mock_db_path),
        patch("stash_ai.tools.embeddings.get_readonly_connection", return_value=wrapped),
        patch(
            "stash_ai.recommendations.engagement.get_stash_db_path",
            return_value=mock_db_path,
        ),
        patch(
            "stash_ai.recommendations.engagement.get_readonly_connection",
            return_value=wrapped,
        ),
    ):
        yield


def _enrich(tool_cls: type, results: list[SimilarityResult]) -> dict[int, dict]:
    """Call ``_enrich_results`` without the tool's heavy ``__init__``.

    The enrich path uses no instance state, so we bypass ``__init__`` (which would
    build an ``EmbeddingStorage`` and touch the real assets dir).
    """
    tool = tool_cls.__new__(tool_cls)
    enriched = tool._enrich_results(results)
    return {item["scene_id"]: item for item in enriched}


# Every enrich path shares the same engagement logic; parametrize over all three.
ENRICH_TOOLS = [QuerySimilarScenesTool, SearchByTextTool, FilterScenesByVisualContentTool]


@pytest.mark.parametrize("tool_cls", ENRICH_TOOLS)
class TestEnrichEngagementScores:
    """Canonical engagement scores from the shared calculator, for each tool."""

    def test_canonical_scores(self, tool_cls: type, patched_enrich_db: None) -> None:
        # Scene 1:  o=2, views=3 (replays=2), rating100=100 (5*) -> 40 + 4 + 7.5 = 51.5
        # Scene 7:  o=4, views=8 (replays=7), rating100=100 (5*) -> 80 + 14 + 7.5 = 101.5
        # Scene 10: o=4, views=10 (replays=9), rating100=100 (5*) -> 80 + 18 + 7.5 = 105.5
        results = [SimilarityResult(scene_id=sid, similarity=0.9) for sid in (1, 7, 10)]
        scenes = _enrich(tool_cls, results)

        assert scenes[1]["engagement_score"] == pytest.approx(51.5)
        assert scenes[7]["engagement_score"] == pytest.approx(101.5)
        assert scenes[10]["engagement_score"] == pytest.approx(105.5)

    def test_rating_only_scene(self, tool_cls: type, patched_enrich_db: None) -> None:
        # Scene 6: no views, no o, rating100=40 (2*) -> 0 + 0 + 3.0 = 3.0
        # The old formula omitted the rating term entirely (would have been 0.0).
        scenes = _enrich(tool_cls, [SimilarityResult(scene_id=6, similarity=0.5)])

        assert scenes[6]["engagement_score"] == pytest.approx(3.0)
        assert scenes[6]["view_count"] == 0
        assert scenes[6]["o_count"] == 0
        assert scenes[6]["replay_count"] == 0

    def test_unwatched_unrated_scene(self, tool_cls: type, patched_enrich_db: None) -> None:
        # Scene 14: unwatched, unrated -> score 0 (no penalty for unrated)
        scenes = _enrich(tool_cls, [SimilarityResult(scene_id=14, similarity=0.5)])

        assert scenes[14]["engagement_score"] == pytest.approx(0.0)
        assert scenes[14]["view_count"] == 0
        assert scenes[14]["o_count"] == 0


@pytest.mark.parametrize("tool_cls", ENRICH_TOOLS)
class TestEnrichContract:
    """Migration preserves each tool's existing output contract."""

    def test_counts_and_fields_present(self, tool_cls: type, patched_enrich_db: None) -> None:
        scenes = _enrich(tool_cls, [SimilarityResult(scene_id=1, similarity=0.9)])

        scene = scenes[1]
        assert scene["view_count"] == 3
        assert scene["o_count"] == 2
        assert scene["replay_count"] == 2  # views beyond the first
        assert scene["similarity"] == pytest.approx(0.9)
        assert "formatted" in scene
        assert "url" in scene

    def test_rating_does_not_dominate(self, tool_cls: type, patched_enrich_db: None) -> None:
        # Regression guard: treating rating100 as raw stars would have scored
        # scene 1 at 194 (40 + 4 + 100*1.5). Canonical (rating100/20) is 51.5.
        scenes = _enrich(tool_cls, [SimilarityResult(scene_id=1, similarity=0.9)])

        assert scenes[1]["engagement_score"] == pytest.approx(51.5)

    def test_missing_db_returns_zero_engagement(self, tool_cls: type, tmp_path: Path) -> None:
        nonexistent = tmp_path / "missing.sqlite"
        with patch("stash_ai.tools.embeddings.get_stash_db_path", return_value=nonexistent):
            scenes = _enrich(tool_cls, [SimilarityResult(scene_id=1, similarity=0.9)])

        assert scenes[1]["engagement_score"] == pytest.approx(0.0)
        assert scenes[1]["view_count"] == 0
