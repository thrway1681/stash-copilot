"""Tests for EnrichSceneResultsTool engagement scoring (issue #1, commit 4).

These pin the behaviour after migrating the tool off its inline engagement
query + formula onto the single EngagementCalculator (ADR-0004 canonical
formula: o_count*20 + replays*2 + stars*1.5, with stars = rating100 / 20).
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stash_ai.tools.database import EnrichSceneResultsTool
from tests.conftest import NonClosingConnection


@pytest.fixture
def patched_enrich_db(mock_db: sqlite3.Connection, tmp_path: Path) -> Generator[None, None, None]:
    """Patch the DB accessors in BOTH the tools and engagement modules.

    EnrichSceneResultsTool reads metadata through ``stash_ai.tools.database``
    and engagement counts through ``EngagementCalculator`` (which binds the
    accessors in ``stash_ai.recommendations.engagement``); both must point at
    the same in-memory fixture DB.
    """
    mock_db_path = tmp_path / "stash-go.sqlite"
    mock_db_path.touch()
    wrapped = NonClosingConnection(mock_db)

    with (
        patch("stash_ai.tools.database.get_stash_db_path", return_value=mock_db_path),
        patch("stash_ai.tools.database.get_readonly_connection", return_value=wrapped),
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


def _by_id(scenes: list[dict]) -> dict[int, dict]:
    return {s["scene_id"]: s for s in scenes}


class TestEnrichEngagementScores:
    """Canonical engagement scores from the shared calculator."""

    def test_canonical_scores(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        # Scene 1:  o=2, views=3 (replays=2), rating100=100 (5*) -> 40 + 4 + 7.5 = 51.5
        # Scene 7:  o=4, views=8 (replays=7), rating100=100 (5*) -> 80 + 14 + 7.5 = 101.5
        # Scene 10: o=4, views=10 (replays=9), rating100=100 (5*) -> 80 + 18 + 7.5 = 105.5
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[1, 7, 10])

        assert result["success"] is True
        scenes = _by_id(result["data"]["scenes"])
        assert scenes[1]["engagement_score"] == pytest.approx(51.5)
        assert scenes[7]["engagement_score"] == pytest.approx(101.5)
        assert scenes[10]["engagement_score"] == pytest.approx(105.5)

    def test_rating_only_scene(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        # Scene 6: no views, no o, rating100=40 (2*) -> 0 + 0 + 3.0 = 3.0
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[6])

        scene = result["data"]["scenes"][0]
        assert scene["engagement_score"] == pytest.approx(3.0)
        assert scene["view_count"] == 0
        assert scene["o_count"] == 0
        assert scene["replay_count"] == 0

    def test_unwatched_unrated_scene(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        # Scene 14: unwatched, unrated -> score 0, rating None (no penalty)
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[14])

        scene = result["data"]["scenes"][0]
        assert scene["engagement_score"] == pytest.approx(0.0)
        assert scene["rating"] is None


class TestEnrichRatingScale:
    """The rating field is reported on the 0-5 star scale (rating100 / 20)."""

    def test_rating_converted_to_stars(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        # rating100=100 -> 5.0 stars (not the raw 100 the old code reported)
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[1])

        scene = result["data"]["scenes"][0]
        assert scene["rating"] == pytest.approx(5.0)

    def test_rating_does_not_dominate_score(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        # Regression guard for the old ~20x bug: treating rating100 as stars made
        # scene 1 score 194 (40 + 4 + 100*1.5). Canonical is 51.5.
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[1])

        assert result["data"]["scenes"][0]["engagement_score"] == pytest.approx(51.5)


class TestEnrichOrderingAndFields:
    """Migration preserves the tool's existing contract."""

    def test_input_order_preserved(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[10, 1, 7])

        ordered = [s["scene_id"] for s in result["data"]["scenes"]]
        assert ordered == [10, 1, 7]

    def test_engagement_fields_excluded_when_not_requested(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_enrich_db: None
    ) -> None:
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[1], include_fields=["studio"])

        scene = result["data"]["scenes"][0]
        assert "engagement_score" not in scene
        assert "rating" not in scene

    def test_empty_scene_ids_errors(self, mock_stash: MagicMock) -> None:
        tool = EnrichSceneResultsTool(mock_stash)
        result = tool.execute(scene_ids=[])

        assert result["success"] is False
        assert result["error"] is not None
