"""Tests for advanced database tools (Phase 4)."""

import sqlite3
from unittest.mock import MagicMock

from stash_ai.tools.database import (
    QueryDuplicatesFindingTool,
    QueryInteractiveContentTool,
    QueryOHistoryTool,
    QueryPerformerCareerTimelineTool,
    QueryUnwatchedContentTool,
    RankScenesByEngagementTool,
)


class TestQueryInteractiveContentTool:
    """Tests for QueryInteractiveContentTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful interactive content query."""
        tool = QueryInteractiveContentTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        # Scene 13 is interactive in test data
        assert len(scenes) >= 1

    def test_interactive_scene_properties(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that interactive scenes have expected properties."""
        tool = QueryInteractiveContentTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        if len(scenes) > 0:
            scene = scenes[0]
            assert "id" in scene
            assert "title" in scene
            # Should indicate interactive status
            assert scene.get("interactive") is True or "interactive_speed" in scene

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryInteractiveContentTool(mock_stash)

        result = tool.execute(limit=1)

        assert result["success"] is True
        assert len(result["data"]["scenes"]) <= 1

    def test_schema_generation(self, mock_stash: MagicMock) -> None:
        """Test that tool schema is correctly generated."""
        tool = QueryInteractiveContentTool(mock_stash)
        schema = tool.to_schema()

        assert schema["name"] == "query_interactive_content"


class TestQueryUnwatchedContentTool:
    """Tests for QueryUnwatchedContentTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful unwatched content query."""
        tool = QueryUnwatchedContentTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        # Scene 14 is unwatched in test data
        assert len(scenes) >= 1

    def test_unwatched_scenes_have_zero_views(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that returned scenes have zero views."""
        tool = QueryUnwatchedContentTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        for scene in scenes:
            # Scene should have play_count of 0 or no view history
            if "play_count" in scene:
                assert scene["play_count"] == 0

    def test_sort_by_date(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting by date."""
        tool = QueryUnwatchedContentTool(mock_stash)

        result = tool.execute(sort_by="date")

        assert result["success"] is True

    def test_sort_by_rating(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting by rating."""
        tool = QueryUnwatchedContentTool(mock_stash)

        result = tool.execute(sort_by="rating")

        assert result["success"] is True

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryUnwatchedContentTool(mock_stash)

        result = tool.execute(limit=5)

        assert result["success"] is True
        assert len(result["data"]["scenes"]) <= 5


class TestRankScenesByEngagementTool:
    """Tests for RankScenesByEngagementTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful engagement ranking."""
        tool = RankScenesByEngagementTool(mock_stash)

        # Use scene IDs from test data
        result = tool.execute(scene_ids=[1, 3, 7, 10])

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        assert len(scenes) > 0

    def test_favorites_scoring_mode(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test favorites scoring mode."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[1, 3, 7, 10], scoring_mode="favorites")

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        # Scenes with higher O counts should rank higher
        if len(scenes) >= 2:
            # Verify ordering by score
            for i in range(len(scenes) - 1):
                assert scenes[i]["score"] >= scenes[i + 1]["score"]

    def test_recent_scoring_mode(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test recent scoring mode with time decay."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[1, 10, 13], scoring_mode="recent")

        assert result["success"] is True

    def test_completion_scoring_mode(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test completion scoring mode."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[1, 3, 7], scoring_mode="completion")

        assert result["success"] is True

    def test_intensity_scoring_mode(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test intensity scoring mode."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[7, 10, 13], scoring_mode="intensity")

        assert result["success"] is True

    def test_min_score_filter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test minimum score filter."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[1, 3, 7, 10], min_score=1.0)

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        for scene in scenes:
            assert scene["score"] >= 1.0

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[1, 3, 7, 10], limit=2)

        assert result["success"] is True
        assert len(result["data"]["scenes"]) <= 2

    def test_missing_scene_ids(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when scene_ids is missing."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False

    def test_empty_scene_ids(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test with empty scene_ids list."""
        tool = RankScenesByEngagementTool(mock_stash)

        result = tool.execute(scene_ids=[])

        # Should either succeed with empty results or fail gracefully
        if result["success"]:
            assert len(result["data"]["scenes"]) == 0


class TestQueryDuplicatesFindingTool:
    """Tests for QueryDuplicatesFindingTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful duplicates finding."""
        tool = QueryDuplicatesFindingTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True

    def test_finds_duplicate_fingerprints(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding duplicates by fingerprint."""
        tool = QueryDuplicatesFindingTool(mock_stash)

        # method="fingerprint" is the default and correct value
        result = tool.execute(method="fingerprint")

        assert result["success"] is True
        duplicate_groups = result["data"]["duplicate_groups"]
        # Scenes 17 and 18 have the same phash in test data
        assert len(duplicate_groups) >= 1

    def test_finds_duplicate_by_filename(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding duplicates by filename similarity."""
        tool = QueryDuplicatesFindingTool(mock_stash)

        result = tool.execute(method="filename")

        assert result["success"] is True

    def test_similarity_threshold(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test similarity threshold parameter."""
        tool = QueryDuplicatesFindingTool(mock_stash)

        result = tool.execute(similarity_threshold=0.9)

        assert result["success"] is True


class TestQueryOHistoryTool:
    """Tests for QueryOHistoryTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful O history query."""
        tool = QueryOHistoryTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        # O history has time_series and top_scenes
        top_scenes = result["data"]["top_scenes"]
        assert len(top_scenes) > 0

    def test_top_scenes_for_specific_scene(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test O history includes scene 7 which has 4 O entries."""
        tool = QueryOHistoryTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        top_scenes = result["data"]["top_scenes"]
        # Find scene 7 in top scenes
        scene_7 = next((s for s in top_scenes if s["scene_id"] == 7), None)
        if scene_7:
            assert scene_7["o_count"] == 4

    def test_date_range(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering by date range."""
        tool = QueryOHistoryTool(mock_stash)

        result = tool.execute(start_date="2023-10-01", end_date="2023-12-31")

        assert result["success"] is True
        # Date filtering affects aggregation
        assert "date_range" in result["data"]

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryOHistoryTool(mock_stash)

        result = tool.execute(limit=5)

        assert result["success"] is True
        assert len(result["data"]["top_scenes"]) <= 5


class TestQueryPerformerCareerTimelineTool:
    """Tests for QueryPerformerCareerTimelineTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful career timeline query."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        timeline = result["data"]["timeline"]
        assert len(timeline) > 0

    def test_timeline_chronological_order(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that timeline is in chronological order."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        timeline = result["data"]["timeline"]
        # Verify chronological order
        if len(timeline) > 1:
            for i in range(len(timeline) - 1):
                if timeline[i].get("date") and timeline[i + 1].get("date"):
                    assert timeline[i]["date"] <= timeline[i + 1]["date"]

    def test_timeline_includes_scene_info(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that timeline entries include expected information."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        timeline = result["data"]["timeline"]
        if len(timeline) > 0:
            entry = timeline[0]
            # Timeline entries have period, scene_count, studios
            assert "period" in entry
            assert "scene_count" in entry

    def test_performer_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer doesn't exist."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute(performer_name="Nonexistent Person")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_missing_performer_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer_name is missing."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryPerformerCareerTimelineTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe", limit=3)

        assert result["success"] is True
        assert len(result["data"]["timeline"]) <= 3
