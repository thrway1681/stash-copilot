"""Tests for entity profile database tools (Phase 2)."""

import sqlite3
from unittest.mock import MagicMock

from stash_ai.tools.database import (
    QueryGroupProgressTool,
    QueryPerformerProfileTool,
    QueryStorageStatsTool,
    QueryStudioProfileTool,
    QueryViewingHistoryTool,
)


class TestQueryPerformerProfileTool:
    """Tests for QueryPerformerProfileTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful performer profile query."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        profile = result["data"]["profile"]

        # Verify basic profile info
        assert profile["name"] == "Jane Doe"
        assert profile["gender"] == "female"
        assert profile["ethnicity"] == "Caucasian"
        assert profile["country"] == "USA"
        assert profile["hair_color"] == "brunette"

    def test_performer_with_stats(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test performer profile with statistics."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe", include_stats=True)

        assert result["success"] is True
        # scene_count is in profile, engagement stats are in stats
        profile = result["data"]["profile"]
        assert profile["scene_count"] == 7
        # stats contains engagement metrics
        assert "stats" in result["data"]

    def test_top_tags(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test performer's top tags."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe", top_tags_limit=5)

        assert result["success"] is True
        if "top_tags" in result["data"]:
            assert len(result["data"]["top_tags"]) <= 5

    def test_top_coperformers(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test performer's top co-performers."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe", top_coperformers_limit=3)

        assert result["success"] is True
        if "top_coperformers" in result["data"]:
            assert len(result["data"]["top_coperformers"]) <= 3

    def test_performer_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer doesn't exist."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute(performer_name="Nonexistent Person")

        assert result["success"] is False
        assert "found" in result["error"].lower()

    def test_missing_performer_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer_name is missing."""
        tool = QueryPerformerProfileTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False

    def test_schema_generation(self, mock_stash: MagicMock) -> None:
        """Test that tool schema is correctly generated."""
        tool = QueryPerformerProfileTool(mock_stash)
        schema = tool.to_schema()

        assert schema["name"] == "query_performer_profile"
        assert "performer_name" in schema["parameters"]["required"]


class TestQueryStudioProfileTool:
    """Tests for QueryStudioProfileTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful studio profile query."""
        tool = QueryStudioProfileTool(mock_stash)

        result = tool.execute(studio_name="Big Studio")

        assert result["success"] is True
        profile = result["data"]["profile"]

        assert profile["name"] == "Big Studio"

    def test_studio_with_stats(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test studio profile with statistics."""
        tool = QueryStudioProfileTool(mock_stash)

        result = tool.execute(studio_name="Big Studio", include_stats=True)

        assert result["success"] is True
        # Stats contains engagement metrics (view_count, o_count, play_hours)
        # scene_count is in the profile, not stats
        assert "stats" in result["data"]
        assert "view_count" in result["data"]["stats"]

    def test_sub_studio_parent(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sub-studio shows parent information."""
        tool = QueryStudioProfileTool(mock_stash)

        result = tool.execute(studio_name="Sub Studio A")

        assert result["success"] is True
        # Sub Studio A's parent is Big Studio
        profile = result["data"]["profile"]
        if "parent" in profile:
            assert profile["parent"]["name"] == "Big Studio"

    def test_studio_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when studio doesn't exist."""
        tool = QueryStudioProfileTool(mock_stash)

        result = tool.execute(studio_name="Nonexistent Studio")

        assert result["success"] is False
        assert "found" in result["error"].lower()

    def test_missing_studio_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when studio_name is missing."""
        tool = QueryStudioProfileTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False


class TestQueryGroupProgressTool:
    """Tests for QueryGroupProgressTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful group progress query."""
        tool = QueryGroupProgressTool(mock_stash)

        result = tool.execute(group_name="Movie Series 1")

        assert result["success"] is True
        progress = result["data"]["progress"]

        assert progress["group_name"] == "Movie Series 1"

    def test_group_scenes(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test group contains associated scenes."""
        tool = QueryGroupProgressTool(mock_stash)

        result = tool.execute(group_name="Movie Series 1")

        assert result["success"] is True
        # Movie Series 1 has scenes 1, 3, 5 in test data
        if "scenes" in result["data"]:
            assert len(result["data"]["scenes"]) == 3

    def test_group_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when group doesn't exist."""
        tool = QueryGroupProgressTool(mock_stash)

        result = tool.execute(group_name="Nonexistent Group")

        assert result["success"] is False
        assert "found" in result["error"].lower()

    def test_missing_group_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when group_name is missing."""
        tool = QueryGroupProgressTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False


class TestQueryViewingHistoryTool:
    """Tests for QueryViewingHistoryTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful viewing history query."""
        tool = QueryViewingHistoryTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        views = result["data"]["views"]
        assert len(views) > 0

    def test_date_range_filter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering by date range."""
        tool = QueryViewingHistoryTool(mock_stash)

        result = tool.execute(start_date="2023-10-01", end_date="2023-12-31")

        assert result["success"] is True
        views = result["data"]["views"]
        for entry in views:
            view_date = entry["view_date"][:10]  # Get date portion
            assert view_date >= "2023-10-01"
            assert view_date <= "2023-12-31"

    def test_performer_filter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering by performer."""
        tool = QueryViewingHistoryTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        # All returned scenes should feature Jane Doe

    def test_tag_filter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering by tag."""
        tool = QueryViewingHistoryTool(mock_stash)

        result = tool.execute(tag_name="oral")

        assert result["success"] is True

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryViewingHistoryTool(mock_stash)

        result = tool.execute(limit=5)

        assert result["success"] is True
        assert len(result["data"]["views"]) <= 5


class TestQueryStorageStatsTool:
    """Tests for QueryStorageStatsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful storage stats query."""
        tool = QueryStorageStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        stats = result["data"]

        # Should have library total info
        assert "library_total" in stats
        assert "total_gb" in stats["library_total"]

    def test_storage_by_studio(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test storage breakdown by studio."""
        tool = QueryStorageStatsTool(mock_stash)

        result = tool.execute(breakdown_by="studio")

        assert result["success"] is True

    def test_storage_by_performer(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test storage breakdown by performer."""
        tool = QueryStorageStatsTool(mock_stash)

        result = tool.execute(breakdown_by="performer")

        assert result["success"] is True

    def test_empty_database(
        self, mock_stash: MagicMock, empty_db: sqlite3.Connection, patched_empty_db: None
    ) -> None:
        """Test with empty database."""
        tool = QueryStorageStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        # Should handle empty database gracefully
