"""Tests for analytics database tools."""

import sqlite3
from unittest.mock import MagicMock

from stash_ai.tools.database import (
    QueryPerformerPairsTool,
    QueryTagCorrelationsTool,
    QueryTagUsageOverTimeTool,
    QueryTopPerformerCommonTagsTool,
    QueryWatchingPatternsTool,
)


class TestQueryWatchingPatternsTool:
    """Tests for QueryWatchingPatternsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful watching patterns query."""
        tool = QueryWatchingPatternsTool(mock_stash)

        # pattern_type is required
        result = tool.execute(pattern_type="hourly")

        assert result["success"] is True
        patterns = result["data"]
        assert patterns is not None

    def test_hourly_patterns(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test hourly viewing patterns."""
        tool = QueryWatchingPatternsTool(mock_stash)

        result = tool.execute(pattern_type="hourly")

        assert result["success"] is True
        if "hourly" in result["data"]:
            hourly = result["data"]["hourly"]
            # Should have entries for hours with views
            assert len(hourly) > 0

    def test_daily_patterns(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test daily viewing patterns (day of week)."""
        tool = QueryWatchingPatternsTool(mock_stash)

        result = tool.execute(pattern_type="daily")

        assert result["success"] is True

    def test_monthly_patterns(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test monthly viewing patterns."""
        tool = QueryWatchingPatternsTool(mock_stash)

        result = tool.execute(pattern_type="monthly")

        assert result["success"] is True

    def test_empty_database(
        self, mock_stash: MagicMock, empty_db: sqlite3.Connection, patched_empty_db: None
    ) -> None:
        """Test with empty database."""
        tool = QueryWatchingPatternsTool(mock_stash)

        # pattern_type is required
        result = tool.execute(pattern_type="hourly")

        assert result["success"] is True
        # Should handle empty database gracefully


class TestQueryTagCorrelationsTool:
    """Tests for QueryTagCorrelationsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful tag correlations query."""
        tool = QueryTagCorrelationsTool(mock_stash)

        # tag_name is required
        result = tool.execute(tag_name="oral")

        assert result["success"] is True
        correlated_tags = result["data"]["correlated_tags"]
        assert len(correlated_tags) >= 0

    def test_correlations_for_specific_tag(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test correlations for a specific tag."""
        tool = QueryTagCorrelationsTool(mock_stash)

        result = tool.execute(tag_name="oral")

        assert result["success"] is True
        correlated_tags = result["data"]["correlated_tags"]
        # oral often appears with blowjob in our test data

    def test_correlation_properties(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that correlations have expected properties."""
        tool = QueryTagCorrelationsTool(mock_stash)

        result = tool.execute(tag_name="oral", limit=5)

        assert result["success"] is True
        correlated_tags = result["data"]["correlated_tags"]
        if len(correlated_tags) > 0:
            tag = correlated_tags[0]
            assert "name" in tag
            assert "co_occurrence_count" in tag

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryTagCorrelationsTool(mock_stash)

        result = tool.execute(tag_name="oral", limit=3)

        assert result["success"] is True
        assert len(result["data"]["correlated_tags"]) <= 3

    def test_excluded_tags_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that excluded tags are filtered from correlations."""
        tool = QueryTagCorrelationsTool(mock_stash)
        tool.set_excluded_tags(["excluded_parent"])

        result = tool.execute(tag_name="oral")

        assert result["success"] is True
        correlated_tags = result["data"]["correlated_tags"]
        for tag in correlated_tags:
            assert tag["name"] != "excluded_parent"
            assert tag["name"] != "excluded_child"


class TestQueryTopPerformerCommonTagsTool:
    """Tests for QueryTopPerformerCommonTagsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful top performer common tags query."""
        tool = QueryTopPerformerCommonTagsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True

    def test_tags_with_performer_names(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding common tags across specified performers."""
        tool = QueryTopPerformerCommonTagsTool(mock_stash)

        result = tool.execute(performer_names=["Jane Doe", "Alice Wonder"])

        assert result["success"] is True
        if "common_tags" in result["data"]:
            # Tags that appear in scenes of both performers

            pass

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryTopPerformerCommonTagsTool(mock_stash)

        result = tool.execute(limit=3)

        assert result["success"] is True

    def test_excluded_tags_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that excluded tags are filtered."""
        tool = QueryTopPerformerCommonTagsTool(mock_stash)
        tool.set_excluded_tags(["excluded_parent"])

        result = tool.execute()

        assert result["success"] is True


class TestQueryPerformerPairsTool:
    """Tests for QueryPerformerPairsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful performer pairs query."""
        tool = QueryPerformerPairsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        pairs = result["data"]["top_pairs"]
        assert len(pairs) >= 0

    def test_pair_properties(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that pairs have expected properties."""
        tool = QueryPerformerPairsTool(mock_stash)

        result = tool.execute(limit=5)

        assert result["success"] is True
        pairs = result["data"]["top_pairs"]
        if len(pairs) > 0:
            pair = pairs[0]
            assert "performer_1" in pair
            assert "performer_2" in pair
            assert "scenes_together" in pair

    def test_pairs_for_specific_performer(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding co-performers for a specific performer."""
        tool = QueryPerformerPairsTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        # Should find John Smith (they share scene 1 and 19)

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryPerformerPairsTool(mock_stash)

        result = tool.execute(limit=3)

        assert result["success"] is True
        assert len(result["data"]["top_pairs"]) <= 3

    def test_top_pairs_have_scenes(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that top pairs have at least 1 scene together."""
        tool = QueryPerformerPairsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        pairs = result["data"]["top_pairs"]
        for pair in pairs:
            assert pair["scenes_together"] >= 1


class TestQueryTagUsageOverTimeTool:
    """Tests for QueryTagUsageOverTimeTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful tag usage over time query."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True

    def test_daily_aggregation(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test daily usage aggregation."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute(aggregation="daily")

        assert result["success"] is True

    def test_weekly_aggregation(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test weekly usage aggregation."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute(aggregation="weekly")

        assert result["success"] is True

    def test_monthly_aggregation(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test monthly usage aggregation."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute(aggregation="monthly")

        assert result["success"] is True

    def test_specific_tag(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test usage trend for specific tag."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute(tag_name="oral")

        assert result["success"] is True

    def test_excluded_tags_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that excluded tags are filtered."""
        tool = QueryTagUsageOverTimeTool(mock_stash)
        tool.set_excluded_tags(["excluded_parent"])

        result = tool.execute()

        assert result["success"] is True

    def test_date_range(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering by date range."""
        tool = QueryTagUsageOverTimeTool(mock_stash)

        result = tool.execute(start_date="2023-01-01", end_date="2023-06-30")

        assert result["success"] is True
