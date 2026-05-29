"""Tests for basic database tools (Phase 1 and core tools)."""

import sqlite3
from unittest.mock import MagicMock

from stash_ai.tools.database import (
    QueryAllPerformersTool,
    QueryAllTagsTool,
    QueryFavoritesTool,
    QueryLibraryStatsTool,
    QueryPerformersByAttributeTool,
    QueryPerformerTagsTool,
    QueryResumePointsTool,
    QueryScenesByDateTool,
    QueryScenesByRatingTool,
    QueryTagPerformersTool,
    QueryTopPerformersTool,
    QueryTopStudiosTool,
    QueryTopTagsTool,
    QueryViewingStatsTool,
)


class TestQueryPerformerTagsTool:
    """Tests for QueryPerformerTagsTool."""

    def test_execute_success_by_exact_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful query with exact performer name."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe", limit=10)

        assert result["success"] is True
        assert result["error"] is None
        assert result["data"] is not None

        # Verify performer data
        performer = result["data"]["performer"]
        assert performer["id"] == 1
        assert performer["name"] == "Jane Doe"
        assert "JD" in performer["aliases"]

        # Verify scene count (Jane Doe has 7 scenes)
        assert result["data"]["scene_count"] == 7

        # Verify tags are returned
        scene_tags = result["data"]["scene_tags"]
        assert len(scene_tags) > 0
        assert all("id" in tag and "name" in tag and "count" in tag for tag in scene_tags)

    def test_execute_success_by_alias(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding performer by alias."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="JD")

        assert result["success"] is True
        assert result["data"]["performer"]["name"] == "Jane Doe"

    def test_execute_success_by_partial_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test finding performer by partial name match."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="Wonder")

        assert result["success"] is True
        assert result["data"]["performer"]["name"] == "Alice Wonder"

    def test_missing_required_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when required parameter is missing."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False
        assert "required" in result["error"].lower()
        assert result["data"] is None

    def test_empty_performer_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer_name is empty."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_performer_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer doesn't exist."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="Nonexistent Person")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_excluded_tags_are_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that excluded tags and their children are filtered."""
        tool = QueryPerformerTagsTool(mock_stash)
        tool.set_excluded_tags(["oral"])

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is True
        scene_tags = result["data"]["scene_tags"]
        tag_names = [t["name"] for t in scene_tags]

        # oral and its children (blowjob, deepthroat) should be excluded
        assert "oral" not in tag_names
        assert "blowjob" not in tag_names
        assert "deepthroat" not in tag_names

    def test_weighted_by_views_option(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test weighted_by_views parameter."""
        tool = QueryPerformerTagsTool(mock_stash)

        # With weighting
        result_weighted = tool.execute(performer_name="Jane Doe", weighted_by_views=True)
        # Without weighting
        result_unweighted = tool.execute(performer_name="Jane Doe", weighted_by_views=False)

        assert result_weighted["success"] is True
        assert result_unweighted["success"] is True

        # Both should return tags, but counts may differ
        assert len(result_weighted["data"]["scene_tags"]) > 0
        assert len(result_unweighted["data"]["scene_tags"]) > 0

    def test_schema_generation(self, mock_stash: MagicMock) -> None:
        """Test that tool schema is correctly generated."""
        tool = QueryPerformerTagsTool(mock_stash)
        schema = tool.to_schema()

        assert schema["name"] == "query_performer_tags"
        assert "performer_name" in schema["parameters"]["required"]
        assert "performer_name" in schema["parameters"]["properties"]


class TestQueryTagPerformersTool:
    """Tests for QueryTagPerformersTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful tag performers query."""
        tool = QueryTagPerformersTool(mock_stash)

        result = tool.execute(tag_name="oral")

        assert result["success"] is True
        assert result["data"]["tag"]["name"] == "oral"
        assert len(result["data"]["performers"]) > 0

        # Each performer should have scene_count
        for performer in result["data"]["performers"]:
            assert "id" in performer
            assert "name" in performer
            assert "scene_count" in performer

    def test_tag_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when tag doesn't exist."""
        tool = QueryTagPerformersTool(mock_stash)

        result = tool.execute(tag_name="nonexistent_tag_xyz")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_missing_tag_name(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when tag_name is missing."""
        tool = QueryTagPerformersTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_partial_tag_match(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test partial tag name matching."""
        tool = QueryTagPerformersTool(mock_stash)

        result = tool.execute(tag_name="blow")  # Should match "blowjob"

        assert result["success"] is True
        assert "blowjob" in result["data"]["tag"]["name"].lower()


class TestQueryViewingStatsTool:
    """Tests for QueryViewingStatsTool."""

    def test_top_scenes(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving top scenes by views."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute(stat_type="top_scenes", limit=5)

        assert result["success"] is True
        assert "top_scenes" in result["data"]
        scenes = result["data"]["top_scenes"]
        assert len(scenes) <= 5

        # Verify ordering by view count (descending)
        if len(scenes) > 1:
            for i in range(len(scenes) - 1):
                assert scenes[i]["view_count"] >= scenes[i + 1]["view_count"]

    def test_top_performers(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving top performers by views."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute(stat_type="top_performers", limit=5)

        assert result["success"] is True
        assert "top_performers" in result["data"]
        performers = result["data"]["top_performers"]
        assert len(performers) <= 5

    def test_top_tags(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving top tags by views."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute(stat_type="top_tags", limit=5)

        assert result["success"] is True
        assert "top_tags" in result["data"]

    def test_recent_views(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving recent views."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute(stat_type="recent_views", limit=5)

        assert result["success"] is True

    def test_invalid_stat_type(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error for invalid stat_type."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute(stat_type="invalid_type")

        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    def test_missing_stat_type(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when stat_type is missing."""
        tool = QueryViewingStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False
        assert "required" in result["error"].lower()


class TestQueryTopPerformersTool:
    """Tests for QueryTopPerformersTool."""

    def test_sort_by_view_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting performers by view count."""
        tool = QueryTopPerformersTool(mock_stash)

        result = tool.execute(sort_by="view_count", limit=5)

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) > 0

    def test_sort_by_scene_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting performers by scene count."""
        tool = QueryTopPerformersTool(mock_stash)

        result = tool.execute(sort_by="scene_count", limit=5)

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) > 0

    def test_sort_by_o_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting performers by O count."""
        tool = QueryTopPerformersTool(mock_stash)

        result = tool.execute(sort_by="o_count", limit=5)

        assert result["success"] is True

    def test_default_parameters(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test with default parameters."""
        tool = QueryTopPerformersTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        # Default limit is 10
        assert len(result["data"]["performers"]) <= 10


class TestQueryTopTagsTool:
    """Tests for QueryTopTagsTool."""

    def test_sort_by_view_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting tags by view count."""
        tool = QueryTopTagsTool(mock_stash)

        result = tool.execute(sort_by="view_count", limit=5)

        assert result["success"] is True
        tags = result["data"]["tags"]
        assert len(tags) > 0

    def test_sort_by_scene_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting tags by scene count."""
        tool = QueryTopTagsTool(mock_stash)

        result = tool.execute(sort_by="scene_count", limit=5)

        assert result["success"] is True

    def test_excluded_tags_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that excluded tags are filtered from results."""
        tool = QueryTopTagsTool(mock_stash)
        tool.set_excluded_tags(["excluded_parent"])

        result = tool.execute(limit=20)

        assert result["success"] is True
        tag_names = [t["name"] for t in result["data"]["tags"]]

        # Both excluded_parent and excluded_child should be filtered
        assert "excluded_parent" not in tag_names
        assert "excluded_child" not in tag_names

    def test_per_query_exclusions(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test per-query tag exclusions via exclude_tags parameter."""
        tool = QueryTopTagsTool(mock_stash)

        result = tool.execute(limit=20, exclude_tags="oral,position")

        assert result["success"] is True
        tag_names = [t["name"] for t in result["data"]["tags"]]

        # These and their children should be excluded
        assert "oral" not in tag_names
        assert "blowjob" not in tag_names
        assert "position" not in tag_names


class TestQueryTopStudiosTool:
    """Tests for QueryTopStudiosTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful studios query."""
        tool = QueryTopStudiosTool(mock_stash)

        result = tool.execute(limit=5)

        assert result["success"] is True
        studios = result["data"]["studios"]
        assert len(studios) > 0

    def test_sort_by_view_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test sorting studios by view count."""
        tool = QueryTopStudiosTool(mock_stash)

        result = tool.execute(sort_by="view_count", limit=5)

        assert result["success"] is True


class TestQueryLibraryStatsTool:
    """Tests for QueryLibraryStatsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful library stats query."""
        tool = QueryLibraryStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        data = result["data"]

        # Verify expected stats are present
        assert "scene_count" in data
        assert "performer_count" in data
        assert "tag_count" in data
        assert "studio_count" in data

        # Verify counts match test data
        assert data["scene_count"] == 20
        assert data["performer_count"] == 5
        assert data["tag_count"] == 12
        assert data["studio_count"] == 6

    def test_watched_count(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test watched scene count calculation."""
        tool = QueryLibraryStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        # Watched count should be scenes that have view_dates entries
        assert "watched_scene_count" in result["data"]
        assert result["data"]["watched_scene_count"] > 0

    def test_empty_database(
        self, mock_stash: MagicMock, empty_db: sqlite3.Connection, patched_empty_db: None
    ) -> None:
        """Test with empty database."""
        tool = QueryLibraryStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        assert result["data"]["scene_count"] == 0


class TestQueryPerformersByAttributeTool:
    """Tests for QueryPerformersByAttributeTool."""

    def test_filter_by_gender(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering performers by gender."""
        tool = QueryPerformersByAttributeTool(mock_stash)

        result = tool.execute(gender="female")

        assert result["success"] is True
        performers = result["data"]["performers"]
        # We have 3 female performers in test data
        assert len(performers) == 3
        for p in performers:
            assert "female" in p["gender"].lower()

    def test_filter_by_hair_color(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering performers by hair color."""
        tool = QueryPerformersByAttributeTool(mock_stash)

        result = tool.execute(hair_color="brunette")

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) >= 1
        for p in performers:
            assert "brunette" in p["hair_color"].lower()

    def test_filter_by_ethnicity(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering performers by ethnicity."""
        tool = QueryPerformersByAttributeTool(mock_stash)

        result = tool.execute(ethnicity="Asian")

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) >= 1

    def test_multiple_filters(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test multiple attribute filters."""
        tool = QueryPerformersByAttributeTool(mock_stash)

        result = tool.execute(gender="female", hair_color="brunette")

        assert result["success"] is True
        performers = result["data"]["performers"]
        for p in performers:
            assert "female" in p["gender"].lower()
            assert "brunette" in p["hair_color"].lower()


class TestQueryScenesByDateTool:
    """Tests for QueryScenesByDateTool."""

    def test_filter_by_date_range(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes by date range."""
        tool = QueryScenesByDateTool(mock_stash)

        result = tool.execute(start_date="2023-06-01", end_date="2023-08-31")

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        assert len(scenes) > 0

    def test_filter_by_date_from_only(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes with only start_date."""
        tool = QueryScenesByDateTool(mock_stash)

        result = tool.execute(start_date="2023-10-01")

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        assert len(scenes) >= 0  # May be empty if no scenes match

    def test_filter_by_date_to_only(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes with only end_date."""
        tool = QueryScenesByDateTool(mock_stash)

        result = tool.execute(end_date="2022-12-31")

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        assert len(scenes) >= 0  # May be empty if no scenes match


class TestQueryFavoritesTool:
    """Tests for QueryFavoritesTool."""

    def test_favorite_performers(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving favorite performers."""
        tool = QueryFavoritesTool(mock_stash)

        result = tool.execute(entity_type="performers")

        assert result["success"] is True
        performers = result["data"]["favorite_performers"]
        # We have 2 favorite performers in test data
        assert len(performers) == 2

    def test_favorite_tags(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving favorite tags."""
        tool = QueryFavoritesTool(mock_stash)

        result = tool.execute(entity_type="tags")

        assert result["success"] is True
        tags = result["data"]["favorite_tags"]
        # We have 1 favorite tag in test data (brunette)
        assert len(tags) == 1

    def test_favorite_studios(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving favorite studios."""
        tool = QueryFavoritesTool(mock_stash)

        result = tool.execute(entity_type="studios")

        assert result["success"] is True
        studios = result["data"]["favorite_studios"]
        # We have 1 favorite studio in test data
        assert len(studios) == 1

    def test_all_favorites(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving all favorites (default)."""
        tool = QueryFavoritesTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        assert "total_favorites" in result["data"]


class TestQueryResumePointsTool:
    """Tests for QueryResumePointsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving scenes with resume points."""
        tool = QueryResumePointsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        # We have scenes with resume_time > 0 in test data
        assert len(scenes) > 0
        for scene in scenes:
            assert scene["resume_time_seconds"] > 0

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryResumePointsTool(mock_stash)

        result = tool.execute(limit=2)

        assert result["success"] is True
        assert len(result["data"]["scenes"]) <= 2


class TestQueryScenesByRatingTool:
    """Tests for QueryScenesByRatingTool."""

    def test_filter_by_min_rating(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes by minimum rating (100 = 5 stars)."""
        tool = QueryScenesByRatingTool(mock_stash)

        # min_rating=100 returns 5-star scenes only
        result = tool.execute(min_rating=100)

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        assert len(scenes) > 0
        for scene in scenes:
            assert scene["rating_100"] >= 100

    def test_filter_by_max_rating(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes by maximum rating (60 = 3 stars)."""
        tool = QueryScenesByRatingTool(mock_stash)

        # max_rating=60 returns 3-star and below scenes
        result = tool.execute(min_rating=1, max_rating=60)

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        for scene in scenes:
            assert scene["rating_100"] <= 60

    def test_filter_by_rating_range(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test filtering scenes by rating range (60-80 = 3-4 stars)."""
        tool = QueryScenesByRatingTool(mock_stash)

        result = tool.execute(min_rating=60, max_rating=80)

        assert result["success"] is True
        scenes = result["data"]["scenes"]
        for scene in scenes:
            assert 60 <= scene["rating_100"] <= 80


class TestQueryAllTagsTool:
    """Tests for QueryAllTagsTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving all tags."""
        tool = QueryAllTagsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        tags = result["data"]["tags"]
        assert len(tags) > 0

    def test_search_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test search parameter."""
        tool = QueryAllTagsTool(mock_stash)

        result = tool.execute(search="oral")

        assert result["success"] is True
        tags = result["data"]["tags"]
        # Should find "oral" tag
        tag_names = [t["name"].lower() for t in tags]
        assert any("oral" in name for name in tag_names)

    def test_favorites_only(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test favorites_only parameter."""
        tool = QueryAllTagsTool(mock_stash)

        result = tool.execute(favorites_only=True)

        assert result["success"] is True
        tags = result["data"]["tags"]
        for tag in tags:
            assert tag["favorite"] == 1

    def test_excluded_tags_filtered(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that plugin-level excluded tags are filtered."""
        tool = QueryAllTagsTool(mock_stash)
        tool.set_excluded_tags(["excluded_parent"])

        result = tool.execute()

        assert result["success"] is True
        tag_names = [t["name"] for t in result["data"]["tags"]]
        assert "excluded_parent" not in tag_names
        assert "excluded_child" not in tag_names


class TestQueryAllPerformersTool:
    """Tests for QueryAllPerformersTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test retrieving all performers."""
        tool = QueryAllPerformersTool(mock_stash)

        result = tool.execute()

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) == 5  # We have 5 performers in test data

    def test_search_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test search parameter."""
        tool = QueryAllPerformersTool(mock_stash)

        result = tool.execute(search="Jane")

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) >= 1
        assert any("Jane" in p["name"] for p in performers)

    def test_include_stats(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test include_stats parameter."""
        tool = QueryAllPerformersTool(mock_stash)

        result = tool.execute(include_stats=True)

        assert result["success"] is True
        performers = result["data"]["performers"]
        # When include_stats is True, should have scene_count
        if len(performers) > 0:
            assert "scene_count" in performers[0]

    def test_limit_parameter(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test limit parameter."""
        tool = QueryAllPerformersTool(mock_stash)

        result = tool.execute(limit=2)

        assert result["success"] is True
        assert len(result["data"]["performers"]) <= 2


class TestDatabaseNotFound:
    """Tests for database-not-found error handling."""

    def test_performer_tags_db_not_found(
        self, mock_stash: MagicMock, patched_nonexistent_db: None
    ) -> None:
        """Test QueryPerformerTagsTool when database doesn't exist."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_library_stats_db_not_found(
        self, mock_stash: MagicMock, patched_nonexistent_db: None
    ) -> None:
        """Test QueryLibraryStatsTool when database doesn't exist."""
        tool = QueryLibraryStatsTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestDatabaseError:
    """Tests for database error handling."""

    def test_performer_tags_db_error(self, mock_stash: MagicMock, patched_failing_db: None) -> None:
        """Test QueryPerformerTagsTool when database throws error."""
        tool = QueryPerformerTagsTool(mock_stash)

        result = tool.execute(performer_name="Jane Doe")

        assert result["success"] is False
        assert "database error" in result["error"].lower()
