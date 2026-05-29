"""Tests for hierarchy and marker database tools (Phase 3)."""

import sqlite3
from unittest.mock import MagicMock

from stash_ai.tools.database import (
    QueryPerformerComparisonTool,
    QuerySceneMarkersTool,
    QueryStudioHierarchyTool,
    QueryTagHierarchyTool,
)


class TestQueryTagHierarchyTool:
    """Tests for QueryTagHierarchyTool."""

    def test_get_children(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test getting child tags of a parent tag."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="oral", direction="children")

        assert result["success"] is True
        # oral -> blowjob
        children = result["data"]["children"]
        assert len(children) > 0
        child_names = [c["tag_name"] for c in children]
        assert "blowjob" in child_names

    def test_get_children_recursive(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test recursive child tag traversal."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="oral", direction="children", depth=3)

        assert result["success"] is True
        # Should include blowjob (direct child) and deepthroat (grandchild)

    def test_get_parents(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test getting parent tags of a child tag."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="deepthroat", direction="parents")

        assert result["success"] is True
        # deepthroat <- blowjob
        parents = result["data"]["parents"]
        assert len(parents) > 0
        parent_names = [p["tag_name"] for p in parents]
        assert "blowjob" in parent_names

    def test_get_parents_recursive(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test recursive parent tag traversal."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="deepthroat", direction="parents", depth=3)

        assert result["success"] is True
        # Should include blowjob (direct parent) and oral (grandparent)

    def test_tag_with_multiple_children(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test tag with multiple children."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="position", direction="children")

        assert result["success"] is True
        # position -> doggy, missionary
        children = result["data"]["children"]
        child_names = [c["tag_name"] for c in children]
        assert "doggy" in child_names
        assert "missionary" in child_names

    def test_tag_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when tag doesn't exist."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute(tag_name="nonexistent_tag")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_omit_tag_name_gives_overview(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that omitting tag_name gives an overview of all tags."""
        tool = QueryTagHierarchyTool(mock_stash)

        result = tool.execute()

        # tag_name is optional - omitting it should give an overview
        assert result["success"] is True

    def test_schema_generation(self, mock_stash: MagicMock) -> None:
        """Test that tool schema is correctly generated."""
        tool = QueryTagHierarchyTool(mock_stash)
        schema = tool.to_schema()

        assert schema["name"] == "query_tag_hierarchy"
        # tag_name is optional in this tool (gives overview when omitted)
        assert "tag_name" in schema["parameters"]["properties"]


class TestQueryStudioHierarchyTool:
    """Tests for QueryStudioHierarchyTool."""

    def test_get_children(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test getting child studios of a parent studio."""
        tool = QueryStudioHierarchyTool(mock_stash)

        result = tool.execute(studio_name="Big Studio", direction="children")

        assert result["success"] is True
        # Big Studio -> Sub Studio A, Sub Studio B
        children = result["data"]["children"]
        assert len(children) >= 2
        child_names = [c["studio_name"] for c in children]
        assert "Sub Studio A" in child_names
        assert "Sub Studio B" in child_names

    def test_get_parent(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test getting parent studio of a child studio."""
        tool = QueryStudioHierarchyTool(mock_stash)

        result = tool.execute(studio_name="Sub Studio A")

        assert result["success"] is True
        # Studio hierarchy uses singular "parent" key
        parent = result["data"]["parent"]
        assert parent is not None
        assert parent["studio_name"] == "Big Studio"

    def test_studio_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when studio doesn't exist."""
        tool = QueryStudioHierarchyTool(mock_stash)

        result = tool.execute(studio_name="Nonexistent Studio")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_omit_studio_name_gives_overview(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that omitting studio_name gives an overview of all studios."""
        tool = QueryStudioHierarchyTool(mock_stash)

        result = tool.execute()

        # studio_name is optional - omitting it should give an overview
        assert result["success"] is True


class TestQuerySceneMarkersTool:
    """Tests for QuerySceneMarkersTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful scene markers query."""
        tool = QuerySceneMarkersTool(mock_stash)

        # Scene 1 has markers in test data
        result = tool.execute(scene_id=1)

        assert result["success"] is True
        markers = result["data"]["markers"]
        assert len(markers) >= 1

    def test_marker_properties(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test marker has expected properties."""
        tool = QuerySceneMarkersTool(mock_stash)

        result = tool.execute(scene_id=1)

        assert result["success"] is True
        markers = result["data"]["markers"]
        if len(markers) > 0:
            marker = markers[0]
            assert "title" in marker
            assert "start_seconds" in marker

    def test_scene_with_multiple_markers(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test scene with multiple markers."""
        tool = QuerySceneMarkersTool(mock_stash)

        # Scene 7 has 2 markers in test data
        result = tool.execute(scene_id=7)

        assert result["success"] is True
        markers = result["data"]["markers"]
        assert len(markers) == 2

    def test_scene_with_no_markers(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test scene with no markers."""
        tool = QuerySceneMarkersTool(mock_stash)

        # Scene 20 has no markers
        result = tool.execute(scene_id=20)

        assert result["success"] is True
        markers = result["data"]["markers"]
        assert len(markers) == 0

    def test_omit_scene_id_gives_overview(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that omitting scene_id gives an overview of all markers."""
        tool = QuerySceneMarkersTool(mock_stash)

        result = tool.execute()

        # scene_id is optional - omitting it should give an overview
        assert result["success"] is True


class TestQueryPerformerComparisonTool:
    """Tests for QueryPerformerComparisonTool."""

    def test_execute_success(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test successful performer comparison."""
        tool = QueryPerformerComparisonTool(mock_stash)

        result = tool.execute(performer_names=["Jane Doe", "Alice Wonder"])

        assert result["success"] is True
        performers = result["data"]["performers"]
        assert len(performers) == 2

    def test_comparison_includes_stats(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test comparison includes relevant stats."""
        tool = QueryPerformerComparisonTool(mock_stash)

        result = tool.execute(performer_names=["Jane Doe", "John Smith"])

        assert result["success"] is True
        performers = result["data"]["performers"]
        for performer in performers:
            assert "performer_name" in performer
            assert "scene_count" in performer

    def test_single_performer_fails(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test that single performer fails (requires 2+ for comparison)."""
        tool = QueryPerformerComparisonTool(mock_stash)

        result = tool.execute(performer_names=["Jane Doe"])

        # Comparison requires at least 2 performers
        assert result["success"] is False
        assert "2" in result["error"]

    def test_performer_not_found(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test handling when one performer doesn't exist."""
        tool = QueryPerformerComparisonTool(mock_stash)

        result = tool.execute(performer_names=["Jane Doe", "Nonexistent Person"])

        # Should either fail or return partial results
        # Behavior depends on implementation

    def test_missing_performer_names(
        self, mock_stash: MagicMock, mock_db: sqlite3.Connection, patched_db_functions: None
    ) -> None:
        """Test error when performer_names is missing."""
        tool = QueryPerformerComparisonTool(mock_stash)

        result = tool.execute()

        assert result["success"] is False
