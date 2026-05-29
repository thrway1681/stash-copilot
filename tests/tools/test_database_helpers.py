"""Tests for module-level database helper functions."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetStashDbPath:
    """Tests for get_stash_db_path function."""

    def test_finds_database_in_home_stash(self, tmp_path: Path) -> None:
        """Test finding database in ~/.stash/ directory."""
        from stash_ai.tools.database import get_stash_db_path

        # Create mock .stash directory with database
        stash_dir = tmp_path / ".stash"
        stash_dir.mkdir()
        db_file = stash_dir / "stash-go.sqlite"
        db_file.touch()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = get_stash_db_path()
            assert result == db_file

    def test_returns_default_when_not_found(self, tmp_path: Path) -> None:
        """Test fallback to default path when database not found."""
        from stash_ai.tools.database import get_stash_db_path

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = get_stash_db_path()
            # Should return default path even if file doesn't exist
            assert result == tmp_path / ".stash" / "stash-go.sqlite"

    def test_handles_permission_error(self, tmp_path: Path) -> None:
        """Test graceful handling of permission errors."""
        from stash_ai.tools.database import get_stash_db_path

        # Create a mock that raises PermissionError
        mock_path = MagicMock()
        mock_path.exists.side_effect = PermissionError("No access")

        with patch("pathlib.Path.home", return_value=tmp_path):
            # Should not raise, should return default
            result = get_stash_db_path()
            assert isinstance(result, Path)


class TestGetReadonlyConnection:
    """Tests for get_readonly_connection function."""

    def test_connection_is_readonly(self, tmp_path: Path) -> None:
        """Test that connection is truly read-only."""
        from stash_ai.tools.database import get_readonly_connection

        # Create test database with some data
        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO test (id) VALUES (1)")
        conn.commit()
        conn.close()

        # Get read-only connection
        ro_conn = get_readonly_connection(db_path)

        # Verify reads work
        cursor = ro_conn.cursor()
        cursor.execute("SELECT * FROM test")
        rows = cursor.fetchall()
        assert len(rows) == 1

        # Verify writes fail
        with pytest.raises(sqlite3.OperationalError):
            cursor.execute("INSERT INTO test (id) VALUES (2)")

        ro_conn.close()

    def test_row_factory_is_set(self, tmp_path: Path) -> None:
        """Test that row_factory is set for dict-like access."""
        from stash_ai.tools.database import get_readonly_connection

        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'test')")
        conn.commit()
        conn.close()

        ro_conn = get_readonly_connection(db_path)
        cursor = ro_conn.cursor()
        cursor.execute("SELECT * FROM test")
        row = cursor.fetchone()

        # Should be accessible by column name
        assert row["id"] == 1
        assert row["name"] == "test"

        ro_conn.close()


class TestGetExcludedTagIdsWithChildren:
    """Tests for get_excluded_tag_ids_with_children function."""

    def test_returns_empty_for_empty_input(self, mock_db: sqlite3.Connection) -> None:
        """Test empty input returns empty set."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        result = get_excluded_tag_ids_with_children(cursor, [])

        assert result == set()

    def test_returns_empty_for_nonexistent_tags(self, mock_db: sqlite3.Connection) -> None:
        """Test that nonexistent tag names return empty set."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        result = get_excluded_tag_ids_with_children(cursor, ["nonexistent_tag", "also_nonexistent"])

        assert result == set()

    def test_returns_direct_tag_id(self, mock_db: sqlite3.Connection) -> None:
        """Test that direct tag IDs are returned."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        # "brunette" has no children
        result = get_excluded_tag_ids_with_children(cursor, ["brunette"])

        assert 8 in result  # brunette is id 8

    def test_returns_direct_and_children(self, mock_db: sqlite3.Connection) -> None:
        """Test that both direct tags and children are returned."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        # oral (1) -> blowjob (2) -> deepthroat (3)
        result = get_excluded_tag_ids_with_children(cursor, ["oral"])

        assert 1 in result  # oral (direct)
        assert 2 in result  # blowjob (child)
        assert 3 in result  # deepthroat (grandchild)

    def test_returns_grandchildren(self, mock_db: sqlite3.Connection) -> None:
        """Test that grandchildren are included via recursive CTE."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        # blowjob (2) -> deepthroat (3)
        result = get_excluded_tag_ids_with_children(cursor, ["blowjob"])

        assert 2 in result  # blowjob (direct)
        assert 3 in result  # deepthroat (child)
        assert 1 not in result  # oral should NOT be included (it's parent)

    def test_case_insensitive_matching(self, mock_db: sqlite3.Connection) -> None:
        """Test case-insensitive tag name matching."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()

        # Test various cases
        result_lower = get_excluded_tag_ids_with_children(cursor, ["oral"])
        result_upper = get_excluded_tag_ids_with_children(cursor, ["ORAL"])
        result_mixed = get_excluded_tag_ids_with_children(cursor, ["Oral"])

        # All should find the same tag
        assert 1 in result_lower
        assert 1 in result_upper
        assert 1 in result_mixed

    def test_multiple_excluded_tags(self, mock_db: sqlite3.Connection) -> None:
        """Test excluding multiple tags at once."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        result = get_excluded_tag_ids_with_children(cursor, ["oral", "position"])

        # oral hierarchy: 1 -> 2 -> 3
        assert 1 in result
        assert 2 in result
        assert 3 in result

        # position hierarchy: 5 -> 6, 7
        assert 5 in result
        assert 6 in result
        assert 7 in result

    def test_excluded_parent_with_child(self, mock_db: sqlite3.Connection) -> None:
        """Test the excluded_parent -> excluded_child hierarchy."""
        from stash_ai.tools.database import get_excluded_tag_ids_with_children

        cursor = mock_db.cursor()
        result = get_excluded_tag_ids_with_children(cursor, ["excluded_parent"])

        assert 10 in result  # excluded_parent
        assert 11 in result  # excluded_child
