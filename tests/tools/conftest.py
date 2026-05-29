"""Tool-specific test fixtures."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def patched_nonexistent_db(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Patch get_stash_db_path to return a path that doesn't exist.

    Useful for testing database-not-found error handling.

    Args:
        tmp_path: Temporary directory

    Yields:
        Path: The nonexistent path for assertions
    """
    nonexistent_path = tmp_path / "nonexistent.sqlite"

    with patch("stash_ai.tools.database.get_stash_db_path", return_value=nonexistent_path):
        yield nonexistent_path


@pytest.fixture
def patched_failing_db(tmp_path: Path) -> Generator[None, None, None]:
    """
    Patch database functions to simulate a database error.

    Useful for testing SQLite error handling.

    Args:
        tmp_path: Temporary directory

    Yields:
        None: Control returns to test while patches are active
    """
    from unittest.mock import MagicMock

    mock_db_path = tmp_path / "test.sqlite"
    mock_db_path.touch()

    # Create a mock connection that raises an error on execute
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = sqlite3.Error("Test database error")
    mock_conn.cursor.return_value = mock_cursor

    with (
        patch("stash_ai.tools.database.get_stash_db_path", return_value=mock_db_path),
        patch("stash_ai.tools.database.get_readonly_connection", return_value=mock_conn),
    ):
        yield
