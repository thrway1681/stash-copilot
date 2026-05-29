"""Shared test fixtures for all test modules."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.mock_stash import create_mock_stash
from tests.fixtures.schema import create_mock_schema, populate_test_data


class NonClosingConnection:
    """
    Wrapper around sqlite3.Connection that makes close() a no-op.

    This is needed because tools close the connection after each use,
    but we want to reuse the same connection across multiple tool calls
    in a single test.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def close(self) -> None:
        """No-op close to keep connection open for reuse."""
        pass

    def cursor(self) -> sqlite3.Cursor:
        """Return a cursor from the underlying connection."""
        return self._conn.cursor()

    def execute(self, sql: str, parameters: Any = None) -> sqlite3.Cursor:
        """Execute SQL on the underlying connection."""
        if parameters is None:
            return self._conn.execute(sql)
        return self._conn.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying connection."""
        return getattr(self._conn, name)


@pytest.fixture
def mock_stash() -> MagicMock:
    """
    Create a mock StashInterface for tool initialization.

    Returns:
        MagicMock: A mock StashInterface
    """
    return create_mock_stash()


def _sqlite_greatest(*args: Any) -> Any:
    """SQLite GREATEST function implementation."""
    non_none = [a for a in args if a is not None]
    return max(non_none) if non_none else None


def _sqlite_least(*args: Any) -> Any:
    """SQLite LEAST function implementation."""
    non_none = [a for a in args if a is not None]
    return min(non_none) if non_none else None


@pytest.fixture
def mock_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Create an in-memory SQLite database with Stash schema and test data.

    The database has the complete Stash schema with deterministic test data
    that can be used for predictable assertions.

    Yields:
        sqlite3.Connection: In-memory database with row_factory set
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Register custom functions that Stash tools may use
    conn.create_function("GREATEST", -1, _sqlite_greatest)
    conn.create_function("LEAST", -1, _sqlite_least)

    # Create schema
    create_mock_schema(conn)

    # Populate with test data
    populate_test_data(conn)

    yield conn

    conn.close()


@pytest.fixture
def patched_db_functions(
    mock_db: sqlite3.Connection, tmp_path: Path
) -> Generator[None, None, None]:
    """
    Patch get_stash_db_path and get_readonly_connection to use mock database.

    This fixture patches the module-level database functions so that tools
    use the in-memory mock database instead of the real Stash database.

    The connection is wrapped to prevent close() from actually closing it,
    allowing multiple tool calls to reuse the same database connection.

    Args:
        mock_db: The mock database connection
        tmp_path: Temporary directory for mock db path

    Yields:
        None: Control returns to test while patches are active
    """
    mock_db_path = tmp_path / "stash-go.sqlite"
    mock_db_path.touch()  # Create empty file for exists() check

    # Wrap the connection to prevent close() from actually closing it
    wrapped_conn = NonClosingConnection(mock_db)

    with (
        patch("stash_ai.tools.database.get_stash_db_path", return_value=mock_db_path),
        patch("stash_ai.tools.database.get_readonly_connection", return_value=wrapped_conn),
    ):
        yield


@pytest.fixture
def empty_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Create an in-memory SQLite database with schema but no data.

    Useful for testing empty database scenarios.

    Yields:
        sqlite3.Connection: Empty database with schema only
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Register custom functions that Stash tools may use
    conn.create_function("GREATEST", -1, _sqlite_greatest)
    conn.create_function("LEAST", -1, _sqlite_least)

    # Create schema only, no data
    create_mock_schema(conn)

    yield conn

    conn.close()


@pytest.fixture
def patched_empty_db(empty_db: sqlite3.Connection, tmp_path: Path) -> Generator[None, None, None]:
    """
    Patch database functions to use an empty mock database.

    Args:
        empty_db: The empty database connection
        tmp_path: Temporary directory for mock db path

    Yields:
        None: Control returns to test while patches are active
    """
    mock_db_path = tmp_path / "stash-go.sqlite"
    mock_db_path.touch()

    # Wrap the connection to prevent close() from actually closing it
    wrapped_conn = NonClosingConnection(empty_db)

    with (
        patch("stash_ai.tools.database.get_stash_db_path", return_value=mock_db_path),
        patch("stash_ai.tools.database.get_readonly_connection", return_value=wrapped_conn),
    ):
        yield
