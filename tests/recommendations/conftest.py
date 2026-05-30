"""Fixtures for recommendation module tests."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import NonClosingConnection
from tests.fixtures.schema import create_mock_schema, populate_test_data


@pytest.fixture
def engagement_db() -> Generator[sqlite3.Connection, None, None]:
    """In-memory Stash DB with test data for engagement tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_mock_schema(conn)
    populate_test_data(conn)
    yield conn
    conn.close()


@pytest.fixture
def patched_engagement_db(
    engagement_db: sqlite3.Connection, tmp_path: Path
) -> Generator[None, None, None]:
    """Patch get_stash_db_path and get_readonly_connection in the engagement module."""
    mock_db_path = tmp_path / "stash-go.sqlite"
    mock_db_path.touch()

    wrapped = NonClosingConnection(engagement_db)

    with (
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
