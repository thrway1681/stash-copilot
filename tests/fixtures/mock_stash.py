"""Mock StashInterface for testing."""

from unittest.mock import MagicMock


def create_mock_stash() -> MagicMock:
    """
    Create a mock StashInterface object.

    The database tools don't actually use the StashInterface for queries
    (they use direct SQLite access), but they require it for initialization.

    Returns:
        MagicMock: A mock StashInterface with common methods stubbed
    """
    mock = MagicMock()

    # Stub common methods that might be called
    mock.graphql = MagicMock(return_value=None)
    mock.log = MagicMock()
    mock.find_scenes = MagicMock(return_value=[])
    mock.find_performers = MagicMock(return_value=[])
    mock.find_tags = MagicMock(return_value=[])
    mock.find_studios = MagicMock(return_value=[])

    return mock
