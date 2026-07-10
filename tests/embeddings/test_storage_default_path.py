"""Tests for the default location of the embeddings store."""

from pathlib import Path

from stash_ai import paths
from stash_ai.embeddings.storage import EmbeddingStorage


def test_default_db_path_resolves_under_data_dir() -> None:
    """The default store lives in the update-safe data directory."""
    storage = EmbeddingStorage()

    assert Path(storage.db_path) == paths.embeddings_db_path()
