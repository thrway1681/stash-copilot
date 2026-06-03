"""Tests for EmbeddingStorage.get_embeddings (bulk fetch)."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from stash_ai.embeddings.storage import EmbeddingStorage


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary embedding-store database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db: str) -> EmbeddingStorage:
    """Create storage instance with a temp database."""
    return EmbeddingStorage(db_path=temp_db, model_key="test_model")


def _store(storage: EmbeddingStorage, scene_id: int) -> None:
    """Store a simple scene embedding with a distinct visual vector."""
    storage.store_embedding(
        scene_id=scene_id,
        composite_embedding=[float(scene_id), 0.1, 0.2],
        text_model="text-test",
        visual_embedding=[float(scene_id) + 0.5, 0.3, 0.4],
        visual_model="visual-test",
    )


class TestGetEmbeddingsBulk:
    def test_empty_input_returns_empty_dict(self, storage: EmbeddingStorage) -> None:
        assert storage.get_embeddings([]) == {}

    def test_returns_records_keyed_by_scene_id(self, storage: EmbeddingStorage) -> None:
        for sid in (1, 2, 3):
            _store(storage, sid)

        result = storage.get_embeddings([1, 2, 3])

        assert set(result.keys()) == {1, 2, 3}
        assert result[1]["scene_id"] == 1
        # Stored as float32, so compare with tolerance.
        assert result[1]["composite_embedding"] == pytest.approx([1.0, 0.1, 0.2])
        assert result[2]["visual_embedding"] == pytest.approx([2.5, 0.3, 0.4])

    def test_matches_per_scene_get_embedding(self, storage: EmbeddingStorage) -> None:
        for sid in (10, 20):
            _store(storage, sid)

        bulk = storage.get_embeddings([10, 20])

        assert bulk[10] == storage.get_embedding(10)
        assert bulk[20] == storage.get_embedding(20)

    def test_missing_ids_are_absent(self, storage: EmbeddingStorage) -> None:
        _store(storage, 1)

        result = storage.get_embeddings([1, 999])

        assert set(result.keys()) == {1}
        assert 999 not in result

    def test_respects_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store(model_a, 1)
        _store(model_b, 2)

        result_a = model_a.get_embeddings([1, 2])

        # Only the scene stored under model_a's key is visible to model_a.
        assert set(result_a.keys()) == {1}

    def test_handles_chunk_boundary(self, storage: EmbeddingStorage) -> None:
        # More than one IN-clause chunk (chunk_size=900) to exercise batching.
        ids = list(range(1, 1001))
        for sid in ids:
            _store(storage, sid)

        result = storage.get_embeddings(ids)

        assert len(result) == 1000
        assert result[1]["scene_id"] == 1
        assert result[1000]["scene_id"] == 1000
