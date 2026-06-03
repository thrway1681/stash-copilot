"""Tests for the frame-embedding operations on EmbeddingStorage.

Exercises the new high-level frame reads added in issue #3 commit 3
(``count_frame_embeddings``, ``get_scene_ids_with_frame_embeddings``,
``iter_frame_embeddings``, ``sample_frame_embeddings``) through the public
interface against a temporary fixture store -- never against internal SQL.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from stash_ai.embeddings.storage import (
    EmbeddingStorage,
    FrameEmbeddingBatch,
    FrameEmbeddingSample,
)

MODEL_KEY = "test_model"


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary embedding-store database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db: str) -> EmbeddingStorage:
    """Create a storage instance keyed to MODEL_KEY."""
    return EmbeddingStorage(db_path=temp_db, model_key=MODEL_KEY)


def _store_frame(store: EmbeddingStorage, scene_id: int, frame_index: int, value: float) -> None:
    """Store one 3-dim frame embedding with a value-derived vector."""
    store.store_frame_embedding(
        scene_id=scene_id,
        frame_index=frame_index,
        timestamp=float(frame_index),
        embedding=[value, value + 0.1, value + 0.2],
    )


class TestCountFrameEmbeddings:
    def test_empty_store_returns_zero(self, storage: EmbeddingStorage) -> None:
        assert storage.count_frame_embeddings(MODEL_KEY) == 0

    def test_counts_all_frames(self, storage: EmbeddingStorage) -> None:
        _store_frame(storage, 1, 0, 0.1)
        _store_frame(storage, 1, 1, 0.2)
        _store_frame(storage, 2, 0, 0.3)

        assert storage.count_frame_embeddings(MODEL_KEY) == 3

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store_frame(model_a, 1, 0, 0.1)
        _store_frame(model_b, 2, 0, 0.2)
        _store_frame(model_b, 2, 1, 0.3)

        assert model_a.count_frame_embeddings("model_a") == 1
        assert model_a.count_frame_embeddings("model_b") == 2
        assert model_a.count_frame_embeddings("missing") == 0


class TestGetSceneIdsWithFrameEmbeddings:
    def test_empty_store_returns_empty_list(self, storage: EmbeddingStorage) -> None:
        assert storage.get_scene_ids_with_frame_embeddings(MODEL_KEY) == []

    def test_returns_distinct_sorted_scene_ids(self, storage: EmbeddingStorage) -> None:
        # Out-of-order, with duplicates across frames of the same scene.
        _store_frame(storage, 3, 0, 0.1)
        _store_frame(storage, 1, 0, 0.2)
        _store_frame(storage, 1, 1, 0.3)
        _store_frame(storage, 2, 0, 0.4)

        assert storage.get_scene_ids_with_frame_embeddings(MODEL_KEY) == [1, 2, 3]

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store_frame(model_a, 10, 0, 0.1)
        _store_frame(model_b, 20, 0, 0.2)

        assert model_a.get_scene_ids_with_frame_embeddings("model_a") == [10]
        assert model_a.get_scene_ids_with_frame_embeddings("model_b") == [20]


class TestIterFrameEmbeddings:
    def test_empty_store_yields_nothing(self, storage: EmbeddingStorage) -> None:
        assert list(storage.iter_frame_embeddings(MODEL_KEY)) == []

    def test_rejects_non_positive_batch_size(self, storage: EmbeddingStorage) -> None:
        with pytest.raises(ValueError):
            list(storage.iter_frame_embeddings(MODEL_KEY, batch_size=0))

    def test_yields_all_frames_ordered_and_row_aligned(self, storage: EmbeddingStorage) -> None:
        # Insert out of order; iteration must come back (scene_id, frame_index).
        _store_frame(storage, 2, 0, 2.0)
        _store_frame(storage, 1, 1, 1.1)
        _store_frame(storage, 1, 0, 1.0)

        batches = list(storage.iter_frame_embeddings(MODEL_KEY, batch_size=10))

        assert len(batches) == 1
        batch = batches[0]
        assert isinstance(batch, FrameEmbeddingBatch)
        assert len(batch) == 3
        assert batch.scene_ids.tolist() == [1, 1, 2]
        assert batch.frame_indices.tolist() == [0, 1, 0]
        assert batch.timestamps.tolist() == pytest.approx([0.0, 1.0, 0.0])
        # Row 0 is scene 1 / frame 0, stored with value 1.0.
        assert batch.embeddings[0].tolist() == pytest.approx([1.0, 1.1, 1.2])
        assert batch.embeddings.dtype == np.float32

    def test_batches_respect_batch_size(self, storage: EmbeddingStorage) -> None:
        for frame_index in range(5):
            _store_frame(storage, 1, frame_index, float(frame_index))

        batches = list(storage.iter_frame_embeddings(MODEL_KEY, batch_size=2))

        assert [len(b) for b in batches] == [2, 2, 1]
        # Concatenating all batches recovers every frame in order.
        all_indices = [int(fi) for b in batches for fi in b.frame_indices.tolist()]
        assert all_indices == [0, 1, 2, 3, 4]

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store_frame(model_a, 1, 0, 0.1)
        _store_frame(model_b, 2, 0, 0.2)

        batches = list(model_a.iter_frame_embeddings("model_a"))

        assert sum(len(b) for b in batches) == 1
        assert batches[0].scene_ids.tolist() == [1]


class TestSampleFrameEmbeddings:
    def test_empty_store_returns_empty_sample(self, storage: EmbeddingStorage) -> None:
        sample = storage.sample_frame_embeddings(MODEL_KEY, n=10)
        assert isinstance(sample, FrameEmbeddingSample)
        assert len(sample) == 0
        assert sample.keys == []
        assert sample.timestamps == {}

    def test_rejects_non_positive_n(self, storage: EmbeddingStorage) -> None:
        with pytest.raises(ValueError):
            storage.sample_frame_embeddings(MODEL_KEY, n=0)

    def test_returns_all_when_under_cap(self, storage: EmbeddingStorage) -> None:
        # value -> [value, value+0.1, value+0.2] per _store_frame.
        values = {(1, 0): 1.0, (1, 1): 1.1, (2, 0): 2.0}
        for (scene_id, frame_index), value in values.items():
            _store_frame(storage, scene_id, frame_index, value)

        sample = storage.sample_frame_embeddings(MODEL_KEY, n=100)

        assert len(sample) == 3
        assert set(sample.keys) == set(values)
        assert sample.embeddings.shape == (3, 3)
        assert sample.embeddings.dtype == np.float32
        # Keys, timestamps and embedding rows stay aligned.
        for row, key in enumerate(sample.keys):
            assert sample.timestamps[key] == pytest.approx(float(key[1]))
            value = values[key]
            assert sample.embeddings[row].tolist() == pytest.approx(
                [value, value + 0.1, value + 0.2]
            )

    def test_caps_sample_size(self, storage: EmbeddingStorage) -> None:
        for frame_index in range(50):
            _store_frame(storage, 1, frame_index, float(frame_index))

        sample = storage.sample_frame_embeddings(MODEL_KEY, n=10)

        # Two-phase sampling selects exactly n rowids when total > n.
        assert len(sample) == 10
        assert sample.embeddings.shape == (10, 3)
        # Every returned key is a real, distinct stored frame.
        assert len(set(sample.keys)) == 10
        for scene_id, frame_index in sample.keys:
            assert scene_id == 1
            assert 0 <= frame_index < 50

    def test_excludes_given_keys(self, storage: EmbeddingStorage) -> None:
        _store_frame(storage, 1, 0, 1.0)
        _store_frame(storage, 1, 1, 1.1)
        _store_frame(storage, 2, 0, 2.0)

        sample = storage.sample_frame_embeddings(MODEL_KEY, n=100, exclude_keys={(1, 0), (2, 0)})

        assert set(sample.keys) == {(1, 1)}
        assert (1, 0) not in sample.timestamps
        assert sample.embeddings.shape == (1, 3)

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store_frame(model_a, 1, 0, 0.1)
        _store_frame(model_b, 2, 0, 0.2)

        sample = model_a.sample_frame_embeddings("model_a", n=100)

        assert set(sample.keys) == {(1, 0)}
