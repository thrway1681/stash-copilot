"""Tests for the tag-gap store operations on EmbeddingStorage.

Exercises the high-level reads/writes the tag-gap task needs (issue #3
commit 6) through the public interface against a temporary fixture store --
never against internal SQL:

* ``get_tag_gap_threshold`` / ``set_tag_gap_threshold`` (typed threshold cache)
* ``get_frame_tag_best_similarities`` (slow-path percentile input)
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from stash_ai.embeddings.storage import EmbeddingStorage, FrameTagCoverageRecord

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


def _store_coverage(
    store: EmbeddingStorage, scene_id: int, frame_index: int, similarity: float
) -> None:
    """Persist one frame-tag coverage row with the given best similarity."""
    store.save_frame_tag_coverage_batch(
        [
            FrameTagCoverageRecord(
                scene_id=scene_id,
                frame_index=frame_index,
                model_key=store.model_key,
                best_tag="tag",
                best_similarity=similarity,
                is_covered=False,
            )
        ]
    )


class TestTagGapThresholdCache:
    def test_missing_threshold_returns_none(self, storage: EmbeddingStorage) -> None:
        assert storage.get_tag_gap_threshold(MODEL_KEY) is None

    def test_set_then_get_roundtrips(self, storage: EmbeddingStorage) -> None:
        storage.set_tag_gap_threshold(MODEL_KEY, 0.275)

        assert storage.get_tag_gap_threshold(MODEL_KEY) == pytest.approx(0.275)

    def test_set_overwrites_previous(self, storage: EmbeddingStorage) -> None:
        storage.set_tag_gap_threshold(MODEL_KEY, 0.1)
        storage.set_tag_gap_threshold(MODEL_KEY, 0.9)

        assert storage.get_tag_gap_threshold(MODEL_KEY) == pytest.approx(0.9)

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")

        model_a.set_tag_gap_threshold("model_a", 0.2)
        model_b.set_tag_gap_threshold("model_b", 0.8)

        assert model_a.get_tag_gap_threshold("model_a") == pytest.approx(0.2)
        assert model_a.get_tag_gap_threshold("model_b") == pytest.approx(0.8)
        assert model_a.get_tag_gap_threshold("missing") is None


class TestGetFrameTagBestSimilarities:
    def test_empty_store_returns_empty_list(self, storage: EmbeddingStorage) -> None:
        assert storage.get_frame_tag_best_similarities(MODEL_KEY) == []

    def test_returns_every_best_similarity(self, storage: EmbeddingStorage) -> None:
        _store_coverage(storage, 1, 0, 0.10)
        _store_coverage(storage, 1, 1, 0.20)
        _store_coverage(storage, 2, 0, 0.30)

        sims = storage.get_frame_tag_best_similarities(MODEL_KEY)

        assert sorted(sims) == pytest.approx([0.10, 0.20, 0.30])

    def test_scoped_by_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        _store_coverage(model_a, 1, 0, 0.5)
        _store_coverage(model_b, 2, 0, 0.6)
        _store_coverage(model_b, 2, 1, 0.7)

        assert model_a.get_frame_tag_best_similarities("model_a") == pytest.approx([0.5])
        assert sorted(model_a.get_frame_tag_best_similarities("model_b")) == pytest.approx(
            [0.6, 0.7]
        )
