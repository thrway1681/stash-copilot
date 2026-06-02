"""Tests that the FAISS index build streams frames through the store seam.

Issue #3 commit 4 migrated ``FrameSearchIndex.build`` off the store's raw
connection / ``_unpack_embedding`` and onto the public ``count_frame_embeddings``
+ ``iter_frame_embeddings`` operations. These tests verify the index still
builds correctly -- including the multi-batch streaming path -- against a
temporary fixture store, never against internal SQL.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from stash_ai.embeddings.frame_search import FrameSearchIndex
from stash_ai.embeddings.storage import EmbeddingStorage

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


@pytest.fixture
def assets_dir() -> Generator[str, None, None]:
    """A throwaway directory for the index files build() writes."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def _store_frame(store: EmbeddingStorage, scene_id: int, frame_index: int, value: float) -> None:
    """Store one 3-dim frame embedding with a value-derived vector."""
    store.store_frame_embedding(
        scene_id=scene_id,
        frame_index=frame_index,
        timestamp=float(frame_index),
        embedding=[value, value + 0.1, value + 0.2],
    )


def test_build_raises_when_no_frames(storage: EmbeddingStorage, assets_dir: str) -> None:
    index = FrameSearchIndex(assets_dir=assets_dir, model_key=MODEL_KEY)
    with pytest.raises(ValueError, match="No frame embeddings"):
        index.build(storage)


def test_build_indexes_all_frames_across_batches(
    storage: EmbeddingStorage, assets_dir: str
) -> None:
    # Three scenes, several frames each, inserted out of order.
    frames = [(2, 0), (1, 1), (1, 0), (3, 0), (3, 1), (3, 2)]
    for scene_id, frame_index in frames:
        _store_frame(storage, scene_id, frame_index, float(scene_id) + frame_index / 10.0)

    index = FrameSearchIndex(assets_dir=assets_dir, model_key=MODEL_KEY)
    # batch_size=2 forces the streaming loop across multiple batches.
    info = index.build(storage, batch_size=2)

    assert info.frame_count == len(frames)
    assert info.scene_count == 3
    assert info.dimensions == 3
    assert info.model_key == MODEL_KEY

    # The index and its metadata are persisted and reloadable.
    assert index.exists
    assert Path(index.index_path).exists()
    assert Path(index.meta_path).exists()
    assert Path(index.info_path).exists()


def test_build_metadata_is_row_aligned(storage: EmbeddingStorage, assets_dir: str) -> None:
    # Frames spanning two scenes; build must keep scene/frame/timestamp aligned.
    _store_frame(storage, 5, 0, 5.0)
    _store_frame(storage, 5, 1, 5.1)
    _store_frame(storage, 7, 0, 7.0)

    index = FrameSearchIndex(assets_dir=assets_dir, model_key=MODEL_KEY)
    index.build(storage, batch_size=2)

    # Reload from disk to confirm what was persisted, not in-memory state.
    reloaded = FrameSearchIndex(assets_dir=assets_dir, model_key=MODEL_KEY)
    reloaded.load()

    assert reloaded._scene_ids is not None
    assert reloaded._frame_indices is not None
    # Frames are ordered by (scene_id, frame_index).
    assert reloaded._scene_ids.tolist() == [5, 5, 7]
    assert reloaded._frame_indices.tolist() == [0, 1, 0]
