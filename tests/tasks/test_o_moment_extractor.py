"""Tests for O-moment extraction functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.recommendations.types import (
    OMomentData,
    OMomentExtractionConfig,
    OMomentMarker,
)
from stash_ai.tasks.frame_extractor import FrameExtractionConfig, FrameExtractor
from stash_ai.tasks.o_moment_extractor import OMomentExtractor


class TestOMomentEmbeddingStorage:
    """Tests for O-moment embedding storage."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def storage(self, temp_db):
        """Create storage instance with temp database."""
        return EmbeddingStorage(db_path=temp_db, model_key="test_model")

    def test_store_and_retrieve_o_moment_embedding(self, storage):
        """Test storing and retrieving O-moment embeddings."""
        scene_id = 123
        o_event_index = 0
        marker_id = 456
        center_timestamp = 120.5
        window_seconds = 120.0
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        frame_count = 12

        # Store embedding
        storage.store_o_moment_embedding(
            scene_id=scene_id,
            o_event_index=o_event_index,
            marker_id=marker_id,
            center_timestamp=center_timestamp,
            window_seconds=window_seconds,
            embedding=embedding,
            frame_count=frame_count,
        )

        # Retrieve embedding
        result = storage.get_o_moment_embedding(scene_id, o_event_index)

        assert result is not None
        assert result["scene_id"] == scene_id
        assert result["o_event_index"] == o_event_index
        assert result["marker_id"] == marker_id
        assert result["center_timestamp"] == center_timestamp
        assert result["window_seconds"] == window_seconds
        assert result["frame_count"] == frame_count
        assert result["model_key"] == "test_model"
        assert len(result["embedding"]) == len(embedding)
        np.testing.assert_array_almost_equal(result["embedding"], embedding)

    def test_has_o_moment_embedding(self, storage):
        """Test checking if O-moment embedding exists."""
        scene_id = 123
        o_event_index = 0

        # Should not exist initially
        assert not storage.has_o_moment_embedding(scene_id, o_event_index)

        # Store embedding
        storage.store_o_moment_embedding(
            scene_id=scene_id,
            o_event_index=o_event_index,
            marker_id=456,
            center_timestamp=120.0,
            window_seconds=120.0,
            embedding=[0.1, 0.2, 0.3],
            frame_count=12,
        )

        # Should exist now
        assert storage.has_o_moment_embedding(scene_id, o_event_index)

        # Different index should not exist
        assert not storage.has_o_moment_embedding(scene_id, 1)

    def test_get_all_o_moment_embeddings_for_scene(self, storage):
        """Test getting all O-moment embeddings for a scene."""
        scene_id = 123

        # Store multiple O-moments for the same scene
        for i in range(3):
            storage.store_o_moment_embedding(
                scene_id=scene_id,
                o_event_index=i,
                marker_id=456 + i,
                center_timestamp=60.0 * (i + 1),
                window_seconds=120.0,
                embedding=[0.1 * (i + 1)] * 5,
                frame_count=12,
            )

        # Retrieve all
        results = storage.get_all_o_moment_embeddings_for_scene(scene_id)

        assert len(results) == 3
        assert results[0]["o_event_index"] == 0
        assert results[1]["o_event_index"] == 1
        assert results[2]["o_event_index"] == 2

    def test_get_scenes_with_o_moments(self, storage):
        """Test getting list of scenes with O-moment embeddings."""
        # Store O-moments for different scenes
        for scene_id in [1, 2, 3]:
            storage.store_o_moment_embedding(
                scene_id=scene_id,
                o_event_index=0,
                marker_id=scene_id * 10,
                center_timestamp=60.0,
                window_seconds=120.0,
                embedding=[0.1] * 5,
                frame_count=12,
            )

        scene_ids = storage.get_scenes_with_o_moments()

        assert len(scene_ids) == 3
        assert set(scene_ids) == {1, 2, 3}

    def test_delete_o_moment_embedding(self, storage):
        """Test deleting O-moment embedding."""
        scene_id = 123
        o_event_index = 0

        # Store embedding
        storage.store_o_moment_embedding(
            scene_id=scene_id,
            o_event_index=o_event_index,
            marker_id=456,
            center_timestamp=120.0,
            window_seconds=120.0,
            embedding=[0.1, 0.2, 0.3],
            frame_count=12,
        )

        assert storage.has_o_moment_embedding(scene_id, o_event_index)

        # Delete embedding
        deleted = storage.delete_o_moment_embedding(scene_id, o_event_index)

        assert deleted
        assert not storage.has_o_moment_embedding(scene_id, o_event_index)

    def test_clear_all_o_moments(self, storage):
        """Test clearing all O-moment embeddings."""
        # Store multiple O-moments
        for scene_id in [1, 2, 3]:
            storage.store_o_moment_embedding(
                scene_id=scene_id,
                o_event_index=0,
                marker_id=scene_id * 10,
                center_timestamp=60.0,
                window_seconds=120.0,
                embedding=[0.1] * 5,
                frame_count=12,
            )

        # Clear all
        deleted = storage.clear_all_o_moments()

        assert deleted == 3
        assert len(storage.get_scenes_with_o_moments()) == 0

    def test_get_o_moment_stats(self, storage):
        """Test getting O-moment statistics."""
        # Store O-moments for different scenes
        for scene_id in [1, 2]:
            for i in range(2):
                storage.store_o_moment_embedding(
                    scene_id=scene_id,
                    o_event_index=i,
                    marker_id=scene_id * 10 + i,
                    center_timestamp=60.0 * (i + 1),
                    window_seconds=120.0,
                    embedding=[0.1] * 5,
                    frame_count=12,
                )

        stats = storage.get_o_moment_stats()

        assert stats["model_key"] == "test_model"
        assert stats["total_o_moments"] == 4
        assert stats["scenes_with_o_moments"] == 2

    def test_find_similar_o_moments(self, storage):
        """Test finding similar O-moment embeddings."""
        # Store O-moments with different embeddings
        embeddings = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Scene 1
            [0.9, 0.1, 0.0, 0.0, 0.0],  # Scene 2 - similar to scene 1
            [0.0, 0.0, 0.0, 0.0, 1.0],  # Scene 3 - different
        ]

        # Normalize embeddings
        for i, emb in enumerate(embeddings, 1):
            arr = np.array(emb, dtype=np.float32)
            arr = arr / np.linalg.norm(arr)
            storage.store_o_moment_embedding(
                scene_id=i,
                o_event_index=0,
                marker_id=i * 10,
                center_timestamp=60.0,
                window_seconds=120.0,
                embedding=arr.tolist(),
                frame_count=12,
            )

        # Query with embedding similar to scene 1
        query = [1.0, 0.0, 0.0, 0.0, 0.0]
        results = storage.find_similar_o_moments(
            query_embedding=query,
            limit=10,
            min_similarity=0.5,
        )

        # Should find scenes 1 and 2 (similar), but not scene 3
        assert len(results) == 2
        scene_ids = [r.scene_id for r in results]
        assert 1 in scene_ids
        assert 2 in scene_ids
        assert 3 not in scene_ids

        # Scene 1 should be most similar
        assert results[0].scene_id == 1
        assert results[0].similarity > results[1].similarity

    def test_model_key_isolation(self, temp_db):
        """Test that different model keys are isolated."""
        storage1 = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        storage2 = EmbeddingStorage(db_path=temp_db, model_key="model_b")

        # Store with model_a
        storage1.store_o_moment_embedding(
            scene_id=1,
            o_event_index=0,
            marker_id=10,
            center_timestamp=60.0,
            window_seconds=120.0,
            embedding=[0.1] * 5,
            frame_count=12,
        )

        # model_a should see it
        assert storage1.has_o_moment_embedding(1, 0)

        # model_b should NOT see it
        assert not storage2.has_o_moment_embedding(1, 0)

        # Store with model_b
        storage2.store_o_moment_embedding(
            scene_id=1,
            o_event_index=0,
            marker_id=10,
            center_timestamp=60.0,
            window_seconds=120.0,
            embedding=[0.2] * 5,  # Different embedding
            frame_count=12,
        )

        # Both should see their own
        result1 = storage1.get_o_moment_embedding(1, 0)
        result2 = storage2.get_o_moment_embedding(1, 0)

        assert result1 is not None
        assert result2 is not None
        assert result1["embedding"] != result2["embedding"]


class TestOMomentExtractionConfig:
    """Tests for O-moment extraction configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OMomentExtractionConfig()

        assert config.window_seconds == 120.0
        assert config.frames_per_window == 12
        assert config.o_tag_name == "O"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OMomentExtractionConfig(
            window_seconds=60.0,
            frames_per_window=6,
            o_tag_name="CustomTag",
        )

        assert config.window_seconds == 60.0
        assert config.frames_per_window == 6
        assert config.o_tag_name == "CustomTag"


class TestOMomentMarkerTypes:
    """Tests for O-moment type definitions."""

    def test_o_moment_marker(self):
        """Test OMomentMarker type."""
        marker: OMomentMarker = {
            "marker_id": 123,
            "scene_id": 456,
            "seconds": 120.5,
            "end_seconds": None,
            "created_at": "2025-01-06T12:00:00",
        }

        assert marker["marker_id"] == 123
        assert marker["scene_id"] == 456
        assert marker["seconds"] == 120.5
        assert marker["end_seconds"] is None

    def test_o_moment_data(self):
        """Test OMomentData type."""
        marker: OMomentMarker = {
            "marker_id": 123,
            "scene_id": 456,
            "seconds": 120.5,
            "end_seconds": None,
            "created_at": "2025-01-06T12:00:00",
        }

        data: OMomentData = {
            "scene_id": 456,
            "marker": marker,
            "o_event_index": 0,
        }

        assert data["scene_id"] == 456
        assert data["marker"]["marker_id"] == 123
        assert data["o_event_index"] == 0


class TestOMomentExtractor:
    """Tests for O-moment extractor logic."""

    @pytest.fixture
    def mock_frame_extractor(self):
        """Create a mock frame extractor."""
        mock = MagicMock(spec=FrameExtractor)
        mock.config = FrameExtractionConfig(frame_width=640)
        return mock

    @pytest.fixture
    def extractor(self, mock_frame_extractor):
        """Create an extractor with default config."""
        return OMomentExtractor(
            frame_extractor=mock_frame_extractor,
            config=OMomentExtractionConfig(),
        )

    def test_extract_o_moment_frames_zero_frames_per_window(self, mock_frame_extractor):
        """Test that frames_per_window <= 0 doesn't cause ZeroDivisionError."""
        log_messages = []

        def log_callback(msg, level):
            log_messages.append((msg, level))

        extractor = OMomentExtractor(
            frame_extractor=mock_frame_extractor,
            config=OMomentExtractionConfig(frames_per_window=0),
            log_callback=log_callback,
        )

        # Should not raise ZeroDivisionError and should use default 12
        frames, start, end = extractor.extract_o_moment_frames(
            scene_id=1,
            video_path="/fake/path.mp4",
            center_position=60.0,
            duration=120.0,
        )

        # Should have logged a warning about invalid frames_per_window
        assert any("Invalid frames_per_window" in msg for msg, level in log_messages)

    def test_extract_o_moment_frames_negative_frames_per_window(self, mock_frame_extractor):
        """Test that negative frames_per_window uses default value."""
        log_messages = []

        def log_callback(msg, level):
            log_messages.append((msg, level))

        extractor = OMomentExtractor(
            frame_extractor=mock_frame_extractor,
            config=OMomentExtractionConfig(frames_per_window=-5),
            log_callback=log_callback,
        )

        frames, start, end = extractor.extract_o_moment_frames(
            scene_id=1,
            video_path="/fake/path.mp4",
            center_position=60.0,
            duration=120.0,
        )

        # Should have logged a warning
        assert any("Invalid frames_per_window: -5" in msg for msg, level in log_messages)

    def test_create_o_moment_embedding_empty_frames(self, extractor):
        """Test that empty frames_base64 returns None with warning."""
        log_messages = []
        extractor.log = lambda msg, level: log_messages.append((msg, level))

        mock_embedder = MagicMock()

        result = extractor.create_o_moment_embedding([], mock_embedder)

        assert result is None
        assert any("No frames to embed" in msg for msg, level in log_messages)
        # Embedder should not be called with empty list
        mock_embedder.embed_images.assert_not_called()

    def test_create_o_moment_embedding_none_embeddings(self, extractor):
        """Test handling when embedder returns no embeddings."""
        log_messages = []
        extractor.log = lambda msg, level: log_messages.append((msg, level))

        mock_embedder = MagicMock()
        mock_embedder.embed_images.return_value = []

        result = extractor.create_o_moment_embedding(["base64data"], mock_embedder)

        assert result is None
        assert any("No embeddings returned" in msg for msg, level in log_messages)
