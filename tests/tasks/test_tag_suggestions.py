"""Tests for tag suggestion storage methods and types."""

import pytest
from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.tasks.tag_suggestions import (
    TagSuggestion,
    EvidenceFrame,
    TagSuggestionsResult,
)


@pytest.fixture
def storage(tmp_path):
    """Create a temporary storage instance."""
    db_path = str(tmp_path / "test.sqlite")
    return EmbeddingStorage(db_path=db_path, model_key="test")


class TestDismissedTagStorage:
    """Tests for dismissed tag suggestion storage."""

    def test_save_dismissed_tag(self, storage):
        """Save a dismissed tag for a scene."""
        storage.save_dismissed_tag(scene_id=1, tag_id=100)
        dismissed = storage.get_dismissed_tags(scene_id=1)
        assert 100 in dismissed

    def test_get_dismissed_tags_empty(self, storage):
        """Returns empty set when no dismissals exist."""
        dismissed = storage.get_dismissed_tags(scene_id=999)
        assert dismissed == set()

    def test_save_dismissed_tag_idempotent(self, storage):
        """Dismissing same tag twice doesn't error."""
        storage.save_dismissed_tag(scene_id=1, tag_id=100)
        storage.save_dismissed_tag(scene_id=1, tag_id=100)
        dismissed = storage.get_dismissed_tags(scene_id=1)
        assert len(dismissed) == 1

    def test_clear_dismissed_tags(self, storage):
        """Clear all dismissals for a scene."""
        storage.save_dismissed_tag(scene_id=1, tag_id=100)
        storage.save_dismissed_tag(scene_id=1, tag_id=101)
        count = storage.clear_dismissed_tags(scene_id=1)
        assert count == 2
        assert storage.get_dismissed_tags(scene_id=1) == set()

    def test_dismissed_tags_per_scene(self, storage):
        """Dismissals are scene-specific."""
        storage.save_dismissed_tag(scene_id=1, tag_id=100)
        storage.save_dismissed_tag(scene_id=2, tag_id=200)
        assert storage.get_dismissed_tags(scene_id=1) == {100}
        assert storage.get_dismissed_tags(scene_id=2) == {200}


class TestSimilarityComputation:
    """Tests for frame-to-tag similarity computation."""

    @pytest.fixture
    def mock_storage(self, tmp_path):
        """Create storage with mock frame/tag embeddings."""
        db_path = str(tmp_path / "test.sqlite")
        return EmbeddingStorage(db_path=db_path, model_key="test")

    def test_compute_similarities_basic(self, mock_storage):
        """Compute similarities between frames and tags."""
        import numpy as np
        from unittest.mock import MagicMock
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        task = TagSuggestionsTask(
            stash=MagicMock(),
            storage=mock_storage,
            log_callback=lambda msg, lvl: None,
        )

        # 3 frames, 2 tags, 4-dimensional embeddings
        frame_embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Frame 0 - matches tag 0
            [0.0, 1.0, 0.0, 0.0],  # Frame 1 - matches tag 1
            [0.7, 0.7, 0.0, 0.0],  # Frame 2 - between both
        ])
        tag_embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Tag 0
            [0.0, 1.0, 0.0, 0.0],  # Tag 1
        ])

        similarities = task._compute_similarities(frame_embeddings, tag_embeddings)

        # Shape should be (3 frames, 2 tags)
        assert similarities.shape == (3, 2)

        # Frame 0 should perfectly match tag 0
        assert similarities[0, 0] == pytest.approx(1.0, abs=0.01)
        assert similarities[0, 1] == pytest.approx(0.0, abs=0.01)

        # Frame 1 should perfectly match tag 1
        assert similarities[1, 0] == pytest.approx(0.0, abs=0.01)
        assert similarities[1, 1] == pytest.approx(1.0, abs=0.01)

    def test_aggregate_votes(self, mock_storage):
        """Aggregate frame votes into tag suggestions."""
        import numpy as np
        from unittest.mock import MagicMock
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        task = TagSuggestionsTask(
            stash=MagicMock(),
            storage=mock_storage,
            log_callback=lambda msg, lvl: None,
        )

        # Similarity matrix: 5 frames x 2 tags
        similarities = np.array([
            [0.8, 0.2],  # Frame 0 votes tag 0
            [0.7, 0.3],  # Frame 1 votes tag 0
            [0.2, 0.9],  # Frame 2 votes tag 1
            [0.3, 0.1],  # Frame 3 - below threshold
            [0.6, 0.4],  # Frame 4 votes tag 0
        ])

        tag_info = [
            {"id": 100, "name": "tag_a"},
            {"id": 101, "name": "tag_b"},
        ]

        votes = task._aggregate_votes(
            similarities, tag_info, threshold=0.50
        )

        # tag_a (index 0) should have 3 votes (frames 0, 1, 4)
        tag_a = next(v for v in votes if v["tag_name"] == "tag_a")
        assert tag_a["frame_count"] == 3
        assert tag_a["max_similarity"] == pytest.approx(0.8, abs=0.01)

        # tag_b (index 1) should have 1 vote (frame 2)
        tag_b = next(v for v in votes if v["tag_name"] == "tag_b")
        assert tag_b["frame_count"] == 1
        assert tag_b["max_similarity"] == pytest.approx(0.9, abs=0.01)


class TestTagSuggestionTypes:
    """Tests for tag suggestion data types."""

    def test_evidence_frame_creation(self):
        """EvidenceFrame stores frame metadata."""
        frame = EvidenceFrame(
            frame_index=45,
            similarity=0.82,
            timestamp="2:15",
            thumbnail_path="assets/embedded_frames/scene_123/frame_0045.jpg",
        )
        assert frame.frame_index == 45
        assert frame.similarity == 0.82
        assert frame.timestamp == "2:15"

    def test_tag_suggestion_creation(self):
        """TagSuggestion aggregates evidence."""
        suggestion = TagSuggestion(
            tag_id=100,
            tag_name="blowjob",
            max_similarity=0.82,
            mean_similarity=0.65,
            frame_count=12,
            evidence_frames=[
                EvidenceFrame(45, 0.82, "2:15", "path/frame_0045.jpg"),
            ],
        )
        assert suggestion.tag_name == "blowjob"
        assert suggestion.frame_count == 12
        assert len(suggestion.evidence_frames) == 1

    def test_suggestions_result_success(self):
        """TagSuggestionsResult for successful computation."""
        result = TagSuggestionsResult(
            status="complete",
            scene_id=123,
            suggestions=[],
            error=None,
        )
        assert result["status"] == "complete"
        assert result["scene_id"] == 123

    def test_suggestions_result_error(self):
        """TagSuggestionsResult for error state."""
        result = TagSuggestionsResult(
            status="error",
            scene_id=123,
            suggestions=[],
            error="No embeddings found",
        )
        assert result["status"] == "error"
        assert result["error"] == "No embeddings found"


class TestTagSuggestionPipeline:
    """Integration tests for the full suggestion pipeline."""

    @pytest.fixture
    def storage_with_embeddings(self, tmp_path):
        """Storage with pre-populated frame and tag embeddings."""
        storage = EmbeddingStorage(
            db_path=str(tmp_path / "test.sqlite"), model_key="test"
        )

        # Add frame embeddings for scene 1
        storage.store_frame_embedding(
            scene_id=1,
            frame_index=0,
            timestamp=10.0,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        storage.store_frame_embedding(
            scene_id=1,
            frame_index=1,
            timestamp=20.0,
            embedding=[0.0, 1.0, 0.0, 0.0],
        )

        # Add tag embeddings
        storage.save_tag_embeddings_batch(
            entries=[
                ("oral", [1.0, 0.0, 0.0, 0.0], "stash_tag"),
                ("brunette", [0.0, 1.0, 0.0, 0.0], "stash_tag"),
                ("blonde", [0.0, 0.0, 1.0, 0.0], "stash_tag"),
            ],
            model_key="test",
        )

        return storage

    @pytest.fixture
    def mock_stash_with_tags(self):
        """Mock StashInterface with tag data."""
        from unittest.mock import MagicMock

        stash = MagicMock()

        def call_gql_side_effect(query, variables=None):
            if "findTags" in query:
                return {
                    "findTags": {
                        "tags": [
                            {"id": "100", "name": "oral"},
                            {"id": "101", "name": "brunette"},
                            {"id": "102", "name": "blonde"},
                        ]
                    }
                }
            if "findScene" in query:
                return {"findScene": {"tags": []}}
            return {}

        stash.call_GQL.side_effect = call_gql_side_effect
        return stash

    def test_run_returns_suggestions(self, storage_with_embeddings, mock_stash_with_tags):
        """Full pipeline returns ranked suggestions."""
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        task = TagSuggestionsTask(
            stash=mock_stash_with_tags,
            storage=storage_with_embeddings,
            log_callback=lambda msg, lvl: None,
            model_key="test",
        )

        result = task.run(scene_id=1)

        assert result["status"] == "complete"
        assert len(result["suggestions"]) == 2  # oral and brunette match

        names = [s["tag_name"] for s in result["suggestions"]]
        assert "oral" in names
        assert "brunette" in names
        assert "blonde" not in names  # No matching frames

    def test_run_excludes_existing_tags(
        self, storage_with_embeddings, mock_stash_with_tags
    ):
        """Suggestions exclude tags already on the scene."""
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        # Scene already has "oral" tag
        def call_gql_side_effect(query, variables=None):
            if "findTags" in query:
                return {
                    "findTags": {
                        "tags": [
                            {"id": "100", "name": "oral"},
                            {"id": "101", "name": "brunette"},
                        ]
                    }
                }
            if "findScene" in query:
                return {"findScene": {"tags": [{"id": "100"}]}}
            return {}

        mock_stash_with_tags.call_GQL.side_effect = call_gql_side_effect

        task = TagSuggestionsTask(
            stash=mock_stash_with_tags,
            storage=storage_with_embeddings,
            log_callback=lambda msg, lvl: None,
            model_key="test",
        )

        result = task.run(scene_id=1)

        names = [s["tag_name"] for s in result["suggestions"]]
        assert "oral" not in names
        assert "brunette" in names

    def test_run_excludes_dismissed_tags(
        self, storage_with_embeddings, mock_stash_with_tags
    ):
        """Suggestions exclude dismissed tags."""
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        storage_with_embeddings.save_dismissed_tag(scene_id=1, tag_id=100)

        task = TagSuggestionsTask(
            stash=mock_stash_with_tags,
            storage=storage_with_embeddings,
            log_callback=lambda msg, lvl: None,
            model_key="test",
        )

        result = task.run(scene_id=1)

        names = [s["tag_name"] for s in result["suggestions"]]
        assert "oral" not in names  # Dismissed
        assert "brunette" in names

    def test_run_no_embeddings(self, tmp_path, mock_stash_with_tags):
        """Returns error when scene has no embeddings."""
        from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

        empty_storage = EmbeddingStorage(
            db_path=str(tmp_path / "empty.sqlite"), model_key="test"
        )

        task = TagSuggestionsTask(
            stash=mock_stash_with_tags,
            storage=empty_storage,
            log_callback=lambda msg, lvl: None,
            model_key="test",
        )

        result = task.run(scene_id=1)

        assert result["status"] == "no_embeddings"
        assert "No frame embeddings" in result["error"]
