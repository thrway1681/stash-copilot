"""Tests for image labeling storage and task logic."""

import pytest
from stash_ai.embeddings.storage import EmbeddingStorage


@pytest.fixture
def storage(tmp_path):
    """Create a temporary storage instance."""
    db_path = str(tmp_path / "test.sqlite")
    return EmbeddingStorage(db_path=db_path, model_key="test")


class TestLabelingSchema:
    """Tests for labeling database schema."""

    def test_labeling_sessions_table_exists(self, storage):
        """Labeling sessions table should exist after migration."""
        conn = storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='labeling_sessions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_frame_annotations_table_exists(self, storage):
        """Frame annotations table should exist after migration."""
        conn = storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_annotations'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_labeling_progress_table_exists(self, storage):
        """Labeling progress table should exist after migration."""
        conn = storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='labeling_progress'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestLabelingSessionStorage:
    """Tests for labeling session CRUD operations."""

    def test_create_session(self, storage):
        """Create session, retrieve it, check status='active' and batch_size."""
        session_id = storage.create_labeling_session(
            sampling_method="random",
            batch_size=50,
            total_frames=200,
        )
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        session = storage.get_labeling_session(session_id)
        assert session is not None
        assert session["status"] == "active"
        assert session["batch_size"] == 50
        assert session["total_frames"] == 200
        assert session["sampling_method"] == "random"
        assert session["labeled_count"] == 0
        assert session["skipped_count"] == 0

    def test_update_session_counts(self, storage):
        """Create session, update labeled_count=5 and skipped_count=2, verify."""
        session_id = storage.create_labeling_session(
            sampling_method="stratified",
            batch_size=30,
            total_frames=100,
        )
        storage.update_labeling_session(
            session_id, labeled_count=5, skipped_count=2
        )

        session = storage.get_labeling_session(session_id)
        assert session is not None
        assert session["labeled_count"] == 5
        assert session["skipped_count"] == 2

    def test_list_active_sessions(self, storage):
        """Create 2 sessions, complete one, list active should return only 1."""
        sid1 = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )
        sid2 = storage.create_labeling_session(
            sampling_method="random", batch_size=20, total_frames=100
        )

        # Complete the first session
        storage.update_labeling_session(sid1, status="complete")

        active = storage.list_labeling_sessions(status="active")
        assert len(active) == 1
        assert active[0]["session_id"] == sid2

        # All sessions should return both
        all_sessions = storage.list_labeling_sessions()
        assert len(all_sessions) == 2


class TestAnnotationStorage:
    """Tests for annotation CRUD operations."""

    def test_save_and_retrieve_annotations(self, storage):
        """Save 2 annotations (1 confirmed, 1 rejected), retrieve, verify counts."""
        session_id = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )

        annotations = [
            {
                "scene_id": 1,
                "frame_index": 5,
                "tag_text": "blowjob",
                "tag_source": "stash_tag",
                "label": "confirmed",
                "similarity_score": 0.92,
            },
            {
                "scene_id": 1,
                "frame_index": 5,
                "tag_text": "anal",
                "tag_source": "curated",
                "label": "rejected",
                "similarity_score": 0.45,
            },
        ]

        storage.save_annotations(session_id, annotations)

        retrieved = storage.get_annotations(session_id)
        assert len(retrieved) == 2

        confirmed = [a for a in retrieved if a["label"] == "confirmed"]
        rejected = [a for a in retrieved if a["label"] == "rejected"]
        assert len(confirmed) == 1
        assert len(rejected) == 1
        assert confirmed[0]["tag_text"] == "blowjob"
        assert rejected[0]["tag_text"] == "anal"

    def test_get_all_confirmed_annotations(self, storage):
        """Save confirmed annotations across 2 sessions, get all confirmed, verify count=2."""
        sid1 = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )
        sid2 = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )

        storage.save_annotations(
            sid1,
            [
                {
                    "scene_id": 1,
                    "frame_index": 5,
                    "tag_text": "blowjob",
                    "tag_source": "stash_tag",
                    "label": "confirmed",
                    "similarity_score": 0.92,
                },
                {
                    "scene_id": 1,
                    "frame_index": 5,
                    "tag_text": "anal",
                    "tag_source": "curated",
                    "label": "rejected",
                    "similarity_score": 0.45,
                },
            ],
        )
        storage.save_annotations(
            sid2,
            [
                {
                    "scene_id": 2,
                    "frame_index": 10,
                    "tag_text": "brunette",
                    "tag_source": "stash_tag",
                    "label": "confirmed",
                    "similarity_score": 0.88,
                },
            ],
        )

        confirmed = storage.get_all_confirmed_annotations()
        assert len(confirmed) == 2
        tags = {a["tag_text"] for a in confirmed}
        assert tags == {"blowjob", "brunette"}

    def test_get_labeled_frames(self, storage):
        """Save annotation + update progress to 'labeled', verify (scene_id, frame_index) in labeled set."""
        session_id = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )

        storage.save_annotations(
            session_id,
            [
                {
                    "scene_id": 3,
                    "frame_index": 7,
                    "tag_text": "cowgirl",
                    "tag_source": "stash_tag",
                    "label": "confirmed",
                    "similarity_score": 0.80,
                },
            ],
        )
        storage.update_labeling_progress(session_id, scene_id=3, frame_index=7, status="labeled")

        labeled = storage.get_labeled_frame_keys()
        assert (3, 7) in labeled

    def test_get_unembedded_manual_tags(self, storage):
        """Save manual tag annotation, verify it appears in unembedded list (no embedding exists)."""
        session_id = storage.create_labeling_session(
            sampling_method="random", batch_size=10, total_frames=50
        )

        storage.save_annotations(
            session_id,
            [
                {
                    "scene_id": 1,
                    "frame_index": 0,
                    "tag_text": "custom_new_tag",
                    "tag_source": "manual",
                    "label": "confirmed",
                    "similarity_score": None,
                },
            ],
        )

        unembedded = storage.get_unembedded_manual_tags(model_key="test")
        assert "custom_new_tag" in unembedded


import numpy as np
from unittest.mock import MagicMock


class TestUncertaintySampling:
    """Tests for the uncertainty scoring algorithm."""

    def test_uncertainty_score_high_for_ambiguous_frame(self):
        """Frame with many tags in confusion zone should score high."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        # Similarities for one frame against 5 tags
        # 3 tags are in confusion zone (0.25-0.35)
        frame_sims = np.array([0.30, 0.28, 0.33, 0.80, 0.10])
        score = task._compute_uncertainty(frame_sims, low=0.25, high=0.35)
        assert score == 3  # 3 tags in zone

    def test_uncertainty_score_zero_for_clear_frame(self):
        """Frame with no ambiguous tags should score 0."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        # All tags either clearly match or clearly don't
        frame_sims = np.array([0.90, 0.85, 0.05, 0.02])
        score = task._compute_uncertainty(frame_sims, low=0.25, high=0.35)
        assert score == 0

    def test_select_uncertain_frames(self):
        """Should select frames with highest uncertainty first."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        # 4 frames × 3 tags
        similarities = np.array([
            [0.90, 0.05, 0.02],   # Frame 0: clear (uncertainty 0)
            [0.30, 0.28, 0.33],   # Frame 1: very uncertain (3)
            [0.80, 0.31, 0.05],   # Frame 2: somewhat uncertain (1)
            [0.29, 0.26, 0.85],   # Frame 3: uncertain (2)
        ], dtype=np.float32)

        frame_keys = [(1, 0), (1, 1), (1, 2), (1, 3)]  # (scene_id, frame_index)
        selected = task._rank_by_uncertainty(
            similarities, frame_keys, low=0.25, high=0.35, limit=3
        )

        # Should be ordered: frame 1 (score=3), frame 3 (score=2), frame 2 (score=1)
        assert selected[0] == (1, 1)
        assert selected[1] == (1, 3)
        assert selected[2] == (1, 2)


class TestSyncAnnotations:
    """Tests for annotation syncing."""

    def test_sync_updates_storage(self, storage):
        """Syncing annotations should persist them to DB."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=storage,
            log_callback=lambda msg, lvl: None,
        )

        session_id = storage.create_labeling_session("uncertainty", 100, 100)
        payload = {
            "session_id": session_id,
            "annotations": [
                {
                    "scene_id": 1, "frame_index": 10,
                    "tag_text": "blowjob", "tag_source": "suggested",
                    "label": "confirmed", "similarity_score": 0.32,
                },
            ],
            "progress": [
                {"scene_id": 1, "frame_index": 10, "status": "labeled"},
            ],
        }

        task.sync_annotations(payload)

        annotations = storage.get_annotations(session_id)
        assert len(annotations) == 1
        assert annotations[0]["label"] == "confirmed"

        session = storage.get_labeling_session(session_id)
        assert session["labeled_count"] == 1


class TestExportDataset:
    """Tests for WebDataset export."""

    def test_generate_caption(self):
        """Auto-generate caption from confirmed tags."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        tags = ["blowjob", "brunette", "POV"]
        caption = task._generate_caption(tags, "a scene featuring {tags}")
        assert caption == "a scene featuring blowjob, brunette, and POV"

    def test_generate_caption_single_tag(self):
        """Caption with single tag."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        tags = ["solo"]
        caption = task._generate_caption(tags, "a scene featuring {tags}")
        assert caption == "a scene featuring solo"

    def test_generate_caption_two_tags(self):
        """Caption with two tags uses 'and'."""
        from stash_ai.tasks.labeling import LabelingTask

        task = LabelingTask(
            stash=MagicMock(),
            storage=MagicMock(),
            log_callback=lambda msg, lvl: None,
        )

        tags = ["blowjob", "POV"]
        caption = task._generate_caption(tags, "a scene featuring {tags}")
        assert caption == "a scene featuring blowjob and POV"
