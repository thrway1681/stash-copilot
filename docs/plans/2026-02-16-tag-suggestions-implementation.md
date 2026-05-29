# Embedding-Based Tag Suggestions - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Tags tab to scene sidebar that suggests tags using frame-to-tag embedding similarity with visual evidence thumbnails.

**Architecture:** Frame-centric voting algorithm: each frame votes for matching tags, aggregate votes to rank suggestions, show top-matching frames as evidence. Pure embedding-based, no LLM calls.

**Tech Stack:** Python (backend task), SQLite (storage), JavaScript (UI), CSS (styling), numpy (similarity computation)

---

## Task 1: Add Storage Schema for Dismissed Tags

**Files:**
- Modify: `stash_ai/embeddings/storage.py:127` (SCHEMA_VERSION)
- Modify: `stash_ai/embeddings/storage.py:202-203` (migration chain)
- Test: `tests/tasks/test_tag_suggestions.py` (new file)

**Step 1: Write the failing test for dismissed tag storage**

Create `tests/tasks/test_tag_suggestions.py`:

```python
"""Tests for tag suggestion storage methods."""

import pytest
from stash_ai.embeddings.storage import EmbeddingStorage


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py -v`
Expected: FAIL with "AttributeError: 'EmbeddingStorage' object has no attribute 'save_dismissed_tag'"

**Step 3: Add schema migration v11**

In `stash_ai/embeddings/storage.py`, update SCHEMA_VERSION:

```python
# Database schema version for migrations
SCHEMA_VERSION = 11  # Added dismissed_tag_suggestions table
```

Add migration check after line 203:

```python
if current_version < 11:
    self._migrate_to_v11(cursor)
```

Add migration method (after `_migrate_to_v10`):

```python
def _migrate_to_v11(self, cursor: sqlite3.Cursor) -> None:
    """Add dismissed_tag_suggestions table (v11)."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dismissed_tag_suggestions (
            scene_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            dismissed_at TEXT NOT NULL,
            PRIMARY KEY (scene_id, tag_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dismissed_scene
        ON dismissed_tag_suggestions(scene_id)
        """
    )
```

**Step 4: Add storage methods**

Add to `EmbeddingStorage` class (near other save/get methods):

```python
def save_dismissed_tag(self, scene_id: int, tag_id: int) -> None:
    """Record that a tag suggestion was dismissed for a scene."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO dismissed_tag_suggestions
        (scene_id, tag_id, dismissed_at)
        VALUES (?, ?, ?)
        """,
        (scene_id, tag_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def get_dismissed_tags(self, scene_id: int) -> set[int]:
    """Get all dismissed tag IDs for a scene."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT tag_id FROM dismissed_tag_suggestions
        WHERE scene_id = ?
        """,
        (scene_id,),
    )
    result = {row["tag_id"] for row in cursor.fetchall()}
    conn.close()
    return result

def clear_dismissed_tags(self, scene_id: int) -> int:
    """Clear all dismissals for a scene. Returns count deleted."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM dismissed_tag_suggestions
        WHERE scene_id = ?
        """,
        (scene_id,),
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add stash_ai/embeddings/storage.py tests/tasks/test_tag_suggestions.py
git commit -m "feat(storage): add dismissed_tag_suggestions table and methods"
```

---

## Task 2: Create TagSuggestion Types

**Files:**
- Create: `stash_ai/tasks/tag_suggestions.py`
- Test: `tests/tasks/test_tag_suggestions.py` (extend)

**Step 1: Write the test for TagSuggestion dataclass**

Add to `tests/tasks/test_tag_suggestions.py`:

```python
from stash_ai.tasks.tag_suggestions import (
    TagSuggestion,
    EvidenceFrame,
    TagSuggestionsResult,
)


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
        assert result.status == "complete"
        assert result.scene_id == 123

    def test_suggestions_result_error(self):
        """TagSuggestionsResult for error state."""
        result = TagSuggestionsResult(
            status="error",
            scene_id=123,
            suggestions=[],
            error="No embeddings found",
        )
        assert result.status == "error"
        assert result.error == "No embeddings found"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestTagSuggestionTypes -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Create the types file**

Create `stash_ai/tasks/tag_suggestions.py`:

```python
"""Tag suggestions task - embedding-based tag recommendations with evidence frames.

Uses frame-centric voting: each frame votes for matching tags based on
cosine similarity between frame embeddings and tag embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    pass


@dataclass
class EvidenceFrame:
    """A frame that matched a tag, used as visual evidence."""

    frame_index: int
    similarity: float
    timestamp: str  # "MM:SS" format
    thumbnail_path: str  # Relative path to frame image


@dataclass
class TagSuggestion:
    """A tag suggestion with supporting evidence."""

    tag_id: int
    tag_name: str
    max_similarity: float  # Highest single-frame match
    mean_similarity: float  # Average across matching frames
    frame_count: int  # Frames with similarity >= threshold
    evidence_frames: list[EvidenceFrame] = field(default_factory=list)


class TagSuggestionsResult(TypedDict):
    """Result from tag suggestion computation."""

    status: str  # "complete", "error", "no_embeddings"
    scene_id: int
    suggestions: list[dict]  # Serialized TagSuggestion objects
    error: str | None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestTagSuggestionTypes -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add stash_ai/tasks/tag_suggestions.py tests/tasks/test_tag_suggestions.py
git commit -m "feat(types): add TagSuggestion and EvidenceFrame dataclasses"
```

---

## Task 3: Implement Core Similarity Computation

**Files:**
- Modify: `stash_ai/tasks/tag_suggestions.py`
- Test: `tests/tasks/test_tag_suggestions.py` (extend)

**Step 1: Write the test for similarity computation**

Add to `tests/tasks/test_tag_suggestions.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch

from stash_ai.tasks.tag_suggestions import TagSuggestionsTask


class TestSimilarityComputation:
    """Tests for frame-to-tag similarity computation."""

    @pytest.fixture
    def mock_storage(self, tmp_path):
        """Create storage with mock frame/tag embeddings."""
        from stash_ai.embeddings.storage import EmbeddingStorage

        storage = EmbeddingStorage(
            db_path=str(tmp_path / "test.sqlite"), model_key="test"
        )
        return storage

    def test_compute_similarities_basic(self, mock_storage):
        """Compute similarities between frames and tags."""
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
            similarities, tag_info, threshold=0.30
        )

        # tag_a (index 0) should have 3 votes (frames 0, 1, 4)
        assert votes[0]["frame_count"] == 3
        assert votes[0]["max_similarity"] == pytest.approx(0.8, abs=0.01)

        # tag_b (index 1) should have 1 vote (frame 2)
        assert votes[1]["frame_count"] == 1
        assert votes[1]["max_similarity"] == pytest.approx(0.9, abs=0.01)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestSimilarityComputation -v`
Expected: FAIL with "AttributeError: 'TagSuggestionsTask' object has no attribute '_compute_similarities'"

**Step 3: Implement TagSuggestionsTask with similarity methods**

Add to `stash_ai/tasks/tag_suggestions.py`:

```python
from collections.abc import Callable
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stashapi.stashapp import StashInterface

from stash_ai.embeddings.storage import EmbeddingStorage


class TagSuggestionsTask:
    """Compute embedding-based tag suggestions for a scene.

    Uses frame-centric voting: each frame votes for tags based on
    cosine similarity. Tags with many votes rank higher.
    """

    # Minimum similarity for a frame to vote for a tag
    SIMILARITY_THRESHOLD = 0.30

    # Maximum suggestions to return
    MAX_SUGGESTIONS = 20

    # Number of evidence frames per suggestion
    EVIDENCE_FRAME_COUNT = 5

    def __init__(
        self,
        stash: StashInterface,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def _compute_similarities(
        self,
        frame_embeddings: NDArray[np.float32],
        tag_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between all frames and all tags.

        Args:
            frame_embeddings: (N, D) array of frame embeddings
            tag_embeddings: (T, D) array of tag embeddings

        Returns:
            (N, T) similarity matrix
        """
        # Normalize embeddings for cosine similarity
        frame_norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        tag_norms = np.linalg.norm(tag_embeddings, axis=1, keepdims=True)

        frame_normalized = frame_embeddings / (frame_norms + 1e-8)
        tag_normalized = tag_embeddings / (tag_norms + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(frame_normalized, tag_normalized.T)

    def _aggregate_votes(
        self,
        similarities: NDArray[np.float32],
        tag_info: list[dict[str, Any]],
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Aggregate frame votes into tag suggestions.

        Args:
            similarities: (N, T) similarity matrix
            tag_info: List of {"id": int, "name": str} for each tag
            threshold: Minimum similarity for a vote

        Returns:
            List of vote aggregations per tag
        """
        results = []

        for tag_idx, tag in enumerate(tag_info):
            tag_similarities = similarities[:, tag_idx]

            # Find frames that vote for this tag
            voting_mask = tag_similarities >= threshold
            voting_frames = np.where(voting_mask)[0]

            if len(voting_frames) == 0:
                continue

            voting_sims = tag_similarities[voting_mask]

            results.append({
                "tag_id": tag["id"],
                "tag_name": tag["name"],
                "frame_count": len(voting_frames),
                "max_similarity": float(np.max(voting_sims)),
                "mean_similarity": float(np.mean(voting_sims)),
                "voting_frames": voting_frames.tolist(),
                "voting_similarities": voting_sims.tolist(),
            })

        return results
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestSimilarityComputation -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add stash_ai/tasks/tag_suggestions.py tests/tasks/test_tag_suggestions.py
git commit -m "feat(task): add similarity computation and vote aggregation"
```

---

## Task 4: Implement Full Suggestion Pipeline

**Files:**
- Modify: `stash_ai/tasks/tag_suggestions.py`
- Test: `tests/tasks/test_tag_suggestions.py` (extend)

**Step 1: Write the test for full pipeline**

Add to `tests/tasks/test_tag_suggestions.py`:

```python
class TestTagSuggestionPipeline:
    """Integration tests for the full suggestion pipeline."""

    @pytest.fixture
    def storage_with_embeddings(self, tmp_path):
        """Storage with pre-populated frame and tag embeddings."""
        from stash_ai.embeddings.storage import EmbeddingStorage

        storage = EmbeddingStorage(
            db_path=str(tmp_path / "test.sqlite"), model_key="test"
        )

        # Add frame embeddings for scene 1
        # Frame 0: matches "oral" tag
        # Frame 1: matches "brunette" tag
        storage.save_frame_embedding(
            scene_id=1,
            frame_index=0,
            timestamp=10.0,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        storage.save_frame_embedding(
            scene_id=1,
            frame_index=1,
            timestamp=20.0,
            embedding=[0.0, 1.0, 0.0, 0.0],
        )

        # Add tag embeddings
        storage.save_tag_embeddings_batch([
            ("oral", "stash_tag", [1.0, 0.0, 0.0, 0.0]),
            ("brunette", "stash_tag", [0.0, 1.0, 0.0, 0.0]),
            ("blonde", "stash_tag", [0.0, 0.0, 1.0, 0.0]),  # No matching frames
        ])

        return storage

    @pytest.fixture
    def mock_stash_with_tags(self):
        """Mock StashInterface with tag data."""
        stash = MagicMock()
        stash.call_GQL.return_value = {
            "findTags": {
                "tags": [
                    {"id": "100", "name": "oral"},
                    {"id": "101", "name": "brunette"},
                    {"id": "102", "name": "blonde"},
                ]
            },
            "findScene": {"tags": []},  # Scene has no existing tags
        }
        return stash

    def test_run_returns_suggestions(
        self, storage_with_embeddings, mock_stash_with_tags
    ):
        """Full pipeline returns ranked suggestions."""
        task = TagSuggestionsTask(
            stash=mock_stash_with_tags,
            storage=storage_with_embeddings,
            log_callback=lambda msg, lvl: None,
            model_key="test",
        )

        result = task.run(scene_id=1)

        assert result["status"] == "complete"
        assert len(result["suggestions"]) == 2  # oral and brunette

        # Should be sorted by frame_count (both have 1), then max_similarity
        names = [s["tag_name"] for s in result["suggestions"]]
        assert "oral" in names
        assert "brunette" in names
        assert "blonde" not in names  # No matching frames

    def test_run_excludes_existing_tags(
        self, storage_with_embeddings, mock_stash_with_tags
    ):
        """Suggestions exclude tags already on the scene."""
        # Scene already has "oral" tag
        mock_stash_with_tags.call_GQL.return_value = {
            "findTags": {
                "tags": [
                    {"id": "100", "name": "oral"},
                    {"id": "101", "name": "brunette"},
                ]
            },
            "findScene": {"tags": [{"id": "100"}]},
        }

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestTagSuggestionPipeline -v`
Expected: FAIL with "AttributeError: 'TagSuggestionsTask' object has no attribute 'run'"

**Step 3: Implement the run() method**

Add to `TagSuggestionsTask` in `stash_ai/tasks/tag_suggestions.py`:

```python
def run(self, scene_id: int) -> TagSuggestionsResult:
    """Compute tag suggestions for a scene.

    Args:
        scene_id: The scene to analyze

    Returns:
        TagSuggestionsResult with suggestions or error
    """
    try:
        # 1. Load frame embeddings
        frame_data = self.storage.get_frame_embeddings(scene_id)
        if not frame_data:
            return TagSuggestionsResult(
                status="no_embeddings",
                scene_id=scene_id,
                suggestions=[],
                error="No frame embeddings found for this scene",
            )

        frame_embeddings = np.array([f["embedding"] for f in frame_data])
        frame_timestamps = [f["timestamp"] for f in frame_data]

        # 2. Load tag embeddings
        tag_embeddings_data = self.storage.get_all_tag_embeddings()
        if not tag_embeddings_data:
            return TagSuggestionsResult(
                status="no_embeddings",
                scene_id=scene_id,
                suggestions=[],
                error="No tag embeddings found. Run tag embedding first.",
            )

        # 3. Get tag info from Stash (for IDs and names)
        tag_info = self._get_stash_tags()
        if not tag_info:
            return TagSuggestionsResult(
                status="error",
                scene_id=scene_id,
                suggestions=[],
                error="Failed to load tags from Stash",
            )

        # 4. Build tag embedding matrix (only for tags that exist in Stash)
        tag_name_to_embedding = {
            t["text"]: t["embedding"] for t in tag_embeddings_data
        }
        valid_tags = []
        tag_embeddings_list = []
        for tag in tag_info:
            name_lower = tag["name"].lower()
            if name_lower in tag_name_to_embedding:
                valid_tags.append(tag)
                tag_embeddings_list.append(tag_name_to_embedding[name_lower])

        if not valid_tags:
            return TagSuggestionsResult(
                status="error",
                scene_id=scene_id,
                suggestions=[],
                error="No tags have embeddings",
            )

        tag_embeddings = np.array(tag_embeddings_list)

        # 5. Compute similarities
        similarities = self._compute_similarities(frame_embeddings, tag_embeddings)

        # 6. Aggregate votes
        votes = self._aggregate_votes(similarities, valid_tags)

        # 7. Get exclusions
        existing_tag_ids = self._get_scene_tag_ids(scene_id)
        dismissed_tag_ids = self.storage.get_dismissed_tags(scene_id)
        excluded_ids = existing_tag_ids | dismissed_tag_ids

        # 8. Filter and rank
        filtered_votes = [
            v for v in votes if v["tag_id"] not in excluded_ids
        ]

        # Sort by frame_count desc, then max_similarity desc
        filtered_votes.sort(
            key=lambda v: (v["frame_count"], v["max_similarity"]),
            reverse=True,
        )

        # 9. Build suggestions with evidence
        suggestions = []
        for vote in filtered_votes[: self.MAX_SUGGESTIONS]:
            evidence = self._build_evidence_frames(
                scene_id,
                vote["voting_frames"],
                vote["voting_similarities"],
                frame_timestamps,
            )
            suggestions.append({
                "tag_id": vote["tag_id"],
                "tag_name": vote["tag_name"],
                "max_similarity": vote["max_similarity"],
                "mean_similarity": vote["mean_similarity"],
                "frame_count": vote["frame_count"],
                "evidence_frames": evidence,
            })

        return TagSuggestionsResult(
            status="complete",
            scene_id=scene_id,
            suggestions=suggestions,
            error=None,
        )

    except Exception as e:
        self.log(f"Tag suggestion error: {e}", "error")
        return TagSuggestionsResult(
            status="error",
            scene_id=scene_id,
            suggestions=[],
            error=str(e),
        )

def _get_stash_tags(self) -> list[dict[str, Any]]:
    """Get all tags from Stash, excluding bracketed system tags."""
    try:
        result = self.stash.call_GQL(
            """
            query FindTags {
                findTags(filter: { per_page: -1 }) {
                    tags {
                        id
                        name
                    }
                }
            }
            """
        )
        if not result or "findTags" not in result:
            return []

        # Filter out bracketed tags
        tags = []
        for t in result["findTags"]["tags"]:
            name = t["name"]
            if not any(c in name for c in "[](){}<>"):
                tags.append({"id": int(t["id"]), "name": name})
        return tags

    except Exception as e:
        self.log(f"Failed to get tags: {e}", "warning")
        return []

def _get_scene_tag_ids(self, scene_id: int) -> set[int]:
    """Get tag IDs already on the scene."""
    try:
        result = self.stash.call_GQL(
            """
            query FindScene($id: ID!) {
                findScene(id: $id) {
                    tags { id }
                }
            }
            """,
            {"id": str(scene_id)},
        )
        if not result or "findScene" not in result:
            return set()

        return {int(t["id"]) for t in result["findScene"]["tags"]}

    except Exception as e:
        self.log(f"Failed to get scene tags: {e}", "warning")
        return set()

def _build_evidence_frames(
    self,
    scene_id: int,
    frame_indices: list[int],
    similarities: list[float],
    timestamps: list[float],
) -> list[dict[str, Any]]:
    """Build evidence frame list for a suggestion.

    Returns top EVIDENCE_FRAME_COUNT frames sorted by similarity.
    """
    # Pair indices with similarities and sort
    paired = list(zip(frame_indices, similarities))
    paired.sort(key=lambda x: x[1], reverse=True)

    evidence = []
    for frame_idx, sim in paired[: self.EVIDENCE_FRAME_COUNT]:
        # Format timestamp as MM:SS
        ts = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        ts_str = f"{minutes}:{seconds:02d}"

        # Frame path (relative to plugin dir)
        frame_path = (
            f"assets/embedded_frames/scene_{scene_id}/"
            f"frame_{frame_idx:04d}.jpg"
        )

        evidence.append({
            "frame_index": frame_idx,
            "similarity": round(sim, 3),
            "timestamp": ts_str,
            "thumbnail_path": frame_path,
        })

    return evidence
```

**Step 4: Add missing storage method**

The test needs `get_all_tag_embeddings()` and `save_tag_embeddings_batch()` - these should already exist. Verify by checking storage.py. If `save_frame_embedding` is missing, add a minimal version:

```python
def save_frame_embedding(
    self,
    scene_id: int,
    frame_index: int,
    timestamp: float,
    embedding: list[float],
) -> None:
    """Save a single frame embedding (for testing)."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO frame_embeddings
        (scene_id, frame_index, timestamp, embedding, model_key, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            scene_id,
            frame_index,
            timestamp,
            self._pack_embedding(embedding),
            self.model_key,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_tag_suggestions.py::TestTagSuggestionPipeline -v`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add stash_ai/tasks/tag_suggestions.py tests/tasks/test_tag_suggestions.py
git commit -m "feat(task): implement tag suggestion pipeline with filtering"
```

---

## Task 5: Add Plugin Task Modes

**Files:**
- Modify: `stash-copilot.py`
- Test: Manual testing via Stash UI

**Step 1: Add task mode handlers**

In `stash-copilot.py`, add import at top:

```python
from stash_ai.tasks.tag_suggestions import TagSuggestionsTask
```

Add task modes in the `run()` method's mode dispatcher (follow existing pattern):

```python
elif mode == "get_tag_suggestions":
    self.run_get_tag_suggestions(args)
elif mode == "apply_suggested_tag":
    self.run_apply_suggested_tag(args)
elif mode == "dismiss_suggested_tag":
    self.run_dismiss_suggested_tag(args)
elif mode == "clear_dismissed_tags":
    self.run_clear_dismissed_tags(args)
```

Add handler methods:

```python
def run_get_tag_suggestions(self, args: dict[str, Any]) -> None:
    """Get embedding-based tag suggestions for a scene."""
    scene_id = args.get("scene_id")
    if not scene_id:
        self.log("Missing scene_id", "error")
        return

    scene_id = int(scene_id)
    request_id = args.get("request_id", "")

    self.log(f"Computing tag suggestions for scene {scene_id}", "info")

    storage = EmbeddingStorage(model_key="siglip")
    task = TagSuggestionsTask(
        stash=self.stash,
        storage=storage,
        log_callback=self.log,
        model_key="siglip",
    )

    result = task.run(scene_id=scene_id)

    # Save result for frontend polling
    if request_id:
        assets_dir = Path(__file__).parent / "assets"
        result_path = assets_dir / f"tag_suggestions_{request_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    if result["status"] == "complete":
        self.log(f"Found {len(result['suggestions'])} tag suggestions", "info")
    else:
        self.log(f"Tag suggestions: {result['error']}", "warning")

def run_apply_suggested_tag(self, args: dict[str, Any]) -> None:
    """Apply a suggested tag to a scene."""
    scene_id = int(args.get("scene_id", 0))
    tag_id = int(args.get("tag_id", 0))

    if not scene_id or not tag_id:
        self.log("Missing scene_id or tag_id", "error")
        return

    try:
        # Get current tags
        result = self.stash.call_GQL(
            """
            query FindScene($id: ID!) {
                findScene(id: $id) { tags { id } }
            }
            """,
            {"id": str(scene_id)},
        )

        current_ids = [int(t["id"]) for t in result["findScene"]["tags"]]
        if tag_id in current_ids:
            self.log("Tag already on scene", "info")
            return

        new_ids = current_ids + [tag_id]

        # Update scene
        self.stash.call_GQL(
            """
            mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }
            """,
            {"input": {"id": str(scene_id), "tag_ids": [str(i) for i in new_ids]}},
        )

        self.log(f"Applied tag {tag_id} to scene {scene_id}", "info")

    except Exception as e:
        self.log(f"Failed to apply tag: {e}", "error")

def run_dismiss_suggested_tag(self, args: dict[str, Any]) -> None:
    """Dismiss a tag suggestion for a scene."""
    scene_id = int(args.get("scene_id", 0))
    tag_id = int(args.get("tag_id", 0))

    if not scene_id or not tag_id:
        self.log("Missing scene_id or tag_id", "error")
        return

    storage = EmbeddingStorage(model_key="siglip")
    storage.save_dismissed_tag(scene_id, tag_id)
    self.log(f"Dismissed tag {tag_id} for scene {scene_id}", "info")

def run_clear_dismissed_tags(self, args: dict[str, Any]) -> None:
    """Clear all dismissed tags for a scene."""
    scene_id = int(args.get("scene_id", 0))

    if not scene_id:
        self.log("Missing scene_id", "error")
        return

    storage = EmbeddingStorage(model_key="siglip")
    count = storage.clear_dismissed_tags(scene_id)
    self.log(f"Cleared {count} dismissed tags for scene {scene_id}", "info")
```

**Step 2: Register tasks in plugin YAML (if needed)**

Check `stash-copilot.yml` - if tasks need explicit registration, add:

```yaml
tasks:
  - name: "Get Tag Suggestions"
    description: "Get embedding-based tag suggestions for a scene"
    mode: "get_tag_suggestions"
```

**Step 3: Commit**

```bash
git add stash-copilot.py
git commit -m "feat(plugin): add tag suggestion task modes"
```

---

## Task 6: Add Tags Tab UI (JavaScript)

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Add state object for Tags tab**

Find existing state objects (like `similarState`, `sceneRecsState`) and add:

```javascript
// Tag suggestions state
const tagSuggestionState = {
    sceneId: null,
    suggestions: [],
    loading: false,
    error: null,
    currentPage: 0,
    suggestionsPerPage: 5,
};
```

**Step 2: Register Tags tab in sidebar injection**

Find `injectSceneTabs()` and add the Tags tab alongside existing tabs:

```javascript
// Add Tags tab nav item
const tagsNavItem = document.createElement('li');
tagsNavItem.className = 'nav-item stash-copilot-tab-nav';
tagsNavItem.innerHTML = `
    <button class="nav-link stash-copilot-sidebar-tab" data-copilot-tab="tags">
        <span class="tab-icon">🏷️</span>
        <span class="tab-label">Tags</span>
    </button>
`;
navTabs.appendChild(tagsNavItem);

// Add Tags content pane
const tagsPane = document.createElement('div');
tagsPane.className = 'tab-pane stash-copilot-tab-pane';
tagsPane.dataset.copilotPane = 'tags';
tabContent.appendChild(tagsPane);
```

**Step 3: Add content loader for Tags tab**

In `loadSidebarTabContent()`, add case for tags:

```javascript
case 'tags':
    renderSidebarTagsContent(container, sceneId);
    break;
```

**Step 4: Implement renderSidebarTagsContent()**

```javascript
function renderSidebarTagsContent(container, sceneId) {
    container.innerHTML = `
        <div class="stash-copilot-sidebar-tags">
            <div class="stash-copilot-sidebar-header">
                <button class="stash-copilot-btn stash-copilot-btn-primary stash-copilot-suggest-tags-btn">
                    Suggest Tags
                </button>
                <button class="stash-copilot-btn stash-copilot-btn-secondary stash-copilot-clear-dismissed-btn">
                    Clear Dismissed
                </button>
            </div>
            <div class="stash-copilot-tags-content">
                <div class="stash-copilot-tags-placeholder">
                    Click "Suggest Tags" to analyze this scene
                </div>
            </div>
        </div>
    `;

    // Suggest Tags button handler
    const suggestBtn = container.querySelector('.stash-copilot-suggest-tags-btn');
    suggestBtn.addEventListener('click', () => {
        runTagSuggestions(container, sceneId);
    });

    // Clear Dismissed button handler
    const clearBtn = container.querySelector('.stash-copilot-clear-dismissed-btn');
    clearBtn.addEventListener('click', async () => {
        await runPluginTask('clear_dismissed_tags', { scene_id: sceneId });
        showToast('Dismissed tags cleared');
    });
}
```

**Step 5: Implement runTagSuggestions()**

```javascript
async function runTagSuggestions(container, sceneId) {
    const contentEl = container.querySelector('.stash-copilot-tags-content');
    const requestId = `tags_${sceneId}_${Date.now()}`;

    // Show loading state
    contentEl.innerHTML = `
        <div class="stash-copilot-loading">
            <div class="stash-copilot-spinner"></div>
            <span>Analyzing scene...</span>
        </div>
    `;

    try {
        await runPluginTask('get_tag_suggestions', {
            scene_id: sceneId,
            request_id: requestId,
        });

        // Poll for results
        pollTagSuggestions(contentEl, sceneId, requestId);
    } catch (error) {
        contentEl.innerHTML = `
            <div class="stash-copilot-error">
                Error: ${error.message}
            </div>
        `;
    }
}

async function pollTagSuggestions(contentEl, sceneId, requestId) {
    const maxAttempts = 30;
    const pollInterval = 1000;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const response = await fetch(
                `/plugin/stash-copilot/assets/tag_suggestions_${requestId}.json`
            );

            if (response.ok) {
                const data = await response.json();
                if (data.status === 'complete') {
                    tagSuggestionState.suggestions = data.suggestions;
                    tagSuggestionState.currentPage = 0;
                    renderTagSuggestions(contentEl, sceneId);
                    return;
                } else if (data.status === 'error' || data.status === 'no_embeddings') {
                    contentEl.innerHTML = `
                        <div class="stash-copilot-error">
                            ${data.error || 'Unknown error'}
                        </div>
                    `;
                    return;
                }
            }
        } catch (e) {
            // File not ready yet
        }

        await new Promise(r => setTimeout(r, pollInterval));
    }

    contentEl.innerHTML = `
        <div class="stash-copilot-error">
            Timeout waiting for results
        </div>
    `;
}
```

**Step 6: Implement renderTagSuggestions()**

```javascript
function renderTagSuggestions(contentEl, sceneId) {
    const { suggestions, currentPage, suggestionsPerPage } = tagSuggestionState;
    const totalPages = Math.ceil(suggestions.length / suggestionsPerPage);
    const start = currentPage * suggestionsPerPage;
    const pageSuggestions = suggestions.slice(start, start + suggestionsPerPage);

    if (suggestions.length === 0) {
        contentEl.innerHTML = `
            <div class="stash-copilot-tags-empty">
                No tag suggestions found for this scene
            </div>
        `;
        return;
    }

    let html = '<div class="stash-copilot-suggestions-list">';

    for (const suggestion of pageSuggestions) {
        const scorePercent = Math.round(suggestion.max_similarity * 100);
        const scoreClass = scorePercent >= 70 ? 'high-confidence' : '';

        html += `
            <div class="stash-copilot-suggestion-card" data-tag-id="${suggestion.tag_id}">
                <div class="stash-copilot-suggestion-header">
                    <span class="stash-copilot-tag-name">${suggestion.tag_name.toUpperCase()}</span>
                    <span class="stash-copilot-tag-score ${scoreClass}">${scorePercent}%</span>
                </div>
                <div class="stash-copilot-evidence-frames">
                    ${suggestion.evidence_frames.map(frame => `
                        <div class="stash-copilot-evidence-frame"
                             data-timestamp="${frame.timestamp}"
                             title="${frame.timestamp}">
                            <img src="/plugin/stash-copilot/${frame.thumbnail_path}"
                                 alt="Frame ${frame.frame_index}"
                                 onerror="this.src='/assets/missing-thumbnail.png'">
                            <span class="frame-similarity">${Math.round(frame.similarity * 100)}%</span>
                        </div>
                    `).join('')}
                </div>
                <div class="stash-copilot-suggestion-meta">
                    ${suggestion.frame_count} frames matched
                </div>
                <div class="stash-copilot-suggestion-actions">
                    <button class="stash-copilot-btn stash-copilot-btn-apply"
                            data-scene-id="${sceneId}"
                            data-tag-id="${suggestion.tag_id}">
                        ✓ Apply
                    </button>
                    <button class="stash-copilot-btn stash-copilot-btn-dismiss"
                            data-scene-id="${sceneId}"
                            data-tag-id="${suggestion.tag_id}">
                        ✕ Dismiss
                    </button>
                </div>
            </div>
        `;
    }

    html += '</div>';

    // Pagination
    if (totalPages > 1) {
        html += `
            <div class="stash-copilot-pagination">
                <button class="stash-copilot-btn stash-copilot-btn-prev"
                        ${currentPage === 0 ? 'disabled' : ''}>◀</button>
                <span>${currentPage + 1} / ${totalPages}</span>
                <button class="stash-copilot-btn stash-copilot-btn-next"
                        ${currentPage >= totalPages - 1 ? 'disabled' : ''}>▶</button>
            </div>
        `;
    }

    contentEl.innerHTML = html;

    // Add event listeners
    setupTagSuggestionEvents(contentEl, sceneId);
}
```

**Step 7: Implement event handlers**

```javascript
function setupTagSuggestionEvents(contentEl, sceneId) {
    // Apply button handlers
    contentEl.querySelectorAll('.stash-copilot-btn-apply').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const tagId = e.target.dataset.tagId;
            btn.disabled = true;
            btn.innerHTML = '...';

            await runPluginTask('apply_suggested_tag', {
                scene_id: sceneId,
                tag_id: tagId,
            });

            // Remove card with animation
            const card = e.target.closest('.stash-copilot-suggestion-card');
            card.style.opacity = '0';
            card.style.transform = 'translateX(20px)';
            setTimeout(() => {
                // Remove from state and re-render
                tagSuggestionState.suggestions = tagSuggestionState.suggestions
                    .filter(s => s.tag_id !== parseInt(tagId));
                renderTagSuggestions(contentEl, sceneId);
            }, 300);

            showToast('Tag applied');
        });
    });

    // Dismiss button handlers
    contentEl.querySelectorAll('.stash-copilot-btn-dismiss').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const tagId = e.target.dataset.tagId;
            btn.disabled = true;

            await runPluginTask('dismiss_suggested_tag', {
                scene_id: sceneId,
                tag_id: tagId,
            });

            // Remove card with animation
            const card = e.target.closest('.stash-copilot-suggestion-card');
            card.style.opacity = '0';
            card.style.transform = 'translateX(-20px)';
            setTimeout(() => {
                tagSuggestionState.suggestions = tagSuggestionState.suggestions
                    .filter(s => s.tag_id !== parseInt(tagId));
                renderTagSuggestions(contentEl, sceneId);
            }, 300);

            showToast('Tag dismissed');
        });
    });

    // Pagination handlers
    const prevBtn = contentEl.querySelector('.stash-copilot-btn-prev');
    const nextBtn = contentEl.querySelector('.stash-copilot-btn-next');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            tagSuggestionState.currentPage--;
            renderTagSuggestions(contentEl, sceneId);
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            tagSuggestionState.currentPage++;
            renderTagSuggestions(contentEl, sceneId);
        });
    }

    // Evidence frame click -> seek video
    contentEl.querySelectorAll('.stash-copilot-evidence-frame').forEach(frame => {
        frame.addEventListener('click', () => {
            const timestamp = frame.dataset.timestamp;
            seekVideoToTimestamp(timestamp);
        });
    });
}

function seekVideoToTimestamp(timestamp) {
    // Parse "MM:SS" format
    const [minutes, seconds] = timestamp.split(':').map(Number);
    const totalSeconds = minutes * 60 + seconds;

    // Find video player and seek
    const video = document.querySelector('video');
    if (video) {
        video.currentTime = totalSeconds;
    }
}
```

**Step 8: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(ui): add Tags tab with suggestion cards and evidence frames"
```

---

## Task 7: Add Tags Tab CSS Styling

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add Tags tab styles**

```css
/* Tags Tab - Cyan theme */
.stash-copilot-sidebar-tags {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
}

.stash-copilot-sidebar-tags .stash-copilot-sidebar-header {
    display: flex;
    gap: 8px;
}

.stash-copilot-suggest-tags-btn {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: white;
    flex: 1;
}

.stash-copilot-suggest-tags-btn:hover {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
}

.stash-copilot-clear-dismissed-btn {
    background: rgba(255, 255, 255, 0.1);
    color: #9ca3af;
}

/* Suggestion Cards */
.stash-copilot-suggestions-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.stash-copilot-suggestion-card {
    background: rgba(6, 182, 212, 0.05);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 8px;
    padding: 12px;
    transition: all 0.3s ease;
}

.stash-copilot-suggestion-card:hover {
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.15);
}

.stash-copilot-suggestion-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.stash-copilot-tag-name {
    font-weight: 600;
    font-size: 14px;
    color: #e5e7eb;
    letter-spacing: 0.5px;
}

.stash-copilot-tag-score {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}

.stash-copilot-tag-score.high-confidence {
    animation: pulse-cyan 2s ease-in-out infinite;
}

@keyframes pulse-cyan {
    0%, 100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(6, 182, 212, 0); }
}

/* Evidence Frames */
.stash-copilot-evidence-frames {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
    overflow-x: auto;
}

.stash-copilot-evidence-frame {
    position: relative;
    width: 60px;
    height: 40px;
    flex-shrink: 0;
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.stash-copilot-evidence-frame:hover {
    transform: scale(1.1);
    z-index: 1;
}

.stash-copilot-evidence-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.stash-copilot-evidence-frame .frame-similarity {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.7);
    color: #06b6d4;
    font-size: 10px;
    text-align: center;
    padding: 1px 0;
}

/* Meta info */
.stash-copilot-suggestion-meta {
    font-size: 11px;
    color: #6b7280;
    margin-bottom: 8px;
}

/* Action buttons */
.stash-copilot-suggestion-actions {
    display: flex;
    gap: 8px;
}

.stash-copilot-btn-apply {
    flex: 1;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
}

.stash-copilot-btn-apply:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
}

.stash-copilot-btn-dismiss {
    flex: 1;
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s ease;
}

.stash-copilot-btn-dismiss:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.5);
}

/* Placeholder and empty states */
.stash-copilot-tags-placeholder,
.stash-copilot-tags-empty {
    text-align: center;
    color: #6b7280;
    padding: 24px;
    font-size: 13px;
}

/* Pagination */
.stash-copilot-pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    color: #9ca3af;
    font-size: 12px;
}

.stash-copilot-btn-prev,
.stash-copilot-btn-next {
    background: rgba(6, 182, 212, 0.1);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
}

.stash-copilot-btn-prev:disabled,
.stash-copilot-btn-next:disabled {
    opacity: 0.3;
    cursor: not-allowed;
}
```

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(css): add Tags tab styling with cyan theme"
```

---

## Task 8: Integration Testing

**Files:**
- Test: Playwright MCP via manual testing

**Step 1: Navigate to a scene with embeddings**

Use Playwright MCP to:
1. Navigate to a scene that has been embedded
2. Click on the Tags tab
3. Click "Suggest Tags"
4. Verify suggestions appear with thumbnails

**Step 2: Test Apply and Dismiss**

1. Click "Apply" on a suggestion
2. Verify the card animates out
3. Verify the tag appears in scene's tag list
4. Click "Dismiss" on another suggestion
5. Verify it disappears and doesn't return on refresh

**Step 3: Test edge cases**

1. Scene without embeddings - verify error message
2. Scene with no matching tags - verify empty message
3. Clear dismissed - verify dismissed tags can reappear

**Step 4: Check logs for errors**

```bash
tail -50 ~/.stash/stash.log | grep -i "error\|warn\|tag"
```

**Step 5: Take screenshot of working feature**

Save to `tests/screenshots/tag-suggestions-working.png`

**Step 6: Commit test results**

```bash
git add tests/screenshots/
git commit -m "test: add integration test screenshots for tag suggestions"
```

---

## Task 9: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add Tags Tab section to Scene Page UI Architecture**

Under the existing tab documentation, add:

```markdown
### Tag Suggestions Tab

**Purpose:** Suggest tags for scenes using frame-to-tag embedding similarity.

**Algorithm:**
1. Compare each frame embedding against all tag embeddings (cosine similarity)
2. Each frame "votes" for tags it matches (similarity ≥ 0.30)
3. Rank tags by vote count, then max similarity
4. Show top 20 suggestions with evidence frames

**UI Components:**
- Suggest Tags button - triggers computation
- Suggestion cards with evidence thumbnails
- Apply/Dismiss buttons per suggestion
- Pagination (5 per page)

**Storage:**
- `dismissed_tag_suggestions` table tracks per-scene dismissals

**Plugin Modes:**
- `get_tag_suggestions` - compute suggestions
- `apply_suggested_tag` - add tag to scene
- `dismiss_suggested_tag` - record dismissal
- `clear_dismissed_tags` - clear dismissals
```

**Step 2: Update Architecture Diagram**

Add tag suggestions to the Tasks section.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add tag suggestions documentation to CLAUDE.md"
```

---

## Summary

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1 | Storage schema for dismissed tags | 15 min |
| 2 | TagSuggestion types | 10 min |
| 3 | Similarity computation | 20 min |
| 4 | Full suggestion pipeline | 30 min |
| 5 | Plugin task modes | 15 min |
| 6 | Tags tab JavaScript UI | 45 min |
| 7 | Tags tab CSS styling | 20 min |
| 8 | Integration testing | 20 min |
| 9 | Documentation update | 10 min |

**Total estimated time:** ~3 hours
