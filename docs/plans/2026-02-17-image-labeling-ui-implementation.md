# Image Labeling UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a human-in-the-loop annotation UI for labeling frame images with tags, using uncertainty sampling (active learning), to create training datasets for fine-tuning OpenCLIP.

**Architecture:** Frontend-heavy preload — Python backend prepares batches of 200 uncertain frames with suggested tags, JS frontend handles all labeling interaction locally, syncs annotations back periodically. Export to WebDataset `.tar` format.

**Tech Stack:** Python (numpy, SQLite, tarfile), JavaScript (vanilla, injected into Stash SPA), CSS (custom properties theming)

**Design Doc:** `docs/plans/2026-02-17-image-labeling-ui-design.md`

---

## Task 1: Database Schema Migration (v12)

Add three new tables for labeling sessions, annotations, and progress tracking.

**Files:**
- Modify: `stash_ai/embeddings/storage.py:127` (bump SCHEMA_VERSION to 12)
- Modify: `stash_ai/embeddings/storage.py:205` (add migration call)
- Modify: `stash_ai/embeddings/storage.py:663` (add `_migrate_to_v12` after v11)
- Test: `tests/tasks/test_labeling.py`

**Step 1: Write the failing test**

```python
# tests/tasks/test_labeling.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: FAIL (tables don't exist yet)

**Step 3: Implement the migration**

In `stash_ai/embeddings/storage.py`:

1. Change `SCHEMA_VERSION = 11` → `SCHEMA_VERSION = 12`
2. After `if current_version < 11:` block, add:
   ```python
   if current_version < 12:
       self._migrate_to_v12(cursor)
   ```
3. Add migration method after `_migrate_to_v11`:
   ```python
   def _migrate_to_v12(self, cursor: sqlite3.Cursor) -> None:
       """Add labeling tables (v12)."""
       cursor.execute(
           """
           CREATE TABLE IF NOT EXISTS labeling_sessions (
               session_id TEXT PRIMARY KEY,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'active',
               sampling_method TEXT NOT NULL,
               batch_size INTEGER NOT NULL,
               total_frames INTEGER NOT NULL,
               labeled_count INTEGER NOT NULL DEFAULT 0,
               skipped_count INTEGER NOT NULL DEFAULT 0,
               config_json TEXT
           )
           """
       )
       cursor.execute(
           """
           CREATE TABLE IF NOT EXISTS frame_annotations (
               annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL REFERENCES labeling_sessions(session_id),
               scene_id INTEGER NOT NULL,
               frame_index INTEGER NOT NULL,
               image_source TEXT NOT NULL DEFAULT 'extracted_frame',
               tag_text TEXT NOT NULL,
               tag_source TEXT NOT NULL,
               label TEXT NOT NULL,
               similarity_score REAL,
               labeled_at TEXT NOT NULL,
               UNIQUE(session_id, scene_id, frame_index, tag_text)
           )
           """
       )
       cursor.execute(
           """
           CREATE INDEX IF NOT EXISTS idx_annotations_session
           ON frame_annotations(session_id)
           """
       )
       cursor.execute(
           """
           CREATE INDEX IF NOT EXISTS idx_annotations_frame
           ON frame_annotations(scene_id, frame_index)
           """
       )
       cursor.execute(
           """
           CREATE TABLE IF NOT EXISTS labeling_progress (
               scene_id INTEGER NOT NULL,
               frame_index INTEGER NOT NULL,
               image_source TEXT NOT NULL DEFAULT 'extracted_frame',
               session_id TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               PRIMARY KEY (scene_id, frame_index, session_id)
           )
           """
       )
   ```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add stash_ai/embeddings/storage.py tests/tasks/test_labeling.py
git commit -m "feat(labeling): add database schema migration v12 for labeling tables"
```

---

## Task 2: Storage Methods for Labeling

Add CRUD methods to `EmbeddingStorage` for labeling sessions, annotations, and progress.

**Files:**
- Modify: `stash_ai/embeddings/storage.py` (add methods at end of class)
- Test: `tests/tasks/test_labeling.py` (extend)

**Step 1: Write failing tests for session management**

```python
# Append to tests/tasks/test_labeling.py

class TestLabelingSessionStorage:
    """Tests for labeling session CRUD."""

    def test_create_session(self, storage):
        """Create a labeling session and retrieve it."""
        session_id = storage.create_labeling_session(
            sampling_method="uncertainty",
            batch_size=200,
            total_frames=200,
        )
        assert session_id is not None
        session = storage.get_labeling_session(session_id)
        assert session is not None
        assert session["status"] == "active"
        assert session["batch_size"] == 200

    def test_update_session_counts(self, storage):
        """Update labeled/skipped counts."""
        session_id = storage.create_labeling_session(
            sampling_method="uncertainty",
            batch_size=100,
            total_frames=100,
        )
        storage.update_labeling_session(session_id, labeled_count=5, skipped_count=2)
        session = storage.get_labeling_session(session_id)
        assert session["labeled_count"] == 5
        assert session["skipped_count"] == 2

    def test_list_active_sessions(self, storage):
        """List only active sessions."""
        sid1 = storage.create_labeling_session("uncertainty", 100, 100)
        sid2 = storage.create_labeling_session("uncertainty", 100, 100)
        storage.update_labeling_session(sid1, status="completed")
        active = storage.list_labeling_sessions(status="active")
        assert len(active) == 1
        assert active[0]["session_id"] == sid2


class TestAnnotationStorage:
    """Tests for frame annotation storage."""

    def test_save_and_retrieve_annotations(self, storage):
        """Save annotations and retrieve them."""
        session_id = storage.create_labeling_session("uncertainty", 100, 100)
        annotations = [
            {
                "scene_id": 1,
                "frame_index": 10,
                "image_source": "extracted_frame",
                "tag_text": "blowjob",
                "tag_source": "suggested",
                "label": "confirmed",
                "similarity_score": 0.32,
            },
            {
                "scene_id": 1,
                "frame_index": 10,
                "image_source": "extracted_frame",
                "tag_text": "brunette",
                "tag_source": "suggested",
                "label": "rejected",
                "similarity_score": 0.28,
            },
        ]
        storage.save_annotations(session_id, annotations)
        saved = storage.get_annotations(session_id)
        assert len(saved) == 2
        confirmed = [a for a in saved if a["label"] == "confirmed"]
        assert len(confirmed) == 1
        assert confirmed[0]["tag_text"] == "blowjob"

    def test_get_all_confirmed_annotations(self, storage):
        """Get all confirmed annotations across sessions."""
        sid1 = storage.create_labeling_session("uncertainty", 100, 100)
        sid2 = storage.create_labeling_session("uncertainty", 100, 100)
        storage.save_annotations(sid1, [
            {"scene_id": 1, "frame_index": 10, "image_source": "extracted_frame",
             "tag_text": "blowjob", "tag_source": "suggested", "label": "confirmed",
             "similarity_score": 0.32},
        ])
        storage.save_annotations(sid2, [
            {"scene_id": 2, "frame_index": 20, "image_source": "extracted_frame",
             "tag_text": "POV", "tag_source": "manual", "label": "confirmed",
             "similarity_score": None},
        ])
        confirmed = storage.get_all_confirmed_annotations()
        assert len(confirmed) == 2

    def test_get_labeled_frames(self, storage):
        """Get set of already-labeled frame identifiers."""
        sid = storage.create_labeling_session("uncertainty", 100, 100)
        storage.save_annotations(sid, [
            {"scene_id": 1, "frame_index": 10, "image_source": "extracted_frame",
             "tag_text": "blowjob", "tag_source": "suggested", "label": "confirmed",
             "similarity_score": 0.32},
        ])
        storage.update_labeling_progress(sid, scene_id=1, frame_index=10, status="labeled")
        labeled = storage.get_labeled_frame_keys()
        assert (1, 10) in labeled

    def test_get_unembedded_manual_tags(self, storage):
        """Find manual tags that don't have CLIP embeddings yet."""
        sid = storage.create_labeling_session("uncertainty", 100, 100)
        storage.save_annotations(sid, [
            {"scene_id": 1, "frame_index": 10, "image_source": "extracted_frame",
             "tag_text": "reverse cowgirl", "tag_source": "manual", "label": "confirmed",
             "similarity_score": None},
        ])
        unembedded = storage.get_unembedded_manual_tags("test")
        assert "reverse cowgirl" in unembedded
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: FAIL (methods don't exist)

**Step 3: Implement storage methods**

Add to `EmbeddingStorage` class in `storage.py`:

```python
# --- Labeling Session Methods ---

def create_labeling_session(
    self,
    sampling_method: str,
    batch_size: int,
    total_frames: int,
    config_json: str | None = None,
) -> str:
    """Create a new labeling session. Returns session_id."""
    import uuid
    from datetime import datetime, timezone

    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = self._get_connection()
    conn.execute(
        """INSERT INTO labeling_sessions
        (session_id, created_at, updated_at, status, sampling_method,
         batch_size, total_frames, config_json)
        VALUES (?, ?, ?, 'active', ?, ?, ?, ?)""",
        (session_id, now, now, sampling_method, batch_size, total_frames, config_json),
    )
    conn.commit()
    conn.close()
    return session_id

def get_labeling_session(self, session_id: str) -> dict[str, Any] | None:
    """Retrieve a labeling session by ID."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM labeling_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_labeling_session(self, session_id: str, **kwargs: Any) -> None:
    """Update labeling session fields."""
    from datetime import datetime, timezone

    allowed = {"status", "labeled_count", "skipped_count", "config_json"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [session_id]

    conn = self._get_connection()
    conn.execute(
        f"UPDATE labeling_sessions SET {set_clause} WHERE session_id = ?",
        values,
    )
    conn.commit()
    conn.close()

def list_labeling_sessions(self, status: str | None = None) -> list[dict[str, Any]]:
    """List labeling sessions, optionally filtered by status."""
    conn = self._get_connection()
    cursor = conn.cursor()
    if status:
        cursor.execute(
            "SELECT * FROM labeling_sessions WHERE status = ? ORDER BY created_at DESC",
            (status,),
        )
    else:
        cursor.execute("SELECT * FROM labeling_sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Annotation Methods ---

def save_annotations(self, session_id: str, annotations: list[dict[str, Any]]) -> None:
    """Bulk save frame annotations."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn = self._get_connection()
    for ann in annotations:
        conn.execute(
            """INSERT OR REPLACE INTO frame_annotations
            (session_id, scene_id, frame_index, image_source, tag_text,
             tag_source, label, similarity_score, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                ann["scene_id"],
                ann["frame_index"],
                ann.get("image_source", "extracted_frame"),
                ann["tag_text"],
                ann["tag_source"],
                ann["label"],
                ann.get("similarity_score"),
                now,
            ),
        )
    conn.commit()
    conn.close()

def get_annotations(self, session_id: str) -> list[dict[str, Any]]:
    """Get all annotations for a session."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM frame_annotations WHERE session_id = ?",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_confirmed_annotations(self) -> list[dict[str, Any]]:
    """Get all confirmed annotations across all sessions."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM frame_annotations WHERE label = 'confirmed' ORDER BY scene_id, frame_index"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Progress Methods ---

def update_labeling_progress(
    self, session_id: str, scene_id: int, frame_index: int, status: str
) -> None:
    """Update labeling progress for a frame."""
    conn = self._get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO labeling_progress
        (scene_id, frame_index, image_source, session_id, status)
        VALUES (?, ?, 'extracted_frame', ?, ?)""",
        (scene_id, frame_index, session_id, status),
    )
    conn.commit()
    conn.close()

def get_labeled_frame_keys(self) -> set[tuple[int, int]]:
    """Get set of (scene_id, frame_index) tuples that have been labeled in any session."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT scene_id, frame_index FROM labeling_progress WHERE status = 'labeled'"
    )
    keys = {(row["scene_id"], row["frame_index"]) for row in cursor.fetchall()}
    conn.close()
    return keys

def get_unembedded_manual_tags(self, model_key: str) -> list[str]:
    """Find manual tags from labeling that don't have CLIP embeddings."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT DISTINCT fa.tag_text
        FROM frame_annotations fa
        WHERE fa.tag_source = 'manual'
        AND fa.tag_text NOT IN (
            SELECT text FROM tag_embeddings WHERE model_key = ?
        )""",
        (model_key,),
    )
    tags = [row["tag_text"] for row in cursor.fetchall()]
    conn.close()
    return tags
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add stash_ai/embeddings/storage.py tests/tasks/test_labeling.py
git commit -m "feat(labeling): add storage CRUD methods for sessions, annotations, progress"
```

---

## Task 3: Labeling Types

Define typed dataclasses for the labeling task input/output.

**Files:**
- Create: `stash_ai/tasks/labeling_types.py`

**Step 1: Create types file**

```python
# stash_ai/tasks/labeling_types.py
"""Type definitions for the image labeling task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class FrameSuggestion(TypedDict):
    """A suggested tag for a specific frame."""

    tag_text: str
    tag_source: str  # "stash_tag" | "curated" | "user"
    similarity: float


class LabelingFrameItem(TypedDict):
    """A single frame in the labeling batch."""

    scene_id: int
    frame_index: int
    frame_path: str  # Relative path: "assets/embedded_frames/scene_X/frame_Y.jpg"
    timestamp: str  # "MM:SS" format
    uncertainty_score: float
    suggested_tags: list[FrameSuggestion]
    scene_tags: list[str]  # Existing tags on parent scene
    scene_title: str


class LabelingSessionResult(TypedDict):
    """Result from PrepareSession task."""

    status: str  # "complete" | "error" | "no_embeddings"
    session_id: str
    batch: list[LabelingFrameItem]
    vocabulary: list[str]  # Full tag vocabulary for autocomplete
    error: str | None


class AnnotationPayload(TypedDict):
    """Payload sent from JS to sync annotations."""

    session_id: str
    annotations: list[dict[str, Any]]  # List of annotation dicts
    progress: list[dict[str, Any]]  # List of {scene_id, frame_index, status}


class ExportResult(TypedDict):
    """Result from ExportDataset task."""

    status: str  # "complete" | "error"
    export_path: str
    total_images: int
    total_tags: int
    error: str | None


@dataclass
class LabelingConfig:
    """Configuration for labeling sessions."""

    batch_size: int = 200
    uncertainty_low: float = 0.25
    uncertainty_high: float = 0.35
    max_suggested_tags: int = 10
    caption_template: str = "a scene featuring {tags}"

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> LabelingConfig:
        """Create from raw Stash plugin settings."""
        return cls(
            batch_size=int(settings.get("label_batch_size", 200)),
            uncertainty_low=float(settings.get("label_uncertainty_low", 0.25)),
            uncertainty_high=float(settings.get("label_uncertainty_high", 0.35)),
            max_suggested_tags=int(settings.get("label_suggested_tags", 10)),
            caption_template=settings.get(
                "label_caption_template", "a scene featuring {tags}"
            ),
        )
```

**Step 2: Commit**

```bash
git add stash_ai/tasks/labeling_types.py
git commit -m "feat(labeling): add type definitions for labeling task"
```

---

## Task 4: PrepareSession Task — Uncertainty Sampling

Core backend task that selects frames by uncertainty and prepares the batch JSON.

**Files:**
- Create: `stash_ai/tasks/labeling.py`
- Test: `tests/tasks/test_labeling.py` (extend)

**Step 1: Write failing test for uncertainty scoring**

```python
# Append to tests/tasks/test_labeling.py

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_labeling.py::TestUncertaintySampling -v`
Expected: FAIL (module doesn't exist)

**Step 3: Implement LabelingTask**

```python
# stash_ai/tasks/labeling.py
"""Image labeling task — uncertainty sampling and session management.

Prepares batches of frames ranked by uncertainty for human annotation.
Uses frame-to-tag similarity to identify frames where the model is
least confident, maximizing the value of each human label.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stashapi.stashapp import StashInterface

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.tasks.labeling_types import (
    FrameSuggestion,
    LabelingConfig,
    LabelingFrameItem,
    LabelingSessionResult,
)


class LabelingTask:
    """Prepare labeling sessions with uncertainty-sampled frames."""

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

    def _compute_uncertainty(
        self,
        frame_similarities: NDArray[np.float32],
        low: float,
        high: float,
    ) -> int:
        """Count how many tags fall in the confusion zone for a single frame.

        Args:
            frame_similarities: (T,) array of similarities to each tag
            low: Lower bound of confusion zone
            high: Upper bound of confusion zone

        Returns:
            Number of tags in the confusion zone
        """
        return int(np.sum((frame_similarities >= low) & (frame_similarities <= high)))

    def _rank_by_uncertainty(
        self,
        similarities: NDArray[np.float32],
        frame_keys: list[tuple[int, int]],
        low: float,
        high: float,
        limit: int,
    ) -> list[tuple[int, int]]:
        """Rank frames by uncertainty score and return top `limit`.

        Args:
            similarities: (N, T) matrix of frame-to-tag similarities
            frame_keys: List of (scene_id, frame_index) for each row
            low: Lower bound of confusion zone
            high: Upper bound of confusion zone
            limit: Max frames to return

        Returns:
            List of (scene_id, frame_index) sorted by uncertainty descending
        """
        scores = []
        for i in range(similarities.shape[0]):
            score = self._compute_uncertainty(similarities[i], low, high)
            scores.append((score, i))

        # Sort by uncertainty descending, then by index for stability
        scores.sort(key=lambda x: (-x[0], x[1]))

        return [frame_keys[idx] for _, idx in scores[:limit]]

    def _get_suggested_tags(
        self,
        frame_sims: NDArray[np.float32],
        tag_info: list[dict[str, str]],
        max_tags: int,
    ) -> list[FrameSuggestion]:
        """Get top suggested tags for a single frame, ordered by similarity.

        Args:
            frame_sims: (T,) similarities to each tag
            tag_info: List of {"text": str, "source": str} for each tag
            max_tags: Maximum suggestions to return

        Returns:
            Top tags sorted by similarity descending
        """
        indexed = [(float(frame_sims[i]), i) for i in range(len(tag_info))]
        indexed.sort(key=lambda x: -x[0])

        suggestions: list[FrameSuggestion] = []
        for sim, idx in indexed[:max_tags]:
            suggestions.append(
                FrameSuggestion(
                    tag_text=tag_info[idx]["text"],
                    tag_source=tag_info[idx]["source"],
                    similarity=round(sim, 4),
                )
            )
        return suggestions

    def _get_scene_tags(self, scene_id: int) -> list[str]:
        """Get existing tags for a scene from Stash."""
        try:
            scene = self.stash.find_scene(scene_id)
            if scene and "tags" in scene:
                return [t["name"] for t in scene["tags"]]
        except Exception:
            pass
        return []

    def _get_scene_title(self, scene_id: int) -> str:
        """Get scene title from Stash."""
        try:
            scene = self.stash.find_scene(scene_id)
            if scene:
                title = scene.get("title", "")
                if not title and scene.get("files"):
                    path = scene["files"][0].get("path", "")
                    title = Path(path).stem
                return title or f"Scene {scene_id}"
        except Exception:
            pass
        return f"Scene {scene_id}"

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}:{secs:02d}"

    def prepare_session(self, config: LabelingConfig) -> LabelingSessionResult:
        """Prepare a labeling session with uncertainty-sampled frames.

        Args:
            config: Labeling configuration

        Returns:
            LabelingSessionResult with batch of frames and metadata
        """
        self.log("Preparing labeling session...", "info")

        # 1. Load all frame embeddings across all scenes
        self.log("Loading frame embeddings...", "info")
        conn = self.storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT scene_id, frame_index, timestamp, embedding
            FROM frame_embeddings WHERE model_key = ?
            ORDER BY scene_id, frame_index""",
            (self.model_key,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return LabelingSessionResult(
                status="no_embeddings",
                session_id="",
                batch=[],
                vocabulary=[],
                error="No frame embeddings found. Run 'Embed All Scenes' first.",
            )

        self.log(f"Loaded {len(rows)} frame embeddings", "info")

        # 2. Exclude already-labeled frames
        labeled_keys = self.storage.get_labeled_frame_keys()
        self.log(f"Excluding {len(labeled_keys)} already-labeled frames", "info")

        frame_embeddings_list: list[list[float]] = []
        frame_keys: list[tuple[int, int]] = []
        frame_timestamps: dict[tuple[int, int], float] = {}

        for row in rows:
            key = (row["scene_id"], row["frame_index"])
            if key in labeled_keys:
                continue
            frame_embeddings_list.append(
                self.storage._unpack_embedding(row["embedding"])
            )
            frame_keys.append(key)
            frame_timestamps[key] = row["timestamp"]

        if not frame_keys:
            return LabelingSessionResult(
                status="complete",
                session_id="",
                batch=[],
                vocabulary=[],
                error="All frames have been labeled!",
            )

        frame_embeddings = np.array(frame_embeddings_list, dtype=np.float32)
        self.log(f"{len(frame_keys)} unlabeled frames available", "info")

        # 3. Load tag embeddings
        tag_data = self.storage.get_all_tag_embeddings(self.model_key)
        if not tag_data:
            return LabelingSessionResult(
                status="error",
                session_id="",
                batch=[],
                vocabulary=[],
                error="No tag embeddings found. Ensure tag vocabulary is built.",
            )

        tag_info = [{"text": t["text"], "source": t["source"]} for t in tag_data]
        tag_embeddings = np.array(
            [t["embedding"] for t in tag_data], dtype=np.float32
        )

        # 4. Compute similarity matrix
        self.log("Computing frame-tag similarities...", "info")
        frame_norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        tag_norms = np.linalg.norm(tag_embeddings, axis=1, keepdims=True)
        frame_normalized = frame_embeddings / (frame_norms + 1e-8)
        tag_normalized = tag_embeddings / (tag_norms + 1e-8)
        similarities = np.dot(frame_normalized, tag_normalized.T)

        # 5. Rank by uncertainty
        self.log("Ranking frames by uncertainty...", "info")
        selected_keys = self._rank_by_uncertainty(
            similarities,
            frame_keys,
            low=config.uncertainty_low,
            high=config.uncertainty_high,
            limit=config.batch_size,
        )

        # 6. Build batch items
        self.log(f"Building batch of {len(selected_keys)} frames...", "info")

        # Cache scene data to avoid repeated API calls
        scene_tags_cache: dict[int, list[str]] = {}
        scene_title_cache: dict[int, str] = {}

        # Build key-to-index mapping for fast lookup
        key_to_idx = {k: i for i, k in enumerate(frame_keys)}

        plugin_dir = Path(__file__).parent.parent.parent
        batch: list[LabelingFrameItem] = []

        for scene_id, frame_index in selected_keys:
            idx = key_to_idx[(scene_id, frame_index)]

            # Get suggested tags for this frame
            suggested = self._get_suggested_tags(
                similarities[idx], tag_info, config.max_suggested_tags
            )

            # Cache scene metadata
            if scene_id not in scene_tags_cache:
                scene_tags_cache[scene_id] = self._get_scene_tags(scene_id)
                scene_title_cache[scene_id] = self._get_scene_title(scene_id)

            frame_path = str(
                plugin_dir
                / "assets"
                / "embedded_frames"
                / f"scene_{scene_id}"
                / f"frame_{frame_index:04d}.jpg"
            )

            timestamp = frame_timestamps.get((scene_id, frame_index), 0.0)

            batch.append(
                LabelingFrameItem(
                    scene_id=scene_id,
                    frame_index=frame_index,
                    frame_path=frame_path,
                    timestamp=self._format_timestamp(timestamp),
                    uncertainty_score=float(
                        self._compute_uncertainty(
                            similarities[idx],
                            config.uncertainty_low,
                            config.uncertainty_high,
                        )
                    ),
                    suggested_tags=suggested,
                    scene_tags=scene_tags_cache[scene_id],
                    scene_title=scene_title_cache[scene_id],
                )
            )

        # 7. Create session in DB
        session_id = self.storage.create_labeling_session(
            sampling_method="uncertainty",
            batch_size=config.batch_size,
            total_frames=len(batch),
            config_json=json.dumps({
                "uncertainty_low": config.uncertainty_low,
                "uncertainty_high": config.uncertainty_high,
                "max_suggested_tags": config.max_suggested_tags,
                "model_key": self.model_key,
            }),
        )

        # 8. Build vocabulary list for autocomplete
        vocabulary = sorted(set(t["text"] for t in tag_info))

        self.log(
            f"Session {session_id} ready: {len(batch)} frames, "
            f"{len(vocabulary)} vocabulary items",
            "info",
        )

        return LabelingSessionResult(
            status="complete",
            session_id=session_id,
            batch=batch,
            vocabulary=vocabulary,
            error=None,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add stash_ai/tasks/labeling.py tests/tasks/test_labeling.py
git commit -m "feat(labeling): implement PrepareSession with uncertainty sampling"
```

---

## Task 5: SyncAnnotations and ExportDataset Tasks

Add the sync and export methods to `LabelingTask`.

**Files:**
- Modify: `stash_ai/tasks/labeling.py`
- Test: `tests/tasks/test_labeling.py` (extend)

**Step 1: Write failing test for sync**

```python
# Append to tests/tasks/test_labeling.py

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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_labeling.py::TestSyncAnnotations tests/tasks/test_labeling.py::TestExportDataset -v`
Expected: FAIL

**Step 3: Implement sync and export**

Add to `LabelingTask` class in `stash_ai/tasks/labeling.py`:

```python
def sync_annotations(self, payload: dict[str, Any]) -> None:
    """Sync annotations from frontend to storage.

    Args:
        payload: Dict with session_id, annotations list, progress list
    """
    session_id = payload["session_id"]
    annotations = payload.get("annotations", [])
    progress = payload.get("progress", [])

    if annotations:
        self.storage.save_annotations(session_id, annotations)

    for p in progress:
        self.storage.update_labeling_progress(
            session_id,
            scene_id=p["scene_id"],
            frame_index=p["frame_index"],
            status=p["status"],
        )

    # Update session counts
    labeled = sum(1 for p in progress if p["status"] == "labeled")
    skipped = sum(1 for p in progress if p["status"] == "skipped")

    session = self.storage.get_labeling_session(session_id)
    if session:
        self.storage.update_labeling_session(
            session_id,
            labeled_count=session["labeled_count"] + labeled,
            skipped_count=session["skipped_count"] + skipped,
        )

    self.log(
        f"Synced {len(annotations)} annotations, {labeled} labeled, {skipped} skipped",
        "info",
    )

def _generate_caption(self, tags: list[str], template: str) -> str:
    """Generate a caption from confirmed tags.

    Args:
        tags: List of confirmed tag strings
        template: Caption template with {tags} placeholder

    Returns:
        Generated caption string
    """
    if len(tags) == 0:
        return ""
    elif len(tags) == 1:
        tag_str = tags[0]
    elif len(tags) == 2:
        tag_str = f"{tags[0]} and {tags[1]}"
    else:
        tag_str = ", ".join(tags[:-1]) + f", and {tags[-1]}"

    return template.replace("{tags}", tag_str)

def _generate_negative_caption(self, tags: list[str]) -> str:
    """Generate a negative caption from rejected tags."""
    if not tags:
        return ""
    return "not featuring " + ", not featuring ".join(tags)

def export_dataset(
    self,
    config: LabelingConfig,
    output_dir: Path | None = None,
    include_negatives: bool = True,
) -> dict[str, Any]:
    """Export labeled data as WebDataset tar.

    Args:
        config: Labeling configuration (for caption template)
        output_dir: Where to save the tar (defaults to assets/exports/)
        include_negatives: Whether to include negative caption files

    Returns:
        ExportResult dict with status and metadata
    """
    import tarfile
    from datetime import datetime, timezone
    from collections import defaultdict

    self.log("Exporting dataset...", "info")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "assets" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Group confirmed annotations by frame
    all_annotations = self.storage.get_all_confirmed_annotations()
    if not all_annotations:
        return {
            "status": "error",
            "export_path": "",
            "total_images": 0,
            "total_tags": 0,
            "error": "No confirmed annotations to export.",
        }

    # Group by (scene_id, frame_index)
    frame_tags: dict[tuple[int, int], list[str]] = defaultdict(list)
    for ann in all_annotations:
        key = (ann["scene_id"], ann["frame_index"])
        frame_tags[key].append(ann["tag_text"])

    # Also collect rejected tags for negatives
    rejected_tags: dict[tuple[int, int], list[str]] = defaultdict(list)
    if include_negatives:
        conn = self.storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT scene_id, frame_index, tag_text FROM frame_annotations WHERE label = 'rejected'"
        )
        for row in cursor.fetchall():
            key = (row["scene_id"], row["frame_index"])
            rejected_tags[key].append(row["tag_text"])
        conn.close()

    # 2. Build tar
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tar_path = output_dir / f"dataset_{timestamp}.tar"
    plugin_dir = Path(__file__).parent.parent.parent

    tag_counts: dict[str, int] = defaultdict(int)
    total_images = 0
    sessions_included: set[str] = set()

    with tarfile.open(tar_path, "w") as tar:
        for (scene_id, frame_index), tags in frame_tags.items():
            frame_path = (
                plugin_dir
                / "assets"
                / "embedded_frames"
                / f"scene_{scene_id}"
                / f"frame_{frame_index:04d}.jpg"
            )

            if not frame_path.exists():
                self.log(f"Frame not found: {frame_path}", "warning")
                continue

            # Add image
            base_name = f"scene{scene_id}_frame{frame_index:04d}"
            tar.add(str(frame_path), arcname=f"{base_name}.jpg")

            # Add caption
            caption = self._generate_caption(tags, config.caption_template)
            caption_bytes = caption.encode("utf-8")
            import io
            import tarfile as _tf

            caption_info = _tf.TarInfo(name=f"{base_name}.txt")
            caption_info.size = len(caption_bytes)
            tar.addfile(caption_info, io.BytesIO(caption_bytes))

            # Add negative caption if requested
            if include_negatives and (scene_id, frame_index) in rejected_tags:
                neg_caption = self._generate_negative_caption(
                    rejected_tags[(scene_id, frame_index)]
                )
                if neg_caption:
                    neg_bytes = neg_caption.encode("utf-8")
                    neg_info = _tf.TarInfo(name=f"{base_name}_neg.txt")
                    neg_info.size = len(neg_bytes)
                    tar.addfile(neg_info, io.BytesIO(neg_bytes))

            total_images += 1
            for tag in tags:
                tag_counts[tag] += 1

        # Add metadata
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_images": total_images,
            "total_tags": len(tag_counts),
            "caption_template": config.caption_template,
            "include_negatives": include_negatives,
            "tag_stats": dict(tag_counts),
        }
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        meta_info = tarfile.TarInfo(name="metadata.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))

    self.log(f"Exported {total_images} images to {tar_path}", "info")

    return {
        "status": "complete",
        "export_path": str(tar_path),
        "total_images": total_images,
        "total_tags": len(tag_counts),
        "error": None,
    }
```

Add `import io` to the top of the file imports.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_labeling.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add stash_ai/tasks/labeling.py tests/tasks/test_labeling.py
git commit -m "feat(labeling): implement SyncAnnotations and ExportDataset tasks"
```

---

## Task 6: Task Registration and Dispatch

Register new tasks in YAML and wire dispatch in `stash-copilot.py`.

**Files:**
- Modify: `stash-copilot.yml` (add task entries before hooks section)
- Modify: `stash-copilot.py` (add dispatch cases + handler methods)

**Step 1: Add YAML task definitions**

Insert before the `hooks:` section (line 506) in `stash-copilot.yml`:

```yaml
  - name: Prepare Labeling Session
    description: Prepare a batch of frames for labeling using uncertainty sampling
    defaultArgs:
      mode: prepare_labeling_session
      batch_size: "200"
      request_id: ""

  - name: Sync Labeling Annotations
    description: Sync annotation labels from the labeling UI
    defaultArgs:
      mode: sync_labeling_annotations
      request_id: ""
      payload: ""

  - name: Export Labeling Dataset
    description: Export labeled data as WebDataset for training
    defaultArgs:
      mode: export_labeling_dataset
      request_id: ""
      include_negatives: "true"

  - name: Get Labeling Sessions
    description: List labeling sessions with progress stats
    defaultArgs:
      mode: get_labeling_sessions
      request_id: ""
```

**Step 2: Add dispatch cases in `stash-copilot.py`**

Add before the `else: self.error(...)` line (line 373) in `run_task`:

```python
elif task_name == "prepare_labeling_session":
    self.run_prepare_labeling_session(args)
elif task_name == "sync_labeling_annotations":
    self.run_sync_labeling_annotations(args)
elif task_name == "export_labeling_dataset":
    self.run_export_labeling_dataset(args)
elif task_name == "get_labeling_sessions":
    self.run_get_labeling_sessions(args)
```

**Step 3: Add handler methods**

Add these methods to `StashCopilotPlugin` class in `stash-copilot.py`:

```python
def run_prepare_labeling_session(self, args: dict[str, Any]) -> None:
    """Prepare a labeling session with uncertainty-sampled frames."""
    request_id = args.get("request_id", "")
    batch_size = int(args.get("batch_size", 200))

    self.log(f"Preparing labeling session (batch_size={batch_size}), request_id={request_id}", "info")

    try:
        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.storage import EmbeddingStorage
        from stash_ai.embeddings.tag_vocabulary import TagVocabulary
        from stash_ai.tasks.labeling import LabelingTask
        from stash_ai.tasks.labeling_types import LabelingConfig

        plugin_settings = self.get_plugin_settings("stash-copilot")

        # Determine model key
        image_provider = plugin_settings.get("image_embedding_provider")
        image_model = plugin_settings.get("image_embedding_model")
        image_device = plugin_settings.get("image_embedding_device") or "auto"

        if image_provider and image_model:
            embedding_config = EmbeddingConfig(
                provider=image_provider, model=image_model, device=image_device
            )
            model_key = embedding_config.model_key
        else:
            model_key = "siglip"

        storage = EmbeddingStorage(model_key=model_key)

        # Sync tag vocabulary before preparing session
        self.log("Syncing tag vocabulary...", "info")
        tag_vocab = TagVocabulary(
            storage=storage, model_key=model_key, log_callback=self.log
        )
        stash_tags = [t["name"] for t in self.stash.find_tags(f={})]
        tag_vocab.ensure_embeddings(stash_tags=stash_tags)

        # Build labeling config
        config = LabelingConfig(
            batch_size=batch_size,
            uncertainty_low=float(plugin_settings.get("label_uncertainty_low", 0.25)),
            uncertainty_high=float(plugin_settings.get("label_uncertainty_high", 0.35)),
            max_suggested_tags=int(plugin_settings.get("label_suggested_tags", 10)),
            caption_template=plugin_settings.get(
                "label_caption_template", "a scene featuring {tags}"
            ),
        )

        task = LabelingTask(
            stash=self.stash,
            storage=storage,
            log_callback=self.log,
            model_key=model_key,
        )

        result = task.prepare_session(config)

        # Write result JSON
        import json
        from pathlib import Path

        assets_dir = Path(__file__).parent / "assets"
        assets_dir.mkdir(exist_ok=True)
        result_file = assets_dir / f"labeling_session_{request_id}.json"
        result_file.write_text(json.dumps(result, indent=2))

        self.log(f"Labeling session written to {result_file}", "info")

    except Exception as e:
        self.log(f"Error preparing labeling session: {e}", "error")
        import json, traceback
        from pathlib import Path

        assets_dir = Path(__file__).parent / "assets"
        result_file = assets_dir / f"labeling_session_{request_id}.json"
        result_file.write_text(json.dumps({
            "status": "error",
            "session_id": "",
            "batch": [],
            "vocabulary": [],
            "error": str(e),
        }))

def run_sync_labeling_annotations(self, args: dict[str, Any]) -> None:
    """Sync annotations from the labeling UI."""
    request_id = args.get("request_id", "")
    payload_json = args.get("payload", "{}")

    try:
        import json
        from pathlib import Path
        from stash_ai.embeddings.storage import EmbeddingStorage
        from stash_ai.tasks.labeling import LabelingTask

        payload = json.loads(payload_json)
        storage = EmbeddingStorage()

        task = LabelingTask(
            stash=self.stash,
            storage=storage,
            log_callback=self.log,
        )

        task.sync_annotations(payload)

        assets_dir = Path(__file__).parent / "assets"
        result_file = assets_dir / f"labeling_sync_{request_id}.json"
        result_file.write_text(json.dumps({"status": "complete"}))

    except Exception as e:
        self.log(f"Error syncing annotations: {e}", "error")

def run_export_labeling_dataset(self, args: dict[str, Any]) -> None:
    """Export labeled data as WebDataset."""
    request_id = args.get("request_id", "")
    include_negatives = args.get("include_negatives", "true").lower() == "true"

    self.log(f"Exporting labeling dataset, request_id={request_id}", "info")

    try:
        import json
        from pathlib import Path
        from stash_ai.embeddings.storage import EmbeddingStorage
        from stash_ai.tasks.labeling import LabelingTask
        from stash_ai.tasks.labeling_types import LabelingConfig

        plugin_settings = self.get_plugin_settings("stash-copilot")
        storage = EmbeddingStorage()
        config = LabelingConfig.from_plugin_settings(plugin_settings)

        task = LabelingTask(
            stash=self.stash,
            storage=storage,
            log_callback=self.log,
        )

        result = task.export_dataset(config, include_negatives=include_negatives)

        assets_dir = Path(__file__).parent / "assets"
        result_file = assets_dir / f"labeling_export_{request_id}.json"
        result_file.write_text(json.dumps(result, indent=2))

        self.log(f"Export result written to {result_file}", "info")

    except Exception as e:
        self.log(f"Error exporting dataset: {e}", "error")
        import json
        from pathlib import Path

        assets_dir = Path(__file__).parent / "assets"
        result_file = assets_dir / f"labeling_export_{request_id}.json"
        result_file.write_text(json.dumps({
            "status": "error",
            "export_path": "",
            "total_images": 0,
            "total_tags": 0,
            "error": str(e),
        }))

def run_get_labeling_sessions(self, args: dict[str, Any]) -> None:
    """List labeling sessions."""
    request_id = args.get("request_id", "")

    try:
        import json
        from pathlib import Path
        from stash_ai.embeddings.storage import EmbeddingStorage

        storage = EmbeddingStorage()
        sessions = storage.list_labeling_sessions()

        assets_dir = Path(__file__).parent / "assets"
        result_file = assets_dir / f"labeling_sessions_{request_id}.json"
        result_file.write_text(json.dumps({
            "status": "complete",
            "sessions": sessions,
        }, indent=2))

    except Exception as e:
        self.log(f"Error listing sessions: {e}", "error")
```

**Step 4: Commit**

```bash
git add stash-copilot.yml stash-copilot.py
git commit -m "feat(labeling): register tasks in YAML and wire dispatch handlers"
```

---

## Task 7: Frontend — Page Injection and Session Management

Add URL detection for `/plugins/stash-copilot/label` and the page skeleton.

**Files:**
- Modify: `stash-copilot.js` (add route detection, page rendering, session management)

**Step 1: Add route detection**

In `onPageChange()` function (around line 12318), add a new route:

```javascript
} else if (path === '/plugins/stash-copilot/label') {
    setTimeout(renderLabelingPage, 100);
}
```

**Step 2: Add labeling page state**

Add near other state objects:

```javascript
const labelingState = {
    initialized: false,
    sessionId: null,
    batch: [],
    vocabulary: [],
    currentIndex: 0,
    annotations: {},       // key: "sceneId_frameIndex", value: {tags: {tagText: label}}
    pendingSync: [],       // Annotations waiting to be synced
    viewMode: 'single',   // 'single' | 'grid'
    gridSelection: null,   // Currently selected grid item index
    syncTimer: null,
    isLoading: false,
};
```

**Step 3: Implement `renderLabelingPage()`**

This follows the same pattern as `renderSearchPage()` — find main content area and replace with the labeling UI.

```javascript
function renderLabelingPage() {
    log('Rendering labeling page...');

    let mainContent = document.querySelector('.main');
    if (!mainContent) {
        mainContent = document.querySelector('#root > div:last-child');
    }
    if (!mainContent) {
        mainContent = document.querySelector('.container-fluid') || document.querySelector('#root');
    }
    if (!mainContent) {
        log('Could not find main content area', 'error');
        return;
    }

    mainContent.innerHTML = `
        <div class="stash-copilot-label-page">
            <div class="stash-copilot-label-header">
                <div class="stash-copilot-label-header-left">
                    <a href="/scenes" class="stash-copilot-label-back-btn">← Back</a>
                    <h1 class="stash-copilot-label-title">
                        <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
                            <line x1="7" y1="7" x2="7.01" y2="7"/>
                        </svg>
                        Image Labeling
                    </h1>
                </div>
                <div class="stash-copilot-label-header-center">
                    <span class="stash-copilot-label-progress-text">No session</span>
                    <div class="stash-copilot-label-progress-bar">
                        <div class="stash-copilot-label-progress-fill"></div>
                    </div>
                </div>
                <div class="stash-copilot-label-header-right">
                    <button class="stash-copilot-label-view-toggle" data-mode="single" title="Toggle view (G)">
                        <span class="view-single active">▣</span>
                        <span class="view-grid">⊞</span>
                    </button>
                    <button class="stash-copilot-label-export-btn" title="Export dataset">Export</button>
                    <button class="stash-copilot-label-settings-btn" title="Settings">⚙</button>
                </div>
            </div>

            <div class="stash-copilot-label-body">
                <!-- Initial state: session picker -->
                <div class="stash-copilot-label-intro">
                    <div class="stash-copilot-label-intro-icon">🏷️</div>
                    <h2>Start Labeling Session</h2>
                    <p>Label images with tags to create training data for embedding model fine-tuning.
                       Uses uncertainty sampling to show you images where the model needs the most help.</p>
                    <div class="stash-copilot-label-session-controls">
                        <label>Batch size:
                            <input type="number" class="stash-copilot-label-batch-input" value="200" min="10" max="1000" step="10">
                        </label>
                        <button class="stash-copilot-label-start-btn">Start New Session</button>
                    </div>
                    <div class="stash-copilot-label-previous-sessions"></div>
                </div>

                <!-- Loading state -->
                <div class="stash-copilot-label-loading" style="display: none;">
                    <div class="stash-copilot-spinner"></div>
                    <span class="stash-copilot-label-loading-status">Preparing session...</span>
                </div>

                <!-- Single view mode -->
                <div class="stash-copilot-label-single" style="display: none;">
                    <div class="stash-copilot-label-image-area">
                        <img class="stash-copilot-label-image" src="" alt="Frame to label">
                        <div class="stash-copilot-label-image-meta"></div>
                    </div>
                    <div class="stash-copilot-label-tag-panel">
                        <h3 class="stash-copilot-label-section-title">Suggested Tags</h3>
                        <div class="stash-copilot-label-suggestions"></div>
                        <h3 class="stash-copilot-label-section-title">Scene Tags</h3>
                        <div class="stash-copilot-label-scene-tags"></div>
                        <h3 class="stash-copilot-label-section-title">Add Tag</h3>
                        <div class="stash-copilot-label-autocomplete">
                            <input type="text" class="stash-copilot-label-tag-input"
                                   placeholder="Type to search tags... (/)"
                                   autocomplete="off">
                            <div class="stash-copilot-label-autocomplete-dropdown"></div>
                        </div>
                        <div class="stash-copilot-label-manual-tags"></div>
                    </div>
                </div>

                <!-- Grid view mode -->
                <div class="stash-copilot-label-grid" style="display: none;">
                    <div class="stash-copilot-label-grid-images"></div>
                    <div class="stash-copilot-label-tag-panel">
                        <!-- Same tag panel, reused -->
                    </div>
                </div>
            </div>

            <div class="stash-copilot-label-footer" style="display: none;">
                <div class="stash-copilot-label-nav">
                    <button class="stash-copilot-label-prev-btn" title="Previous (←)">← Prev</button>
                    <span class="stash-copilot-label-position">0 / 0</span>
                    <button class="stash-copilot-label-next-btn" title="Next (→)">Next →</button>
                </div>
                <div class="stash-copilot-label-actions">
                    <button class="stash-copilot-label-skip-btn" title="Skip (S)">Skip</button>
                    <button class="stash-copilot-label-save-btn" title="Save & Next (Enter)">Save & Next</button>
                </div>
            </div>
        </div>
    `;

    setupLabelingEvents(mainContent);
    loadPreviousSessions(mainContent);
}
```

**Step 4: Implement session management functions**

```javascript
function setupLabelingEvents(container) {
    // Start new session
    const startBtn = container.querySelector('.stash-copilot-label-start-btn');
    startBtn.addEventListener('click', () => {
        const batchInput = container.querySelector('.stash-copilot-label-batch-input');
        const batchSize = parseInt(batchInput.value, 10) || 200;
        startLabelingSession(container, batchSize);
    });

    // Navigation
    const prevBtn = container.querySelector('.stash-copilot-label-prev-btn');
    const nextBtn = container.querySelector('.stash-copilot-label-next-btn');
    prevBtn.addEventListener('click', () => navigateLabeling(container, -1));
    nextBtn.addEventListener('click', () => navigateLabeling(container, 1));

    // Actions
    const skipBtn = container.querySelector('.stash-copilot-label-skip-btn');
    const saveBtn = container.querySelector('.stash-copilot-label-save-btn');
    skipBtn.addEventListener('click', () => skipFrame(container));
    saveBtn.addEventListener('click', () => saveAndNext(container));

    // View toggle
    const viewToggle = container.querySelector('.stash-copilot-label-view-toggle');
    viewToggle.addEventListener('click', () => toggleViewMode(container));

    // Export
    const exportBtn = container.querySelector('.stash-copilot-label-export-btn');
    exportBtn.addEventListener('click', () => exportDataset(container));

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => handleLabelingKeyboard(e, container));
}

async function startLabelingSession(container, batchSize) {
    const introEl = container.querySelector('.stash-copilot-label-intro');
    const loadingEl = container.querySelector('.stash-copilot-label-loading');

    introEl.style.display = 'none';
    loadingEl.style.display = 'flex';

    const requestId = `label_${Date.now()}`;
    labelingState.isLoading = true;

    try {
        await runPluginTask('Prepare Labeling Session', {
            batch_size: String(batchSize),
            request_id: requestId,
        });

        // Poll for results
        await pollLabelingSession(container, requestId);
    } catch (error) {
        log(`Error starting session: ${error.message}`, 'error');
        loadingEl.style.display = 'none';
        introEl.style.display = 'block';
    }
}

async function pollLabelingSession(container, requestId) {
    const loadingEl = container.querySelector('.stash-copilot-label-loading');
    const statusEl = loadingEl.querySelector('.stash-copilot-label-loading-status');

    const maxAttempts = 120;  // 2 minute timeout
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const resp = await fetch(
                `/plugin/stash-copilot/assets/labeling_session_${requestId}.json?t=${Date.now()}`,
                { cache: 'no-store' }
            );
            if (resp.ok) {
                const data = await resp.json();
                if (data.status === 'complete') {
                    labelingState.sessionId = data.session_id;
                    labelingState.batch = data.batch;
                    labelingState.vocabulary = data.vocabulary;
                    labelingState.currentIndex = 0;
                    labelingState.annotations = {};
                    labelingState.isLoading = false;

                    loadingEl.style.display = 'none';
                    showLabelingUI(container);
                    renderCurrentFrame(container);
                    return;
                } else if (data.status === 'error' || data.status === 'no_embeddings') {
                    loadingEl.style.display = 'none';
                    const introEl = container.querySelector('.stash-copilot-label-intro');
                    introEl.style.display = 'block';
                    alert(data.error || 'Failed to prepare session');
                    return;
                }
                if (statusEl) statusEl.textContent = 'Computing uncertainty scores...';
            }
        } catch (e) {
            // File not ready yet
        }
        await new Promise(r => setTimeout(r, 1000));
    }

    // Timeout
    loadingEl.style.display = 'none';
    const introEl = container.querySelector('.stash-copilot-label-intro');
    introEl.style.display = 'block';
    alert('Session preparation timed out');
}

function showLabelingUI(container) {
    const singleView = container.querySelector('.stash-copilot-label-single');
    const footer = container.querySelector('.stash-copilot-label-footer');

    if (labelingState.viewMode === 'single') {
        singleView.style.display = 'flex';
    }
    footer.style.display = 'flex';
    updateProgress(container);
}
```

**Step 5: Implement frame rendering and tag interaction**

This is the core interaction — rendering the current frame with its tags. See Task 8 for full details.

**Step 6: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(labeling): add page injection, session management, and polling"
```

---

## Task 8: Frontend — Single View Rendering and Tag Interaction

Render the current frame with suggested tags, scene tags, and autocomplete input.

**Files:**
- Modify: `stash-copilot.js` (add rendering functions)

**Step 1: Implement frame rendering**

```javascript
function renderCurrentFrame(container) {
    const item = labelingState.batch[labelingState.currentIndex];
    if (!item) return;

    const imageEl = container.querySelector('.stash-copilot-label-image');
    const metaEl = container.querySelector('.stash-copilot-label-image-meta');
    const suggestionsEl = container.querySelector('.stash-copilot-label-suggestions');
    const sceneTagsEl = container.querySelector('.stash-copilot-label-scene-tags');
    const manualTagsEl = container.querySelector('.stash-copilot-label-manual-tags');

    // Load image — use plugin asset path
    const framePath = item.frame_path.replace(/^.*?assets\//, '/plugin/stash-copilot/assets/');
    imageEl.src = framePath;
    imageEl.alt = `Scene ${item.scene_id} - Frame ${item.frame_index}`;

    // Meta info
    metaEl.innerHTML = `
        <span class="stash-copilot-label-meta-scene">
            <a href="/scenes/${item.scene_id}" target="_blank">${escapeHtml(item.scene_title)}</a>
        </span>
        <span class="stash-copilot-label-meta-time">${item.timestamp}</span>
        <span class="stash-copilot-label-meta-uncertainty" title="Uncertainty score">
            ⚡ ${item.uncertainty_score}
        </span>
    `;

    // Get existing annotations for this frame
    const frameKey = `${item.scene_id}_${item.frame_index}`;
    const existing = labelingState.annotations[frameKey] || {};

    // Render suggested tags
    suggestionsEl.innerHTML = item.suggested_tags.map((tag, idx) => {
        const label = existing[tag.tag_text] || 'undecided';
        return `
            <div class="stash-copilot-label-tag-row" data-tag="${escapeHtml(tag.tag_text)}" data-state="${label}">
                <button class="stash-copilot-label-tag-toggle" data-key="${idx + 1}">
                    <span class="tag-state-icon">${label === 'confirmed' ? '✓' : label === 'rejected' ? '✗' : '?'}</span>
                </button>
                <span class="stash-copilot-label-tag-name">${escapeHtml(tag.tag_text)}</span>
                <span class="stash-copilot-label-tag-sim">${(tag.similarity * 100).toFixed(0)}%</span>
                <span class="stash-copilot-label-tag-key">${idx + 1}</span>
            </div>
        `;
    }).join('');

    // Setup tag toggle events
    suggestionsEl.querySelectorAll('.stash-copilot-label-tag-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const row = btn.closest('.stash-copilot-label-tag-row');
            toggleTagState(row, frameKey);
        });
    });

    // Render scene tags (read-only)
    sceneTagsEl.innerHTML = item.scene_tags.map(tag =>
        `<span class="stash-copilot-label-scene-tag-pill">${escapeHtml(tag)}</span>`
    ).join('');

    // Render manually added tags
    const manualTags = Object.entries(existing)
        .filter(([text, label]) => label === 'confirmed' && !item.suggested_tags.some(s => s.tag_text === text))
        .map(([text]) => text);

    manualTagsEl.innerHTML = manualTags.map(tag =>
        `<span class="stash-copilot-label-manual-tag">
            ${escapeHtml(tag)}
            <button class="stash-copilot-label-remove-tag" data-tag="${escapeHtml(tag)}">×</button>
        </span>`
    ).join('');

    // Update position
    updateProgress(container);
}

function toggleTagState(row, frameKey) {
    const tagText = row.dataset.tag;
    const currentState = row.dataset.state;
    const states = ['undecided', 'confirmed', 'rejected'];
    const nextIdx = (states.indexOf(currentState) + 1) % states.length;
    const newState = states[nextIdx];

    row.dataset.state = newState;
    const icon = row.querySelector('.tag-state-icon');
    icon.textContent = newState === 'confirmed' ? '✓' : newState === 'rejected' ? '✗' : '?';

    // Store annotation
    if (!labelingState.annotations[frameKey]) {
        labelingState.annotations[frameKey] = {};
    }
    labelingState.annotations[frameKey][tagText] = newState;
}

function updateProgress(container) {
    const total = labelingState.batch.length;
    const current = labelingState.currentIndex + 1;
    const labeled = Object.keys(labelingState.annotations).length;
    const pct = total > 0 ? Math.round((labeled / total) * 100) : 0;

    const posEl = container.querySelector('.stash-copilot-label-position');
    const progressText = container.querySelector('.stash-copilot-label-progress-text');
    const progressFill = container.querySelector('.stash-copilot-label-progress-fill');

    if (posEl) posEl.textContent = `${current} / ${total}`;
    if (progressText) progressText.textContent = `Session: ${labeled}/${total} labeled (${pct}%)`;
    if (progressFill) progressFill.style.width = `${pct}%`;
}
```

**Step 2: Implement autocomplete**

```javascript
function setupAutocomplete(container) {
    const input = container.querySelector('.stash-copilot-label-tag-input');
    const dropdown = container.querySelector('.stash-copilot-label-autocomplete-dropdown');

    input.addEventListener('input', () => {
        const query = input.value.trim().toLowerCase();
        if (query.length < 2) {
            dropdown.style.display = 'none';
            return;
        }

        // Fuzzy filter vocabulary
        const matches = labelingState.vocabulary
            .filter(tag => tag.toLowerCase().includes(query))
            .slice(0, 10);

        if (matches.length === 0) {
            // Offer to create new tag
            dropdown.innerHTML = `
                <div class="stash-copilot-label-ac-item stash-copilot-label-ac-new"
                     data-tag="${escapeHtml(input.value.trim())}">
                    + Create: "${escapeHtml(input.value.trim())}"
                </div>
            `;
        } else {
            dropdown.innerHTML = matches.map(tag =>
                `<div class="stash-copilot-label-ac-item" data-tag="${escapeHtml(tag)}">
                    ${escapeHtml(tag)}
                </div>`
            ).join('');

            // Add "create new" option if exact match not found
            if (!matches.some(t => t.toLowerCase() === query)) {
                dropdown.innerHTML += `
                    <div class="stash-copilot-label-ac-item stash-copilot-label-ac-new"
                         data-tag="${escapeHtml(input.value.trim())}">
                        + Create: "${escapeHtml(input.value.trim())}"
                    </div>
                `;
            }
        }

        dropdown.style.display = 'block';

        // Click to add tag
        dropdown.querySelectorAll('.stash-copilot-label-ac-item').forEach(item => {
            item.addEventListener('click', () => {
                const tagText = item.dataset.tag;
                addManualTag(container, tagText);
                input.value = '';
                dropdown.style.display = 'none';
            });
        });
    });

    // Hide dropdown on blur (with delay for click)
    input.addEventListener('blur', () => {
        setTimeout(() => { dropdown.style.display = 'none'; }, 200);
    });
}

function addManualTag(container, tagText) {
    const item = labelingState.batch[labelingState.currentIndex];
    if (!item) return;

    const frameKey = `${item.scene_id}_${item.frame_index}`;
    if (!labelingState.annotations[frameKey]) {
        labelingState.annotations[frameKey] = {};
    }
    labelingState.annotations[frameKey][tagText] = 'confirmed';

    // Add to vocabulary if new
    if (!labelingState.vocabulary.includes(tagText)) {
        labelingState.vocabulary.push(tagText);
        labelingState.vocabulary.sort();
    }

    renderCurrentFrame(container);
}
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(labeling): implement single view rendering, tag toggles, autocomplete"
```

---

## Task 9: Frontend — Navigation, Sync, and Keyboard Shortcuts

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Implement navigation and sync**

```javascript
function navigateLabeling(container, direction) {
    const newIndex = labelingState.currentIndex + direction;
    if (newIndex < 0 || newIndex >= labelingState.batch.length) return;

    labelingState.currentIndex = newIndex;
    renderCurrentFrame(container);
}

function skipFrame(container) {
    const item = labelingState.batch[labelingState.currentIndex];
    if (!item) return;

    const frameKey = `${item.scene_id}_${item.frame_index}`;
    // Mark as skipped (but don't clear any existing annotations)
    labelingState.pendingSync.push({
        type: 'progress',
        scene_id: item.scene_id,
        frame_index: item.frame_index,
        status: 'skipped',
    });

    navigateLabeling(container, 1);
    maybeSyncAnnotations();
}

function saveAndNext(container) {
    const item = labelingState.batch[labelingState.currentIndex];
    if (!item) return;

    const frameKey = `${item.scene_id}_${item.frame_index}`;
    const annotations = labelingState.annotations[frameKey] || {};

    // Queue annotations for sync
    for (const [tagText, label] of Object.entries(annotations)) {
        const suggested = item.suggested_tags.find(s => s.tag_text === tagText);
        labelingState.pendingSync.push({
            type: 'annotation',
            scene_id: item.scene_id,
            frame_index: item.frame_index,
            tag_text: tagText,
            tag_source: suggested ? 'suggested' : 'manual',
            label: label,
            similarity_score: suggested ? suggested.similarity : null,
        });
    }

    // Mark as labeled
    labelingState.pendingSync.push({
        type: 'progress',
        scene_id: item.scene_id,
        frame_index: item.frame_index,
        status: 'labeled',
    });

    navigateLabeling(container, 1);
    maybeSyncAnnotations();
}

function maybeSyncAnnotations() {
    // Sync every 30 items or on explicit request
    if (labelingState.pendingSync.length >= 30) {
        syncAnnotationsNow();
    }
}

async function syncAnnotationsNow() {
    if (labelingState.pendingSync.length === 0) return;
    if (!labelingState.sessionId) return;

    const items = [...labelingState.pendingSync];
    labelingState.pendingSync = [];

    const annotations = items
        .filter(i => i.type === 'annotation')
        .map(({ type, ...rest }) => rest);
    const progress = items
        .filter(i => i.type === 'progress')
        .map(({ type, ...rest }) => rest);

    const payload = {
        session_id: labelingState.sessionId,
        annotations: annotations,
        progress: progress,
    };

    const requestId = `sync_${Date.now()}`;

    try {
        await runPluginTask('Sync Labeling Annotations', {
            request_id: requestId,
            payload: JSON.stringify(payload),
        });
        log(`Synced ${annotations.length} annotations, ${progress.length} progress updates`);
    } catch (e) {
        log(`Sync failed: ${e.message}`, 'error');
        // Re-queue failed items
        labelingState.pendingSync.push(...items);
    }
}

// Sync on page unload
window.addEventListener('beforeunload', () => {
    if (labelingState.pendingSync.length > 0) {
        syncAnnotationsNow();
    }
});

function handleLabelingKeyboard(e, container) {
    // Only active when on labeling page
    if (!document.querySelector('.stash-copilot-label-page')) return;

    // Don't intercept when typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') {
            e.target.blur();
            e.preventDefault();
        }
        return;
    }

    switch (e.key) {
        case 'ArrowRight':
        case 'd':
        case 'D':
            navigateLabeling(container, 1);
            e.preventDefault();
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            navigateLabeling(container, -1);
            e.preventDefault();
            break;
        case 'Enter':
            saveAndNext(container);
            e.preventDefault();
            break;
        case 's':
        case 'S':
            skipFrame(container);
            e.preventDefault();
            break;
        case '/':
            const input = container.querySelector('.stash-copilot-label-tag-input');
            if (input) input.focus();
            e.preventDefault();
            break;
        case 'g':
        case 'G':
            toggleViewMode(container);
            e.preventDefault();
            break;
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
            toggleSuggestedTagByIndex(container, parseInt(e.key, 10) - 1);
            e.preventDefault();
            break;
    }
}

function toggleSuggestedTagByIndex(container, index) {
    const rows = container.querySelectorAll('.stash-copilot-label-tag-row');
    if (index < rows.length) {
        const item = labelingState.batch[labelingState.currentIndex];
        const frameKey = `${item.scene_id}_${item.frame_index}`;
        toggleTagState(rows[index], frameKey);
    }
}

function toggleViewMode(container) {
    labelingState.viewMode = labelingState.viewMode === 'single' ? 'grid' : 'single';
    const singleView = container.querySelector('.stash-copilot-label-single');
    const gridView = container.querySelector('.stash-copilot-label-grid');
    const toggle = container.querySelector('.stash-copilot-label-view-toggle');

    if (labelingState.viewMode === 'single') {
        singleView.style.display = 'flex';
        gridView.style.display = 'none';
        toggle.querySelector('.view-single').classList.add('active');
        toggle.querySelector('.view-grid').classList.remove('active');
        renderCurrentFrame(container);
    } else {
        singleView.style.display = 'none';
        gridView.style.display = 'flex';
        toggle.querySelector('.view-single').classList.remove('active');
        toggle.querySelector('.view-grid').classList.add('active');
        renderGridView(container);
    }
}
```

**Step 2: Implement grid view rendering**

```javascript
function renderGridView(container) {
    const gridImages = container.querySelector('.stash-copilot-label-grid-images');
    if (!gridImages) return;

    const startIdx = labelingState.currentIndex;
    const endIdx = Math.min(startIdx + 6, labelingState.batch.length);

    gridImages.innerHTML = '';
    for (let i = startIdx; i < endIdx; i++) {
        const item = labelingState.batch[i];
        const frameKey = `${item.scene_id}_${item.frame_index}`;
        const hasAnnotations = !!labelingState.annotations[frameKey];
        const isSelected = i === labelingState.currentIndex;
        const framePath = item.frame_path.replace(/^.*?assets\//, '/plugin/stash-copilot/assets/');

        const cell = document.createElement('div');
        cell.className = `stash-copilot-label-grid-cell${isSelected ? ' selected' : ''}${hasAnnotations ? ' labeled' : ''}`;
        cell.dataset.index = i;
        cell.innerHTML = `
            <img src="${framePath}" alt="Frame ${item.frame_index}">
            <div class="stash-copilot-label-grid-overlay">
                <span>${item.scene_title}</span>
                <span>${item.timestamp}</span>
            </div>
        `;
        cell.addEventListener('click', () => {
            labelingState.currentIndex = i;
            renderGridView(container);
            renderCurrentFrame(container);
        });
        gridImages.appendChild(cell);
    }
}
```

**Step 3: Implement export**

```javascript
async function exportDataset(container) {
    // Sync remaining annotations first
    await syncAnnotationsNow();

    const requestId = `export_${Date.now()}`;
    const exportBtn = container.querySelector('.stash-copilot-label-export-btn');
    exportBtn.disabled = true;
    exportBtn.textContent = 'Exporting...';

    try {
        await runPluginTask('Export Labeling Dataset', {
            request_id: requestId,
            include_negatives: 'true',
        });

        // Poll for result
        for (let i = 0; i < 60; i++) {
            try {
                const resp = await fetch(
                    `/plugin/stash-copilot/assets/labeling_export_${requestId}.json?t=${Date.now()}`,
                    { cache: 'no-store' }
                );
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        alert(`Exported ${data.total_images} images with ${data.total_tags} tags to:\n${data.export_path}`);
                        break;
                    } else if (data.status === 'error') {
                        alert(`Export failed: ${data.error}`);
                        break;
                    }
                }
            } catch (e) { /* not ready */ }
            await new Promise(r => setTimeout(r, 1000));
        }
    } catch (e) {
        alert(`Export failed: ${e.message}`);
    } finally {
        exportBtn.disabled = false;
        exportBtn.textContent = 'Export';
    }
}
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(labeling): add navigation, keyboard shortcuts, sync, grid view, export"
```

---

## Task 10: CSS Styling

Add styling for the labeling page with a distinctive theme color.

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add labeling page CSS**

Use a **warm amber/gold theme** (`#f59e0b`) to distinguish from existing cyan/green/purple themes. This represents the "annotation/labeling" concept — warm, focused, productive.

```css
/* ===== IMAGE LABELING PAGE ===== */

.stash-copilot-label-page {
    --label-accent: #f59e0b;
    --label-accent-rgb: 245, 158, 11;
    --label-bg: #0f0f14;
    --label-surface: #1a1a24;
    --label-border: #2a2a3a;
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--label-bg);
    color: #e0e0e0;
}

/* Header */
.stash-copilot-label-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: var(--label-surface);
    border-bottom: 1px solid var(--label-border);
    gap: 16px;
}

.stash-copilot-label-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.stash-copilot-label-back-btn {
    color: #999;
    text-decoration: none;
    font-size: 14px;
}
.stash-copilot-label-back-btn:hover {
    color: rgba(var(--label-accent-rgb), 1);
}

.stash-copilot-label-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 600;
    margin: 0;
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 1), #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stash-copilot-label-title svg {
    stroke: rgba(var(--label-accent-rgb), 1);
}

/* Progress bar */
.stash-copilot-label-header-center {
    flex: 1;
    max-width: 400px;
    text-align: center;
}

.stash-copilot-label-progress-text {
    font-size: 12px;
    color: #999;
    margin-bottom: 4px;
    display: block;
}

.stash-copilot-label-progress-bar {
    height: 4px;
    background: var(--label-border);
    border-radius: 2px;
    overflow: hidden;
}

.stash-copilot-label-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, rgba(var(--label-accent-rgb), 0.8), #fbbf24);
    border-radius: 2px;
    transition: width 0.3s ease;
    width: 0%;
}

/* Header right actions */
.stash-copilot-label-header-right {
    display: flex;
    gap: 8px;
    align-items: center;
}

.stash-copilot-label-view-toggle,
.stash-copilot-label-export-btn,
.stash-copilot-label-settings-btn {
    background: var(--label-border);
    border: 1px solid #3a3a4a;
    color: #ccc;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}

.stash-copilot-label-export-btn:hover,
.stash-copilot-label-settings-btn:hover {
    background: rgba(var(--label-accent-rgb), 0.2);
    border-color: rgba(var(--label-accent-rgb), 0.5);
    color: rgba(var(--label-accent-rgb), 1);
}

.stash-copilot-label-view-toggle .active {
    color: rgba(var(--label-accent-rgb), 1);
}

/* Body */
.stash-copilot-label-body {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

/* Intro state */
.stash-copilot-label-intro {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    gap: 16px;
    text-align: center;
    padding: 40px;
}

.stash-copilot-label-intro-icon {
    font-size: 48px;
}

.stash-copilot-label-intro h2 {
    font-size: 24px;
    margin: 0;
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 1), #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stash-copilot-label-intro p {
    color: #999;
    max-width: 500px;
    line-height: 1.5;
}

.stash-copilot-label-session-controls {
    display: flex;
    gap: 12px;
    align-items: center;
}

.stash-copilot-label-batch-input {
    width: 80px;
    background: var(--label-surface);
    border: 1px solid var(--label-border);
    color: #e0e0e0;
    padding: 8px;
    border-radius: 6px;
    text-align: center;
}

.stash-copilot-label-start-btn {
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 0.8), rgba(var(--label-accent-rgb), 0.6));
    border: none;
    color: #000;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}

.stash-copilot-label-start-btn:hover {
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 1), #fbbf24);
    box-shadow: 0 0 20px rgba(var(--label-accent-rgb), 0.3);
}

/* Loading */
.stash-copilot-label-loading {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
}

/* Single view layout */
.stash-copilot-label-single {
    flex: 1;
    display: flex;
    gap: 0;
    overflow: hidden;
}

.stash-copilot-label-image-area {
    flex: 0 0 65%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #000;
    position: relative;
    padding: 16px;
}

.stash-copilot-label-image {
    max-width: 100%;
    max-height: calc(100vh - 200px);
    object-fit: contain;
    border-radius: 4px;
}

.stash-copilot-label-image-meta {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 6px;
    margin-top: 8px;
    font-size: 12px;
    color: #999;
}

.stash-copilot-label-image-meta a {
    color: rgba(var(--label-accent-rgb), 1);
    text-decoration: none;
}

/* Tag panel */
.stash-copilot-label-tag-panel {
    flex: 0 0 35%;
    background: var(--label-surface);
    border-left: 1px solid var(--label-border);
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.stash-copilot-label-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin: 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--label-border);
}

/* Tag rows */
.stash-copilot-label-tag-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    transition: all 0.15s;
    cursor: pointer;
}

.stash-copilot-label-tag-row:hover {
    background: rgba(255, 255, 255, 0.05);
}

.stash-copilot-label-tag-row[data-state="confirmed"] {
    background: rgba(16, 185, 129, 0.15);
    border-left: 3px solid #10b981;
}

.stash-copilot-label-tag-row[data-state="rejected"] {
    background: rgba(239, 68, 68, 0.15);
    border-left: 3px solid #ef4444;
    opacity: 0.7;
}

.stash-copilot-label-tag-row[data-state="undecided"] {
    border-left: 3px solid #555;
}

.stash-copilot-label-tag-toggle {
    background: none;
    border: none;
    font-size: 16px;
    cursor: pointer;
    padding: 2px;
    min-width: 24px;
}

.stash-copilot-label-tag-name {
    flex: 1;
    font-size: 13px;
    color: #ddd;
}

.stash-copilot-label-tag-sim {
    font-size: 11px;
    color: #777;
    font-family: monospace;
}

.stash-copilot-label-tag-key {
    font-size: 10px;
    color: #555;
    background: rgba(255, 255, 255, 0.05);
    padding: 1px 5px;
    border-radius: 3px;
    font-family: monospace;
}

/* Scene tags pills */
.stash-copilot-label-scene-tag-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.08);
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    color: #999;
    margin: 2px;
}

/* Autocomplete */
.stash-copilot-label-autocomplete {
    position: relative;
}

.stash-copilot-label-tag-input {
    width: 100%;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--label-border);
    color: #e0e0e0;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    box-sizing: border-box;
}

.stash-copilot-label-tag-input:focus {
    border-color: rgba(var(--label-accent-rgb), 0.5);
    outline: none;
    box-shadow: 0 0 8px rgba(var(--label-accent-rgb), 0.2);
}

.stash-copilot-label-autocomplete-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: #1e1e2e;
    border: 1px solid var(--label-border);
    border-radius: 6px;
    max-height: 200px;
    overflow-y: auto;
    z-index: 100;
    display: none;
}

.stash-copilot-label-ac-item {
    padding: 8px 12px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.1s;
}

.stash-copilot-label-ac-item:hover {
    background: rgba(var(--label-accent-rgb), 0.15);
}

.stash-copilot-label-ac-new {
    color: rgba(var(--label-accent-rgb), 1);
    font-style: italic;
}

/* Manual tags */
.stash-copilot-label-manual-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(var(--label-accent-rgb), 0.15);
    border: 1px solid rgba(var(--label-accent-rgb), 0.3);
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 12px;
    color: rgba(var(--label-accent-rgb), 1);
    margin: 2px;
}

.stash-copilot-label-remove-tag {
    background: none;
    border: none;
    color: rgba(var(--label-accent-rgb), 0.6);
    cursor: pointer;
    font-size: 14px;
    padding: 0 2px;
}

/* Grid view */
.stash-copilot-label-grid {
    flex: 1;
    display: flex;
    overflow: hidden;
}

.stash-copilot-label-grid-images {
    flex: 0 0 65%;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 4px;
    padding: 4px;
    background: #000;
}

.stash-copilot-label-grid-cell {
    position: relative;
    cursor: pointer;
    border: 2px solid transparent;
    border-radius: 4px;
    overflow: hidden;
    transition: border-color 0.2s;
}

.stash-copilot-label-grid-cell img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.stash-copilot-label-grid-cell.selected {
    border-color: rgba(var(--label-accent-rgb), 1);
    box-shadow: 0 0 12px rgba(var(--label-accent-rgb), 0.3);
}

.stash-copilot-label-grid-cell.labeled {
    border-color: #10b981;
}

.stash-copilot-label-grid-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
    padding: 16px 8px 6px;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #ccc;
}

/* Footer navigation */
.stash-copilot-label-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: var(--label-surface);
    border-top: 1px solid var(--label-border);
}

.stash-copilot-label-nav,
.stash-copilot-label-actions {
    display: flex;
    gap: 12px;
    align-items: center;
}

.stash-copilot-label-position {
    font-family: monospace;
    font-size: 14px;
    color: #999;
    min-width: 80px;
    text-align: center;
}

.stash-copilot-label-prev-btn,
.stash-copilot-label-next-btn,
.stash-copilot-label-skip-btn {
    background: var(--label-border);
    border: 1px solid #3a3a4a;
    color: #ccc;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}

.stash-copilot-label-prev-btn:hover,
.stash-copilot-label-next-btn:hover {
    background: rgba(var(--label-accent-rgb), 0.2);
    border-color: rgba(var(--label-accent-rgb), 0.4);
}

.stash-copilot-label-save-btn {
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 0.8), rgba(var(--label-accent-rgb), 0.6));
    border: none;
    color: #000;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}

.stash-copilot-label-save-btn:hover {
    background: linear-gradient(135deg, rgba(var(--label-accent-rgb), 1), #fbbf24);
    box-shadow: 0 0 16px rgba(var(--label-accent-rgb), 0.3);
}
```

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(labeling): add amber/gold themed CSS for labeling page"
```

---

## Task 11: Frontend — Previous Sessions and Resume

Allow resuming a previous session from the intro screen.

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Implement session listing**

```javascript
async function loadPreviousSessions(container) {
    const sessionsEl = container.querySelector('.stash-copilot-label-previous-sessions');
    if (!sessionsEl) return;

    const requestId = `sessions_${Date.now()}`;

    try {
        await runPluginTask('Get Labeling Sessions', { request_id: requestId });

        // Quick poll (sessions list is fast)
        for (let i = 0; i < 10; i++) {
            try {
                const resp = await fetch(
                    `/plugin/stash-copilot/assets/labeling_sessions_${requestId}.json?t=${Date.now()}`,
                    { cache: 'no-store' }
                );
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete' && data.sessions.length > 0) {
                        sessionsEl.innerHTML = `
                            <h3 style="font-size: 14px; color: #999; margin-top: 24px;">Previous Sessions</h3>
                            ${data.sessions.slice(0, 5).map(s => `
                                <div class="stash-copilot-label-session-card" data-session-id="${s.session_id}">
                                    <span class="session-status ${s.status}">${s.status}</span>
                                    <span class="session-progress">${s.labeled_count}/${s.total_frames} labeled</span>
                                    <span class="session-date">${new Date(s.created_at).toLocaleDateString()}</span>
                                </div>
                            `).join('')}
                        `;
                        return;
                    }
                }
            } catch (e) { /* not ready */ }
            await new Promise(r => setTimeout(r, 500));
        }
    } catch (e) {
        log(`Failed to load sessions: ${e.message}`, 'error');
    }
}
```

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(labeling): add previous sessions listing on intro screen"
```

---

## Task 12: Navigation Link

Add a link to the labeling page from the navbar dropdown.

**Files:**
- Modify: `stash-copilot.js` (find the `createNavbarButton` function and add labeling link)

**Step 1: Find existing navbar dropdown items and add labeling link**

In the existing `createNavbarButton()` function, add a new dropdown item alongside the existing items (AI Search, Taste Map, etc.):

```javascript
// Add alongside existing dropdown items
{
    label: '🏷️ Image Labeling',
    href: '/plugins/stash-copilot/label',
    description: 'Label images for model training'
}
```

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(labeling): add navigation link in navbar dropdown"
```

---

## Task 13: Integration Testing via Playwright

Test the full workflow: navigate to page, start session, label frames, export.

**Files:**
- Tests performed via Playwright MCP tools (no test file — follows project testing convention)

**Step 1: Navigate to labeling page**

```
browser_navigate(url="http://localhost:9999/plugins/stash-copilot/label")
browser_snapshot()
browser_take_screenshot(filename="~/.stash/plugins/stash-copilot/tests/screenshots/labeling-intro.png")
```

**Step 2: Start a session**

```
browser_click(element="Start New Session button")
browser_wait_for(text="Computing uncertainty")
browser_wait_for(time=10)  // Wait for processing
browser_snapshot()
browser_take_screenshot(filename="~/.stash/plugins/stash-copilot/tests/screenshots/labeling-session.png")
```

**Step 3: Test tag toggling**

```
browser_click(element="First suggested tag toggle")
browser_snapshot()
browser_take_screenshot(filename="~/.stash/plugins/stash-copilot/tests/screenshots/labeling-tag-toggle.png")
```

**Step 4: Test keyboard navigation**

```
browser_press_key(key="Enter")  // Save & Next
browser_snapshot()
browser_press_key(key="1")  // Toggle first tag
browser_press_key(key="2")  // Toggle second tag
browser_press_key(key="s")  // Skip
```

**Step 5: Check logs for errors**

```bash
tail -50 ~/.stash/stash.log | grep -i "error\|warn\|exception"
```

**Step 6: Take final screenshot**

```
browser_take_screenshot(filename="~/.stash/plugins/stash-copilot/tests/screenshots/labeling-complete.png")
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Database schema migration v12 | `storage.py`, `test_labeling.py` |
| 2 | Storage CRUD methods | `storage.py`, `test_labeling.py` |
| 3 | Labeling types | `labeling_types.py` |
| 4 | PrepareSession task (uncertainty sampling) | `labeling.py`, `test_labeling.py` |
| 5 | SyncAnnotations + ExportDataset tasks | `labeling.py`, `test_labeling.py` |
| 6 | Task registration + dispatch | `stash-copilot.yml`, `stash-copilot.py` |
| 7 | Frontend page injection + session management | `stash-copilot.js` |
| 8 | Single view rendering + tag interaction | `stash-copilot.js` |
| 9 | Navigation, keyboard shortcuts, sync, grid | `stash-copilot.js` |
| 10 | CSS styling (amber/gold theme) | `stash-copilot.css` |
| 11 | Previous sessions + resume | `stash-copilot.js` |
| 12 | Navigation link in navbar | `stash-copilot.js` |
| 13 | Playwright integration testing | Screenshots |

**Estimated commits:** 12 (one per task, testing is inline)
