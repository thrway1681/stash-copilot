# Tag Gap Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect visual content in scenes not covered by any existing tag, flag those scenes, and provide context to help users create new tags.

**Architecture:** Frame-level OpenCLIP image embeddings are compared against tag text embeddings via cosine similarity. Frames below an adaptive threshold (bottom quartile) are "uncovered". Results stored in SQLite, surfaced in AI Insights modal (bulk report) and scene sidebar (per-scene detail). Cross-scene similarity computed on-demand, no stored clusters.

**Tech Stack:** Python (numpy, sqlite3), JavaScript (vanilla DOM), existing OpenCLIP/TagVocabulary infrastructure.

---

### Task 1: Schema Migration v10 — `frame_tag_coverage` Table

**Files:**
- Modify: `stash_ai/embeddings/storage.py` (lines 116, 189-190, after line 606)

**Step 1: Add the TypedDict for frame tag coverage records**

At the top of `storage.py`, after the `SceneSegment` TypedDict (around line 103), add:

```python
class FrameTagCoverageRecord(TypedDict):
    """Represents a frame's tag coverage analysis result."""

    scene_id: int
    frame_index: int
    model_key: str
    best_tag: str
    best_similarity: float
    is_covered: bool
```

**Step 2: Bump schema version and add migration call**

Change line 116:
```python
SCHEMA_VERSION = 10  # Added frame_tag_coverage table
```

In `_init_database()`, after line 190 (`self._migrate_to_v9(cursor)`), add:
```python
        if current_version < 10:
            self._migrate_to_v10(cursor)
```

**Step 3: Write the migration method**

After `_migrate_to_v9` (after line 606), add:

```python
    def _migrate_to_v10(self, cursor: sqlite3.Cursor) -> None:
        """Add frame_tag_coverage table for tag gap detection (v10)."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_tag_coverage (
                scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                best_tag TEXT NOT NULL,
                best_similarity REAL NOT NULL,
                is_covered INTEGER NOT NULL,
                PRIMARY KEY (scene_id, frame_index, model_key)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_tag_coverage_scene
            ON frame_tag_coverage(scene_id, model_key)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_tag_coverage_uncovered
            ON frame_tag_coverage(is_covered, model_key)
        """)
```

**Step 4: Verify migration runs**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.embeddings.storage import EmbeddingStorage
s = EmbeddingStorage()
conn = s._get_connection()
cursor = conn.cursor()
cursor.execute(\"SELECT value FROM schema_info WHERE key = 'version'\")
print(f'Schema version: {cursor.fetchone()[\"value\"]}')
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='frame_tag_coverage'\")
print(f'Table exists: {cursor.fetchone() is not None}')
conn.close()
"
```

Expected: `Schema version: 10` and `Table exists: True`

**Step 5: Commit**

```bash
git add stash_ai/embeddings/storage.py
git commit -m "feat(tag-gaps): add frame_tag_coverage table schema migration v10"
```

---

### Task 2: Storage Methods for Tag Gap Coverage

**Files:**
- Modify: `stash_ai/embeddings/storage.py` (add methods after existing tag embedding methods, around line 3185)

**Step 1: Add `save_frame_tag_coverage_batch` method**

```python
    def save_frame_tag_coverage_batch(
        self,
        rows: List[FrameTagCoverageRecord],
    ) -> None:
        """Batch insert/replace frame tag coverage results.

        Args:
            rows: List of FrameTagCoverageRecord dicts.
        """
        if not rows:
            return

        conn = self._get_connection()
        conn.executemany(
            """INSERT OR REPLACE INTO frame_tag_coverage
            (scene_id, frame_index, model_key, best_tag, best_similarity, is_covered)
            VALUES (:scene_id, :frame_index, :model_key, :best_tag, :best_similarity, :is_covered)""",
            rows,
        )
        conn.commit()
        conn.close()
```

**Step 2: Add `get_scene_tag_coverage` method**

```python
    def get_scene_tag_coverage(
        self,
        scene_id: int,
    ) -> List[FrameTagCoverageRecord]:
        """Get all frame coverage records for a scene.

        Args:
            scene_id: Stash scene ID.

        Returns:
            List of FrameTagCoverageRecord ordered by frame_index.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT scene_id, frame_index, model_key, best_tag, best_similarity, is_covered
            FROM frame_tag_coverage
            WHERE scene_id = ? AND model_key = ?
            ORDER BY frame_index""",
            (scene_id, self.model_key),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            FrameTagCoverageRecord(
                scene_id=r["scene_id"],
                frame_index=r["frame_index"],
                model_key=r["model_key"],
                best_tag=r["best_tag"],
                best_similarity=r["best_similarity"],
                is_covered=bool(r["is_covered"]),
            )
            for r in rows
        ]
```

**Step 3: Add `get_coverage_summary` method**

```python
    def get_coverage_summary(
        self,
    ) -> List[Dict[str, Any]]:
        """Get per-scene coverage summary for bulk report.

        Returns:
            List of dicts with scene_id, total_frames, uncovered_frames, coverage_ratio,
            sorted by coverage_ratio ascending (most uncovered first).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT
                scene_id,
                COUNT(*) as total_frames,
                SUM(CASE WHEN NOT is_covered THEN 1 ELSE 0 END) as uncovered_frames
            FROM frame_tag_coverage
            WHERE model_key = ?
            GROUP BY scene_id
            ORDER BY CAST(SUM(CASE WHEN NOT is_covered THEN 1 ELSE 0 END) AS REAL) / COUNT(*) DESC""",
            (self.model_key,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "scene_id": r["scene_id"],
                "total_frames": r["total_frames"],
                "uncovered_frames": r["uncovered_frames"],
                "coverage_ratio": 1.0 - (r["uncovered_frames"] / r["total_frames"])
                if r["total_frames"] > 0
                else 1.0,
            }
            for r in rows
        ]
```

**Step 4: Add `get_uncovered_frame_embeddings` method**

```python
    def get_uncovered_frame_embeddings(
        self,
        scene_id: int,
    ) -> NDArray[np.float32]:
        """Get embeddings for uncovered frames of a scene.

        Joins frame_tag_coverage with frame_embeddings to get the actual
        embedding vectors for frames marked as uncovered.

        Args:
            scene_id: Stash scene ID.

        Returns:
            Numpy array of shape (N, dims) with uncovered frame embeddings.
            Empty array if no uncovered frames.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT fe.embedding
            FROM frame_tag_coverage ftc
            JOIN frame_embeddings fe
                ON ftc.scene_id = fe.scene_id
                AND ftc.frame_index = fe.frame_index
                AND ftc.model_key = fe.model_key
            WHERE ftc.scene_id = ? AND ftc.model_key = ? AND NOT ftc.is_covered
            ORDER BY ftc.frame_index""",
            (scene_id, self.model_key),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return np.empty((0, 0), dtype=np.float32)

        dims = len(rows[0]["embedding"]) // 4
        arr = np.empty((len(rows), dims), dtype=np.float32)
        for i, r in enumerate(rows):
            arr[i] = np.frombuffer(r["embedding"], dtype=np.float32)
        return arr
```

**Step 5: Add `get_scenes_with_tag_coverage` method**

```python
    def get_scenes_with_tag_coverage(self) -> List[int]:
        """Get list of scene IDs that have tag coverage data.

        Returns:
            List of scene_id values.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT scene_id FROM frame_tag_coverage WHERE model_key = ?",
            (self.model_key,),
        )
        scene_ids = [r["scene_id"] for r in cursor.fetchall()]
        conn.close()
        return scene_ids
```

**Step 6: Add `delete_scene_tag_coverage` method**

```python
    def delete_scene_tag_coverage(self, scene_id: int) -> None:
        """Delete tag coverage data for a scene.

        Args:
            scene_id: Stash scene ID.
        """
        conn = self._get_connection()
        conn.execute(
            "DELETE FROM frame_tag_coverage WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        conn.commit()
        conn.close()
```

**Step 7: Add `update_coverage_threshold` method**

```python
    def update_coverage_threshold(self, threshold: float) -> int:
        """Update is_covered flags based on a new threshold.

        Re-evaluates all frames against the threshold without recomputing similarities.

        Args:
            threshold: New similarity threshold. Frames with best_similarity >= threshold
                are marked as covered.

        Returns:
            Number of rows updated.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE frame_tag_coverage
            SET is_covered = CASE WHEN best_similarity >= ? THEN 1 ELSE 0 END
            WHERE model_key = ?""",
            (threshold, self.model_key),
        )
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return updated
```

**Step 8: Verify methods work**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.embeddings.storage import EmbeddingStorage, FrameTagCoverageRecord
s = EmbeddingStorage()

# Test save and retrieve
rows = [
    FrameTagCoverageRecord(scene_id=1, frame_index=0, model_key=s.model_key, best_tag='test', best_similarity=0.3, is_covered=True),
    FrameTagCoverageRecord(scene_id=1, frame_index=1, model_key=s.model_key, best_tag='test2', best_similarity=0.1, is_covered=False),
]
s.save_frame_tag_coverage_batch(rows)
result = s.get_scene_tag_coverage(1)
print(f'Saved and retrieved {len(result)} rows')
print(f'First row covered: {result[0][\"is_covered\"]}, Second row covered: {result[1][\"is_covered\"]}')

# Test summary
summary = s.get_coverage_summary()
print(f'Summary has {len(summary)} scenes')

# Cleanup test data
s.delete_scene_tag_coverage(1)
print('Cleanup done')
"
```

Expected: Saved/retrieved 2 rows, correct covered flags, summary with 1 scene, cleanup successful.

**Step 9: Commit**

```bash
git add stash_ai/embeddings/storage.py
git commit -m "feat(tag-gaps): add storage methods for frame tag coverage"
```

---

### Task 3: TagGapDetectionTask — Core Algorithm

**Files:**
- Create: `stash_ai/tasks/tag_gap_detection.py`

**Step 1: Create the task file with types and constructor**

```python
"""Tag gap detection task — finds visual content not covered by existing tags."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

import numpy as np

if TYPE_CHECKING:
    from stashapi.stashapp import StashInterface

from stash_ai.embeddings.storage import EmbeddingStorage, FrameTagCoverageRecord
from stash_ai.embeddings.tag_vocabulary import TagVocabulary


class SceneCoverageSummary(TypedDict):
    """Per-scene coverage summary for the report."""

    scene_id: int
    total_frames: int
    uncovered_frames: int
    coverage_ratio: float
    top_uncovered_tags: List[Dict[str, Any]]  # [{"tag": str, "similarity": float}]


class TagGapReport(TypedDict):
    """Full tag gap detection report."""

    status: str  # "complete" | "error"
    threshold: float
    total_scenes: int
    avg_coverage: float
    flagged_scenes: int
    scenes: List[SceneCoverageSummary]
    error: Optional[str]


class TagGapDetectionTask:
    """Detect visual content in scenes not covered by any existing tag.

    Algorithm:
        1. Ensure all Stash tags have text embeddings
        2. Load tag embeddings into a matrix
        3. For each scene's frame embeddings, compute best-match tag similarity
        4. Compute adaptive threshold (bottom quartile of all best-match scores)
        5. Mark frames below threshold as uncovered
        6. Save results and generate report
    """

    def __init__(
        self,
        stash: "StashInterface",
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        self.stash = stash
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)
        self.model_key = model_key
        self.storage = EmbeddingStorage(model_key=model_key)
```

**Step 2: Add the main `run` method**

```python
    def run(
        self,
        request_id: str = "",
        force: bool = False,
    ) -> TagGapReport:
        """Run tag gap detection across all embedded scenes.

        Args:
            request_id: Unique ID for result file (for frontend polling).
            force: If True, recompute all scenes even if cached.

        Returns:
            TagGapReport with coverage summary.
        """
        try:
            total_steps = 5
            self.progress(0, total_steps)

            # Step 1: Ensure tag embeddings
            self.log("Step 1/5: Ensuring tag embeddings...", "info")
            tag_matrix, tag_names = self._load_tag_embeddings()
            if tag_matrix.shape[0] == 0:
                report: TagGapReport = {
                    "status": "error",
                    "threshold": 0.0,
                    "total_scenes": 0,
                    "avg_coverage": 0.0,
                    "flagged_scenes": 0,
                    "scenes": [],
                    "error": "No tag embeddings found. Ensure tags exist in Stash.",
                }
                self._save_results(report, request_id)
                return report
            self.log(f"  Loaded {tag_matrix.shape[0]} tag embeddings", "info")
            self.progress(1, total_steps)

            # Step 2: Get scenes to process
            self.log("Step 2/5: Finding scenes with frame embeddings...", "info")
            scene_ids = self._get_scenes_to_process(force)
            if not scene_ids:
                report = {
                    "status": "error",
                    "threshold": 0.0,
                    "total_scenes": 0,
                    "avg_coverage": 0.0,
                    "flagged_scenes": 0,
                    "scenes": [],
                    "error": "No scenes with frame embeddings found. Run 'Embed All Scenes' first.",
                }
                self._save_results(report, request_id)
                return report
            self.log(f"  {len(scene_ids)} scenes to process", "info")
            self.progress(2, total_steps)

            # Step 3: Compute per-frame best-match similarities
            self.log("Step 3/5: Computing frame-tag similarities...", "info")
            all_best_similarities = self._compute_all_similarities(
                scene_ids, tag_matrix, tag_names
            )
            self.progress(3, total_steps)

            # Step 4: Compute adaptive threshold (bottom quartile)
            self.log("Step 4/5: Computing adaptive threshold...", "info")
            threshold = self._compute_threshold(all_best_similarities)
            self.log(f"  Adaptive threshold (Q25): {threshold:.4f}", "info")

            # Update is_covered flags with computed threshold
            updated = self.storage.update_coverage_threshold(threshold)
            self.log(f"  Updated {updated} frame coverage flags", "info")
            self.progress(4, total_steps)

            # Step 5: Generate report
            self.log("Step 5/5: Generating report...", "info")
            report = self._build_report(threshold)
            self._save_results(report, request_id)
            self.progress(5, total_steps)

            self.log(
                f"Tag gap detection complete: {report['avg_coverage']:.0%} avg coverage, "
                f"{report['flagged_scenes']} scenes flagged",
                "info",
            )
            return report

        except Exception as e:
            error_msg = f"Tag gap detection failed: {e}"
            self.log(error_msg, "error")
            report = {
                "status": "error",
                "threshold": 0.0,
                "total_scenes": 0,
                "avg_coverage": 0.0,
                "flagged_scenes": 0,
                "scenes": [],
                "error": error_msg,
            }
            self._save_results(report, request_id)
            return report
```

**Step 3: Add helper methods**

```python
    def _load_tag_embeddings(self) -> tuple[np.ndarray, List[str]]:
        """Load all tag embeddings into a normalized matrix.

        Returns:
            Tuple of (tag_matrix [N_tags x dims], tag_names [N_tags]).
        """
        # Ensure Stash tags are embedded
        vocab = TagVocabulary(
            storage=self.storage,
            model_key=self.model_key,
            log=self.log,
        )
        # Fetch current Stash tags
        tag_result = self.stash.call_GQL(
            """query { allTags { id name } }"""
        )
        stash_tags = [t["name"] for t in (tag_result.get("allTags") or [])]
        vocab.ensure_embeddings(stash_tags=stash_tags)

        # Load all tag embeddings
        all_tags = self.storage.get_all_tag_embeddings(self.model_key)
        if not all_tags:
            return np.empty((0, 0), dtype=np.float32), []

        tag_names = [t["text"] for t in all_tags]
        tag_matrix = np.array(
            [t["embedding"] for t in all_tags], dtype=np.float32
        )
        # L2 normalize rows
        norms = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        tag_matrix = tag_matrix / norms

        return tag_matrix, tag_names

    def _get_scenes_to_process(self, force: bool) -> List[int]:
        """Get scene IDs that need processing.

        Args:
            force: If True, return all scenes with frame embeddings.

        Returns:
            List of scene IDs to process.
        """
        # Get all scenes with frame embeddings
        conn = self.storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT scene_id FROM frame_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        all_scenes = {r["scene_id"] for r in cursor.fetchall()}
        conn.close()

        if force:
            return sorted(all_scenes)

        # Exclude scenes already processed
        existing = set(self.storage.get_scenes_with_tag_coverage())
        return sorted(all_scenes - existing)

    def _compute_all_similarities(
        self,
        scene_ids: List[int],
        tag_matrix: np.ndarray,
        tag_names: List[str],
    ) -> List[float]:
        """Compute per-frame best-match similarities and save to storage.

        This is the core computation. For each frame, find the most similar tag
        via cosine similarity, and record the result.

        Args:
            scene_ids: Scenes to process.
            tag_matrix: Normalized tag embedding matrix [N_tags x dims].
            tag_names: Tag name for each row.

        Returns:
            List of all best_similarity values (for threshold computation).
        """
        all_best_sims: List[float] = []
        total = len(scene_ids)

        for idx, scene_id in enumerate(scene_ids):
            if idx % 10 == 0:
                self.log(f"  Processing scene {idx + 1}/{total}...", "debug")
                # Map scene progress into step 3 (from 2/5 to 3/5)
                self.progress(2 * total + idx, 5 * total)

            # Load frame embeddings for this scene
            conn = self.storage._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT frame_index, embedding
                FROM frame_embeddings
                WHERE scene_id = ? AND model_key = ?
                ORDER BY frame_index""",
                (scene_id, self.model_key),
            )
            frame_rows = cursor.fetchall()
            conn.close()

            if not frame_rows:
                continue

            # Build frame matrix
            dims = len(frame_rows[0]["embedding"]) // 4
            frame_indices = [r["frame_index"] for r in frame_rows]
            frame_matrix = np.empty((len(frame_rows), dims), dtype=np.float32)
            for i, r in enumerate(frame_rows):
                frame_matrix[i] = np.frombuffer(r["embedding"], dtype=np.float32)

            # L2 normalize frames
            norms = np.linalg.norm(frame_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            frame_matrix = frame_matrix / norms

            # Cosine similarity: [N_frames x N_tags]
            similarities = frame_matrix @ tag_matrix.T

            # Best tag per frame
            best_indices = np.argmax(similarities, axis=1)
            best_sims = similarities[np.arange(len(best_indices)), best_indices]

            # Build coverage records (is_covered=False placeholder, updated later by threshold)
            coverage_rows: List[FrameTagCoverageRecord] = []
            for i, frame_idx in enumerate(frame_indices):
                sim = float(best_sims[i])
                all_best_sims.append(sim)
                coverage_rows.append(
                    FrameTagCoverageRecord(
                        scene_id=scene_id,
                        frame_index=frame_idx,
                        model_key=self.model_key,
                        best_tag=tag_names[int(best_indices[i])],
                        best_similarity=round(sim, 5),
                        is_covered=False,  # Placeholder, set by threshold later
                    )
                )

            self.storage.save_frame_tag_coverage_batch(coverage_rows)

        return all_best_sims

    def _compute_threshold(self, all_best_sims: List[float]) -> float:
        """Compute adaptive threshold as bottom quartile (25th percentile).

        Args:
            all_best_sims: All best-match similarity values.

        Returns:
            Threshold value. Frames with similarity >= threshold are "covered".
        """
        if not all_best_sims:
            return 0.0
        arr = np.array(all_best_sims, dtype=np.float32)
        return float(np.percentile(arr, 25))

    def _build_report(self, threshold: float) -> TagGapReport:
        """Build the final report from stored coverage data.

        Args:
            threshold: The adaptive threshold used.

        Returns:
            Complete TagGapReport.
        """
        summary = self.storage.get_coverage_summary()

        scenes: List[SceneCoverageSummary] = []
        for s in summary:
            # Get top uncovered tags (most common best_tag among uncovered frames)
            coverage = self.storage.get_scene_tag_coverage(s["scene_id"])
            uncovered_tags: Dict[str, List[float]] = {}
            for frame in coverage:
                if not frame["is_covered"]:
                    tag = frame["best_tag"]
                    if tag not in uncovered_tags:
                        uncovered_tags[tag] = []
                    uncovered_tags[tag].append(frame["best_similarity"])

            top_tags = sorted(
                [
                    {"tag": tag, "similarity": round(sum(sims) / len(sims), 4), "count": len(sims)}
                    for tag, sims in uncovered_tags.items()
                ],
                key=lambda t: t["count"],
                reverse=True,
            )[:5]

            scenes.append(
                SceneCoverageSummary(
                    scene_id=s["scene_id"],
                    total_frames=s["total_frames"],
                    uncovered_frames=s["uncovered_frames"],
                    coverage_ratio=round(s["coverage_ratio"], 4),
                    top_uncovered_tags=top_tags,
                )
            )

        total = len(scenes)
        avg_cov = sum(s["coverage_ratio"] for s in scenes) / total if total else 0.0
        flagged = sum(1 for s in scenes if s["coverage_ratio"] < 0.75)

        return TagGapReport(
            status="complete",
            threshold=round(threshold, 5),
            total_scenes=total,
            avg_coverage=round(avg_cov, 4),
            flagged_scenes=flagged,
            scenes=scenes,
            error=None,
        )

    def _save_results(self, report: TagGapReport, request_id: str) -> None:
        """Save results to JSON for frontend polling."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        if request_id:
            req_path = os.path.join(assets_dir, f"tag_gaps_{request_id}.json")
            with open(req_path, "w") as f:
                json.dump(report, f)
            self.log(f"Results saved to tag_gaps_{request_id}.json", "debug")

        latest_path = os.path.join(assets_dir, "tag_gaps_latest.json")
        with open(latest_path, "w") as f:
            json.dump(report, f)
        self.log("Results saved to tag_gaps_latest.json", "debug")
```

**Step 4: Verify the module imports cleanly**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask, TagGapReport, SceneCoverageSummary
print('Import successful')
print(f'TagGapReport fields: {list(TagGapReport.__annotations__.keys())}')
"
```

Expected: Import successful, fields listed.

**Step 5: Commit**

```bash
git add stash_ai/tasks/tag_gap_detection.py
git commit -m "feat(tag-gaps): add TagGapDetectionTask with core detection algorithm"
```

---

### Task 4: Find Similar Uncovered Scenes Method

**Files:**
- Modify: `stash_ai/tasks/tag_gap_detection.py` (add method to TagGapDetectionTask)

This method supports the sidebar "similar uncovered scenes" feature. It's on the task class rather than storage because it orchestrates data from multiple storage calls.

**Step 1: Add `find_similar_uncovered` method**

Add to `TagGapDetectionTask` class:

```python
    def find_similar_uncovered(
        self,
        scene_id: int,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find scenes with similar uncovered content.

        Averages the uncovered frame embeddings for the query scene, then
        compares against other scenes' averaged uncovered embeddings.

        Args:
            scene_id: Query scene ID.
            limit: Max results.
            min_similarity: Minimum cosine similarity.

        Returns:
            List of {"scene_id": int, "similarity": float} sorted descending.
        """
        # Get uncovered embeddings for query scene
        query_uncovered = self.storage.get_uncovered_frame_embeddings(scene_id)
        if query_uncovered.size == 0:
            return []

        query_avg = query_uncovered.mean(axis=0)
        query_norm = np.linalg.norm(query_avg)
        if query_norm < 1e-8:
            return []
        query_avg = query_avg / query_norm

        # Get all scenes with coverage data (excluding query scene)
        all_scenes = self.storage.get_scenes_with_tag_coverage()
        results: List[Dict[str, Any]] = []

        for other_id in all_scenes:
            if other_id == scene_id:
                continue

            other_uncovered = self.storage.get_uncovered_frame_embeddings(other_id)
            if other_uncovered.size == 0:
                continue

            other_avg = other_uncovered.mean(axis=0)
            other_norm = np.linalg.norm(other_avg)
            if other_norm < 1e-8:
                continue
            other_avg = other_avg / other_norm

            similarity = float(np.dot(query_avg, other_avg))
            if similarity >= min_similarity:
                results.append({"scene_id": other_id, "similarity": round(similarity, 4)})

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:limit]
```

**Step 2: Add `get_scene_gaps_detail` method for sidebar queries**

This is a convenience method that returns everything the sidebar needs in one call.

```python
    def get_scene_gaps_detail(
        self,
        scene_id: int,
    ) -> Dict[str, Any]:
        """Get complete tag gap detail for a scene (for sidebar display).

        Args:
            scene_id: Stash scene ID.

        Returns:
            Dict with coverage data, uncovered frames, nearest tags, similar scenes.
            Returns {"has_data": False} if no coverage data exists.
        """
        coverage = self.storage.get_scene_tag_coverage(scene_id)
        if not coverage:
            return {"has_data": False}

        total = len(coverage)
        uncovered = [f for f in coverage if not f["is_covered"]]
        covered_count = total - len(uncovered)

        # Group uncovered frames with their nearest tags
        uncovered_frames: List[Dict[str, Any]] = []
        for frame in uncovered:
            uncovered_frames.append({
                "frame_index": frame["frame_index"],
                "timestamp": float(frame["frame_index"]),  # 1fps: frame_index ≈ timestamp
                "best_tag": frame["best_tag"],
                "best_similarity": frame["best_similarity"],
            })

        # Aggregate nearest tags across uncovered frames
        tag_counts: Dict[str, List[float]] = {}
        for frame in uncovered:
            tag = frame["best_tag"]
            if tag not in tag_counts:
                tag_counts[tag] = []
            tag_counts[tag].append(frame["best_similarity"])

        nearest_tags = sorted(
            [
                {
                    "tag": tag,
                    "avg_similarity": round(sum(sims) / len(sims), 4),
                    "frame_count": len(sims),
                }
                for tag, sims in tag_counts.items()
            ],
            key=lambda t: t["frame_count"],
            reverse=True,
        )[:10]

        return {
            "has_data": True,
            "scene_id": scene_id,
            "total_frames": total,
            "covered_frames": covered_count,
            "uncovered_frames": len(uncovered),
            "coverage_ratio": round(covered_count / total, 4) if total > 0 else 1.0,
            "uncovered_frame_list": uncovered_frames,
            "nearest_tags": nearest_tags,
        }
```

**Step 3: Verify imports**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask
t = TagGapDetectionTask.__new__(TagGapDetectionTask)
print('Methods:', [m for m in dir(t) if m.startswith('find_similar') or m.startswith('get_scene_gaps')])
"
```

Expected: Methods include `find_similar_uncovered` and `get_scene_gaps_detail`.

**Step 4: Commit**

```bash
git add stash_ai/tasks/tag_gap_detection.py
git commit -m "feat(tag-gaps): add similar uncovered scenes and per-scene detail methods"
```

---

### Task 5: Wire Up Plugin Task Registration

**Files:**
- Modify: `stash-copilot.py` (add elif branch + handler method)
- Modify: `stash-copilot.yml` (add task definitions)

**Step 1: Add task handler method to `StashCopilotPlugin`**

In `stash-copilot.py`, find an appropriate location near the other `run_*` methods (e.g., after `run_build_taste_map` around line 617) and add:

```python
    def run_detect_tag_gaps(self, args: Dict[str, Any]):
        """Run the tag gap detection task."""
        try:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            self.log("Initializing tag gap detection...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TagGapDetectionTask(
                stash=self.stash,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            force = args.get("force", "false").lower() == "true"
            report = task.run(
                request_id=args.get("request_id", ""),
                force=force,
            )

            if report["status"] == "complete":
                self.log(
                    f"Tag gap detection complete: {report['avg_coverage']:.0%} avg coverage, "
                    f"{report['flagged_scenes']} scenes flagged",
                    "info",
                )
            else:
                self.log(f"Tag gap detection failed: {report.get('error', 'Unknown')}", "error")

        except Exception as e:
            self.error(f"Detect Tag Gaps failed: {e}")
```

**Step 2: Add per-scene query handler**

```python
    def run_get_scene_tag_gaps(self, args: Dict[str, Any]):
        """Get tag gap detail for a specific scene (sidebar query)."""
        try:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            scene_id = args.get("scene_id")
            if not scene_id:
                self.error("scene_id argument required")
                return

            plugin_settings = self.get_plugin_settings("stash-copilot")

            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TagGapDetectionTask(
                stash=self.stash,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            result = task.get_scene_gaps_detail(int(scene_id))

            # Include similar uncovered scenes if data exists
            if result.get("has_data"):
                similar = task.find_similar_uncovered(int(scene_id), limit=8)
                result["similar_uncovered"] = similar

            # Save result for frontend polling
            import json
            import os

            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(plugin_dir, "assets")
            os.makedirs(assets_dir, exist_ok=True)

            request_id = args.get("request_id", f"scene_{scene_id}")
            filepath = os.path.join(assets_dir, f"tag_gaps_scene_{request_id}.json")
            with open(filepath, "w") as f:
                json.dump(result, f)

        except Exception as e:
            self.error(f"Get scene tag gaps failed: {e}")
```

**Step 3: Add elif branches in `run_task`**

In the `run_task()` method (around line 347, before the `else` clause), add:

```python
        elif task_name == "detect_tag_gaps":
            self.run_detect_tag_gaps(args)
        elif task_name == "get_scene_tag_gaps":
            self.run_get_scene_tag_gaps(args)
```

**Step 4: Add YAML task definitions**

In `stash-copilot.yml`, add to the `tasks:` list:

```yaml
  - name: Detect Tag Gaps
    description: Detect visual content in scenes not covered by existing tags
    defaultArgs:
      mode: detect_tag_gaps
      request_id: ""
      force: "false"

  - name: Get Scene Tag Gaps
    description: Get tag gap details for a specific scene
    defaultArgs:
      mode: get_scene_tag_gaps
      scene_id: ""
      request_id: ""
```

**Step 5: Commit**

```bash
git add stash-copilot.py stash-copilot.yml
git commit -m "feat(tag-gaps): wire up Detect Tag Gaps and Get Scene Tag Gaps tasks"
```

---

### Task 6: AI Insights Modal — Tag Gaps Tab HTML & State

**Files:**
- Modify: `stash-copilot.js` (modal state, tab button, panel HTML)

**Step 1: Add `tag_gaps` to the modal state**

Find the `state` object at the top of `stash-copilot.js` (around line 40-80). Look for the `// AI Insights Modal state` comment and add:

```javascript
        tagGapsLoading: false,
        tagGapsData: null,
        tagGapsRequestId: null,
```

**Step 2: Add tab button in `createInsightsModal()`**

In the tabs section (around line 3737, after the Train tab button), add:

```javascript
                    <button class="stash-copilot-insights-tab ${savedTab === 'tag_gaps' ? 'active' : ''}" data-tab="tag_gaps">Tag Gaps</button>
```

**Step 3: Add the Tag Gaps panel HTML**

After the Train panel (before the closing `</div>` of `stash-copilot-insights-body`, around line 4007), add:

```javascript
                    <!-- Tag Gaps Panel -->
                    <div class="stash-copilot-insights-panel ${savedTab === 'tag_gaps' ? 'active' : ''}" data-tab="tag_gaps">
                        <div class="stash-copilot-tag-gaps-container">
                            <div class="stash-copilot-tag-gaps-controls">
                                <button class="btn btn-primary stash-copilot-tag-gaps-detect-btn">Detect Tag Gaps</button>
                                <label class="stash-copilot-tag-gaps-force-label" title="Recompute all scenes, ignoring cached results">
                                    <input type="checkbox" class="stash-copilot-tag-gaps-force-check" /> Force recompute
                                </label>
                                <span class="stash-copilot-tag-gaps-status"></span>
                            </div>
                            <div class="stash-copilot-tag-gaps-summary" style="display:none">
                                <div class="stash-copilot-tag-gaps-stats"></div>
                            </div>
                            <div class="stash-copilot-tag-gaps-results" style="display:none">
                                <div class="stash-copilot-tag-gaps-scene-list"></div>
                            </div>
                            <div class="stash-copilot-tag-gaps-empty">
                                <div class="stash-copilot-empty-state">
                                    <div class="stash-copilot-empty-state-icon">🏷️</div>
                                    <div class="stash-copilot-empty-state-desc">
                                        <h4>Tag Gap Detection</h4>
                                        <p>Detect visual content in your scenes that isn't covered by any existing tag. Uses frame-level embeddings to find gaps in your tag library.</p>
                                    </div>
                                    <div class="stash-copilot-empty-state-tip">
                                        <p>Click <strong>Detect Tag Gaps</strong> to analyze your library. Requires scenes to be embedded first.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(tag-gaps): add Tag Gaps tab HTML to AI Insights modal"
```

---

### Task 7: AI Insights Modal — Tag Gaps Trigger, Polling, and Results Rendering

**Files:**
- Modify: `stash-copilot.js` (add functions after existing modal functions)

**Step 1: Add the detect trigger function**

Find the `setupInsightsModalEvents` function (around line 4641) and add a listener for the detect button. Then add the following functions near the other modal tab functions (e.g., after `buildTasteMap`):

```javascript
    async function detectTagGaps(modal) {
        if (state.tagGapsLoading) return;

        state.tagGapsLoading = true;
        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
        const emptyEl = modal.querySelector('.stash-copilot-tag-gaps-empty');

        detectBtn.disabled = true;
        detectBtn.innerHTML = '<span class="stash-copilot-spinner"></span> Detecting...';
        statusEl.textContent = 'Analyzing frames against tag embeddings...';
        if (emptyEl) emptyEl.style.display = 'none';

        const requestId = `tag_gaps_${Date.now()}`;
        state.tagGapsRequestId = requestId;

        try {
            const forceCheck = modal.querySelector('.stash-copilot-tag-gaps-force-check');
            const force = forceCheck && forceCheck.checked ? 'true' : 'false';

            await runPluginTask('Detect Tag Gaps', { request_id: requestId, force: force });
            pollTagGapsResults(modal, requestId);
        } catch (e) {
            log(`Detect Tag Gaps error: ${e.message}`, 'error');
            state.tagGapsLoading = false;
            detectBtn.disabled = false;
            detectBtn.textContent = 'Detect Tag Gaps';
            statusEl.textContent = `Error: ${e.message}`;
        }
    }
```

**Step 2: Add the polling function**

```javascript
    function pollTagGapsResults(modal, requestId) {
        const resultFile = `/plugin/stash-copilot/assets/tag_gaps_${requestId}.json`;

        const interval = setInterval(async () => {
            if (state.tagGapsRequestId !== requestId) {
                clearInterval(interval);
                return;
            }

            try {
                const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.status === 'complete') {
                        clearInterval(interval);
                        state.tagGapsData = data;
                        state.tagGapsLoading = false;
                        renderTagGapsResults(modal, data);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        state.tagGapsLoading = false;
                        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
                        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
                        if (detectBtn) {
                            detectBtn.disabled = false;
                            detectBtn.textContent = 'Detect Tag Gaps';
                        }
                        if (statusEl) statusEl.textContent = `Error: ${data.error || 'Unknown'}`;
                    }
                }
            } catch (e) {
                // File not ready yet, keep polling
            }
        }, 1500);
    }
```

**Step 3: Add the results renderer**

```javascript
    function renderTagGapsResults(modal, data) {
        const detectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        const statusEl = modal.querySelector('.stash-copilot-tag-gaps-status');
        const summaryEl = modal.querySelector('.stash-copilot-tag-gaps-summary');
        const statsEl = modal.querySelector('.stash-copilot-tag-gaps-stats');
        const resultsEl = modal.querySelector('.stash-copilot-tag-gaps-results');
        const sceneListEl = modal.querySelector('.stash-copilot-tag-gaps-scene-list');

        if (detectBtn) {
            detectBtn.disabled = false;
            detectBtn.textContent = 'Re-detect';
        }
        if (statusEl) statusEl.textContent = '';

        // Summary stats
        if (summaryEl && statsEl) {
            const avgPct = Math.round(data.avg_coverage * 100);
            statsEl.innerHTML = `
                <div class="stash-copilot-tag-gaps-stat-row">
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${avgPct}%</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Avg Coverage</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.flagged_scenes}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Scenes Flagged</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.total_scenes}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Scenes Analyzed</span>
                    </div>
                    <div class="stash-copilot-tag-gaps-stat">
                        <span class="stash-copilot-tag-gaps-stat-value">${data.threshold.toFixed(3)}</span>
                        <span class="stash-copilot-tag-gaps-stat-label">Threshold</span>
                    </div>
                </div>
            `;
            summaryEl.style.display = '';
        }

        // Scene list (only scenes with uncovered content)
        if (resultsEl && sceneListEl) {
            const flagged = data.scenes.filter(s => s.coverage_ratio < 1.0);
            flagged.sort((a, b) => a.coverage_ratio - b.coverage_ratio);

            if (flagged.length === 0) {
                sceneListEl.innerHTML = '<p class="stash-copilot-info">All scenes are fully covered by existing tags.</p>';
            } else {
                sceneListEl.innerHTML = flagged.slice(0, 50).map(scene => {
                    const covPct = Math.round(scene.coverage_ratio * 100);
                    const barColor = covPct > 75 ? '#10b981' : covPct > 50 ? '#f59e0b' : '#ef4444';
                    const tagHints = (scene.top_uncovered_tags || [])
                        .slice(0, 3)
                        .map(t => `<span class="stash-copilot-tag-gaps-hint">${t.tag} (${(t.similarity * 100).toFixed(0)}%)</span>`)
                        .join(' ');

                    return `
                        <div class="stash-copilot-tag-gaps-scene-row" data-scene-id="${scene.scene_id}">
                            <div class="stash-copilot-tag-gaps-scene-info">
                                <a href="/scenes/${scene.scene_id}" class="stash-copilot-tag-gaps-scene-link">Scene ${scene.scene_id}</a>
                                <span class="stash-copilot-tag-gaps-uncovered-count">${scene.uncovered_frames} uncovered</span>
                            </div>
                            <div class="stash-copilot-tag-gaps-coverage-bar-container">
                                <div class="stash-copilot-tag-gaps-coverage-bar" style="width: ${covPct}%; background: ${barColor}"></div>
                                <span class="stash-copilot-tag-gaps-coverage-label">${covPct}%</span>
                            </div>
                            <div class="stash-copilot-tag-gaps-hints">${tagHints}</div>
                        </div>
                    `;
                }).join('');
            }
            resultsEl.style.display = '';
        }
    }
```

**Step 4: Wire up the event listener in `setupInsightsModalEvents`**

In `setupInsightsModalEvents` (around line 4641), add:

```javascript
        // Tag Gaps detect button
        const tagGapsDetectBtn = modal.querySelector('.stash-copilot-tag-gaps-detect-btn');
        if (tagGapsDetectBtn) {
            tagGapsDetectBtn.addEventListener('click', () => detectTagGaps(modal));
        }
```

**Step 5: Auto-load cached results on tab switch**

In the tab switching logic within `setupInsightsModalEvents`, add a handler for the `tag_gaps` tab to auto-load the latest results if they exist:

```javascript
        // In the tab click handler, after switching to tag_gaps tab:
        // Check for and load cached results
        if (state.tagGapsData) {
            renderTagGapsResults(modal, state.tagGapsData);
        } else {
            // Try loading latest results file
            fetch(`/plugin/stash-copilot/assets/tag_gaps_latest.json?t=${Date.now()}`, { cache: 'no-store' })
                .then(r => r.ok ? r.json() : null)
                .then(data => {
                    if (data && data.status === 'complete') {
                        state.tagGapsData = data;
                        renderTagGapsResults(modal, data);
                    }
                })
                .catch(() => {});
        }
```

**Step 6: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(tag-gaps): add tag gap detection trigger, polling, and results rendering"
```

---

### Task 8: AI Insights Modal — Tag Gaps CSS

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add Tag Gaps styles**

Add to the CSS file (find an appropriate location, e.g., after taste map styles):

```css
/* ===== Tag Gaps ===== */
.stash-copilot-tag-gaps-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}

.stash-copilot-tag-gaps-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.stash-copilot-tag-gaps-force-label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.6);
    cursor: pointer;
}

.stash-copilot-tag-gaps-status {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.5);
}

.stash-copilot-tag-gaps-stat-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}

.stash-copilot-tag-gaps-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.stash-copilot-tag-gaps-stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #8b5cf6;
}

.stash-copilot-tag-gaps-stat-label {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stash-copilot-tag-gaps-scene-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 500px;
    overflow-y: auto;
}

.stash-copilot-tag-gaps-scene-row {
    display: grid;
    grid-template-columns: 200px 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    transition: background 0.15s;
}

.stash-copilot-tag-gaps-scene-row:hover {
    background: rgba(255, 255, 255, 0.06);
}

.stash-copilot-tag-gaps-scene-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.stash-copilot-tag-gaps-scene-link {
    color: #e2e8f0;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.9rem;
}

.stash-copilot-tag-gaps-scene-link:hover {
    color: #8b5cf6;
}

.stash-copilot-tag-gaps-uncovered-count {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.4);
}

.stash-copilot-tag-gaps-coverage-bar-container {
    position: relative;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    overflow: hidden;
    min-width: 100px;
}

.stash-copilot-tag-gaps-coverage-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.stash-copilot-tag-gaps-coverage-label {
    position: absolute;
    right: -36px;
    top: -4px;
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.6);
    width: 32px;
    text-align: right;
}

.stash-copilot-tag-gaps-hints {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.stash-copilot-tag-gaps-hint {
    font-size: 0.7rem;
    padding: 2px 6px;
    background: rgba(139, 92, 246, 0.15);
    color: rgba(139, 92, 246, 0.8);
    border-radius: 4px;
    white-space: nowrap;
}
```

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(tag-gaps): add CSS styles for Tag Gaps modal tab"
```

---

### Task 9: Scene Sidebar — Gaps Tab Injection

**Files:**
- Modify: `stash-copilot.js` (sidebar tab injection, state, lazy loading)

**Step 1: Add Gaps to sidebar tab definitions**

In `injectSceneTabs()` (around line 10493), modify the `tabs` array:

```javascript
            const tabs = [
                { key: 'scene-copilot-analyze', label: 'Analyze', icon: '👁' },
                { key: 'scene-copilot-similar', label: 'Similar', icon: '🔍' },
                { key: 'scene-copilot-recs', label: 'Recs', icon: '⭐' },
                { key: 'scene-copilot-gaps', label: 'Gaps', icon: '🏷️' },
            ];
```

**Step 2: Add `gaps` to `sidebarTabState.contentLoaded`**

Update `sidebarTabState` (line 10447-10451):

```javascript
        contentLoaded: {
            analyze: false,
            similar: false,
            recs: false,
            gaps: false,
        }
```

Also update the two places where `contentLoaded` is reset (lines 10478, 10490) to include `gaps: false`.

**Step 3: Add case to `loadSidebarTabContent`**

In the switch statement (around line 10594), add:

```javascript
            case 'gaps':
                renderSidebarGapsContent(container, sceneId);
                break;
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(tag-gaps): add Gaps tab to scene sidebar injection"
```

---

### Task 10: Scene Sidebar — Gaps Tab Content Rendering

**Files:**
- Modify: `stash-copilot.js` (add render function and data fetching)

**Step 1: Add the `renderSidebarGapsContent` function**

Add near the other `renderSidebar*` functions:

```javascript
    function renderSidebarGapsContent(container, sceneId) {
        container.innerHTML = `
            <div class="stash-copilot-sidebar-gaps">
                <div class="stash-copilot-sidebar-header">
                    <span class="stash-copilot-sidebar-title">Tag Gaps</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Loading coverage data...</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-content" style="display: none"></div>
                <div class="stash-copilot-sidebar-gaps-empty" style="display: none">
                    <p>No tag gap data for this scene.</p>
                    <p class="stash-copilot-sidebar-gaps-empty-hint">Run <strong>Detect Tag Gaps</strong> from AI Insights to analyze your library.</p>
                </div>
            </div>
        `;

        loadSidebarGapsData(container, sceneId);
    }
```

**Step 2: Add data fetching function**

```javascript
    async function loadSidebarGapsData(container, sceneId) {
        const loadingEl = container.querySelector('.stash-copilot-sidebar-gaps-loading');
        const contentEl = container.querySelector('.stash-copilot-sidebar-gaps-content');
        const emptyEl = container.querySelector('.stash-copilot-sidebar-gaps-empty');

        try {
            const requestId = `${sceneId}_${Date.now()}`;
            await runPluginTask('Get Scene Tag Gaps', { scene_id: String(sceneId), request_id: requestId });

            // Poll for result
            const resultFile = `/plugin/stash-copilot/assets/tag_gaps_scene_${requestId}.json`;
            const maxAttempts = 30;
            let attempts = 0;

            const poll = setInterval(async () => {
                attempts++;
                if (attempts > maxAttempts) {
                    clearInterval(poll);
                    if (loadingEl) loadingEl.style.display = 'none';
                    if (emptyEl) {
                        emptyEl.style.display = '';
                        emptyEl.querySelector('p').textContent = 'Timed out loading gap data.';
                    }
                    return;
                }

                try {
                    const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                    if (resp.ok) {
                        const data = await resp.json();
                        clearInterval(poll);
                        if (loadingEl) loadingEl.style.display = 'none';

                        if (data.has_data === false) {
                            if (emptyEl) emptyEl.style.display = '';
                        } else {
                            if (contentEl) {
                                contentEl.style.display = '';
                                renderSidebarGapsDetail(contentEl, data, sceneId);
                            }
                        }
                    }
                } catch (e) {
                    // Not ready yet
                }
            }, 1000);
        } catch (e) {
            log(`Load sidebar gaps error: ${e.message}`, 'error');
            if (loadingEl) loadingEl.style.display = 'none';
            if (emptyEl) emptyEl.style.display = '';
        }
    }
```

**Step 3: Add the detail renderer**

```javascript
    function renderSidebarGapsDetail(container, data, sceneId) {
        const covPct = Math.round(data.coverage_ratio * 100);
        const barColor = covPct > 75 ? '#10b981' : covPct > 50 ? '#f59e0b' : '#ef4444';

        let html = `
            <div class="stash-copilot-sidebar-gaps-coverage">
                <div class="stash-copilot-sidebar-gaps-coverage-header">
                    <span>Coverage</span>
                    <span style="color: ${barColor}; font-weight: 600">${covPct}%</span>
                </div>
                <div class="stash-copilot-sidebar-gaps-coverage-bar-wrap">
                    <div class="stash-copilot-sidebar-gaps-coverage-fill" style="width: ${covPct}%; background: ${barColor}"></div>
                </div>
                <div class="stash-copilot-sidebar-gaps-coverage-detail">
                    ${data.covered_frames} covered / ${data.uncovered_frames} uncovered of ${data.total_frames} frames
                </div>
            </div>
        `;

        // Nearest tags for uncovered content
        if (data.nearest_tags && data.nearest_tags.length > 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-nearest">
                    <div class="stash-copilot-sidebar-gaps-section-title">Nearest Tags (not matching)</div>
                    <div class="stash-copilot-sidebar-gaps-tag-list">
                        ${data.nearest_tags.slice(0, 8).map(t => `
                            <div class="stash-copilot-sidebar-gaps-tag-row">
                                <span class="stash-copilot-sidebar-gaps-tag-name">${t.tag}</span>
                                <span class="stash-copilot-sidebar-gaps-tag-sim">${(t.avg_similarity * 100).toFixed(0)}%</span>
                                <span class="stash-copilot-sidebar-gaps-tag-count">${t.frame_count} frames</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Uncovered frames strip
        if (data.uncovered_frame_list && data.uncovered_frame_list.length > 0) {
            const frameDir = `/plugin/stash-copilot/assets/embedded_frames/scene_${sceneId}`;
            html += `
                <div class="stash-copilot-sidebar-gaps-frames">
                    <div class="stash-copilot-sidebar-gaps-section-title">Uncovered Frames</div>
                    <div class="stash-copilot-sidebar-gaps-frame-strip">
                        ${data.uncovered_frame_list.slice(0, 20).map(f => {
                            const frameNum = String(f.frame_index + 1).padStart(4, '0');
                            const src = `${frameDir}/frame_${frameNum}.jpg`;
                            const mins = Math.floor(f.timestamp / 60);
                            const secs = Math.floor(f.timestamp % 60);
                            const ts = `${mins}:${String(secs).padStart(2, '0')}`;
                            return `
                                <div class="stash-copilot-sidebar-gaps-frame" title="${ts} — nearest: ${f.best_tag} (${(f.best_similarity * 100).toFixed(0)}%)">
                                    <img src="${src}" loading="lazy" alt="Frame at ${ts}" />
                                    <span class="stash-copilot-sidebar-gaps-frame-ts">${ts}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        // Similar uncovered scenes
        if (data.similar_uncovered && data.similar_uncovered.length > 0) {
            html += `
                <div class="stash-copilot-sidebar-gaps-similar">
                    <div class="stash-copilot-sidebar-gaps-section-title">Similar Uncovered Content</div>
                    <div class="stash-copilot-sidebar-gaps-similar-list">
                        ${data.similar_uncovered.slice(0, 5).map(s => `
                            <a href="/scenes/${s.scene_id}" class="stash-copilot-sidebar-gaps-similar-item">
                                <span>Scene ${s.scene_id}</span>
                                <span class="stash-copilot-sidebar-gaps-sim-score">${(s.similarity * 100).toFixed(0)}% similar</span>
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(tag-gaps): add Gaps sidebar tab with coverage, frames, and similar scenes"
```

---

### Task 11: Scene Sidebar — Gaps Tab CSS

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add sidebar Gaps styles**

```css
/* ===== Sidebar Gaps Tab ===== */
.stash-copilot-sidebar-gaps {
    padding: 8px 0;
}

.stash-copilot-sidebar-gaps-loading {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.85rem;
}

.stash-copilot-sidebar-gaps-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.stash-copilot-sidebar-gaps-coverage {
    padding: 0 4px;
}

.stash-copilot-sidebar-gaps-coverage-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    margin-bottom: 4px;
    color: rgba(255, 255, 255, 0.7);
}

.stash-copilot-sidebar-gaps-coverage-bar-wrap {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
}

.stash-copilot-sidebar-gaps-coverage-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

.stash-copilot-sidebar-gaps-coverage-detail {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.4);
    margin-top: 4px;
}

.stash-copilot-sidebar-gaps-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.stash-copilot-sidebar-gaps-tag-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stash-copilot-sidebar-gaps-tag-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    font-size: 0.8rem;
}

.stash-copilot-sidebar-gaps-tag-name {
    flex: 1;
    color: rgba(255, 255, 255, 0.8);
}

.stash-copilot-sidebar-gaps-tag-sim {
    color: rgba(139, 92, 246, 0.7);
    font-size: 0.75rem;
}

.stash-copilot-sidebar-gaps-tag-count {
    color: rgba(255, 255, 255, 0.35);
    font-size: 0.7rem;
}

.stash-copilot-sidebar-gaps-frame-strip {
    display: flex;
    gap: 4px;
    overflow-x: auto;
    padding-bottom: 4px;
}

.stash-copilot-sidebar-gaps-frame {
    flex-shrink: 0;
    width: 80px;
    position: relative;
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
}

.stash-copilot-sidebar-gaps-frame img {
    width: 100%;
    height: 45px;
    object-fit: cover;
    display: block;
}

.stash-copilot-sidebar-gaps-frame-ts {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.7);
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.65rem;
    text-align: center;
    padding: 1px 0;
}

.stash-copilot-sidebar-gaps-similar-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stash-copilot-sidebar-gaps-similar-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.8);
    text-decoration: none;
    font-size: 0.8rem;
    transition: background 0.15s;
}

.stash-copilot-sidebar-gaps-similar-item:hover {
    background: rgba(139, 92, 246, 0.1);
    color: #8b5cf6;
}

.stash-copilot-sidebar-gaps-sim-score {
    color: rgba(139, 92, 246, 0.7);
    font-size: 0.75rem;
}

.stash-copilot-sidebar-gaps-empty-hint {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.4);
}
```

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(tag-gaps): add CSS styles for sidebar Gaps tab"
```

---

### Task 12: Integration Test — End-to-End Verification

**Step 1: Verify backend task runs without error**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask
from stash_ai.embeddings.storage import EmbeddingStorage

# Verify the full import chain works
storage = EmbeddingStorage()
print('Storage initialized, schema version:', end=' ')
conn = storage._get_connection()
cursor = conn.cursor()
cursor.execute(\"SELECT value FROM schema_info WHERE key = 'version'\")
print(cursor.fetchone()['value'])

# Check frame_tag_coverage table exists
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='frame_tag_coverage'\")
print(f'frame_tag_coverage table exists: {cursor.fetchone() is not None}')
conn.close()

print('All imports and schema checks passed')
"
```

**Step 2: Verify YAML is valid**

```bash
cd ~/.stash/plugins/stash-copilot && uv run python -c "
import yaml
with open('stash-copilot.yml') as f:
    config = yaml.safe_load(f)
task_names = [t['name'] for t in config['tasks']]
print(f'Total tasks: {len(task_names)}')
assert 'Detect Tag Gaps' in task_names, 'Missing Detect Tag Gaps task'
assert 'Get Scene Tag Gaps' in task_names, 'Missing Get Scene Tag Gaps task'
print('YAML validation passed')
"
```

**Step 3: Check for JS syntax errors**

Open Stash UI in browser, open DevTools console, navigate to any page, and verify no JS errors from stash-copilot.js. Then navigate to the AI Insights modal and verify the Tag Gaps tab appears.

**Step 4: Test via Stash UI**

1. Open AI Insights modal
2. Click "Tag Gaps" tab — should show empty state
3. Click "Detect Tag Gaps" button — should show spinner and progress
4. Wait for completion — should show summary stats and scene list
5. Navigate to a scene page — Gaps tab should appear in sidebar
6. Click Gaps tab — should show coverage bar, uncovered frames, nearest tags

**Step 5: Check logs for errors**

```bash
tail -50 ~/.stash/stash.log | grep -i "error\|warn\|exception\|tag.gap"
```

**Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(tag-gaps): address integration test findings"
```
