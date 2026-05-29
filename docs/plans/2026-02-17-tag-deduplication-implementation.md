# Tag Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pair-by-pair interactive UI for detecting and merging duplicate tags using OpenCLIP embedding similarity.

**Architecture:** Backend task computes all-pairs cosine similarity on cached tag embeddings, filters above 0.75 threshold, returns candidates. Frontend renders a versus-card UI where users review one pair at a time and choose to merge or skip. Merging reassigns scenes and deletes the empty tag.

**Tech Stack:** Python (NumPy, SQLite), JavaScript (injected into Stash task page), Stash GraphQL API

**Design doc:** `docs/plans/2026-02-17-tag-deduplication-design.md`

---

### Task 1: Add dismissed_tag_merges table (SQLite migration v13)

**Files:**
- Modify: `stash_ai/embeddings/storage.py`

**Step 1: Add migration method and bump schema version**

In `storage.py`, change `SCHEMA_VERSION = 12` to `SCHEMA_VERSION = 13`, add the migration call in `_run_migrations()`, and add the migration method:

```python
# In _run_migrations(), after the v12 block:
if current_version < 13:
    self._migrate_to_v13(cursor)
```

```python
def _migrate_to_v13(self, cursor: sqlite3.Cursor) -> None:
    """Add dismissed_tag_merges table (v13)."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dismissed_tag_merges (
            tag_a_name TEXT NOT NULL,
            tag_b_name TEXT NOT NULL,
            dismissed_at TEXT NOT NULL,
            PRIMARY KEY (tag_a_name, tag_b_name)
        )
        """
    )
```

**Step 2: Add storage helper methods**

Add three methods to the `EmbeddingStorage` class:

```python
def save_dismissed_tag_merge(self, tag_a_name: str, tag_b_name: str) -> None:
    """Record that a tag merge pair was dismissed."""
    # Normalize order so (A,B) and (B,A) are the same dismissal
    names = sorted([tag_a_name.lower(), tag_b_name.lower()])
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO dismissed_tag_merges
        (tag_a_name, tag_b_name, dismissed_at)
        VALUES (?, ?, ?)
        """,
        (names[0], names[1], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def get_dismissed_tag_merges(self) -> set[tuple[str, str]]:
    """Get all dismissed tag merge pairs as a set of (name_a, name_b) tuples."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT tag_a_name, tag_b_name FROM dismissed_tag_merges")
    result = {(row["tag_a_name"], row["tag_b_name"]) for row in cursor.fetchall()}
    conn.close()
    return result

def delete_tag_embedding(self, text: str, model_key: str) -> None:
    """Delete a tag embedding by text and model_key."""
    conn = self._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM tag_embeddings WHERE text = ? AND model_key = ?",
        (text.lower(), model_key),
    )
    conn.commit()
    conn.close()
```

**Step 3: Verify migration runs**

Run: `uv run python -c "from stash_ai.embeddings.storage import EmbeddingStorage; s = EmbeddingStorage(); print('v13 OK')"`
Expected: prints "v13 OK" without errors.

**Step 4: Commit**

```bash
git add stash_ai/embeddings/storage.py
git commit -m "feat(tag-dedup): add dismissed_tag_merges table and storage helpers (v13)"
```

---

### Task 2: Create FindDuplicateTagsTask

**Files:**
- Create: `stash_ai/tasks/tag_dedup.py`

**Step 1: Create the task file with types and constructor**

```python
"""Tag deduplication task - find and merge duplicate tags via embedding similarity."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stashapi.stashapp import StashInterface

from stash_ai.embeddings.storage import EmbeddingStorage


class TagInfo(TypedDict):
    """Minimal tag information for dedup candidates."""

    id: int
    name: str
    scene_count: int


class TagDedupCandidate(TypedDict):
    """A pair of tags that may be duplicates."""

    tag_a: TagInfo
    tag_b: TagInfo
    similarity: float
    suggested_keep: str  # "a" or "b"


class FindDuplicateTagsResult(TypedDict):
    """Result from duplicate tag detection."""

    status: str  # "complete", "error", "no_embeddings"
    candidates: list[TagDedupCandidate]
    error: str | None


class MergeTagsResult(TypedDict):
    """Result from merging two tags."""

    status: str  # "complete", "error"
    scenes_updated: int
    error: str | None


class FindDuplicateTagsTask:
    """Find duplicate tags using embedding cosine similarity.

    Computes all-pairs similarity on cached tag embeddings,
    filters above threshold, and returns candidates sorted
    by descending similarity with auto-suggested keep targets.
    """

    SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        stash: StashInterface,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "openclip:ViT-H-14",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def run(self) -> FindDuplicateTagsResult:
        """Find duplicate tag candidates."""
        try:
            # 1. Load tag embeddings (only stash_tag source)
            tag_embeddings_data = self.storage.get_all_tag_embeddings(self.model_key)
            stash_tags_only = [
                t for t in tag_embeddings_data if t["source"] == "stash_tag"
            ]
            if not stash_tags_only:
                return FindDuplicateTagsResult(
                    status="no_embeddings",
                    candidates=[],
                    error="No tag embeddings found. Run 'Build Tag Vocabulary' first.",
                )

            self.log(f"Loaded {len(stash_tags_only)} stash tag embeddings", "info")

            # 2. Build embedding matrix
            tag_names = [t["text"] for t in stash_tags_only]
            embeddings = np.array(
                [t["embedding"] for t in stash_tags_only], dtype=np.float32
            )

            # 3. Compute all-pairs cosine similarity
            similarities = self._compute_all_pairs(embeddings)

            # 4. Extract pairs above threshold
            pairs = self._extract_candidate_pairs(similarities, tag_names)
            self.log(f"Found {len(pairs)} pairs above {self.SIMILARITY_THRESHOLD} threshold", "info")

            if not pairs:
                return FindDuplicateTagsResult(
                    status="complete",
                    candidates=[],
                    error=None,
                )

            # 5. Filter out previously dismissed pairs
            dismissed = self.storage.get_dismissed_tag_merges()
            pairs = [
                (a, b, sim) for a, b, sim in pairs
                if (min(a.lower(), b.lower()), max(a.lower(), b.lower())) not in dismissed
            ]
            self.log(f"{len(pairs)} pairs after excluding dismissed", "info")

            # 6. Get tag IDs and scene counts from Stash
            tag_info_map = self._get_tag_info_with_scene_counts()
            if not tag_info_map:
                return FindDuplicateTagsResult(
                    status="error",
                    candidates=[],
                    error="Failed to load tag info from Stash",
                )

            # 7. Build candidate list with scene counts
            candidates: list[TagDedupCandidate] = []
            for name_a, name_b, sim in pairs:
                info_a = tag_info_map.get(name_a.lower())
                info_b = tag_info_map.get(name_b.lower())
                if not info_a or not info_b:
                    continue

                suggested_keep = "a" if info_a["scene_count"] >= info_b["scene_count"] else "b"
                candidates.append(TagDedupCandidate(
                    tag_a=info_a,
                    tag_b=info_b,
                    similarity=round(sim, 4),
                    suggested_keep=suggested_keep,
                ))

            # Sort by descending similarity
            candidates.sort(key=lambda c: c["similarity"], reverse=True)

            self.log(f"Returning {len(candidates)} dedup candidates", "info")
            return FindDuplicateTagsResult(
                status="complete",
                candidates=candidates,
                error=None,
            )

        except Exception as e:
            self.log(f"Tag dedup error: {e}", "error")
            return FindDuplicateTagsResult(
                status="error",
                candidates=[],
                error=str(e),
            )

    def _compute_all_pairs(
        self, embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute all-pairs cosine similarity matrix.

        Args:
            embeddings: (N, D) array of tag embeddings

        Returns:
            (N, N) similarity matrix
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        return np.dot(normalized, normalized.T)

    def _extract_candidate_pairs(
        self,
        similarities: NDArray[np.float32],
        tag_names: list[str],
    ) -> list[tuple[str, str, float]]:
        """Extract tag pairs above similarity threshold.

        Only checks upper triangle to avoid duplicates.

        Returns:
            List of (tag_a_name, tag_b_name, similarity) sorted descending.
        """
        n = len(tag_names)
        pairs: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(similarities[i, j])
                if sim >= self.SIMILARITY_THRESHOLD:
                    pairs.append((tag_names[i], tag_names[j], sim))

        pairs.sort(key=lambda p: p[2], reverse=True)
        return pairs

    def _get_tag_info_with_scene_counts(self) -> dict[str, TagInfo]:
        """Get tag IDs and scene counts from Stash.

        Returns:
            Dict mapping lowercase tag name to TagInfo.
        """
        try:
            result = self.stash.call_GQL(
                """
                query FindTags {
                    findTags(filter: { per_page: -1 }) {
                        tags {
                            id
                            name
                            scene_count
                        }
                    }
                }
                """
            )
            if not result or "findTags" not in result:
                return {}

            info_map: dict[str, TagInfo] = {}
            for t in result["findTags"]["tags"]:
                name = t["name"]
                if not any(c in name for c in "[](){}<>"):
                    info_map[name.lower()] = TagInfo(
                        id=int(t["id"]),
                        name=name,
                        scene_count=t.get("scene_count", 0),
                    )
            return info_map
        except Exception as e:
            self.log(f"Failed to get tags with scene counts: {e}", "warning")
            return {}
```

**Step 2: Verify import works**

Run: `uv run python -c "from stash_ai.tasks.tag_dedup import FindDuplicateTagsTask; print('import OK')"`
Expected: prints "import OK"

**Step 3: Commit**

```bash
git add stash_ai/tasks/tag_dedup.py
git commit -m "feat(tag-dedup): add FindDuplicateTagsTask with embedding similarity"
```

---

### Task 3: Create MergeTagsTask

**Files:**
- Modify: `stash_ai/tasks/tag_dedup.py`

**Step 1: Add MergeTagsTask class to the existing file**

Append to `stash_ai/tasks/tag_dedup.py`:

```python
class MergeTagsTask:
    """Merge one tag into another: reassign scenes, then delete the source tag.

    Moves all scene associations from remove_tag to keep_tag,
    then deletes remove_tag from Stash and cleans up its embedding.
    """

    def __init__(
        self,
        stash: StashInterface,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "openclip:ViT-H-14",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def run(self, keep_tag_id: int, remove_tag_id: int) -> MergeTagsResult:
        """Merge remove_tag into keep_tag.

        Args:
            keep_tag_id: ID of the tag to keep
            remove_tag_id: ID of the tag to delete after reassignment

        Returns:
            MergeTagsResult with status and scene count
        """
        try:
            # 1. Find all scenes with the remove_tag
            scenes = self._find_scenes_with_tag(remove_tag_id)
            self.log(f"Found {len(scenes)} scenes with tag {remove_tag_id}", "info")

            scenes_updated = 0

            # 2. For each scene: add keep_tag, remove remove_tag
            for scene in scenes:
                scene_id = int(scene["id"])
                existing_tag_ids = {int(t["id"]) for t in scene["tags"]}

                new_tag_ids = existing_tag_ids.copy()
                new_tag_ids.add(keep_tag_id)
                new_tag_ids.discard(remove_tag_id)

                # Only update if tags actually changed
                if new_tag_ids != existing_tag_ids:
                    success = self._update_scene_tags(scene_id, list(new_tag_ids))
                    if not success:
                        return MergeTagsResult(
                            status="error",
                            scenes_updated=scenes_updated,
                            error=f"Failed to update scene {scene_id}. Stopping to prevent partial merge.",
                        )
                    scenes_updated += 1

            # 3. Delete the now-empty tag
            self._destroy_tag(remove_tag_id)
            self.log(f"Deleted tag {remove_tag_id}", "info")

            # 4. Clean up embedding from storage
            # We need the tag name to delete the embedding
            remove_tag_name = self._get_tag_name(remove_tag_id)
            if remove_tag_name:
                self.storage.delete_tag_embedding(remove_tag_name, self.model_key)

            return MergeTagsResult(
                status="complete",
                scenes_updated=scenes_updated,
                error=None,
            )

        except Exception as e:
            self.log(f"Tag merge error: {e}", "error")
            return MergeTagsResult(
                status="error",
                scenes_updated=0,
                error=str(e),
            )

    def _find_scenes_with_tag(self, tag_id: int) -> list[dict[str, Any]]:
        """Find all scenes that have a given tag."""
        try:
            result = self.stash.call_GQL(
                """
                query FindScenes($tag_id: [ID!]) {
                    findScenes(
                        scene_filter: { tags: { value: $tag_id, modifier: INCLUDES } }
                        filter: { per_page: -1 }
                    ) {
                        scenes { id tags { id } }
                    }
                }
                """,
                {"tag_id": [str(tag_id)]},
            )
            if not result or "findScenes" not in result:
                return []
            return result["findScenes"]["scenes"]
        except Exception as e:
            self.log(f"Failed to find scenes with tag {tag_id}: {e}", "warning")
            return []

    def _update_scene_tags(self, scene_id: int, tag_ids: list[int]) -> bool:
        """Update a scene's tags to the given list of tag IDs."""
        try:
            result = self.stash.call_GQL(
                """
                mutation SceneUpdate($id: ID!, $tag_ids: [ID!]) {
                    sceneUpdate(input: { id: $id, tag_ids: $tag_ids }) { id }
                }
                """,
                {"id": str(scene_id), "tag_ids": [str(t) for t in tag_ids]},
            )
            return result is not None and "sceneUpdate" in result
        except Exception as e:
            self.log(f"Failed to update scene {scene_id}: {e}", "warning")
            return False

    def _destroy_tag(self, tag_id: int) -> bool:
        """Delete a tag from Stash."""
        try:
            result = self.stash.call_GQL(
                """
                mutation TagDestroy($id: ID!) {
                    tagDestroy(input: { id: $id })
                }
                """,
                {"id": str(tag_id)},
            )
            return result is not None
        except Exception as e:
            self.log(f"Failed to delete tag {tag_id}: {e}", "warning")
            return False

    def _get_tag_name(self, tag_id: int) -> str | None:
        """Get a tag's name by ID (for embedding cleanup)."""
        try:
            result = self.stash.call_GQL(
                """
                query FindTag($id: ID!) {
                    findTag(id: $id) { name }
                }
                """,
                {"id": str(tag_id)},
            )
            if result and "findTag" in result and result["findTag"]:
                return result["findTag"]["name"]
            return None
        except Exception:
            return None
```

**Step 2: Verify import works**

Run: `uv run python -c "from stash_ai.tasks.tag_dedup import MergeTagsTask; print('import OK')"`
Expected: prints "import OK"

**Step 3: Commit**

```bash
git add stash_ai/tasks/tag_dedup.py
git commit -m "feat(tag-dedup): add MergeTagsTask for scene reassignment and tag deletion"
```

---

### Task 4: Wire tasks into plugin entry point

**Files:**
- Modify: `stash-copilot.py`
- Modify: `stash-copilot.yml`

**Step 1: Add task dispatch routes in `stash-copilot.py`**

In `run_task()` method, add these `elif` branches before the final `else`:

```python
elif task_name == "find_duplicate_tags":
    self.run_find_duplicate_tags(args)
elif task_name == "merge_tags":
    self.run_merge_tags(args)
elif task_name == "dismiss_tag_merge":
    self.run_dismiss_tag_merge(args)
```

**Step 2: Add the handler methods**

Add these methods to `StashCopilotPlugin` (after `run_clear_dismissed_tags` or similar):

```python
def run_find_duplicate_tags(self, args: dict[str, Any]) -> None:
    """Find duplicate tags using embedding similarity."""
    request_id = args.get("request_id", "")

    try:
        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.storage import EmbeddingStorage
        from stash_ai.tasks.tag_dedup import FindDuplicateTagsTask

        plugin_settings = self.get_plugin_settings("stash-copilot")
        image_provider = plugin_settings.get("image_embedding_provider")
        image_model = plugin_settings.get("image_embedding_model")

        if image_provider and image_model:
            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
            )
            model_key = embedding_config.model_key
        else:
            model_key = "openclip:ViT-H-14"

        storage = EmbeddingStorage(model_key=model_key)
        task = FindDuplicateTagsTask(
            stash=self.stash,
            storage=storage,
            log_callback=self.log,
            model_key=model_key,
        )

        result = task.run()

        # Save result for frontend polling
        if request_id:
            assets_dir = os.path.join(PLUGIN_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            result_path = os.path.join(assets_dir, f"tag_dedup_{request_id}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

        if result["status"] == "complete":
            self.log(f"Found {len(result['candidates'])} duplicate tag candidates", "info")
        else:
            self.log(f"Tag dedup: {result.get('error', 'unknown error')}", "warning")

    except Exception as e:
        self.error(f"Find duplicate tags failed: {e}")

def run_merge_tags(self, args: dict[str, Any]) -> None:
    """Merge one tag into another."""
    keep_tag_id = int(args.get("keep_tag_id", 0))
    remove_tag_id = int(args.get("remove_tag_id", 0))
    request_id = args.get("request_id", "")

    if not keep_tag_id or not remove_tag_id:
        self.log("Missing keep_tag_id or remove_tag_id", "error")
        return

    try:
        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.storage import EmbeddingStorage
        from stash_ai.tasks.tag_dedup import MergeTagsTask

        plugin_settings = self.get_plugin_settings("stash-copilot")
        image_provider = plugin_settings.get("image_embedding_provider")
        image_model = plugin_settings.get("image_embedding_model")

        if image_provider and image_model:
            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
            )
            model_key = embedding_config.model_key
        else:
            model_key = "openclip:ViT-H-14"

        storage = EmbeddingStorage(model_key=model_key)
        task = MergeTagsTask(
            stash=self.stash,
            storage=storage,
            log_callback=self.log,
            model_key=model_key,
        )

        result = task.run(keep_tag_id=keep_tag_id, remove_tag_id=remove_tag_id)

        # Save result for frontend
        if request_id:
            assets_dir = os.path.join(PLUGIN_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            result_path = os.path.join(assets_dir, f"tag_merge_{request_id}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

        if result["status"] == "complete":
            self.log(f"Merged tags: {result['scenes_updated']} scenes updated", "info")
        else:
            self.log(f"Tag merge error: {result.get('error')}", "warning")

    except Exception as e:
        self.error(f"Merge tags failed: {e}")

def run_dismiss_tag_merge(self, args: dict[str, Any]) -> None:
    """Dismiss a tag merge candidate (not duplicates)."""
    tag_a_name = args.get("tag_a_name", "")
    tag_b_name = args.get("tag_b_name", "")
    request_id = args.get("request_id", "")

    if not tag_a_name or not tag_b_name:
        self.log("Missing tag_a_name or tag_b_name", "error")
        return

    try:
        from stash_ai.embeddings.storage import EmbeddingStorage

        storage = EmbeddingStorage()
        storage.save_dismissed_tag_merge(tag_a_name, tag_b_name)
        self.log(f"Dismissed merge: {tag_a_name} / {tag_b_name}", "info")

        # Save confirmation for frontend
        if request_id:
            assets_dir = os.path.join(PLUGIN_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            result_path = os.path.join(assets_dir, f"tag_dismiss_{request_id}.json")
            with open(result_path, "w") as f:
                json.dump({"status": "complete"}, f)

    except Exception as e:
        self.log(f"Failed to dismiss tag merge: {e}", "error")
```

**Step 3: Register tasks in `stash-copilot.yml`**

Add after the existing tag tasks section:

```yaml
  - name: Find Duplicate Tags
    description: Find duplicate tags using embedding similarity for review and merge
    defaultArgs:
      mode: find_duplicate_tags
      request_id: ""

  - name: Merge Tags
    description: Merge one tag into another (reassign scenes, delete source tag)
    defaultArgs:
      mode: merge_tags
      keep_tag_id: ""
      remove_tag_id: ""
      request_id: ""

  - name: Dismiss Tag Merge
    description: Dismiss a duplicate tag candidate (mark as not duplicates)
    defaultArgs:
      mode: dismiss_tag_merge
      tag_a_name: ""
      tag_b_name: ""
      request_id: ""
```

**Step 4: Verify plugin can parse the new config**

Run: `uv run python -c "import yaml; d = yaml.safe_load(open('stash-copilot.yml')); print([t['name'] for t in d['tasks'] if 'Tag' in t['name'] or 'tag' in t['name']])"`
Expected: List including "Find Duplicate Tags", "Merge Tags", "Dismiss Tag Merge"

**Step 5: Commit**

```bash
git add stash-copilot.py stash-copilot.yml
git commit -m "feat(tag-dedup): wire find/merge/dismiss tasks into plugin entry point"
```

---

### Task 5: Build the frontend tag dedup UI

**Files:**
- Modify: `stash-copilot.js`

This is the largest task. The UI needs to:
1. Detect when the "Find Duplicate Tags" task is running/complete
2. Render a pair-by-pair review interface on the Stash task page
3. Handle keep left / keep right / skip actions
4. Show progress and end summary

**Step 1: Add the tag dedup state variables**

Find the global `state` object in `stash-copilot.js` and add:

```javascript
// Tag dedup state
tagDedupCandidates: [],
tagDedupCurrentIndex: 0,
tagDedupMergeCount: 0,
tagDedupSkipCount: 0,
tagDedupScenesUpdated: 0,
tagDedupPollInterval: null,
tagDedupRequestId: null,
```

**Step 2: Add the CSS styles**

Add to the CSS injection section of `stash-copilot.js`:

```css
/* Tag Dedup UI */
.stash-copilot-dedup-container {
    max-width: 700px;
    margin: 2rem auto;
    padding: 2rem;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stash-copilot-dedup-header {
    text-align: center;
    margin-bottom: 2rem;
}

.stash-copilot-dedup-header h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stash-copilot-dedup-pair-info {
    color: #9ca3af;
    font-size: 0.9rem;
}

.stash-copilot-dedup-similarity {
    color: #a78bfa;
    font-weight: 600;
}

.stash-copilot-dedup-versus {
    display: flex;
    align-items: stretch;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.stash-copilot-dedup-card {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 2px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
}

.stash-copilot-dedup-card:hover {
    border-color: rgba(167, 139, 250, 0.5);
    background: rgba(167, 139, 250, 0.08);
}

.stash-copilot-dedup-card.selected {
    border-color: #a78bfa;
    background: rgba(167, 139, 250, 0.15);
    box-shadow: 0 0 20px rgba(167, 139, 250, 0.2);
}

.stash-copilot-dedup-card.removing {
    animation: dedupCardFadeOut 0.4s ease forwards;
}

@keyframes dedupCardFadeOut {
    0% { opacity: 1; transform: scale(1); }
    100% { opacity: 0; transform: scale(0.8); }
}

.stash-copilot-dedup-tag-name {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.75rem;
    word-break: break-word;
}

.stash-copilot-dedup-scene-count {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-bottom: 0.5rem;
}

.stash-copilot-dedup-keep-badge {
    display: inline-block;
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stash-copilot-dedup-vs {
    display: flex;
    align-items: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #6b7280;
    user-select: none;
}

.stash-copilot-dedup-actions {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stash-copilot-dedup-btn {
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05);
    color: #e5e7eb;
    font-size: 0.95rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stash-copilot-dedup-btn:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.3);
}

.stash-copilot-dedup-btn.keep-left {
    border-color: rgba(96, 165, 250, 0.4);
    color: #60a5fa;
}

.stash-copilot-dedup-btn.keep-left:hover {
    background: rgba(96, 165, 250, 0.15);
}

.stash-copilot-dedup-btn.keep-right {
    border-color: rgba(167, 139, 250, 0.4);
    color: #a78bfa;
}

.stash-copilot-dedup-btn.keep-right:hover {
    background: rgba(167, 139, 250, 0.15);
}

.stash-copilot-dedup-btn.skip {
    color: #6b7280;
}

.stash-copilot-dedup-progress {
    text-align: center;
    margin-top: 1rem;
}

.stash-copilot-dedup-progress-bar {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.stash-copilot-dedup-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    transition: width 0.3s ease;
}

.stash-copilot-dedup-progress-text {
    font-size: 0.85rem;
    color: #6b7280;
}

.stash-copilot-dedup-summary {
    text-align: center;
    padding: 2rem;
}

.stash-copilot-dedup-summary h3 {
    font-size: 1.3rem;
    margin-bottom: 1rem;
    color: #a78bfa;
}

.stash-copilot-dedup-summary-stats {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
}

.stash-copilot-dedup-stat {
    text-align: center;
}

.stash-copilot-dedup-stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e5e7eb;
}

.stash-copilot-dedup-stat-label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stash-copilot-dedup-keyboard-hint {
    text-align: center;
    font-size: 0.8rem;
    color: #4b5563;
    margin-top: 0.5rem;
}

.stash-copilot-dedup-keyboard-hint kbd {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-family: monospace;
    font-size: 0.75rem;
}

.stash-copilot-dedup-loading {
    text-align: center;
    padding: 3rem;
    color: #9ca3af;
}

.stash-copilot-dedup-empty {
    text-align: center;
    padding: 3rem;
    color: #6b7280;
}
```

**Step 3: Add the main rendering functions**

```javascript
function initTagDedupUI(candidates) {
    state.tagDedupCandidates = candidates;
    state.tagDedupCurrentIndex = 0;
    state.tagDedupMergeCount = 0;
    state.tagDedupSkipCount = 0;
    state.tagDedupScenesUpdated = 0;
    renderTagDedupPair();
}

function renderTagDedupPair() {
    const container = document.querySelector('.stash-copilot-dedup-container');
    if (!container) return;

    const candidates = state.tagDedupCandidates;
    const idx = state.tagDedupCurrentIndex;

    // All pairs reviewed - show summary
    if (idx >= candidates.length) {
        renderTagDedupSummary(container);
        return;
    }

    const candidate = candidates[idx];
    const total = candidates.length;
    const similarityPct = Math.round(candidate.similarity * 100);

    container.innerHTML = `
        <div class="stash-copilot-dedup-header">
            <h2>Tag Deduplication Review</h2>
            <div class="stash-copilot-dedup-pair-info">
                Pair ${idx + 1} of ${total} &middot;
                <span class="stash-copilot-dedup-similarity">${similarityPct}% similar</span>
            </div>
        </div>

        <div class="stash-copilot-dedup-versus">
            <div class="stash-copilot-dedup-card ${candidate.suggested_keep === 'a' ? 'selected' : ''}"
                 data-side="a" onclick="handleDedupKeep('a')">
                <div class="stash-copilot-dedup-tag-name">${escapeHtml(candidate.tag_a.name)}</div>
                <div class="stash-copilot-dedup-scene-count">${candidate.tag_a.scene_count} scenes</div>
                ${candidate.suggested_keep === 'a' ? '<div class="stash-copilot-dedup-keep-badge">KEEP</div>' : ''}
            </div>

            <div class="stash-copilot-dedup-vs">VS</div>

            <div class="stash-copilot-dedup-card ${candidate.suggested_keep === 'b' ? 'selected' : ''}"
                 data-side="b" onclick="handleDedupKeep('b')">
                <div class="stash-copilot-dedup-tag-name">${escapeHtml(candidate.tag_b.name)}</div>
                <div class="stash-copilot-dedup-scene-count">${candidate.tag_b.scene_count} scenes</div>
                ${candidate.suggested_keep === 'b' ? '<div class="stash-copilot-dedup-keep-badge">KEEP</div>' : ''}
            </div>
        </div>

        <div class="stash-copilot-dedup-actions">
            <button class="stash-copilot-dedup-btn keep-left" onclick="handleDedupKeep('a')">
                &larr; Keep Left
            </button>
            <button class="stash-copilot-dedup-btn skip" onclick="handleDedupSkip()">
                Skip
            </button>
            <button class="stash-copilot-dedup-btn keep-right" onclick="handleDedupKeep('b')">
                Keep Right &rarr;
            </button>
        </div>

        <div class="stash-copilot-dedup-keyboard-hint">
            <kbd>&larr;</kbd> Keep Left &nbsp;&nbsp;
            <kbd>&darr;</kbd> Skip &nbsp;&nbsp;
            <kbd>&rarr;</kbd> Keep Right
        </div>

        <div class="stash-copilot-dedup-progress">
            <div class="stash-copilot-dedup-progress-bar">
                <div class="stash-copilot-dedup-progress-fill"
                     style="width: ${Math.round((idx / total) * 100)}%"></div>
            </div>
            <div class="stash-copilot-dedup-progress-text">
                ${state.tagDedupMergeCount} merged &middot; ${state.tagDedupSkipCount} skipped
            </div>
        </div>
    `;
}

function renderTagDedupSummary(container) {
    container.innerHTML = `
        <div class="stash-copilot-dedup-summary">
            <h3>Deduplication Complete</h3>
            <div class="stash-copilot-dedup-summary-stats">
                <div class="stash-copilot-dedup-stat">
                    <div class="stash-copilot-dedup-stat-value">${state.tagDedupMergeCount}</div>
                    <div class="stash-copilot-dedup-stat-label">Tags Merged</div>
                </div>
                <div class="stash-copilot-dedup-stat">
                    <div class="stash-copilot-dedup-stat-value">${state.tagDedupSkipCount}</div>
                    <div class="stash-copilot-dedup-stat-label">Skipped</div>
                </div>
                <div class="stash-copilot-dedup-stat">
                    <div class="stash-copilot-dedup-stat-value">${state.tagDedupScenesUpdated}</div>
                    <div class="stash-copilot-dedup-stat-label">Scenes Updated</div>
                </div>
            </div>
        </div>
    `;
    // Remove keyboard listener
    document.removeEventListener('keydown', handleDedupKeyboard);
}
```

**Step 4: Add action handlers**

```javascript
async function handleDedupKeep(side) {
    const candidate = state.tagDedupCandidates[state.tagDedupCurrentIndex];
    if (!candidate) return;

    const keepTag = side === 'a' ? candidate.tag_a : candidate.tag_b;
    const removeTag = side === 'a' ? candidate.tag_b : candidate.tag_a;
    const removeSide = side === 'a' ? 'b' : 'a';

    // Animate the removing card
    const removeCard = document.querySelector(`.stash-copilot-dedup-card[data-side="${removeSide}"]`);
    if (removeCard) removeCard.classList.add('removing');

    // Execute merge via plugin task
    const requestId = `dedup_merge_${Date.now()}`;
    await runPluginTask('Merge Tags', {
        mode: 'merge_tags',
        keep_tag_id: String(keepTag.id),
        remove_tag_id: String(removeTag.id),
        request_id: requestId,
    });

    // Poll for result
    const resultFile = `/plugin/stash-copilot/assets/tag_merge_${requestId}.json`;
    let attempts = 0;
    const pollMerge = setInterval(async () => {
        attempts++;
        try {
            const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
            if (resp.ok) {
                const data = await resp.json();
                clearInterval(pollMerge);
                if (data.status === 'complete') {
                    state.tagDedupMergeCount++;
                    state.tagDedupScenesUpdated += data.scenes_updated || 0;

                    // Remove the merged tag from remaining candidates too
                    removeMergedTagFromCandidates(removeTag.id);
                } else {
                    log(`Merge failed: ${data.error}`);
                }
                state.tagDedupCurrentIndex++;
                renderTagDedupPair();
            }
        } catch (e) { /* file not ready */ }
        if (attempts > 100) {
            clearInterval(pollMerge);
            log('Merge timed out');
            state.tagDedupCurrentIndex++;
            renderTagDedupPair();
        }
    }, 200);
}

function removeMergedTagFromCandidates(removedTagId) {
    // Filter out any remaining candidates that reference the now-deleted tag
    const remaining = state.tagDedupCandidates.slice(state.tagDedupCurrentIndex + 1);
    const filtered = remaining.filter(
        c => c.tag_a.id !== removedTagId && c.tag_b.id !== removedTagId
    );
    state.tagDedupCandidates = [
        ...state.tagDedupCandidates.slice(0, state.tagDedupCurrentIndex + 1),
        ...filtered,
    ];
}

async function handleDedupSkip() {
    const candidate = state.tagDedupCandidates[state.tagDedupCurrentIndex];
    if (!candidate) return;

    // Dismiss via plugin task
    const requestId = `dedup_dismiss_${Date.now()}`;
    await runPluginTask('Dismiss Tag Merge', {
        mode: 'dismiss_tag_merge',
        tag_a_name: candidate.tag_a.name,
        tag_b_name: candidate.tag_b.name,
        request_id: requestId,
    });

    state.tagDedupSkipCount++;
    state.tagDedupCurrentIndex++;
    renderTagDedupPair();
}

function handleDedupKeyboard(event) {
    // Don't capture if user is typing in an input
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;

    if (event.key === 'ArrowLeft') {
        event.preventDefault();
        handleDedupKeep('a');
    } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        handleDedupKeep('b');
    } else if (event.key === 'ArrowDown' || event.key === 's') {
        event.preventDefault();
        handleDedupSkip();
    }
}
```

**Step 5: Add the trigger and polling for the initial scan**

```javascript
async function startTagDedup() {
    const requestId = `dedup_${Date.now()}`;
    state.tagDedupRequestId = requestId;

    // Inject container into Stash task page
    const container = document.createElement('div');
    container.className = 'stash-copilot-dedup-container';
    container.innerHTML = '<div class="stash-copilot-dedup-loading">Scanning for duplicate tags...</div>';

    // Find appropriate mount point (Stash task results area or body)
    const mountPoint = document.querySelector('.container-fluid') || document.body;
    mountPoint.appendChild(container);

    // Start backend scan
    await runPluginTask('Find Duplicate Tags', {
        mode: 'find_duplicate_tags',
        request_id: requestId,
    });

    // Poll for results
    const resultFile = `/plugin/stash-copilot/assets/tag_dedup_${requestId}.json`;
    state.tagDedupPollInterval = setInterval(async () => {
        if (state.tagDedupRequestId !== requestId) {
            clearInterval(state.tagDedupPollInterval);
            return;
        }

        try {
            const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
            if (resp.ok) {
                const data = await resp.json();
                clearInterval(state.tagDedupPollInterval);
                state.tagDedupPollInterval = null;

                if (data.status === 'complete' && data.candidates.length > 0) {
                    // Add keyboard listener
                    document.addEventListener('keydown', handleDedupKeyboard);
                    initTagDedupUI(data.candidates);
                } else if (data.status === 'complete') {
                    container.innerHTML = '<div class="stash-copilot-dedup-empty">No duplicate tags found above 75% similarity.</div>';
                } else if (data.status === 'no_embeddings') {
                    container.innerHTML = '<div class="stash-copilot-dedup-empty">No tag embeddings found. Please run "Build Tag Vocabulary" first.</div>';
                } else {
                    container.innerHTML = `<div class="stash-copilot-dedup-empty">Error: ${escapeHtml(data.error || 'Unknown')}</div>`;
                }
            }
        } catch (e) { /* file not ready */ }
    }, 200);

    // Timeout after 30s (this should be fast)
    setTimeout(() => {
        if (state.tagDedupPollInterval) {
            clearInterval(state.tagDedupPollInterval);
            container.innerHTML = '<div class="stash-copilot-dedup-empty">Scan timed out. Check logs for errors.</div>';
        }
    }, 30000);
}
```

**Step 6: Hook the dedup UI into the Stash page**

Add page detection in the existing route handler to detect when the user triggers the "Find Duplicate Tags" task. The exact integration point depends on where Stash shows task output — look at how existing tasks (like tag suggestions) inject their UI, and follow that pattern. If Stash uses a task page, hook into its URL pattern. Otherwise, add a button to the tags page.

The simplest approach: add a listener for the plugin task page and call `startTagDedup()` when the task name matches.

**Step 7: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(tag-dedup): add pair-by-pair review UI with keyboard shortcuts"
```

---

### Task 6: Integration testing with Playwright

**Files:**
- Screenshots saved to: `tests/screenshots/`

**Step 1: Navigate to Stash and trigger the task**

Use Playwright MCP to:
1. Navigate to Stash UI
2. Trigger the "Find Duplicate Tags" task
3. Wait for results to appear
4. Screenshot the initial pair view

**Step 2: Test "Keep Left" action**

1. Click "Keep Left" button (or press Left arrow)
2. Wait for merge to complete (poll logs)
3. Verify next pair appears
4. Screenshot the result
5. Check logs: `grep -i "error\|exception\|merge" ~/.stash/stash.log | tail -20`

**Step 3: Test "Skip" action**

1. Click "Skip" button (or press Down arrow)
2. Verify pair advances
3. Screenshot
4. Verify dismissed pair is persisted: run task again and confirm skipped pair doesn't reappear

**Step 4: Test keyboard shortcuts**

1. Press Right arrow → verify "Keep Right" triggers
2. Press Left arrow → verify "Keep Left" triggers
3. Press Down arrow → verify "Skip" triggers

**Step 5: Test edge cases**

1. Run with no embeddings → verify helpful error message, screenshot
2. If no duplicates exist above threshold → verify "No duplicates found" message, screenshot

**Step 6: Commit test screenshots**

```bash
git add tests/screenshots/tag-dedup-*.png
git commit -m "test(tag-dedup): add Playwright integration test screenshots"
```

---

### Task 7: Final review and cleanup

**Files:**
- Review: all modified files
- Modify: `CLAUDE.md` (update architecture diagram if needed)

**Step 1: Code review**

Review all changes for:
- Type annotations on all public methods
- Error handling follows existing patterns
- No security issues (SQL injection, XSS via tag names — use `escapeHtml()`)
- CSS class naming follows `stash-copilot-dedup-*` convention
- Performance within budget (< 2s for similarity scan, < 1 GB memory)

**Step 2: Update architecture references if needed**

If the architecture diagram in CLAUDE.md needs updating (new task module), add `TagDedup` to the Tasks subgraph.

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: update architecture for tag dedup feature"
```
