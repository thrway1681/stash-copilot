# Frame-Level Semantic Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable searching scenes by individual frame embeddings using FAISS, finding specific poses/moments even in brief appearances.

**Architecture:** Add a parallel search path alongside scene-level search. User toggles between modes. Frame search queries a FAISS index, aggregates results to scenes using max-score, and returns the best-matching frame info (thumbnail + timestamp).

**Tech Stack:** Python (FAISS, NumPy), JavaScript (toggle UI, result rendering), SQLite (frame_embeddings table)

---

## Task 1: Add FAISS Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add faiss-cpu to dependencies**

In `pyproject.toml`, add `faiss-cpu` to the dependencies list:

```toml
dependencies = [
    "faiss-cpu>=1.7.4",
    # ... existing dependencies
]
```

Add after line 9 (after `"matplotlib>=3.7.0",`).

**Step 2: Sync dependencies in worktree**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv sync
```

Expected: faiss-cpu installed successfully

**Step 3: Verify FAISS import works**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
```

Expected: FAISS version printed (e.g., "1.7.4")

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
deps: add faiss-cpu for frame-level semantic search

FAISS enables fast vector similarity search across ~4M frame embeddings.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create FrameSearchIndex Class - Types and Init

**Files:**
- Create: `stash_ai/embeddings/frame_search.py`

**Step 1: Create the file with types and class skeleton**

```python
"""FAISS-based frame-level semantic search index."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameMetadata:
    """Metadata for a single frame in the index."""

    scene_id: int
    frame_index: int
    timestamp: float


@dataclass
class FrameMatch:
    """A single frame match from FAISS search."""

    scene_id: int
    frame_index: int
    timestamp: float
    similarity: float


@dataclass
class SceneMatch:
    """Aggregated scene match (best frame per scene)."""

    scene_id: int
    best_frame_index: int
    best_timestamp: float
    similarity: float


@dataclass
class IndexInfo:
    """Metadata about a built index."""

    model_key: str
    frame_count: int
    scene_count: int
    dimensions: int
    created_at: str


class FrameSearchIndex:
    """FAISS-based frame-level search index.

    Builds and queries a FAISS index over all frame embeddings,
    enabling fast similarity search across millions of frames.
    """

    def __init__(self, assets_dir: str, model_key: str = "siglip"):
        """Initialize the frame search index.

        Args:
            assets_dir: Path to assets directory for index storage
            model_key: Embedding model key (e.g., "siglip", "openclip:ViT-H-14")
        """
        self.assets_dir = Path(assets_dir)
        self.model_key = model_key

        # Sanitize model_key for filename (replace : with -)
        safe_key = model_key.replace(":", "-").replace("/", "-")
        self.index_path = self.assets_dir / f"frame_search_{safe_key}.index"
        self.meta_path = self.assets_dir / f"frame_search_{safe_key}_meta.json"

        # Lazy-loaded index and metadata
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[FrameMetadata] = []
        self._info: Optional[IndexInfo] = None

    @property
    def exists(self) -> bool:
        """Check if index files exist on disk."""
        return self.index_path.exists() and self.meta_path.exists()

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded in memory."""
        return self._index is not None
```

**Step 2: Verify file is syntactically correct**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "from stash_ai.embeddings.frame_search import FrameSearchIndex; print('Import OK')"
```

Expected: "Import OK"

**Step 3: Commit**

```bash
git add stash_ai/embeddings/frame_search.py
git commit -m "$(cat <<'EOF'
feat(frame-search): add FrameSearchIndex class skeleton

Types: FrameMetadata, FrameMatch, SceneMatch, IndexInfo
Class: FrameSearchIndex with init and property methods

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement Index Building

**Files:**
- Modify: `stash_ai/embeddings/frame_search.py`

**Step 1: Add build() method**

Add after the `is_loaded` property (around line 85):

```python
    def build(
        self,
        storage: "EmbeddingStorage",
        batch_size: int = 10000,
        progress_callback: Optional[callable] = None,
    ) -> IndexInfo:
        """Build FAISS index from all frame embeddings.

        Args:
            storage: EmbeddingStorage instance to read frames from
            batch_size: Number of frames to process at a time
            progress_callback: Optional callback(current, total) for progress

        Returns:
            IndexInfo with build statistics
        """
        from stash_ai.embeddings.storage import EmbeddingStorage

        # Ensure assets directory exists
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Get all frame embeddings for this model
        conn = storage._get_connection()
        cursor = conn.cursor()

        # First, count total frames
        cursor.execute(
            "SELECT COUNT(*) FROM frame_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        total_frames = cursor.fetchone()[0]

        if total_frames == 0:
            conn.close()
            raise ValueError(
                f"No frame embeddings found for model '{self.model_key}'. "
                "Run 'Embed All Scenes' task first."
            )

        # Get dimensions from first embedding
        cursor.execute(
            "SELECT embedding FROM frame_embeddings WHERE model_key = ? LIMIT 1",
            (self.model_key,),
        )
        first_row = cursor.fetchone()
        first_embedding = storage._unpack_embedding(first_row[0])
        dimensions = len(first_embedding)

        # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dimensions)

        # Collect metadata
        metadata: List[FrameMetadata] = []
        scene_ids = set()

        # Process in batches
        cursor.execute(
            """
            SELECT scene_id, frame_index, timestamp, embedding
            FROM frame_embeddings
            WHERE model_key = ?
            ORDER BY scene_id, frame_index
            """,
            (self.model_key,),
        )

        batch_embeddings = []
        processed = 0

        for row in cursor:
            scene_id, frame_index, timestamp, embedding_blob = row
            embedding = storage._unpack_embedding(embedding_blob)

            batch_embeddings.append(embedding)
            metadata.append(
                FrameMetadata(
                    scene_id=scene_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
            scene_ids.add(scene_id)
            processed += 1

            # Add batch to index when full
            if len(batch_embeddings) >= batch_size:
                vectors = np.array(batch_embeddings, dtype=np.float32)
                index.add(vectors)
                batch_embeddings = []

                if progress_callback:
                    progress_callback(processed, total_frames)

        # Add remaining embeddings
        if batch_embeddings:
            vectors = np.array(batch_embeddings, dtype=np.float32)
            index.add(vectors)

        conn.close()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        # Build info
        info = IndexInfo(
            model_key=self.model_key,
            frame_count=len(metadata),
            scene_count=len(scene_ids),
            dimensions=dimensions,
            created_at=datetime.utcnow().isoformat() + "Z",
        )

        # Save index to disk
        faiss.write_index(index, str(self.index_path))

        # Save metadata to JSON
        meta_dict = {
            "model_key": info.model_key,
            "frame_count": info.frame_count,
            "scene_count": info.scene_count,
            "dimensions": info.dimensions,
            "created_at": info.created_at,
            "frames": [
                {
                    "scene_id": m.scene_id,
                    "frame_index": m.frame_index,
                    "timestamp": m.timestamp,
                }
                for m in metadata
            ],
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta_dict, f)

        # Update instance state
        self._index = index
        self._metadata = metadata
        self._info = info

        return info
```

**Step 2: Add import for EmbeddingStorage type hint**

At the top of the file, add after the existing imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stash_ai.embeddings.storage import EmbeddingStorage
```

Update the typing import line to:
```python
from typing import TYPE_CHECKING, List, Optional
```

**Step 3: Verify build method is syntactically correct**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "from stash_ai.embeddings.frame_search import FrameSearchIndex; print('Import OK')"
```

Expected: "Import OK"

**Step 4: Commit**

```bash
git add stash_ai/embeddings/frame_search.py
git commit -m "$(cat <<'EOF'
feat(frame-search): implement index building

- Batch processing (10k frames at a time) for memory efficiency
- Progress callback support for UI feedback
- Saves FAISS index and JSON metadata to assets/

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement Index Loading and Search

**Files:**
- Modify: `stash_ai/embeddings/frame_search.py`

**Step 1: Add load() method**

Add after the `build()` method:

```python
    def load(self) -> IndexInfo:
        """Load index from disk into memory.

        Returns:
            IndexInfo with index statistics

        Raises:
            FileNotFoundError: If index files don't exist
        """
        if not self.exists:
            raise FileNotFoundError(
                f"Frame search index not found for model '{self.model_key}'. "
                "Run 'Build Frame Search Index' task first."
            )

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))

        # Load metadata
        with open(self.meta_path, "r") as f:
            meta_dict = json.load(f)

        self._metadata = [
            FrameMetadata(
                scene_id=m["scene_id"],
                frame_index=m["frame_index"],
                timestamp=m["timestamp"],
            )
            for m in meta_dict["frames"]
        ]

        self._info = IndexInfo(
            model_key=meta_dict["model_key"],
            frame_count=meta_dict["frame_count"],
            scene_count=meta_dict["scene_count"],
            dimensions=meta_dict["dimensions"],
            created_at=meta_dict["created_at"],
        )

        return self._info

    def ensure_loaded(self) -> IndexInfo:
        """Ensure index is loaded, loading from disk if needed."""
        if not self.is_loaded:
            return self.load()
        return self._info  # type: ignore
```

**Step 2: Add search() method**

Add after `ensure_loaded()`:

```python
    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 1000,
    ) -> List[FrameMatch]:
        """Search for similar frames.

        Args:
            query_embedding: Query vector (must match index dimensions)
            top_k: Number of top matches to return

        Returns:
            List of FrameMatch sorted by similarity (descending)
        """
        self.ensure_loaded()

        # Reshape query to 2D array for FAISS
        query = np.array([query_embedding], dtype=np.float32)

        # Search (D = distances/similarities, I = indices)
        similarities, indices = self._index.search(query, top_k)

        # Build results
        results: List[FrameMatch] = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for not enough results
                break

            meta = self._metadata[idx]
            results.append(
                FrameMatch(
                    scene_id=meta.scene_id,
                    frame_index=meta.frame_index,
                    timestamp=meta.timestamp,
                    similarity=float(sim),
                )
            )

        return results
```

**Step 3: Add aggregate_to_scenes() method**

Add after `search()`:

```python
    def aggregate_to_scenes(
        self,
        frame_matches: List[FrameMatch],
    ) -> List[SceneMatch]:
        """Aggregate frame matches to scene-level using max-score.

        For each scene, keeps only the best-matching frame.

        Args:
            frame_matches: List of frame matches from search()

        Returns:
            List of SceneMatch sorted by similarity (descending)
        """
        # Group by scene, keep max
        scene_best: dict[int, FrameMatch] = {}

        for match in frame_matches:
            if match.scene_id not in scene_best:
                scene_best[match.scene_id] = match
            elif match.similarity > scene_best[match.scene_id].similarity:
                scene_best[match.scene_id] = match

        # Convert to SceneMatch and sort
        results = [
            SceneMatch(
                scene_id=m.scene_id,
                best_frame_index=m.frame_index,
                best_timestamp=m.timestamp,
                similarity=m.similarity,
            )
            for m in scene_best.values()
        ]

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results
```

**Step 4: Add get_info() method**

Add after `aggregate_to_scenes()`:

```python
    def get_info(self) -> Optional[IndexInfo]:
        """Get index info without loading the full index.

        Returns:
            IndexInfo if metadata file exists, None otherwise
        """
        if self._info is not None:
            return self._info

        if not self.meta_path.exists():
            return None

        with open(self.meta_path, "r") as f:
            meta_dict = json.load(f)

        return IndexInfo(
            model_key=meta_dict["model_key"],
            frame_count=meta_dict["frame_count"],
            scene_count=meta_dict["scene_count"],
            dimensions=meta_dict["dimensions"],
            created_at=meta_dict["created_at"],
        )
```

**Step 5: Verify all methods are syntactically correct**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "
from stash_ai.embeddings.frame_search import FrameSearchIndex, FrameMatch, SceneMatch
idx = FrameSearchIndex('/tmp', 'test')
print('All methods accessible:', hasattr(idx, 'search'), hasattr(idx, 'aggregate_to_scenes'))
"
```

Expected: "All methods accessible: True True"

**Step 6: Commit**

```bash
git add stash_ai/embeddings/frame_search.py
git commit -m "$(cat <<'EOF'
feat(frame-search): implement load, search, and aggregation

- load(): Lazy-load index from disk
- search(): FAISS similarity search returning FrameMatch list
- aggregate_to_scenes(): Max-score aggregation for scene-level results
- get_info(): Get index stats without loading full index

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add Build Frame Index Task Registration

**Files:**
- Modify: `stash-copilot.yml`

**Step 1: Add task registration**

Add after "Embed Cached Frames" task (around line 336):

```yaml
  - name: Build Frame Search Index
    description: Build FAISS index for fast frame-level semantic search
    defaultArgs:
      mode: build_frame_index
      model_key: ""        # Optional: defaults to configured image embedding model
```

**Step 2: Commit**

```bash
git add stash-copilot.yml
git commit -m "$(cat <<'EOF'
feat(frame-search): register Build Frame Search Index task

New task in Stash UI to build FAISS index for frame-level search.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Build Frame Index Task Handler

**Files:**
- Modify: `stash-copilot.py`

**Step 1: Add handler method**

Find the `run_cleanup_orphaned` method and add the new handler before it (around line 2100):

```python
    def run_build_frame_index(self, args: Dict[str, Any]):
        """
        Build FAISS index for frame-level semantic search.

        Args:
            args: Task arguments containing:
                - model_key: Optional model key (defaults to configured model)
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.frame_search import FrameSearchIndex
            from stash_ai.embeddings.storage import EmbeddingStorage

            # Get model key from args or settings
            requested_model_key = args.get("model_key", "").strip()

            plugin_settings = self.get_plugin_settings("stash-copilot")

            if requested_model_key:
                model_key = requested_model_key
            else:
                image_provider = plugin_settings.get("image_embedding_provider")
                image_model = plugin_settings.get("image_embedding_model")

                if not image_provider or not image_model:
                    self.error(
                        "Image embedding provider not configured. "
                        "Set up in Plugin Settings first."
                    )
                    return

                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device="cpu",  # Not used for indexing
                )
                model_key = embedding_config.model_key

            self.log(f"Building frame search index for model: {model_key}", "info")

            # Initialize storage and index
            storage = EmbeddingStorage(model_key=model_key)
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(plugin_dir, "assets")

            frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

            # Build with progress reporting
            def progress_callback(current: int, total: int):
                self.progress(current / total)
                if current % 50000 == 0 or current == total:
                    self.log(f"Indexed {current:,} / {total:,} frames", "info")

            info = frame_index.build(
                storage=storage,
                progress_callback=progress_callback,
            )

            self.log(
                f"Frame search index built successfully:\n"
                f"  Model: {info.model_key}\n"
                f"  Frames: {info.frame_count:,}\n"
                f"  Scenes: {info.scene_count:,}\n"
                f"  Dimensions: {info.dimensions}",
                "info",
            )

        except ValueError as e:
            self.error(str(e))
        except Exception as e:
            self.error(f"Failed to build frame search index: {e}")
            import traceback
            self.log(traceback.format_exc(), "debug")
```

**Step 2: Add mode dispatch**

Find the `run()` method's mode dispatch section (around line 350-450). Add the new mode handler:

```python
        elif mode == "build_frame_index":
            self.run_build_frame_index(args)
```

Add after the `embed_cached_frames` case.

**Step 3: Verify syntax**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "import stash_copilot; print('Import OK')"
```

Expected: "Import OK" (or may fail on stdin, but no syntax errors)

**Step 4: Commit**

```bash
git add stash-copilot.py
git commit -m "$(cat <<'EOF'
feat(frame-search): add build_frame_index task handler

- Progress reporting during index build
- Uses configured embedding model or explicit model_key
- Logs index statistics on completion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Search Handler for Frame Search

**Files:**
- Modify: `stash-copilot.py`

**Step 1: Update run_search_by_text to accept frame_search parameter**

In the `run_search_by_text` method, update the docstring and add parameter parsing:

After line 1816 (`requested_model_key = args.get("model_key", "").strip()`), add:

```python
            frame_search = args.get("frame_search", "").lower() == "true"
```

**Step 2: Add frame search branch**

After line 1862 (the "No scene embeddings found" check), add the frame search logic:

```python
            # Frame-level search using FAISS index
            if frame_search:
                from stash_ai.embeddings.frame_search import FrameSearchIndex

                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                assets_dir = os.path.join(plugin_dir, "assets")

                frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

                if not frame_index.exists:
                    self._write_search_result(request_id, {
                        "status": "error",
                        "error": f"Frame search index not built for model '{model_key}'. Run 'Build Frame Search Index' task first."
                    })
                    return

                # Embed the query text
                try:
                    result = embedder.embed_text(query)
                    query_embedding = np.array(result["embedding"], dtype=np.float32)
                except Exception as e:
                    self._write_search_result(request_id, {
                        "status": "error",
                        "error": f"Failed to embed query: {str(e)}"
                    })
                    return

                # Search frames
                frame_matches = frame_index.search(query_embedding, top_k=2000)

                # Aggregate to scenes
                scene_matches = frame_index.aggregate_to_scenes(frame_matches)

                # Apply pagination
                paginated = scene_matches[offset:offset + limit]

                # Fetch scene details
                scene_details = self._get_scene_details_batch([m.scene_id for m in paginated])

                # Build result data with frame info
                result_data = []
                for m in paginated:
                    scene = scene_details.get(m.scene_id, {})
                    # Format frame path
                    frame_path = f"embedded_frames/scene_{m.scene_id}/frame_{m.best_frame_index:04d}.jpg"
                    result_data.append({
                        "scene_id": m.scene_id,
                        "similarity": m.similarity,
                        "best_frame_index": m.best_frame_index,
                        "best_timestamp": m.best_timestamp,
                        "frame_path": frame_path,
                        "scene": scene
                    })

                has_more = len(scene_matches) > (offset + limit)

                self._write_search_result(request_id, {
                    "status": "complete",
                    "query": query,
                    "model_key": model_key,
                    "frame_search": True,
                    "results": result_data,
                    "offset": offset,
                    "limit": limit,
                    "has_more": has_more,
                    "request_id": request_id,
                    "total_scenes": len(scene_matches),
                })

                self.log(f"Frame search complete: {len(result_data)} scenes for '{query}'", "info")
                return
```

**Step 3: Add numpy import**

At the top of the file, ensure numpy is imported. Find the imports section and add if not present:

```python
import numpy as np
```

**Step 4: Verify syntax**

Run:
```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search && uv run python -c "import stash_copilot; print('Import OK')"
```

Expected: "Import OK"

**Step 5: Commit**

```bash
git add stash-copilot.py
git commit -m "$(cat <<'EOF'
feat(frame-search): add frame search mode to search_by_text

When frame_search=true:
- Uses FAISS index instead of scene embeddings
- Returns best-matching frame info per scene
- Includes frame_path for thumbnail display

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update Search Task Registration

**Files:**
- Modify: `stash-copilot.yml`

**Step 1: Add frame_search parameter to Search Scenes by Text task**

Find the "Search Scenes by Text" task (around line 252) and add the parameter:

```yaml
  - name: Search Scenes by Text
    description: Search scenes using natural language (semantic search)
    defaultArgs:
      mode: search_by_text
      query: ""            # Required: Natural language search query
      limit: "240"         # Maximum results (10 pages × 24 per page for fast pagination)
      offset: "0"          # Pagination offset
      request_id: ""       # Unique request ID for frontend
      model_key: ""        # Optional: Embedding model to use (e.g., "openclip:ViT-H-14")
      frame_search: ""     # Optional: Set to "true" for frame-level search (requires index)
```

**Step 2: Commit**

```bash
git add stash-copilot.yml
git commit -m "$(cat <<'EOF'
feat(frame-search): add frame_search parameter to search task

Enables frame-level search when set to "true".

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add Frontend Toggle - State and HTML

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Add frameSearch to searchState**

Find the `searchState` object (around line 69) and add:

```javascript
    const searchState = {
        query: '',
        allResults: [],
        isSearching: false,
        currentPage: 1,
        perPage: 24,
        pagesPerBatch: 10,
        totalFetched: 0,
        hasMoreOnServer: true,
        requestId: null,
        pollInterval: null,
        lastQuery: localStorage.getItem('stash-copilot-last-search') || '',
        availableModels: [],
        selectedModel: localStorage.getItem('stash-copilot-selected-model') || '',
        modelsLoaded: false,
        frameSearch: localStorage.getItem('stash-copilot-frame-search') === 'true'  // NEW
    };
```

**Step 2: Add toggle HTML**

Find the search model selector div (around line 4466-4479) and add the toggle after it:

After the closing `</div>` of `stash-copilot-search-model-selector` (around line 4479), add:

```html
                    <div class="stash-copilot-search-mode-toggle">
                        <span class="stash-copilot-toggle-label">Search Mode:</span>
                        <div class="stash-copilot-toggle-buttons">
                            <button class="stash-copilot-toggle-btn ${!searchState.frameSearch ? 'active' : ''}"
                                    data-mode="scene">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="2" y="2" width="20" height="20" rx="2"/>
                                    <path d="M7 2v20M17 2v20M2 12h20"/>
                                </svg>
                                Scene
                            </button>
                            <button class="stash-copilot-toggle-btn ${searchState.frameSearch ? 'active' : ''}"
                                    data-mode="frame">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="2" y="2" width="20" height="20" rx="2"/>
                                    <circle cx="12" cy="12" r="3"/>
                                    <path d="M2 12h4M18 12h4M12 2v4M12 18v4"/>
                                </svg>
                                Frame
                            </button>
                        </div>
                    </div>
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "$(cat <<'EOF'
feat(frame-search): add search mode toggle to frontend

- frameSearch state with localStorage persistence
- Toggle buttons for Scene/Frame mode

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Add Frontend Toggle - Event Handling

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Add toggle click handler**

Find where search page event listeners are set up. Look for the search button click handler (around line 4720-4740). Add the toggle handler nearby:

```javascript
        // Search mode toggle
        container.querySelectorAll('.stash-copilot-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                const newFrameSearch = mode === 'frame';

                if (newFrameSearch !== searchState.frameSearch) {
                    searchState.frameSearch = newFrameSearch;
                    localStorage.setItem('stash-copilot-frame-search', newFrameSearch);

                    // Update button states
                    container.querySelectorAll('.stash-copilot-toggle-btn').forEach(b => {
                        b.classList.toggle('active', b.dataset.mode === mode);
                    });

                    // Clear cached results and re-search if we have a query
                    searchState.allResults = [];
                    searchState.totalFetched = 0;
                    searchState.hasMoreOnServer = true;

                    if (searchState.query) {
                        performSearch(searchState.query, true);
                    }

                    log(`Search mode changed to: ${mode}`);
                }
            });
        });
```

**Step 2: Update performSearch to include frame_search parameter**

Find the `performSearch` function (around line 4758). In the taskArgs object (around line 4810-4820), add the frame_search parameter:

```javascript
            const taskArgs = {
                query: query,
                limit: batchSize.toString(),
                offset: offset.toString(),
                request_id: requestId
            };

            if (searchState.selectedModel) {
                taskArgs.model_key = searchState.selectedModel;
            }

            // Add frame search flag
            if (searchState.frameSearch) {
                taskArgs.frame_search = 'true';
            }
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "$(cat <<'EOF'
feat(frame-search): add toggle event handling

- Toggle click updates state and localStorage
- performSearch sends frame_search parameter
- Results cleared on mode switch

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Add Frontend - Frame Thumbnail and Timestamp Display

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Update displayCurrentPage for frame results**

Find the `displayCurrentPage` function (around line 4900). In the card rendering section (around line 4940-4955), update to handle frame results:

```javascript
        // Render cards for current page
        if (gridDiv) {
            gridDiv.innerHTML = pageResults.map((item, index) => {
                // Determine thumbnail source
                let thumbnailUrl = null;
                let matchTimestamp = null;

                if (searchState.frameSearch && item.frame_path) {
                    // Use frame thumbnail for frame search results
                    thumbnailUrl = `/plugin/stash-copilot/assets/${item.frame_path}`;
                    matchTimestamp = item.best_timestamp;
                }

                return buildSceneCard({
                    scene: item.scene,
                    score: item.similarity,
                    cardIndex: index,
                    theme: 'search',
                    scoreLabel: 'relevance',
                    overrideThumbnail: thumbnailUrl,
                    matchTimestamp: matchTimestamp
                });
            }).join('');

            // Setup card events
            setupSceneCardEvents(gridDiv, {
                theme: 'search',
                tooltipMode: 'fixed'
            });
        }
```

**Step 2: Update buildSceneCard to accept overrideThumbnail and matchTimestamp**

Find the `buildSceneCard` function. Add parameters to the function signature and use them:

In the options destructuring (find where `scene`, `score`, `cardIndex`, `theme`, `scoreLabel` are extracted):

```javascript
function buildSceneCard({
    scene,
    score,
    cardIndex = 0,
    theme = 'similar',
    scoreLabel = 'match',
    overrideThumbnail = null,
    matchTimestamp = null
}) {
```

Then in the thumbnail rendering section, update to use overrideThumbnail:

```javascript
        // Use override thumbnail if provided (for frame search results)
        const thumbnailSrc = overrideThumbnail || (scene.paths?.screenshot || scene.screenshot || '');
```

And add the timestamp badge in the thumbnail container:

```javascript
        // Match timestamp badge (for frame search)
        const timestampBadge = matchTimestamp !== null
            ? `<span class="stash-copilot-match-timestamp">${formatTimestamp(matchTimestamp)}</span>`
            : '';
```

Then include `timestampBadge` in the thumbnail container HTML.

**Step 3: Add formatTimestamp helper if not exists**

Check if `formatTimestamp` function exists, if not add it:

```javascript
    function formatTimestamp(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "$(cat <<'EOF'
feat(frame-search): display frame thumbnails and timestamps

- overrideThumbnail parameter for frame search results
- matchTimestamp badge shows best-matching frame time
- formatTimestamp helper for MM:SS display

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Add CSS Styling for Toggle and Timestamp Badge

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add toggle button styles**

Find the search page styles section (around line 7700+). Add after the model selector styles:

```css
/* Search Mode Toggle */
.stash-copilot-search-mode-toggle {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-left: 24px;
}

.stash-copilot-toggle-label {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
}

.stash-copilot-toggle-buttons {
    display: flex;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}

.stash-copilot-toggle-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: rgba(255, 255, 255, 0.6);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stash-copilot-toggle-btn:hover {
    color: rgba(255, 255, 255, 0.9);
    background: rgba(255, 255, 255, 0.05);
}

.stash-copilot-toggle-btn.active {
    color: #fff;
    background: linear-gradient(135deg, #10b981, #059669);
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.stash-copilot-toggle-btn svg {
    width: 14px;
    height: 14px;
}
```

**Step 2: Add timestamp badge styles**

Add after the toggle styles:

```css
/* Match Timestamp Badge (Frame Search) */
.stash-copilot-match-timestamp {
    position: absolute;
    bottom: 8px;
    left: 8px;
    padding: 4px 8px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(5, 150, 105, 0.9));
    color: #fff;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    z-index: 2;
}

.stash-copilot-match-timestamp::before {
    content: '⏱ ';
}
```

**Step 3: Commit**

```bash
git add stash-copilot.css
git commit -m "$(cat <<'EOF'
feat(frame-search): add CSS for toggle and timestamp badge

- Search mode toggle buttons (Scene/Frame)
- Match timestamp badge with clock icon
- Consistent green gradient theme

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Add Thumbnail Fallback Handling

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Add onerror handler for frame thumbnails**

In `setupSceneCardEvents` or where card images are processed, add fallback handling:

Find the image elements and add error handling:

```javascript
        // Handle frame thumbnail load errors (fallback to scene screenshot)
        gridDiv.querySelectorAll('.stash-copilot-card img').forEach(img => {
            if (img.src.includes('/embedded_frames/')) {
                img.onerror = function() {
                    // Extract scene ID from path and use scene screenshot
                    const card = this.closest('.stash-copilot-card');
                    if (card && card.dataset.sceneId) {
                        const sceneId = card.dataset.sceneId;
                        // Use Stash's scene screenshot endpoint
                        this.src = `/scene/${sceneId}/screenshot`;
                        this.onerror = null; // Prevent infinite loop
                    }
                };
            }
        });
```

**Step 2: Ensure card has data-scene-id attribute**

In `buildSceneCard`, add the scene ID as a data attribute:

```javascript
    return `<div class="stash-copilot-card" data-theme="${theme}" data-scene-id="${scene.id}">
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "$(cat <<'EOF'
feat(frame-search): add thumbnail fallback for missing frames

If frame image fails to load, falls back to scene screenshot.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Integration Test - Build Index and Search

**Files:** None (manual testing)

**Step 1: Deploy to main plugin directory**

```bash
# Copy updated files from worktree to main plugin
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/pyproject.toml ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/uv.lock ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/stash-copilot.yml ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/stash-copilot.py ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/stash-copilot.js ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/stash-copilot.css ~/.stash/plugins/stash-copilot/
cp ~/.stash/plugins/stash-copilot/.worktrees/frame-search/stash_ai/embeddings/frame_search.py ~/.stash/plugins/stash-copilot/stash_ai/embeddings/
```

**Step 2: Sync dependencies in main plugin**

```bash
cd ~/.stash/plugins/stash-copilot && uv sync
```

**Step 3: Reload Stash plugins**

In Stash UI: Settings → Plugins → Reload Plugins

**Step 4: Run Build Frame Search Index task**

In Stash UI: Settings → Tasks → Run "Build Frame Search Index"

Monitor logs in `~/.stash/stash.log` for progress.

**Step 5: Test semantic search with Frame mode**

1. Navigate to AI Semantic Search page
2. Click "Frame" toggle button
3. Enter a search query (e.g., "close up face")
4. Verify results show:
   - Frame thumbnails (not scene screenshots)
   - Timestamp badges (e.g., "2:34")
   - Similarity scores

**Step 6: Take screenshots**

```bash
# Save test screenshots
mkdir -p ~/.stash/plugins/stash-copilot/tests/screenshots/frame-search/
```

**Step 7: Commit test results (if screenshots added)**

```bash
git add tests/screenshots/frame-search/
git commit -m "$(cat <<'EOF'
test: add frame search integration test screenshots

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final Merge

**Step 1: Ensure all commits are in worktree**

```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/frame-search
git log --oneline -15
```

**Step 2: Merge feature branch to main**

```bash
cd ~/.stash/plugins/stash-copilot
git merge feature/frame-level-semantic-search
```

**Step 3: Clean up worktree**

```bash
git worktree remove .worktrees/frame-search
git branch -d feature/frame-level-semantic-search
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add FAISS dependency | pyproject.toml |
| 2 | Create FrameSearchIndex class skeleton | stash_ai/embeddings/frame_search.py |
| 3 | Implement index building | stash_ai/embeddings/frame_search.py |
| 4 | Implement load, search, aggregation | stash_ai/embeddings/frame_search.py |
| 5 | Register build task | stash-copilot.yml |
| 6 | Add build task handler | stash-copilot.py |
| 7 | Add frame search to search handler | stash-copilot.py |
| 8 | Update search task registration | stash-copilot.yml |
| 9 | Add frontend toggle HTML | stash-copilot.js |
| 10 | Add toggle event handling | stash-copilot.js |
| 11 | Add frame thumbnail display | stash-copilot.js |
| 12 | Add CSS styling | stash-copilot.css |
| 13 | Add thumbnail fallback | stash-copilot.js |
| 14 | Integration test | (manual) |
| 15 | Merge to main | (git) |
