# Taste Map Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-profile recommendation engine with a multi-cluster taste profiling system, visualized as an interactive 2D scatter plot in the AI Insights modal.

**Architecture:** Scene embeddings are clustered via K-Means (optimal k by silhouette score), projected to 2D via UMAP, and auto-labeled by comparing cluster centroids to CLIP text embeddings of a curated vocabulary. The Taste Map tab in the AI Insights modal renders an ECharts scatter plot with a cluster sidebar and tag match panel. The recommendation engine queries per-cluster and merges with proportional sampling.

**Tech Stack:** Python (scikit-learn, umap-learn, OpenCLIP, numpy), JavaScript (ECharts via CDN), SQLite

**Design Doc:** `docs/plans/2026-02-09-taste-map-design.md`

---

## Phase 1: Types & Storage Foundation

### Task 1: Add Taste Map Types

**Files:**
- Modify: `stash_ai/recommendations/types.py` (after line 239)

**Step 1: Add new TypedDicts and dataclasses**

Add after the existing `OMomentProfileInfo` dataclass at the end of the file:

```python
# --- Taste Map Types ---

class TagMatch(TypedDict):
    """A vocabulary phrase matched to a cluster centroid."""
    text: str
    similarity: float
    source: str  # 'stash_tag' | 'curated' | 'user'


class TasteClusterData(TypedDict):
    """Serializable cluster data for JSON output."""
    cluster_id: int
    auto_label: str
    scene_ids: list[int]
    engagement_total: float
    engagement_share: float
    representative_scenes: list[int]
    tag_matches: list[TagMatch]


class TasteMapSceneData(TypedDict):
    """Per-scene data for the taste map visualization."""
    scene_id: int
    x: float
    y: float
    cluster_id: int | None  # None for non-profile scenes
    engagement_score: float
    is_profile: bool
    title: str | None
    thumbnail: str | None
    play_count: int
    o_counter: int


class TasteMapResponse(TypedDict):
    """Full taste map response saved to JSON."""
    status: str  # 'complete' | 'error'
    optimal_k: int
    silhouette_score: float
    clusters: list[TasteClusterData]
    scenes: list[TasteMapSceneData]
    error: str | None


@dataclass
class TasteCluster:
    """Runtime cluster with embedding data (not serialized to JSON)."""
    cluster_id: int
    centroid: "NDArray[np.float32]"
    scene_ids: list[int]
    engagement_total: float
    engagement_share: float
    auto_label: str
    user_label: str | None
    weight_override: float | None
    excluded: bool
    tag_matches: list[TagMatch]


@dataclass
class TasteProfile:
    """Complete taste profile with all clusters."""
    clusters: list[TasteCluster]
    optimal_k: int
    silhouette_score: float
    model_key: str
```

**Step 2: Add numpy import at top of file**

Add `import numpy as np` and `from numpy.typing import NDArray` to the imports if not already present. Check existing imports first — if `numpy` is not imported, add it inside a `TYPE_CHECKING` block since it's only used for type annotations:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
```

**Step 3: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.recommendations.types import TasteCluster, TasteProfile, TasteMapResponse; print('Types OK')"`

Expected: `Types OK`

**Step 4: Commit**

```bash
git add stash_ai/recommendations/types.py
git commit -m "feat(taste-map): add type definitions for clustering and taste map"
```

---

### Task 2: Add SQLite Tables

**Files:**
- Modify: `stash_ai/embeddings/storage.py`

**Step 1: Bump schema version**

Find `SCHEMA_VERSION = 6` (line 116) and change to `SCHEMA_VERSION = 7`.

**Step 2: Add migration method**

Find the `_init_database` method (lines 173-184) and add the v7 migration call. Then add the new migration method after the last `_migrate_to_v6` method. Follow the existing pattern:

```python
def _migrate_to_v7(self) -> None:
    """Add taste map tables: taste_clusters, scene_umap_coords, tag_embeddings."""
    self.log("Migrating to schema v7: taste map tables", "info")

    self._conn.execute("""
        CREATE TABLE IF NOT EXISTS taste_clusters (
            cluster_id INTEGER NOT NULL,
            model_key TEXT NOT NULL,
            centroid BLOB NOT NULL,
            scene_ids TEXT NOT NULL,
            engagement_total REAL NOT NULL,
            engagement_share REAL NOT NULL,
            auto_label TEXT NOT NULL,
            user_label TEXT,
            weight_override REAL,
            excluded INTEGER DEFAULT 0,
            tag_matches TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (cluster_id, model_key)
        )
    """)

    self._conn.execute("""
        CREATE TABLE IF NOT EXISTS scene_umap_coords (
            scene_id INTEGER NOT NULL,
            model_key TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            cluster_id INTEGER,
            created_at TEXT NOT NULL,
            PRIMARY KEY (scene_id, model_key)
        )
    """)

    self._conn.execute("""
        CREATE TABLE IF NOT EXISTS tag_embeddings (
            text TEXT NOT NULL,
            model_key TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (text, model_key)
        )
    """)

    self._conn.commit()
```

**Step 3: Add v7 migration call to _init_database**

In the `_init_database` method, find where migrations are called sequentially and add:

```python
if current_version < 7:
    self._migrate_to_v7()
    self._set_schema_version(7)
```

**Step 4: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.embeddings.storage import EmbeddingStorage; s = EmbeddingStorage(); print('Schema v7 OK')"`

Expected: `Schema v7 OK` (migration runs automatically)

**Step 5: Commit**

```bash
git add stash_ai/embeddings/storage.py
git commit -m "feat(taste-map): add SQLite tables for clusters, UMAP coords, tag embeddings"
```

---

### Task 3: Add CRUD Methods for New Tables

**Files:**
- Modify: `stash_ai/embeddings/storage.py` (add methods after existing CRUD section)

**Step 1: Add taste_clusters CRUD**

```python
# --- Taste Cluster Methods ---

def save_taste_clusters(
    self, clusters: list["TasteCluster"], model_key: str
) -> None:
    """Save taste clusters, replacing any existing for this model_key."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    # Clear existing clusters for this model
    self._conn.execute(
        "DELETE FROM taste_clusters WHERE model_key = ?", (model_key,)
    )

    for cluster in clusters:
        tag_matches_json = json.dumps(cluster.tag_matches)
        scene_ids_json = json.dumps(cluster.scene_ids)
        self._conn.execute(
            """INSERT INTO taste_clusters
            (cluster_id, model_key, centroid, scene_ids, engagement_total,
             engagement_share, auto_label, user_label, weight_override,
             excluded, tag_matches, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cluster.cluster_id,
                model_key,
                self._pack_embedding(cluster.centroid.tolist()),
                scene_ids_json,
                cluster.engagement_total,
                cluster.engagement_share,
                cluster.auto_label,
                cluster.user_label,
                cluster.weight_override,
                1 if cluster.excluded else 0,
                tag_matches_json,
                now,
            ),
        )
    self._conn.commit()

def get_taste_clusters(self, model_key: str) -> list[dict]:
    """Load taste clusters for a model_key."""
    import json

    rows = self._conn.execute(
        "SELECT * FROM taste_clusters WHERE model_key = ? ORDER BY cluster_id",
        (model_key,),
    ).fetchall()

    clusters = []
    for row in rows:
        clusters.append({
            "cluster_id": row[0],
            "model_key": row[1],
            "centroid": self._unpack_embedding(row[2]),
            "scene_ids": json.loads(row[3]),
            "engagement_total": row[4],
            "engagement_share": row[5],
            "auto_label": row[6],
            "user_label": row[7],
            "weight_override": row[8],
            "excluded": bool(row[9]),
            "tag_matches": json.loads(row[10]),
            "created_at": row[11],
        })
    return clusters

def update_taste_cluster(
    self, cluster_id: int, model_key: str, **kwargs: object
) -> None:
    """Update specific fields of a taste cluster (user_label, weight_override, excluded)."""
    allowed = {"user_label", "weight_override", "excluded"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return

    if "excluded" in updates:
        updates["excluded"] = 1 if updates["excluded"] else 0

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [cluster_id, model_key]
    self._conn.execute(
        f"UPDATE taste_clusters SET {set_clause} WHERE cluster_id = ? AND model_key = ?",
        values,
    )
    self._conn.commit()
```

**Step 2: Add scene_umap_coords CRUD**

```python
# --- UMAP Coordinate Methods ---

def save_umap_coords(
    self,
    coords: dict[int, tuple[float, float]],
    cluster_assignments: dict[int, int | None],
    model_key: str,
) -> None:
    """Save UMAP 2D coordinates for scenes."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    # Clear existing coords for this model
    self._conn.execute(
        "DELETE FROM scene_umap_coords WHERE model_key = ?", (model_key,)
    )

    for scene_id, (x, y) in coords.items():
        cluster_id = cluster_assignments.get(scene_id)
        self._conn.execute(
            """INSERT INTO scene_umap_coords
            (scene_id, model_key, x, y, cluster_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (scene_id, model_key, x, y, cluster_id, now),
        )
    self._conn.commit()

def get_umap_coords(self, model_key: str) -> list[dict]:
    """Load all UMAP coordinates for a model_key."""
    rows = self._conn.execute(
        "SELECT scene_id, x, y, cluster_id FROM scene_umap_coords WHERE model_key = ?",
        (model_key,),
    ).fetchall()

    return [
        {"scene_id": r[0], "x": r[1], "y": r[2], "cluster_id": r[3]}
        for r in rows
    ]
```

**Step 3: Add tag_embeddings CRUD**

```python
# --- Tag Embedding Methods ---

def save_tag_embedding(
    self, text: str, model_key: str, embedding: list[float], source: str
) -> None:
    """Save a tag/phrase text embedding."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    self._conn.execute(
        """INSERT OR REPLACE INTO tag_embeddings
        (text, model_key, embedding, source, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (text, model_key, self._pack_embedding(embedding), source, now),
    )
    self._conn.commit()

def save_tag_embeddings_batch(
    self,
    entries: list[tuple[str, list[float], str]],
    model_key: str,
) -> None:
    """Batch save tag embeddings. Each entry is (text, embedding, source)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    for text, embedding, source in entries:
        self._conn.execute(
            """INSERT OR REPLACE INTO tag_embeddings
            (text, model_key, embedding, source, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (text, model_key, self._pack_embedding(embedding), source, now),
        )
    self._conn.commit()

def get_all_tag_embeddings(self, model_key: str) -> list[dict]:
    """Load all tag embeddings for a model_key."""
    rows = self._conn.execute(
        "SELECT text, embedding, source FROM tag_embeddings WHERE model_key = ?",
        (model_key,),
    ).fetchall()

    return [
        {
            "text": r[0],
            "embedding": self._unpack_embedding(r[1]),
            "source": r[2],
        }
        for r in rows
    ]

def get_tag_embedding(self, text: str, model_key: str) -> list[float] | None:
    """Get embedding for a specific tag/phrase."""
    row = self._conn.execute(
        "SELECT embedding FROM tag_embeddings WHERE text = ? AND model_key = ?",
        (text, model_key),
    ).fetchone()

    if row:
        return self._unpack_embedding(row[0])
    return None

def get_tag_embedding_count(self, model_key: str) -> int:
    """Count how many tag embeddings exist for a model_key."""
    row = self._conn.execute(
        "SELECT COUNT(*) FROM tag_embeddings WHERE model_key = ?",
        (model_key,),
    ).fetchone()
    return row[0] if row else 0
```

**Step 4: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.embeddings.storage import EmbeddingStorage
s = EmbeddingStorage()
print('Tag count:', s.get_tag_embedding_count('test'))
print('Clusters:', s.get_taste_clusters('test'))
print('Coords:', s.get_umap_coords('test'))
print('CRUD OK')
"`

Expected: Empty results, no errors, `CRUD OK`

**Step 5: Commit**

```bash
git add stash_ai/embeddings/storage.py
git commit -m "feat(taste-map): add CRUD methods for clusters, UMAP coords, tag embeddings"
```

---

## Phase 2: Tag Vocabulary & Text Embeddings

### Task 4: Expose OpenCLIP Text Embedding

**Files:**
- Modify: `stash_ai/embeddings/providers/openclip.py`

**Step 1: Verify `embed_text` is public**

Read the file and check if `embed_text` exists and is public (not prefixed with `_`). Based on exploration, it exists at lines 234-262. Verify it accepts a single string and returns an `EmbeddingResult` dict with an `"embedding"` key containing a `list[float]`.

If it only handles single strings, add a batch variant:

```python
def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
    """Generate embeddings for multiple texts in a batch."""
    if not texts:
        return []

    tokens = self._tokenizer(texts).to(self.device)
    with torch.no_grad():
        embeddings = self._model.encode_text(tokens)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()

    results: list[EmbeddingResult] = []
    for i, text in enumerate(texts):
        results.append({
            "embedding": embeddings[i].tolist(),
            "model": self.model,
            "dimensions": embeddings.shape[1],
            "tokens_used": None,
        })
    return results
```

**Step 2: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.embeddings.providers.openclip import OpenCLIPProvider
from stash_ai.embeddings.config import EmbeddingConfig
config = EmbeddingConfig(provider='openclip', model='ViT-H-14')
p = OpenCLIPProvider(config)
result = p.embed_text('music video compilation')
print(f'Dims: {result[\"dimensions\"]}, embedding length: {len(result[\"embedding\"])}')
"`

Expected: `Dims: 1024, embedding length: 1024` (or similar)

**Step 3: Commit**

```bash
git add stash_ai/embeddings/providers/openclip.py
git commit -m "feat(taste-map): add batch text embedding to OpenCLIP provider"
```

---

### Task 5: Create Tag Vocabulary Module

**Files:**
- Create: `stash_ai/embeddings/tag_vocabulary.py`

**Step 1: Write the module**

```python
"""Tag vocabulary for cluster auto-labeling via CLIP text embeddings.

Three tiers of label candidates:
- Tier 1: Existing Stash tags from the user's database
- Tier 2: Curated descriptive phrases covering common content categories
- Tier 3: Compound phrases for specific niches
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stash_ai.embeddings.storage import EmbeddingStorage

# --- Tier 2: Curated Content Descriptors ---
CURATED_PHRASES: list[str] = [
    # Act / Position
    "oral sex blowjob",
    "doggy style from behind",
    "missionary position",
    "cowgirl riding on top",
    "reverse cowgirl",
    "solo masturbation",
    "handjob",
    "footjob",
    "anal sex",
    "deepthroat",
    "facesitting",
    "sixty nine position",
    "standing sex",
    "spooning sex",
    # Setting
    "bedroom scene",
    "bathroom shower",
    "outdoor nature",
    "hotel room",
    "pool scene",
    "office scene",
    "kitchen scene",
    "living room couch",
    "car scene",
    "public place",
    # Style
    "POV perspective first person",
    "close-up intimate",
    "wide shot full body",
    "professional studio lighting",
    "amateur homemade",
    "gonzo raw handheld",
    "glamour photography",
    "artistic softcore",
    "compilation montage",
    "behind the scenes",
    # Aesthetic
    "high energy music video",
    "slow sensual romantic",
    "fast cuts editing montage",
    "teasing striptease",
    "rough aggressive intense",
    "gentle tender lovemaking",
    "kinky fetish",
    "cosplay costume",
    "massage oil sensual",
    "dance and rhythm",
    # Body type
    "petite slim small woman",
    "curvy voluptuous woman",
    "athletic fit toned body",
    "tall woman long legs",
    "busty large breasts",
    "flat chested small breasts",
    "thick curvy hips",
    "muscular strong woman",
    # Features
    "blonde hair",
    "brunette dark hair",
    "redhead ginger hair",
    "black hair",
    "short hair pixie cut",
    "long hair flowing",
    "tattoos and piercings",
    "lingerie stockings",
    "high heels",
    "glasses nerdy",
    "natural no makeup",
    "heavy makeup glam",
    "tan skin",
    "pale fair skin",
    "dark skin",
    "asian woman",
    "latina woman",
    "ebony woman",
    # Group
    "solo performer alone",
    "couple two people",
    "threesome three people",
    "group multiple performers",
    "lesbian two women",
    "girl on girl",
    # Category
    "PMV porn music video",
    "compilation best of",
    "full scene complete",
    "trailer preview teaser",
    "virtual reality VR",
    "interactive funscript",
    "jerk off instruction JOI",
    "dirty talk verbal",
    "roleplay fantasy",
    "stepmom taboo",
    "teen young eighteen",
    "milf mature woman",
    "creampie internal",
    "facial cumshot",
    "squirting orgasm",
    "bondage tied up",
    "domination submission",
    "worship body worship",
]

# --- Tier 3: Compound Phrases ---
COMPOUND_PHRASES: list[str] = [
    "petite blonde POV blowjob",
    "curvy brunette solo masturbation",
    "high energy PMV compilation",
    "sensual lesbian massage",
    "rough anal doggy style",
    "intimate couple lovemaking bedroom",
    "amateur girlfriend homemade POV",
    "professional glamour striptease",
    "JOI dirty talk close up",
    "teen petite casting audition",
    "milf busty seduction",
    "interactive VR POV",
    "outdoor public risky",
    "cosplay anime roleplay",
    "oil massage sensual body",
    "facesitting femdom worship",
    "compilation cumshot facial",
    "romantic slow sensual couple",
    "gangbang group rough",
    "squirting intense orgasm",
]


class TagVocabulary:
    """Manages the tag/phrase vocabulary and their CLIP text embeddings."""

    def __init__(
        self,
        storage: "EmbeddingStorage",
        model_key: str,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.model_key = model_key
        self.log = log_callback or (lambda msg, level: None)

    def get_full_vocabulary(self, stash_tags: list[str] | None = None) -> list[tuple[str, str]]:
        """Get complete vocabulary as (text, source) pairs.

        Args:
            stash_tags: Existing tags from the user's Stash database.

        Returns:
            List of (text, source) tuples where source is 'stash_tag', 'curated', or 'user'.
        """
        vocab: list[tuple[str, str]] = []

        # Tier 1: Stash tags
        if stash_tags:
            for tag in stash_tags:
                if tag.strip():
                    vocab.append((tag.strip().lower(), "stash_tag"))

        # Tier 2: Curated phrases
        for phrase in CURATED_PHRASES:
            vocab.append((phrase, "curated"))

        # Tier 3: Compound phrases
        for phrase in COMPOUND_PHRASES:
            vocab.append((phrase, "curated"))

        # Tier 4: User-added phrases (from previous sessions)
        existing = self.storage.get_all_tag_embeddings(self.model_key)
        for entry in existing:
            if entry["source"] == "user":
                vocab.append((entry["text"], "user"))

        # Deduplicate by text (keep first occurrence)
        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for text, source in vocab:
            if text not in seen:
                seen.add(text)
                deduped.append((text, source))

        return deduped

    def ensure_embeddings(
        self,
        stash_tags: list[str] | None = None,
        force_recompute: bool = False,
    ) -> int:
        """Ensure all vocabulary items have embeddings. Returns count of newly embedded items.

        Uses the OpenCLIP text encoder to embed phrases that don't already
        have cached embeddings in the database.
        """
        vocab = self.get_full_vocabulary(stash_tags)
        existing_count = self.storage.get_tag_embedding_count(self.model_key)

        if not force_recompute and existing_count >= len(vocab):
            self.log(f"Tag embeddings already cached: {existing_count} entries", "debug")
            return 0

        # Find which phrases need embedding
        to_embed: list[tuple[str, str]] = []
        for text, source in vocab:
            if force_recompute or self.storage.get_tag_embedding(text, self.model_key) is None:
                to_embed.append((text, source))

        if not to_embed:
            self.log("All tag embeddings already cached", "debug")
            return 0

        self.log(f"Embedding {len(to_embed)} vocabulary items via OpenCLIP text encoder", "info")

        # Import and initialize provider
        from stash_ai.embeddings.providers.openclip import OpenCLIPProvider
        from stash_ai.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(provider="openclip", model=self._get_openclip_model())
        provider = OpenCLIPProvider(config)

        # Batch embed
        texts = [t for t, _ in to_embed]
        results = provider.embed_texts(texts)

        # Save to storage
        entries: list[tuple[str, list[float], str]] = []
        for i, (text, source) in enumerate(to_embed):
            entries.append((text, results[i]["embedding"], source))

        self.storage.save_tag_embeddings_batch(entries, self.model_key)
        self.log(f"Saved {len(entries)} tag embeddings", "info")

        # Cleanup provider
        provider.cleanup()

        return len(entries)

    def embed_custom_phrase(self, phrase: str) -> list[float]:
        """Embed a single custom phrase on the fly. Saves to storage as 'user' source."""
        # Check cache first
        cached = self.storage.get_tag_embedding(phrase.lower(), self.model_key)
        if cached:
            return cached

        from stash_ai.embeddings.providers.openclip import OpenCLIPProvider
        from stash_ai.embeddings.config import EmbeddingConfig

        config = EmbeddingConfig(provider="openclip", model=self._get_openclip_model())
        provider = OpenCLIPProvider(config)

        result = provider.embed_text(phrase.lower())
        embedding = result["embedding"]

        # Save as user phrase
        self.storage.save_tag_embedding(phrase.lower(), self.model_key, embedding, "user")

        provider.cleanup()
        return embedding

    def match_cluster_centroid(
        self,
        centroid: "NDArray[np.float32]",
        top_k: int = 8,
    ) -> list[dict[str, object]]:
        """Find top-k vocabulary matches for a cluster centroid.

        Args:
            centroid: Normalized cluster centroid embedding.
            top_k: Number of top matches to return.

        Returns:
            List of {text, similarity, source} dicts sorted by similarity descending.
        """
        all_tags = self.storage.get_all_tag_embeddings(self.model_key)
        if not all_tags:
            return []

        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        matches: list[dict[str, object]] = []
        for entry in all_tags:
            tag_emb = np.array(entry["embedding"], dtype=np.float32)
            tag_norm = tag_emb / (np.linalg.norm(tag_emb) + 1e-8)
            similarity = float(np.dot(centroid_norm, tag_norm))
            matches.append({
                "text": entry["text"],
                "similarity": round(similarity, 4),
                "source": entry["source"],
            })

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]

    def _get_openclip_model(self) -> str:
        """Extract OpenCLIP model name from the model_key."""
        # model_key format: "openclip:ViT-H-14" or just "ViT-H-14"
        if ":" in self.model_key:
            return self.model_key.split(":", 1)[1]
        return self.model_key
```

**Step 2: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.embeddings.tag_vocabulary import TagVocabulary, CURATED_PHRASES, COMPOUND_PHRASES
print(f'Curated: {len(CURATED_PHRASES)}, Compound: {len(COMPOUND_PHRASES)}')
print(f'Total vocabulary: {len(CURATED_PHRASES) + len(COMPOUND_PHRASES)}')
print('Import OK')
"`

Expected: Counts printed, `Import OK`

**Step 3: Commit**

```bash
git add stash_ai/embeddings/tag_vocabulary.py
git commit -m "feat(taste-map): add tag vocabulary module with curated phrases and CLIP text encoding"
```

---

## Phase 3: Clustering Engine

### Task 6: Create Clustering Module

**Files:**
- Create: `stash_ai/recommendations/clusters.py`

**Step 1: Write the clustering engine**

```python
"""Taste clustering engine using K-Means with silhouette score optimization.

Groups engaged scenes into taste clusters by embedding similarity,
computes weighted centroids, and auto-labels via tag embedding matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from stash_ai.embeddings.storage import EmbeddingStorage
    from stash_ai.embeddings.tag_vocabulary import TagVocabulary
    from stash_ai.recommendations.engagement import EngagementScore
    from stash_ai.recommendations.types import TasteCluster, TasteProfile


MIN_K = 2
MAX_K = 8
FALLBACK_K = 3
MIN_SCENES_FOR_CLUSTERING = 6  # Need at least 6 scenes to form 2 clusters


def find_optimal_k(
    embeddings: "NDArray[np.float32]",
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    log: Callable[[str, str], None] | None = None,
) -> tuple[int, float]:
    """Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Matrix of shape (n_scenes, n_dims).
        min_k: Minimum clusters to try.
        max_k: Maximum clusters to try.
        log: Logging callback.

    Returns:
        Tuple of (optimal_k, best_silhouette_score).
    """
    _log = log or (lambda msg, level: None)
    n_samples = len(embeddings)

    # Cap max_k at n_samples - 1
    max_k = min(max_k, n_samples - 1)
    if max_k < min_k:
        _log(f"Too few scenes ({n_samples}) for clustering range {min_k}-{max_k}, using k={min_k}", "warning")
        return min_k, 0.0

    best_k = FALLBACK_K
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels))
        _log(f"  k={k}: silhouette={score:.4f}", "debug")

        if score > best_score:
            best_score = score
            best_k = k

    _log(f"Optimal k={best_k} (silhouette={best_score:.4f})", "info")
    return best_k, best_score


def cluster_scenes(
    scene_ids: list[int],
    embeddings: "NDArray[np.float32]",
    engagement_scores: dict[int, float],
    optimal_k: int,
) -> tuple[list[list[int]], "NDArray[np.float32]", list[int]]:
    """Run K-Means and compute engagement-weighted centroids.

    Args:
        scene_ids: Scene IDs corresponding to embedding rows.
        embeddings: Matrix of shape (n_scenes, n_dims).
        engagement_scores: Mapping of scene_id -> engagement score.
        optimal_k: Number of clusters.

    Returns:
        Tuple of (cluster_scene_ids, centroids, labels) where:
        - cluster_scene_ids[i] is the list of scene_ids in cluster i
        - centroids is shape (k, n_dims) engagement-weighted centroids
        - labels[i] is the cluster assignment for scene_ids[i]
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings).tolist()

    n_dims = embeddings.shape[1]
    cluster_scene_ids: list[list[int]] = [[] for _ in range(optimal_k)]
    centroids = np.zeros((optimal_k, n_dims), dtype=np.float32)
    centroid_weights = np.zeros(optimal_k, dtype=np.float32)

    for i, scene_id in enumerate(scene_ids):
        cluster_idx = labels[i]
        cluster_scene_ids[cluster_idx].append(scene_id)

        weight = engagement_scores.get(scene_id, 1.0)
        centroids[cluster_idx] += embeddings[i] * weight
        centroid_weights[cluster_idx] += weight

    # Normalize centroids
    for k in range(optimal_k):
        if centroid_weights[k] > 0:
            centroids[k] /= centroid_weights[k]
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(centroids[k])
        if norm > 0:
            centroids[k] /= norm

    return cluster_scene_ids, centroids, labels


def compute_umap_projection(
    embeddings: "NDArray[np.float32]",
    log: Callable[[str, str], None] | None = None,
) -> "NDArray[np.float32]":
    """Project embeddings to 2D using UMAP.

    Args:
        embeddings: Matrix of shape (n_scenes, n_dims).

    Returns:
        2D coordinates of shape (n_scenes, 2).
    """
    import umap

    _log = log or (lambda msg, level: None)
    n_samples = len(embeddings)
    n_neighbors = min(15, max(2, n_samples - 1))
    init_method = "random" if n_samples < 10 else "spectral"

    _log(f"Running UMAP: {n_samples} scenes, n_neighbors={n_neighbors}", "info")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        init=init_method,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32)


def build_taste_profile(
    scene_ids: list[int],
    embeddings: "NDArray[np.float32]",
    engagement_scores: dict[int, float],
    tag_vocabulary: "TagVocabulary",
    model_key: str,
    log: Callable[[str, str], None] | None = None,
) -> "TasteProfile":
    """Build a complete taste profile with clustering, labeling, and UMAP.

    Args:
        scene_ids: Scene IDs for profile scenes.
        embeddings: Visual embeddings matrix (n_scenes, n_dims).
        engagement_scores: scene_id -> engagement score mapping.
        tag_vocabulary: Vocabulary for auto-labeling clusters.
        model_key: Embedding model identifier.
        log: Logging callback.

    Returns:
        Complete TasteProfile with clusters and coordinates.
    """
    from stash_ai.recommendations.types import TasteCluster, TasteProfile

    _log = log or (lambda msg, level: None)
    n_scenes = len(scene_ids)

    if n_scenes < MIN_SCENES_FOR_CLUSTERING:
        _log(f"Only {n_scenes} scenes — too few for clustering (need {MIN_SCENES_FOR_CLUSTERING})", "warning")
        # Create single cluster with all scenes
        total_engagement = sum(engagement_scores.get(sid, 0) for sid in scene_ids)
        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        tag_matches = tag_vocabulary.match_cluster_centroid(centroid, top_k=8)
        auto_label = " / ".join(m["text"] for m in tag_matches[:2]) if tag_matches else "Mixed"

        cluster = TasteCluster(
            cluster_id=0,
            centroid=centroid,
            scene_ids=scene_ids,
            engagement_total=total_engagement,
            engagement_share=1.0,
            auto_label=auto_label,
            user_label=None,
            weight_override=None,
            excluded=False,
            tag_matches=tag_matches,
        )
        return TasteProfile(
            clusters=[cluster],
            optimal_k=1,
            silhouette_score=0.0,
            model_key=model_key,
        )

    # Find optimal k
    _log("Finding optimal cluster count...", "info")
    optimal_k, sil_score = find_optimal_k(embeddings, log=_log)

    # Cluster scenes
    _log(f"Clustering {n_scenes} scenes into {optimal_k} clusters...", "info")
    cluster_scene_ids, centroids, labels = cluster_scenes(
        scene_ids, embeddings, engagement_scores, optimal_k
    )

    # Compute total engagement for share calculation
    total_engagement = sum(engagement_scores.get(sid, 0) for sid in scene_ids)

    # Build cluster objects with auto-labels
    clusters: list[TasteCluster] = []
    for k in range(optimal_k):
        cluster_eng = sum(engagement_scores.get(sid, 0) for sid in cluster_scene_ids[k])
        eng_share = cluster_eng / total_engagement if total_engagement > 0 else 1.0 / optimal_k

        # Auto-label from tag vocabulary
        tag_matches = tag_vocabulary.match_cluster_centroid(centroids[k], top_k=8)
        auto_label = " / ".join(m["text"] for m in tag_matches[:2]) if tag_matches else f"Cluster {k + 1}"

        clusters.append(TasteCluster(
            cluster_id=k,
            centroid=centroids[k],
            scene_ids=cluster_scene_ids[k],
            engagement_total=cluster_eng,
            engagement_share=eng_share,
            auto_label=auto_label,
            user_label=None,
            weight_override=None,
            excluded=False,
            tag_matches=tag_matches,
        ))

    # Sort clusters by engagement share (largest first)
    clusters.sort(key=lambda c: c.engagement_share, reverse=True)
    # Re-assign cluster IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    _log(f"Built {len(clusters)} taste clusters", "info")
    for c in clusters:
        _log(f"  Cluster {c.cluster_id}: '{c.auto_label}' — {len(c.scene_ids)} scenes, {c.engagement_share:.1%} engagement", "info")

    return TasteProfile(
        clusters=clusters,
        optimal_k=optimal_k,
        silhouette_score=sil_score,
        model_key=model_key,
    )
```

**Step 2: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.recommendations.clusters import find_optimal_k, cluster_scenes, compute_umap_projection
import numpy as np
# Quick smoke test with random data
emb = np.random.randn(20, 128).astype(np.float32)
k, score = find_optimal_k(emb)
print(f'Optimal k={k}, silhouette={score:.4f}')
ids = list(range(20))
eng = {i: float(i) for i in ids}
cluster_ids, centroids, labels = cluster_scenes(ids, emb, eng, k)
print(f'Clusters: {[len(c) for c in cluster_ids]}, centroid shape: {centroids.shape}')
coords = compute_umap_projection(emb)
print(f'UMAP coords shape: {coords.shape}')
print('Clustering OK')
"`

Expected: Cluster counts, shapes printed, `Clustering OK`

**Step 3: Commit**

```bash
git add stash_ai/recommendations/clusters.py
git commit -m "feat(taste-map): add clustering engine with K-Means, silhouette scoring, UMAP projection"
```

---

## Phase 4: Task Pipeline

### Task 7: Create Taste Map Task

**Files:**
- Create: `stash_ai/tasks/taste_map.py`

**Step 1: Write the task**

```python
"""Build Taste Map task — orchestrates clustering, UMAP projection, and auto-labeling."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from stashapi.stashapp import StashInterface

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.embeddings.tag_vocabulary import TagVocabulary
from stash_ai.recommendations.clusters import (
    build_taste_profile,
    compute_umap_projection,
)
from stash_ai.recommendations.engagement import EngagementCalculator
from stash_ai.recommendations.types import TasteMapResponse, TasteMapSceneData


class TasteMapTask:
    """Task for building the taste map visualization data."""

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

    def run(
        self,
        request_id: str = "",
        top_scenes: int = 200,
        weights: dict[str, float] | None = None,
        time_decay: dict[str, float] | None = None,
        scoring_method: str = "base_weighted",
    ) -> TasteMapResponse:
        """Run the taste map pipeline.

        Steps:
            1. Load engaged scenes + embeddings
            2. Cluster (K-Means + silhouette)
            3. UMAP projection for ALL embedded scenes
            4. Tag embedding matching for auto-labels
            5. Save results to JSON

        Args:
            request_id: Unique ID for result file.
            top_scenes: Number of top engaged scenes for profile.
            weights: Engagement weight overrides.
            time_decay: Time decay config.
            scoring_method: 'base_weighted' or 'time_decayed'.

        Returns:
            Complete taste map response.
        """
        try:
            total_steps = 5
            self.progress(0, total_steps)

            # Step 1: Load engaged scenes + embeddings
            self.log("Step 1/5: Loading engaged scenes and embeddings...", "info")
            scene_ids, embeddings, engagement_scores = self._load_scene_data(
                top_scenes, weights, time_decay, scoring_method
            )

            if len(scene_ids) == 0:
                self.log("No embedded scenes found", "error")
                response: TasteMapResponse = {
                    "status": "error",
                    "optimal_k": 0,
                    "silhouette_score": 0.0,
                    "clusters": [],
                    "scenes": [],
                    "error": "No embedded scenes found. Run 'Embed All Scenes' first.",
                }
                self._save_results(response, request_id)
                return response

            self.log(f"Loaded {len(scene_ids)} profile scenes", "info")
            self.progress(1, total_steps)

            # Step 2: Ensure tag embeddings
            self.log("Step 2/5: Preparing tag vocabulary embeddings...", "info")
            tag_vocab = TagVocabulary(
                storage=self.storage,
                model_key=self.model_key,
                log_callback=self.log,
            )
            stash_tags = self._get_stash_tags()
            tag_vocab.ensure_embeddings(stash_tags=stash_tags)
            self.progress(2, total_steps)

            # Step 3: Cluster scenes
            self.log("Step 3/5: Clustering scenes...", "info")
            profile = build_taste_profile(
                scene_ids=scene_ids,
                embeddings=embeddings,
                engagement_scores=engagement_scores,
                tag_vocabulary=tag_vocab,
                model_key=self.model_key,
                log=self.log,
            )
            self.progress(3, total_steps)

            # Step 4: UMAP projection (all embedded scenes, not just profile)
            self.log("Step 4/5: Computing UMAP projection...", "info")
            all_scene_ids, all_embeddings = self._load_all_embeddings()
            umap_coords = compute_umap_projection(all_embeddings, log=self.log)

            # Build cluster assignment map for profile scenes
            cluster_map: dict[int, int] = {}
            for cluster in profile.clusters:
                for sid in cluster.scene_ids:
                    cluster_map[sid] = cluster.cluster_id

            # Save coords to storage
            coords_dict: dict[int, tuple[float, float]] = {}
            for i, sid in enumerate(all_scene_ids):
                coords_dict[sid] = (float(umap_coords[i][0]), float(umap_coords[i][1]))
            self.storage.save_umap_coords(coords_dict, cluster_map, self.model_key)
            self.progress(4, total_steps)

            # Step 5: Build response and save
            self.log("Step 5/5: Building response...", "info")

            # Save clusters to storage
            self.storage.save_taste_clusters(profile.clusters, self.model_key)

            # Build scene data for frontend
            scene_details = self._get_scene_details(all_scene_ids)
            scenes_data: list[TasteMapSceneData] = []
            for i, sid in enumerate(all_scene_ids):
                details = scene_details.get(sid, {})
                scenes_data.append({
                    "scene_id": sid,
                    "x": float(umap_coords[i][0]),
                    "y": float(umap_coords[i][1]),
                    "cluster_id": cluster_map.get(sid),
                    "engagement_score": engagement_scores.get(sid, 0.0),
                    "is_profile": sid in engagement_scores,
                    "title": details.get("title"),
                    "thumbnail": self._get_thumbnail_url(sid),
                    "play_count": details.get("play_count", 0),
                    "o_counter": details.get("o_counter", 0),
                })

            # Build cluster data
            clusters_data = []
            for cluster in profile.clusters:
                # Find representative scenes (closest to centroid)
                rep_scenes = self._find_representative_scenes(
                    cluster, embeddings, scene_ids, n=3
                )
                clusters_data.append({
                    "cluster_id": cluster.cluster_id,
                    "auto_label": cluster.auto_label,
                    "scene_ids": cluster.scene_ids,
                    "engagement_total": round(cluster.engagement_total, 2),
                    "engagement_share": round(cluster.engagement_share, 4),
                    "representative_scenes": rep_scenes,
                    "tag_matches": cluster.tag_matches,
                })

            response = TasteMapResponse(
                status="complete",
                optimal_k=profile.optimal_k,
                silhouette_score=round(profile.silhouette_score, 4),
                clusters=clusters_data,
                scenes=scenes_data,
                error=None,
            )

            self._save_results(response, request_id)
            self.progress(5, total_steps)

            self.log(
                f"Taste map complete: {profile.optimal_k} clusters, "
                f"{len(scenes_data)} scenes, silhouette={profile.silhouette_score:.4f}",
                "info",
            )
            return response

        except Exception as e:
            self.log(f"Taste map failed: {e}", "error")
            error_response: TasteMapResponse = {
                "status": "error",
                "optimal_k": 0,
                "silhouette_score": 0.0,
                "clusters": [],
                "scenes": [],
                "error": str(e),
            }
            self._save_results(error_response, request_id)
            return error_response

    def _load_scene_data(
        self,
        top_scenes: int,
        weights: dict[str, float] | None,
        time_decay: dict[str, float] | None,
        scoring_method: str,
    ) -> tuple[list[int], "np.ndarray", dict[int, float]]:
        """Load top engaged scenes with their embeddings."""
        from stash_ai.recommendations.types import EngagementScoringMethod

        calculator = EngagementCalculator(
            weights=weights,
            time_decay=time_decay,
            log_callback=self.log,
        )

        method = (
            EngagementScoringMethod.TIME_DECAYED
            if scoring_method == "time_decayed"
            else EngagementScoringMethod.BASE_WEIGHTED
        )

        # Get more than needed to account for missing embeddings
        scores = calculator.get_top_engaged_scenes(limit=top_scenes * 2, method=method)

        # Filter to scenes with embeddings
        valid_scene_ids = self.storage.get_embedded_scene_ids()
        embedded_set = set(valid_scene_ids) if not isinstance(valid_scene_ids, set) else valid_scene_ids

        scene_ids: list[int] = []
        engagement_map: dict[int, float] = {}

        for score in scores:
            if score.scene_id in embedded_set:
                scene_ids.append(score.scene_id)
                eng = score.time_decayed_score if method == EngagementScoringMethod.TIME_DECAYED else score.raw_score
                engagement_map[score.scene_id] = eng
                if len(scene_ids) >= top_scenes:
                    break

        if not scene_ids:
            return [], np.array([]), {}

        # Load embeddings
        embeddings_list: list[list[float]] = []
        valid_ids: list[int] = []
        for sid in scene_ids:
            emb = self.storage.get_scene_embedding(sid)
            if emb and emb.get("visual_embedding"):
                embeddings_list.append(emb["visual_embedding"])
                valid_ids.append(sid)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        # Filter engagement map to valid IDs
        engagement_map = {sid: engagement_map[sid] for sid in valid_ids}

        return valid_ids, embeddings, engagement_map

    def _load_all_embeddings(self) -> tuple[list[int], "np.ndarray"]:
        """Load ALL scene embeddings for UMAP projection."""
        all_ids = self.storage.get_embedded_scene_ids()
        if isinstance(all_ids, set):
            all_ids = sorted(all_ids)

        embeddings_list: list[list[float]] = []
        valid_ids: list[int] = []

        for sid in all_ids:
            emb = self.storage.get_scene_embedding(sid)
            if emb and emb.get("visual_embedding"):
                embeddings_list.append(emb["visual_embedding"])
                valid_ids.append(sid)

        return valid_ids, np.array(embeddings_list, dtype=np.float32)

    def _get_stash_tags(self) -> list[str]:
        """Fetch all tag names from Stash database."""
        try:
            result = self.stash.find_tags(filter={"per_page": -1}, fragment="name")
            return [t["name"] for t in result if t.get("name")]
        except Exception as e:
            self.log(f"Failed to fetch tags: {e}", "warning")
            return []

    def _get_scene_details(self, scene_ids: list[int]) -> dict[int, dict]:
        """Fetch scene details (title, play_count, o_counter) from Stash."""
        details: dict[int, dict] = {}
        try:
            for sid in scene_ids:
                scene = self.stash.find_scene(sid)
                if scene:
                    details[sid] = {
                        "title": scene.get("title") or scene.get("files", [{}])[0].get("basename", f"Scene {sid}"),
                        "play_count": scene.get("play_count", 0),
                        "o_counter": scene.get("o_counter", 0),
                    }
        except Exception as e:
            self.log(f"Failed to fetch scene details: {e}", "warning")
        return details

    def _get_thumbnail_url(self, scene_id: int) -> str:
        """Get the thumbnail URL for a scene."""
        return f"/scene/{scene_id}/screenshot"

    def _find_representative_scenes(
        self,
        cluster: "Any",
        all_embeddings: "np.ndarray",
        all_scene_ids: list[int],
        n: int = 3,
    ) -> list[int]:
        """Find n scenes closest to the cluster centroid."""
        scene_id_to_idx = {sid: i for i, sid in enumerate(all_scene_ids)}
        distances: list[tuple[int, float]] = []

        for sid in cluster.scene_ids:
            idx = scene_id_to_idx.get(sid)
            if idx is not None:
                emb = all_embeddings[idx]
                dist = float(np.dot(cluster.centroid, emb / (np.linalg.norm(emb) + 1e-8)))
                distances.append((sid, dist))

        distances.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in distances[:n]]

    def _save_results(self, response: TasteMapResponse, request_id: str) -> None:
        """Save results to JSON file for frontend polling."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        filename = f"taste_map_{request_id}.json" if request_id else "taste_map.json"
        filepath = os.path.join(assets_dir, filename)

        with open(filepath, "w") as f:
            json.dump(response, f)

        self.log(f"Results saved to {filename}", "debug")
```

**Step 2: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tasks.taste_map import TasteMapTask; print('Task import OK')"`

Expected: `Task import OK`

**Step 3: Commit**

```bash
git add stash_ai/tasks/taste_map.py
git commit -m "feat(taste-map): add taste map task pipeline"
```

---

### Task 8: Register Task in Plugin Dispatcher

**Files:**
- Modify: `stash-copilot.py`

**Step 1: Add task registration**

Find the task dispatcher in `run_task` method (around line 288-347). Find the pattern of `elif task_name ==` checks and add:

```python
elif task_name == "build_taste_map":
    self.run_build_taste_map(args)
```

**Step 2: Add task handler method**

Add the handler method following the pattern of `run_recommendations`:

```python
def run_build_taste_map(self, args: Dict[str, Any]):
    """Run the Build Taste Map task."""
    try:
        from stash_ai.tasks.taste_map import TasteMapTask

        plugin_settings = self.get_plugin_settings("stash-copilot")
        model_key = args.get("model_key", plugin_settings.get("embedding_model", "siglip"))

        task = TasteMapTask(
            stash=self.stash,
            log_callback=self.log,
            progress_callback=self.progress,
            model_key=model_key,
        )

        response = task.run(
            request_id=args.get("request_id", ""),
            top_scenes=int(args.get("top_scenes", 200)),
            scoring_method=args.get("scoring_method", "base_weighted"),
        )

        if response["status"] == "complete":
            self.log(
                f"Taste map complete: {response['optimal_k']} clusters, "
                f"{len(response['scenes'])} scenes",
                "info",
            )
        else:
            self.log(f"Taste map failed: {response.get('error', 'Unknown error')}", "error")

    except Exception as e:
        self.error(f"Build Taste Map failed: {e}")
```

**Step 3: Add task to plugin YAML**

Check the plugin YAML file (likely `stash-copilot.yml`) and add the new task to the `tasks` list:

```yaml
- name: "Build Taste Map"
  description: "Build taste clusters and 2D visualization of your preferences"
  defaultArgs:
    mode: "build_taste_map"
```

**Step 4: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
import stash_copilot
# Just verify the import doesn't break
print('Plugin import OK')
"` (will fail due to stdin, but checks imports)

**Step 5: Commit**

```bash
git add stash-copilot.py stash-copilot.yml
git commit -m "feat(taste-map): register Build Taste Map task in plugin dispatcher"
```

---

## Phase 5: Cluster Recommendation Engine

### Task 9: Create Cluster-Based Recommendation Engine

**Files:**
- Create: `stash_ai/recommendations/cluster_engine.py`

**Step 1: Write the cluster engine**

```python
"""Cluster-based recommendation engine with proportional sampling.

Replaces single-profile cosine similarity with per-cluster querying
and weighted round-robin merging for diverse recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from stash_ai.embeddings.storage import EmbeddingStorage
    from stash_ai.recommendations.types import RecommendationResult, SceneDetails


class ClusterRecommendationEngine:
    """Query recommendations per-cluster and merge with proportional sampling."""

    def __init__(
        self,
        storage: "EmbeddingStorage",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)

    def get_cluster_recommendations(
        self,
        mode: str,  # 'discover_new' | 'rewatch'
        limit: int = 120,
        min_similarity: float = 0.5,
        exclude_scene_ids: set[int] | None = None,
        watched_scene_ids: set[int] | None = None,
    ) -> list["RecommendationResult"]:
        """Generate recommendations using cluster-based querying.

        Args:
            mode: 'discover_new' (unwatched) or 'rewatch' (watched only).
            limit: Max total results.
            min_similarity: Minimum cosine similarity threshold.
            exclude_scene_ids: Scene IDs to exclude from results.
            watched_scene_ids: Set of watched scene IDs (for mode filtering).

        Returns:
            Merged, deduplicated recommendation results.
        """
        model_key = self.storage.model_key
        clusters = self.storage.get_taste_clusters(model_key)

        if not clusters:
            self.log("No taste clusters found — run 'Build Taste Map' first", "warning")
            return []

        # Filter to active (non-excluded) clusters
        active_clusters = [c for c in clusters if not c["excluded"]]
        if not active_clusters:
            self.log("All clusters are excluded", "warning")
            return []

        # Calculate effective weights
        total_weight = sum(
            c.get("weight_override") or c["engagement_share"]
            for c in active_clusters
        )
        if total_weight <= 0:
            total_weight = 1.0

        cluster_weights: list[tuple[dict, float]] = []
        for c in active_clusters:
            weight = (c.get("weight_override") or c["engagement_share"]) / total_weight
            cluster_weights.append((c, weight))

        self.log(
            f"Querying {len(active_clusters)} clusters "
            f"(weights: {[f'{w:.0%}' for _, w in cluster_weights]})",
            "info",
        )

        # Query each cluster
        per_cluster_results: list[tuple[dict, float, list["RecommendationResult"]]] = []
        for cluster, weight in cluster_weights:
            centroid = np.array(cluster["centroid"], dtype=np.float32)
            cluster_limit = max(10, int(limit * weight * 2))  # Over-fetch for dedup

            results = self._query_single_cluster(
                centroid=centroid,
                limit=cluster_limit,
                min_similarity=min_similarity,
                mode=mode,
                exclude_scene_ids=exclude_scene_ids or set(),
                profile_scene_ids=set(cluster["scene_ids"]),
                watched_scene_ids=watched_scene_ids or set(),
            )

            per_cluster_results.append((cluster, weight, results))
            self.log(
                f"  Cluster '{cluster['auto_label']}': {len(results)} results",
                "debug",
            )

        # Proportional merge
        merged = self._proportional_merge(per_cluster_results, limit)
        self.log(f"Merged {len(merged)} recommendations from {len(active_clusters)} clusters", "info")

        return merged

    def _query_single_cluster(
        self,
        centroid: "np.ndarray",
        limit: int,
        min_similarity: float,
        mode: str,
        exclude_scene_ids: set[int],
        profile_scene_ids: set[int],
        watched_scene_ids: set[int],
    ) -> list["RecommendationResult"]:
        """Query similar scenes for a single cluster centroid."""
        # Get all similar scenes
        raw_results = self.storage.find_similar(
            query_embedding=centroid.tolist(),
            limit=limit * 3,  # Over-fetch to filter
            min_similarity=min_similarity,
        )

        results: list["RecommendationResult"] = []
        for scene_id, similarity in raw_results:
            # Apply mode filter
            if mode == "discover_new":
                if scene_id in watched_scene_ids or scene_id in profile_scene_ids:
                    continue
            elif mode == "rewatch":
                if scene_id not in watched_scene_ids:
                    continue

            # Apply exclusions
            if scene_id in exclude_scene_ids:
                continue

            results.append({
                "scene_id": scene_id,
                "similarity_score": similarity,
                "engagement_score": 0.0,
                "combined_score": similarity,
                "scene": {"id": scene_id},
            })

            if len(results) >= limit:
                break

        return results

    def _proportional_merge(
        self,
        per_cluster_results: list[tuple[dict, float, list["RecommendationResult"]]],
        limit: int,
    ) -> list["RecommendationResult"]:
        """Merge results from multiple clusters with proportional sampling.

        Uses weighted round-robin: if cluster A has 50% weight and cluster B has 30%
        and cluster C has 20%, every 10 results will have ~5 from A, ~3 from B, ~2 from C.
        """
        seen_ids: set[int] = set()
        merged: list["RecommendationResult"] = []

        # Track position in each cluster's results
        positions = [0] * len(per_cluster_results)
        weights = [w for _, w, _ in per_cluster_results]

        # Accumulator-based round-robin
        accumulators = [0.0] * len(per_cluster_results)

        max_iterations = limit * 3  # Safety cap
        iteration = 0

        while len(merged) < limit and iteration < max_iterations:
            iteration += 1

            # Add weights to accumulators
            for i in range(len(accumulators)):
                accumulators[i] += weights[i]

            # Pick the cluster with highest accumulator
            best_idx = max(range(len(accumulators)), key=lambda i: accumulators[i])

            cluster, weight, results = per_cluster_results[best_idx]

            # Try to get next unseen result from this cluster
            added = False
            while positions[best_idx] < len(results):
                result = results[positions[best_idx]]
                positions[best_idx] += 1

                if result["scene_id"] not in seen_ids:
                    seen_ids.add(result["scene_id"])
                    merged.append(result)
                    added = True
                    break

            if added:
                accumulators[best_idx] -= 1.0
            else:
                # This cluster is exhausted, remove its weight
                accumulators[best_idx] = -float("inf")
                weights[best_idx] = 0.0

                # Check if all clusters are exhausted
                if all(w == 0.0 for w in weights):
                    break

        return merged
```

**Step 2: Verify**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.recommendations.cluster_engine import ClusterRecommendationEngine; print('Cluster engine OK')"`

Expected: `Cluster engine OK`

**Step 3: Commit**

```bash
git add stash_ai/recommendations/cluster_engine.py
git commit -m "feat(taste-map): add cluster-based recommendation engine with proportional merge"
```

---

### Task 10: Integrate Cluster Engine into Existing Recommendation Flow

**Files:**
- Modify: `stash_ai/recommendations/engine.py`
- Modify: `stash_ai/tasks/recommendations.py`

**Step 1: Update engine.py**

Read the existing `engine.py` to understand the `_discover_new` and `_rewatch_favorites` methods. Add a method that delegates to the cluster engine when taste clusters exist, falling back to the old single-profile approach when they don't.

Add to the `RecommendationEngine` class:

```python
def _has_taste_clusters(self) -> bool:
    """Check if taste clusters have been built."""
    clusters = self.storage.get_taste_clusters(self.storage.model_key)
    return len(clusters) > 0

def _cluster_discover_new(self, config: "RecommendationConfig") -> list["RecommendationResult"]:
    """Discover new scenes using cluster-based querying."""
    from stash_ai.recommendations.cluster_engine import ClusterRecommendationEngine

    cluster_engine = ClusterRecommendationEngine(
        storage=self.storage,
        log_callback=self.log,
    )

    watched = self._get_watched_scene_ids()
    return cluster_engine.get_cluster_recommendations(
        mode="discover_new",
        limit=config.get("limit", 120),
        min_similarity=config.get("min_similarity", 0.5),
        watched_scene_ids=watched,
    )

def _cluster_rewatch(self, config: "RecommendationConfig") -> list["RecommendationResult"]:
    """Rewatch recommendations using cluster-based querying."""
    from stash_ai.recommendations.cluster_engine import ClusterRecommendationEngine

    cluster_engine = ClusterRecommendationEngine(
        storage=self.storage,
        log_callback=self.log,
    )

    watched = self._get_watched_scene_ids()
    return cluster_engine.get_cluster_recommendations(
        mode="rewatch",
        limit=config.get("limit", 120),
        min_similarity=config.get("min_similarity", 0.3),
        watched_scene_ids=watched,
    )
```

**Step 2: Update the dispatch logic**

In the `generate_recommendations` method (or whatever dispatches to `_discover_new` / `_rewatch_favorites`), add cluster-first logic:

```python
# At the top of the discover_new / rewatch dispatch:
if self._has_taste_clusters():
    self.log("Using cluster-based recommendations", "info")
    if mode == "discover_new":
        return self._cluster_discover_new(config)
    elif mode == "rewatch":
        return self._cluster_rewatch(config)
else:
    self.log("No taste clusters — using single-profile recommendations", "info")
    # Fall through to existing single-profile logic
```

**Step 3: Verify by reading the changes**

Read the modified `engine.py` to ensure the cluster path is correctly integrated and the fallback to single-profile works.

**Step 4: Commit**

```bash
git add stash_ai/recommendations/engine.py
git commit -m "feat(taste-map): integrate cluster engine into recommendation dispatch with single-profile fallback"
```

---

## Phase 6: Frontend — Taste Map Tab

### Task 11: Add ECharts CDN and Theme

**Files:**
- Modify: `stash-copilot.js` (top-level initialization)

**Step 1: Add ECharts CDN loader**

Find where other external scripts are loaded (search for `createElement('script')` or CDN loading). Add an ECharts loader function:

```javascript
function loadECharts() {
    return new Promise((resolve, reject) => {
        if (window.echarts) {
            resolve(window.echarts);
            return;
        }
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js';
        script.onload = () => {
            registerCopilotTheme();
            resolve(window.echarts);
        };
        script.onerror = () => reject(new Error('Failed to load ECharts'));
        document.head.appendChild(script);
    });
}
```

**Step 2: Add the Stash Copilot ECharts theme**

```javascript
function registerCopilotTheme() {
    if (!window.echarts) return;

    const CLUSTER_COLORS = [
        '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'
    ];

    window.echarts.registerTheme('stash-copilot', {
        backgroundColor: 'transparent',
        color: CLUSTER_COLORS,
        textStyle: {
            color: 'rgba(255, 255, 255, 0.7)',
            fontFamily: 'system-ui, -apple-system, monospace',
            fontSize: 12
        },
        title: {
            textStyle: { color: 'rgba(255, 255, 255, 0.9)', fontSize: 14 }
        },
        legend: {
            textStyle: { color: 'rgba(255, 255, 255, 0.7)' }
        },
        tooltip: {
            backgroundColor: 'rgba(10, 10, 15, 0.95)',
            borderColor: 'rgba(139, 92, 246, 0.3)',
            borderWidth: 1,
            textStyle: { color: 'rgba(255, 255, 255, 0.9)', fontSize: 12 },
            extraCssText: 'border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5), 0 0 15px rgba(139,92,246,0.15);'
        },
        xAxis: {
            axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
            splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
            axisLabel: { color: 'rgba(255, 255, 255, 0.4)' }
        },
        yAxis: {
            axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
            splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
            axisLabel: { color: 'rgba(255, 255, 255, 0.4)' }
        }
    });

    log('ECharts stash-copilot theme registered');
}
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(taste-map): add ECharts CDN loader and stash-copilot neon theme"
```

---

### Task 12: Add Taste Map Tab to AI Insights Modal

**Files:**
- Modify: `stash-copilot.js` — `createInsightsModal()` function

**Step 1: Add tab button**

Find the tab buttons section in `createInsightsModal()` (around line 3708-3714). Add the Taste Map tab button after the existing tabs:

```javascript
<button class="stash-copilot-insights-tab ${savedTab === 'taste_map' ? 'active' : ''}" data-tab="taste_map">Taste Map</button>
```

**Step 2: Add panel HTML**

Add the Taste Map panel in the insights body (before the closing `</div>` of the body). This is the full layout:

```javascript
<div class="stash-copilot-insights-panel ${savedTab === 'taste_map' ? 'active' : ''}" data-tab="taste_map">
    <div class="stash-copilot-taste-map-container">
        <div class="stash-copilot-taste-map-toolbar">
            <button class="btn btn-primary stash-copilot-taste-map-build-btn">Build Taste Map</button>
            <span class="stash-copilot-taste-map-status"></span>
        </div>
        <div class="stash-copilot-taste-map-content">
            <div class="stash-copilot-taste-map-main">
                <div class="stash-copilot-taste-map-chart" id="taste-map-chart"></div>
                <div class="stash-copilot-taste-map-tags" style="display: none;">
                    <div class="stash-copilot-taste-map-tags-header">
                        <span class="stash-copilot-taste-map-tags-title">TAG MATCHES</span>
                        <button class="stash-copilot-taste-map-tags-close">&times;</button>
                    </div>
                    <div class="stash-copilot-taste-map-tags-list"></div>
                    <div class="stash-copilot-taste-map-tags-custom">
                        <input type="text" placeholder="Test a phrase..." class="stash-copilot-taste-map-phrase-input">
                        <div class="stash-copilot-taste-map-phrase-result" style="display: none;"></div>
                    </div>
                </div>
            </div>
            <div class="stash-copilot-taste-map-sidebar">
                <div class="stash-copilot-taste-map-sidebar-header">CLUSTERS</div>
                <div class="stash-copilot-taste-map-clusters"></div>
            </div>
        </div>
        <div class="stash-copilot-taste-map-empty">
            <p>Build a Taste Map to visualize your preference clusters.</p>
            <p class="stash-copilot-taste-map-empty-sub">Your most engaged scenes are grouped by visual similarity, auto-labeled, and plotted in 2D space.</p>
        </div>
    </div>
</div>
```

**Step 3: Add tab switch handler**

Find `switchInsightsTab()` and add:

```javascript
if (tabName === 'taste_map') loadTasteMapData(modal);
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(taste-map): add Taste Map tab HTML to AI Insights modal"
```

---

### Task 13: Implement Taste Map Frontend Logic

**Files:**
- Modify: `stash-copilot.js`

**Step 1: Add state**

Add to the global `state` object:

```javascript
tasteMapData: null,           // Full taste map response
tasteMapRequestId: null,      // Current build request ID
tasteMapLoading: false,       // Loading flag
tasteMapSelectedCluster: null, // Currently selected cluster ID
tasteMapChart: null,          // ECharts instance
```

**Step 2: Add `buildTasteMap()` function**

Triggers the backend task and starts polling:

```javascript
async function buildTasteMap(modal) {
    if (state.tasteMapLoading) return;

    state.tasteMapLoading = true;
    const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
    const statusEl = modal.querySelector('.stash-copilot-taste-map-status');

    buildBtn.disabled = true;
    buildBtn.innerHTML = '<span class="stash-copilot-spinner"></span>';
    statusEl.textContent = 'Building taste map...';

    const requestId = `taste_map_${Date.now()}`;
    state.tasteMapRequestId = requestId;

    try {
        await runPluginTask('Build Taste Map', { request_id: requestId });
        pollTasteMapResults(modal, requestId);
    } catch (e) {
        log(`Build Taste Map error: ${e.message}`, 'error');
        state.tasteMapLoading = false;
        buildBtn.disabled = false;
        buildBtn.textContent = 'Build Taste Map';
        statusEl.textContent = `Error: ${e.message}`;
    }
}
```

**Step 3: Add `pollTasteMapResults()` function**

```javascript
function pollTasteMapResults(modal, requestId) {
    const resultFile = `/plugin/stash-copilot/assets/taste_map_${requestId}.json`;

    const interval = setInterval(async () => {
        if (state.tasteMapRequestId !== requestId) {
            clearInterval(interval);
            return;
        }

        try {
            const resp = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
            if (resp.ok) {
                const data = await resp.json();
                if (data.status === 'complete') {
                    clearInterval(interval);
                    state.tasteMapData = data;
                    state.tasteMapLoading = false;
                    renderTasteMap(modal, data);
                } else if (data.status === 'error') {
                    clearInterval(interval);
                    state.tasteMapLoading = false;
                    const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
                    const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
                    buildBtn.disabled = false;
                    buildBtn.textContent = 'Build Taste Map';
                    statusEl.textContent = `Error: ${data.error}`;
                }
            }
        } catch (e) {
            // 404 expected while task is running
        }
    }, 500);

    setTimeout(() => {
        clearInterval(interval);
        if (state.tasteMapLoading) {
            state.tasteMapLoading = false;
            const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
            statusEl.textContent = 'Timed out waiting for results';
        }
    }, 300000); // 5 min timeout (UMAP can be slow)
}
```

**Step 4: Add `renderTasteMap()` function**

This is the main render function that sets up ECharts and the sidebar:

```javascript
async function renderTasteMap(modal, data) {
    const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
    const statusEl = modal.querySelector('.stash-copilot-taste-map-status');
    const emptyEl = modal.querySelector('.stash-copilot-taste-map-empty');
    const contentEl = modal.querySelector('.stash-copilot-taste-map-content');

    buildBtn.disabled = false;
    buildBtn.textContent = 'Rebuild';
    statusEl.textContent = `${data.clusters.length} clusters, ${data.scenes.length} scenes (silhouette: ${data.silhouette_score.toFixed(2)})`;

    emptyEl.style.display = 'none';
    contentEl.style.display = 'flex';

    // Render scatter plot
    await loadECharts();
    renderTasteMapChart(modal, data);

    // Render cluster sidebar
    renderClusterSidebar(modal, data);
}
```

**Step 5: Add `renderTasteMapChart()` function**

The core ECharts scatter plot:

```javascript
function renderTasteMapChart(modal, data) {
    const CLUSTER_COLORS = [
        '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'
    ];

    const chartContainer = modal.querySelector('#taste-map-chart');
    if (!chartContainer) return;

    // Dispose old chart if exists
    if (state.tasteMapChart) {
        state.tasteMapChart.dispose();
    }

    const chart = echarts.init(chartContainer, 'stash-copilot');
    state.tasteMapChart = chart;

    // Build series per cluster + background scenes
    const series = [];

    // Background (non-profile) scenes
    const bgScenes = data.scenes.filter(s => !s.is_profile);
    if (bgScenes.length > 0) {
        series.push({
            name: 'Library',
            type: 'scatter',
            data: bgScenes.map(s => [s.x, s.y, s.scene_id, s.title, s.play_count, s.o_counter]),
            symbolSize: 3,
            itemStyle: {
                color: 'rgba(255, 255, 255, 0.08)',
            },
            emphasis: {
                itemStyle: { color: 'rgba(255, 255, 255, 0.3)', shadowBlur: 5 }
            },
            z: 1,
        });
    }

    // Profile scenes by cluster
    for (const cluster of data.clusters) {
        const clusterScenes = data.scenes.filter(
            s => s.cluster_id === cluster.cluster_id && s.is_profile
        );
        const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];

        series.push({
            name: cluster.auto_label,
            type: 'scatter',
            data: clusterScenes.map(s => [
                s.x, s.y, s.scene_id, s.title || `Scene ${s.scene_id}`,
                s.play_count, s.o_counter, s.engagement_score, cluster.cluster_id
            ]),
            symbolSize: function(val) {
                // Size based on engagement (index 6)
                const eng = val[6] || 1;
                return Math.max(6, Math.min(20, 6 + Math.log(eng + 1) * 3));
            },
            itemStyle: {
                color: color,
                shadowBlur: 8,
                shadowColor: color + '80',
            },
            emphasis: {
                itemStyle: {
                    shadowBlur: 15,
                    shadowColor: color,
                    borderColor: '#fff',
                    borderWidth: 1,
                },
                scale: 1.5,
            },
            z: 10,
        });
    }

    const option = {
        animation: true,
        animationDuration: 1000,
        animationEasing: 'cubicOut',
        grid: { left: 40, right: 20, top: 20, bottom: 40 },
        xAxis: { type: 'value', show: false },
        yAxis: { type: 'value', show: false },
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                if (params.seriesName === 'Library') {
                    const [x, y, id, title, plays, oCnt] = params.data;
                    return `<div style="max-width:200px">
                        <b>${title || 'Scene ' + id}</b><br/>
                        <span style="color:#888">▶ ${plays} 💦 ${oCnt}</span>
                    </div>`;
                }
                const [x, y, id, title, plays, oCnt, eng] = params.data;
                return `<div style="max-width:200px">
                    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${params.color}"></span>
                        <b>${title}</b>
                    </div>
                    <span style="color:#888">▶ ${plays} 💦 ${oCnt}</span><br/>
                    <span style="color:#888">Engagement: ${eng.toFixed(1)}</span>
                </div>`;
            }
        },
        toolbox: {
            feature: {
                dataZoom: { yAxisIndex: 'none', title: { zoom: 'Zoom', back: 'Reset' } },
                brush: { title: { rect: 'Select', clear: 'Clear' } },
                restore: { title: 'Reset' },
            },
            iconStyle: { borderColor: 'rgba(255,255,255,0.4)' },
            emphasis: { iconStyle: { borderColor: '#8b5cf6' } },
        },
        brush: {
            toolbox: ['rect', 'polygon', 'clear'],
            xAxisIndex: 0,
            yAxisIndex: 0,
        },
        dataZoom: [
            { type: 'inside', xAxisIndex: 0 },
            { type: 'inside', yAxisIndex: 0 },
        ],
        series: series,
    };

    chart.setOption(option);

    // Handle click to navigate
    chart.on('click', function(params) {
        if (params.data && params.data[2]) {
            window.location.href = `/scenes/${params.data[2]}`;
        }
    });

    // Handle resize
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(chartContainer);
}
```

**Step 6: Add `renderClusterSidebar()` function**

```javascript
function renderClusterSidebar(modal, data) {
    const CLUSTER_COLORS = [
        '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'
    ];

    const container = modal.querySelector('.stash-copilot-taste-map-clusters');
    if (!container) return;

    container.innerHTML = data.clusters.map(cluster => {
        const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
        const thumbs = cluster.representative_scenes.map(sid => {
            const scene = data.scenes.find(s => s.scene_id === sid);
            const thumb = scene?.thumbnail || `/scene/${sid}/screenshot`;
            return `<img src="${thumb}" class="stash-copilot-taste-map-cluster-thumb" alt="">`;
        }).join('');

        const totalPlays = data.scenes
            .filter(s => cluster.scene_ids.includes(s.scene_id))
            .reduce((sum, s) => sum + s.play_count, 0);
        const totalO = data.scenes
            .filter(s => cluster.scene_ids.includes(s.scene_id))
            .reduce((sum, s) => sum + s.o_counter, 0);

        return `
            <div class="stash-copilot-taste-map-cluster-card"
                 data-cluster-id="${cluster.cluster_id}"
                 style="--cluster-color: ${color}; --cluster-color-rgb: ${hexToRgb(color)}">
                <div class="stash-copilot-taste-map-cluster-header">
                    <span class="stash-copilot-taste-map-cluster-dot" style="background: ${color}"></span>
                    <span class="stash-copilot-taste-map-cluster-label">${cluster.auto_label}</span>
                </div>
                <div class="stash-copilot-taste-map-cluster-stats">
                    <span>${cluster.scene_ids.length} scenes &middot; ${(cluster.engagement_share * 100).toFixed(0)}%</span>
                    <span>▶ ${totalPlays} 💦 ${totalO}</span>
                </div>
                <div class="stash-copilot-taste-map-cluster-thumbs">${thumbs}</div>
                <div class="stash-copilot-taste-map-cluster-weight">
                    <input type="range" min="0" max="200" value="100"
                           class="stash-copilot-taste-map-weight-slider"
                           title="Recommendation weight">
                    <span class="stash-copilot-taste-map-weight-value">100%</span>
                </div>
                <button class="stash-copilot-taste-map-exclude-btn" title="Exclude from recommendations">Exclude</button>
            </div>
        `;
    }).join('');

    // Setup event handlers
    setupClusterCardEvents(modal, data);
}

function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `${r}, ${g}, ${b}`;
}
```

**Step 7: Add `setupClusterCardEvents()` and `setupTasteMapEvents()` functions**

```javascript
function setupClusterCardEvents(modal, data) {
    const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');

    cards.forEach(card => {
        const clusterId = parseInt(card.dataset.clusterId);

        // Click to select/highlight
        card.addEventListener('click', (e) => {
            if (e.target.closest('.stash-copilot-taste-map-weight-slider') ||
                e.target.closest('.stash-copilot-taste-map-exclude-btn')) return;

            cards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            state.tasteMapSelectedCluster = clusterId;

            highlightClusterInChart(clusterId);
            showTagMatches(modal, data.clusters.find(c => c.cluster_id === clusterId));
        });

        // Weight slider
        const slider = card.querySelector('.stash-copilot-taste-map-weight-slider');
        const valueEl = card.querySelector('.stash-copilot-taste-map-weight-value');
        slider.addEventListener('input', () => {
            valueEl.textContent = `${slider.value}%`;
        });

        // Exclude button
        const excludeBtn = card.querySelector('.stash-copilot-taste-map-exclude-btn');
        excludeBtn.addEventListener('click', () => {
            card.classList.toggle('excluded');
            excludeBtn.textContent = card.classList.contains('excluded') ? 'Include' : 'Exclude';
        });
    });
}

function setupTasteMapEvents(modal) {
    // Build button
    const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
    if (buildBtn) {
        buildBtn.addEventListener('click', () => buildTasteMap(modal));
    }

    // Tag panel close
    const closeBtn = modal.querySelector('.stash-copilot-taste-map-tags-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.querySelector('.stash-copilot-taste-map-tags').style.display = 'none';
        });
    }

    // Custom phrase input
    const phraseInput = modal.querySelector('.stash-copilot-taste-map-phrase-input');
    if (phraseInput) {
        let debounceTimer;
        phraseInput.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                testCustomPhrase(modal, phraseInput.value);
            }, 500);
        });
    }
}

function highlightClusterInChart(clusterId) {
    if (!state.tasteMapChart) return;

    const option = state.tasteMapChart.getOption();
    // Dim all series except the selected cluster (+1 for background series)
    option.series.forEach((s, i) => {
        if (i === 0) {
            // Background — always dim
            s.itemStyle = { ...s.itemStyle, opacity: 0.05 };
        } else if (i - 1 === clusterId) {
            // Selected cluster — full brightness
            s.itemStyle = { ...s.itemStyle, opacity: 1.0 };
        } else {
            // Other clusters — dim
            s.itemStyle = { ...s.itemStyle, opacity: 0.15 };
        }
    });
    state.tasteMapChart.setOption({ series: option.series }, { replaceMerge: ['series'] });
}

function showTagMatches(modal, cluster) {
    if (!cluster) return;

    const tagsPanel = modal.querySelector('.stash-copilot-taste-map-tags');
    const tagsList = modal.querySelector('.stash-copilot-taste-map-tags-list');
    const titleEl = modal.querySelector('.stash-copilot-taste-map-tags-title');

    const CLUSTER_COLORS = [
        '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'
    ];
    const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];

    titleEl.innerHTML = `TAG MATCHES FOR: <span style="color:${color}">${cluster.auto_label}</span>`;

    tagsList.innerHTML = cluster.tag_matches.map(match => `
        <div class="stash-copilot-taste-map-tag-row">
            <span class="stash-copilot-taste-map-tag-text">"${match.text}"</span>
            <div class="stash-copilot-taste-map-tag-bar">
                <div class="stash-copilot-taste-map-tag-fill" style="width: ${match.similarity * 100}%; background: ${color}"></div>
            </div>
            <span class="stash-copilot-taste-map-tag-score">${match.similarity.toFixed(2)}</span>
        </div>
    `).join('');

    tagsPanel.style.display = 'block';
}

async function testCustomPhrase(modal, phrase) {
    if (!phrase || phrase.length < 3) {
        modal.querySelector('.stash-copilot-taste-map-phrase-result').style.display = 'none';
        return;
    }

    // TODO: Call backend API to embed phrase and compare against selected cluster
    // For now, show placeholder
    const resultEl = modal.querySelector('.stash-copilot-taste-map-phrase-result');
    resultEl.textContent = `Testing "${phrase}"...`;
    resultEl.style.display = 'block';
}

function loadTasteMapData(modal) {
    // Setup events on first load
    setupTasteMapEvents(modal);

    // If we already have data, render it
    if (state.tasteMapData) {
        renderTasteMap(modal, state.tasteMapData);
    }
}
```

**Step 8: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(taste-map): add full taste map frontend logic — chart, sidebar, tag panel, interactions"
```

---

### Task 14: Add Taste Map CSS

**Files:**
- Modify: `stash-copilot.css`

**Step 1: Add all Taste Map styles**

Add a new section for Taste Map styles. Follow the neon/cyberpunk design system. Key elements:

- `.stash-copilot-taste-map-container` — full height flex layout
- `.stash-copilot-taste-map-content` — two-panel split (70/30)
- `.stash-copilot-taste-map-chart` — fill available space, min-height 400px
- `.stash-copilot-taste-map-sidebar` — 300px fixed width, scrollable
- `.stash-copilot-taste-map-cluster-card` — dark panel with cluster-colored border, glow on active
- `.stash-copilot-taste-map-tags` — slide-up panel with tag match bars
- `.stash-copilot-taste-map-weight-slider` — styled range input matching purple theme
- Neon orb animation for active cluster cards (2s pulse)
- All surfaces use the design system surface colors

This is a substantial CSS file (~200 lines). Write it based on the design system defined in the design document. Key requirements:
- Background: `#0a0a0f`
- Panels: `rgba(15, 15, 25, 0.9)` with `rgba(139, 92, 246, 0.2)` borders
- Active glow: `box-shadow: 0 0 12px rgba(var(--cluster-color-rgb), 0.4)`
- Tag bars: horizontal bars with cluster color fill
- Cluster thumbs: 48x48 rounded, 3 in a row
- Exclude state: 50% opacity, strikethrough label

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(taste-map): add neon/cyberpunk CSS for taste map visualization"
```

---

## Phase 7: Integration & Testing

### Task 15: End-to-End Testing

**Testing protocol (Playwright MCP):**

1. Navigate to `http://localhost:9999`
2. Click "AI Insights" in the navbar
3. Click the "Taste Map" tab
4. Verify the empty state message and "Build Taste Map" button appear
5. Click "Build Taste Map"
6. Wait for the task to complete (monitor logs with `tail -f ~/.stash/stash.log`)
7. Verify:
   - Scatter plot renders with colored clusters
   - Cluster sidebar shows cards with labels, stats, thumbs
   - Clicking a cluster card highlights it in the chart
   - Tag match panel appears with similarity bars
   - Zoom/pan works on the chart
   - Weight sliders are draggable
   - Exclude button toggles state
8. Take screenshots of each state to `tests/screenshots/taste-map-*.png`
9. Switch to "Recs" tab, click "Generate"
10. Verify recommendations use cluster-based engine (check logs for "Using cluster-based recommendations")
11. Check logs for any errors: `grep -i "error\|exception\|traceback" ~/.stash/stash.log`

### Task 16: Review and Polish

Use `superpowers:requesting-code-review` to review all changes before committing the final state. Focus on:
- Type safety across all new Python files
- No dead code or unused imports
- CSS consistency with design system
- Error handling in the task pipeline
- Graceful degradation when no embeddings exist
