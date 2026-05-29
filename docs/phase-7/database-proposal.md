# Phase 7: Database Architecture Proposal

**Generated:** 2026-02-15

⚠️ **THIS IS A PROPOSAL DOCUMENT ONLY** - No database modifications will be made.

## Executive Summary

This document analyzes the current database architecture and proposes improvements for performance, reliability, and maintainability. All recommendations are subject to user approval before implementation.

## Current Database State

### Active Database: `stash_copilot.sqlite`

| Metric | Value |
|--------|-------|
| Size | 19 GB |
| Schema Version | 10 |
| Tables | 11 |
| Total Rows | ~8.2 million |
| Last Updated | 2026-02-13 |

### Stale Databases

| Database | Size | Last Updated | Recommendation |
|----------|------|--------------|----------------|
| `stash_copilot_ViT-bigG-14-dense.sqlite` | 204 MB | 2026-01-18 (27 days) | Archive or delete |
| `stash_copilot_ViT-H-14-sparse.sqlite` | 432 MB | 2026-01-04 (42 days) | Archive or delete |

### Empty Databases (Safe to Remove)

| Location | Size |
|----------|------|
| `~/.stash/plugins/stash-copilot/embeddings.db` | 0 bytes |
| `~/.stash/plugins/stash-copilot/assets/embeddings.db` | 0 bytes |
| `~/.stash/plugins/stash-copilot/stash_ai/embeddings/embeddings.db` | 0 bytes |

## Best Practices Analysis

### 1. Database Design Principles for Embedding Storage

**Applicable to this project:**

| Principle | Current State | Recommendation |
|-----------|---------------|----------------|
| Normalization | Good | Keep multi-table design |
| Indexing | Good | Key indexes exist |
| BLOB Storage | Good | Packed float32 is efficient |
| Transactions | Partial | Add explicit transaction blocks |
| Schema Versioning | Good | Migration system in place |

### 2. SQLite Best Practices

| Practice | Current | Recommendation |
|----------|---------|----------------|
| WAL Mode | Unknown | Enable for concurrent reads |
| PRAGMA optimize | Unknown | Run after large writes |
| Foreign Keys | Not used | Consider adding for integrity |
| VACUUM | Unknown | Schedule periodic VACUUM |

### 3. Embedding-Specific Considerations

| Consideration | Current | Recommendation |
|---------------|---------|----------------|
| Batch Inserts | Yes (32) | Keep, good GPU utilization |
| Index Type | B-tree | Consider ANN index for large scale |
| Compression | None | Consider zstd for cold storage |
| Sharding | None | Not needed at current scale |

## Performance Analysis

### Current Table Sizes

```
frame_embeddings:      4,073,927 rows (~15 GB, 74% of DB)
frame_tag_coverage:    4,073,278 rows (~2 GB, 10% of DB)
scene_embeddings:         12,812 rows (~500 MB, 3% of DB)
scene_umap_coords:        12,756 rows (~50 MB)
frame_embedding_metadata: 12,362 rows (~20 MB)
tag_embeddings:              507 rows (~5 MB)
performer_embeddings:        313 rows (~5 MB)
preference_comparisons:    1,296 rows (<1 MB)
```

### Query Performance

| Query Type | Current | Notes |
|------------|---------|-------|
| Scene embedding lookup | <10ms | Indexed, fast |
| Frame embeddings by scene | ~50ms | Indexed |
| Similarity search (all scenes) | ~100ms | Full scan, acceptable |
| Frame-level search (all frames) | ~2s | FAISS index recommended |

### Bottlenecks Identified

1. **Frame embedding insertion:** Bulk writes during embedding generation
2. **Frame-level search:** 4M+ rows for similarity scan
3. **Tag coverage computation:** 4M+ row scan

## Proposed Improvements

### 1. Enable WAL Mode

**Impact:** Improved concurrent read performance
**Risk:** Low
**Effort:** Low

```python
def enable_wal_mode(conn: sqlite3.Connection) -> None:
    """Enable Write-Ahead Logging for better concurrency."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster with WAL
```

**Pros:**
- Concurrent reads during writes
- Better performance for read-heavy workloads
- Standard SQLite feature

**Cons:**
- Two additional files (-wal, -shm)
- Slightly more complex backup

---

### 2. Add Foreign Key Constraints (Optional)

**Impact:** Data integrity
**Risk:** Medium (requires migration)
**Effort:** Medium

```sql
-- Example for frame_embeddings
ALTER TABLE frame_embeddings
ADD CONSTRAINT fk_scene_embeddings
FOREIGN KEY (scene_id, model_key)
REFERENCES scene_embeddings(scene_id, model_key)
ON DELETE CASCADE;
```

**Pros:**
- Automatic cascade deletes
- Prevents orphaned records
- Self-documenting relationships

**Cons:**
- SQLite foreign keys require recreating tables
- Performance overhead on inserts/deletes

**Recommendation:** Document relationships in code, don't add FK constraints (SQLite limitation makes it complex).

---

### 3. Periodic VACUUM Schedule

**Impact:** Disk space recovery, performance
**Risk:** Low
**Effort:** Low

```python
def vacuum_database(db_path: str) -> None:
    """Reclaim space and defragment database."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("VACUUM")
        conn.execute("PRAGMA optimize")
```

**Schedule:** Run after large delete operations or monthly.

---

### 4. FAISS Index for Frame-Level Search

**Current:** Full table scan for frame similarity
**Proposed:** Maintain FAISS index alongside SQLite

```python
# stash_ai/embeddings/frame_search.py (already partially implemented)
class FrameSearchIndex:
    """FAISS index for fast frame-level similarity search."""

    def __init__(self, db_path: str, model_key: str):
        self.index: Optional[faiss.Index] = None
        self.scene_frame_map: List[Tuple[int, int]] = []  # (scene_id, frame_index)

    def build_index(self, storage: EmbeddingStorage) -> None:
        """Build FAISS index from frame embeddings."""
        embeddings = storage.get_all_frame_embeddings(model_key)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (cosine on normalized)
        self.index.add(embeddings)

    def search(self, query: np.ndarray, k: int = 10) -> List[FrameMatch]:
        """Find k most similar frames."""
        distances, indices = self.index.search(query, k)
        return [self._to_match(idx, dist) for idx, dist in zip(indices[0], distances[0])]
```

**Pros:**
- Orders of magnitude faster frame search
- GPU acceleration available
- Standard vector search solution

**Cons:**
- Separate index file to maintain
- Rebuild required when embeddings change
- Additional memory for index

---

### 5. Archive Stale Databases

**Current:** 3 databases, 2 stale
**Proposed:** Archive or remove stale databases

**Option A: Archive**
```bash
mkdir -p assets/archive/
mv assets/stash_copilot_ViT-bigG-14-dense.sqlite assets/archive/
mv assets/stash_copilot_ViT-H-14-sparse.sqlite assets/archive/
gzip assets/archive/*.sqlite
```

**Option B: Delete (after user confirmation)**
```bash
rm assets/stash_copilot_ViT-bigG-14-dense.sqlite
rm assets/stash_copilot_ViT-H-14-sparse.sqlite
```

**Recommendation:** Archive with compression. Recovery possible if needed.

---

### 6. Remove Empty Database Files

**Current:** 3 empty 0-byte database files
**Proposed:** Delete unused files

```bash
rm embeddings.db
rm assets/embeddings.db
rm stash_ai/embeddings/embeddings.db
```

**Risk:** None (files are empty)

---

### 7. Schema Migration Improvements

**Current:** Linear migration in storage.py
**Proposed:** Separate migration files

```
stash_ai/migrations/
├── __init__.py
├── v01_initial.py
├── v02_frame_embeddings.py
├── v03_performer_embeddings.py
├── v04_taste_clusters.py
├── v05_umap_coords.py
├── v06_preferences.py
├── v07_tag_embeddings.py
├── v08_frame_metadata.py
├── v09_preference_sessions.py
└── v10_tag_coverage.py
```

**Pros:**
- Each migration is isolated
- Easier to understand history
- Can add rollback logic per migration

**Cons:**
- More files to manage
- Requires migration framework

---

### 8. Backup Strategy

**Current:** No backup system
**Proposed:** Automated backup before major operations

```python
def backup_database(db_path: str) -> str:
    """Create timestamped backup before major operations."""
    import shutil
    from datetime import datetime

    backup_dir = Path(db_path).parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"stash_copilot_{timestamp}.sqlite"

    shutil.copy2(db_path, backup_path)
    return str(backup_path)
```

**Schedule:**
- Before schema migrations
- Before "Embed All Scenes" (full regeneration)
- Before "Cleanup Orphaned" operations

---

## Alternative Technologies Considered

### 1. PostgreSQL with pgvector

**Pros:**
- Native vector similarity search
- Better concurrency
- Full ACID compliance

**Cons:**
- Requires separate server
- Overkill for single-user plugin
- More complex deployment

**Recommendation:** Not recommended. SQLite + FAISS is sufficient.

### 2. DuckDB

**Pros:**
- Fast analytical queries
- Better for bulk operations
- Columnar storage

**Cons:**
- Less mature
- Different SQL dialect
- Migration effort

**Recommendation:** Not recommended. No significant benefit for this use case.

### 3. Milvus / Weaviate (Vector Databases)

**Pros:**
- Purpose-built for embeddings
- Automatic indexing
- Scalable

**Cons:**
- Requires separate service
- Massive overkill for 12K scenes
- Complex deployment

**Recommendation:** Not recommended. SQLite + FAISS is sufficient at current scale.

### 4. Redis with Vector Search

**Pros:**
- Fast in-memory search
- Simple setup

**Cons:**
- Memory intensive for 19GB data
- Persistence complexity

**Recommendation:** Not recommended.

## Recommendation Summary

| Improvement | Priority | Effort | Risk | Status |
|-------------|----------|--------|------|--------|
| Enable WAL mode | High | Low | Low | Recommended |
| FAISS index for frames | High | Medium | Low | Recommended |
| Archive stale databases | Medium | Low | Low | Recommended |
| Remove empty files | Medium | Low | None | Recommended |
| Backup system | Medium | Low | Low | Recommended |
| Periodic VACUUM | Low | Low | Low | Recommended |
| Migration file split | Low | Medium | Low | Optional |
| Foreign key constraints | Low | High | Medium | Not recommended |

## Data Integrity Considerations

### Redundancy (Avoiding)

**Current State:** Good
- Single primary database
- model_key scoping prevents duplicate embeddings

**Improvement:** Archive stale databases to eliminate redundant data.

### Reliability

**Current State:** Acceptable
- SQLite is robust
- No foreign keys but application logic handles relationships

**Improvement:**
- Add WAL mode for crash recovery
- Implement backup system

### Recoverability

**Current State:** Limited
- No automated backups
- Stash SQLite separate from plugin SQLite

**Improvement:**
- Automated backups before major operations
- Document recovery procedures

### Robustness

**Current State:** Good
- Schema migrations handle upgrades
- Error handling in storage operations

**Improvement:**
- Add integrity checks (PRAGMA integrity_check)
- Log database statistics periodically

## Implementation Plan (If Approved)

### Immediate (No Code Changes)

1. Archive stale databases
2. Delete empty database files
3. Document backup procedures

### Short-Term (Minor Code Changes)

1. Enable WAL mode in storage.py
2. Add VACUUM to cleanup task
3. Implement backup function

### Medium-Term (New Features)

1. Build FAISS index infrastructure
2. Add integrity check command
3. Implement migration file separation

## Approval Required

This document is for review only. No database modifications will be made without explicit approval.

**Action items awaiting approval:**

1. [ ] Archive stale databases (ViT-bigG-14-dense, ViT-H-14-sparse)
2. [ ] Delete empty database files (3 files, 0 bytes each)
3. [ ] Enable WAL mode
4. [ ] Implement backup system
5. [ ] Build FAISS index for frame search
