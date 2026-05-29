# Phase 3: Architecture Proposal

**Generated:** 2026-02-15

## Executive Summary

This document proposes architectural improvements for Stash Copilot based on analysis of the current codebase. The recommendations focus on maintainability, testability, and reducing code duplication while preserving the existing functionality.

## Current Architecture Analysis

### Strengths

1. **Clean Provider Pattern:** LLM and embedding providers use decorator-based registration with factory pattern - easily extensible
2. **Comprehensive Type Coverage:** TypedDict and dataclasses throughout provide good type safety
3. **Modular Tasks:** Each task is an independent class with clear input/output contracts
4. **Multi-Model Support:** `model_key` scoping allows multiple embedding models to coexist

### Weaknesses

1. **Large Monolithic Files:**
   - `stash-copilot.js`: 15,411 lines
   - `stash-copilot.css`: 11,156 lines
   - `storage.py`: Complex with 10 schema migrations

2. **Duplicated Code:**
   - 8 `formatDuration()` implementations in JS
   - 3 `escapeHtml()` implementations
   - Similar error handling patterns in LLM providers

3. **Mixed Concerns:**
   - Frontend state management scattered across multiple objects
   - Database operations mixed into task handlers

4. **Implicit Dependencies:**
   - Features depend on database tables being present
   - No explicit dependency injection for testability

## Proposed Architecture Changes

### 1. Frontend Module Organization

**Current:** Single 15K line JavaScript file
**Proposed:** Feature-based module structure

```
stash-copilot/
├── frontend/
│   ├── core/
│   │   ├── api.js          # GraphQL + plugin task calls
│   │   ├── state.js        # Unified state management
│   │   └── utils.js        # formatDuration, escapeHtml, etc.
│   ├── components/
│   │   ├── cards.js        # Unified scene card system
│   │   ├── modals.js       # Modal base functionality
│   │   ├── tabs.js         # Tab navigation
│   │   └── tooltips.js     # Tooltip system
│   ├── features/
│   │   ├── navbar.js       # Navbar + dropdown
│   │   ├── insights/       # Main insights modal
│   │   ├── vision/         # Scene vision analysis
│   │   ├── similar/        # Similar scenes
│   │   ├── recommendations/# All recommendation modes
│   │   ├── search/         # Semantic search page
│   │   ├── preferences/    # Preference training
│   │   └── tastemap/       # 3D visualization
│   └── index.js            # Entry point, initialization
├── stash-copilot.js        # Build output (bundled)
└── stash-copilot.css       # (or also modular)
```

**Pros:**
- Easier to navigate and understand
- Feature isolation improves testability
- Shared utilities prevent duplication
- Better code organization for collaboration

**Cons:**
- Requires build system (esbuild, webpack, or similar)
- More complex development setup
- Initial migration effort

**Recommendation:** Implement with esbuild (fast, minimal config)

---

### 2. Backend Service Layer

**Current:** Task handlers directly call storage, LLM providers
**Proposed:** Service abstraction layer

```python
stash_ai/
├── services/
│   ├── embedding_service.py    # Coordinates embedding operations
│   ├── recommendation_service.py # Recommendation business logic
│   ├── vision_service.py       # Vision analysis coordination
│   └── preference_service.py   # Preference learning orchestration
├── repositories/
│   ├── scene_repository.py     # Scene data access
│   ├── embedding_repository.py # Embedding data access
│   └── preference_repository.py # Preference data access
└── tasks/                      # Thin handlers that use services
```

**Pattern:** Repository + Service Layer

```python
# Current (task directly accesses storage)
class EmbedScenesTask:
    def run(self):
        storage = EmbeddingStorage()
        scenes = self.stash.find_scenes()
        # ... business logic mixed with data access

# Proposed (service layer)
class EmbeddingService:
    def __init__(self, scene_repo: SceneRepository,
                 embedding_repo: EmbeddingRepository,
                 provider_factory: EmbeddingProviderFactory):
        self.scene_repo = scene_repo
        self.embedding_repo = embedding_repo
        self.provider_factory = provider_factory

    def embed_scenes(self, config: EmbeddingConfig) -> EmbedResult:
        scenes = self.scene_repo.get_unembedded(config.model_key)
        provider = self.provider_factory.get(config)
        # ... business logic isolated
```

**Pros:**
- Clear separation of concerns
- Easier to test (mock repositories)
- Reusable business logic across tasks
- Single responsibility per class

**Cons:**
- More classes to maintain
- Indirection adds complexity
- Requires refactoring existing tasks

**Recommendation:** Implement incrementally, starting with most-used services

---

### 3. Dependency Injection Container

**Current:** Direct instantiation in task handlers
**Proposed:** Simple DI container for testability

```python
# stash_ai/container.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Container:
    """Simple dependency container for testability."""
    stash: StashInterface
    llm_config: LLMConfig
    embedding_config: EmbeddingConfig

    # Lazy-loaded services
    _embedding_service: Optional[EmbeddingService] = None
    _recommendation_service: Optional[RecommendationService] = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                scene_repo=SceneRepository(self.stash),
                embedding_repo=EmbeddingRepository(),
                provider_factory=EmbeddingProviderFactory(self.embedding_config)
            )
        return self._embedding_service
```

**Pros:**
- Testable (inject mocks)
- Explicit dependencies
- Lazy loading preserves startup performance
- No external library required

**Cons:**
- Manual wiring
- Need to update container for new services

**Recommendation:** Use simple dataclass-based container, no framework

---

### 4. CSS Module Organization

**Current:** Single 11K line CSS file
**Proposed:** Feature-based CSS organization

```
styles/
├── base/
│   ├── variables.css       # CSS custom properties
│   ├── typography.css      # Font scales
│   └── animations.css      # Shared keyframes
├── components/
│   ├── cards.css          # Scene card styles
│   ├── modals.css         # Modal styles
│   ├── buttons.css        # Button variants
│   └── tooltips.css       # Tooltip styles
├── features/
│   ├── navbar.css         # Navbar styles
│   ├── insights.css       # Insights modal tabs
│   ├── vision.css         # Vision analysis
│   ├── similar.css        # Similar scenes
│   ├── recommendations.css # Recommendations
│   ├── search.css         # Search page
│   └── preferences.css    # Preference training
└── index.css              # Imports all modules
```

**Build:** Concatenate with CSS bundler or native @import

**Pros:**
- Feature isolation
- Easier to find styles
- Shared variables reduce duplication

**Cons:**
- Multiple files to manage
- Need build step for production

---

### 5. Database Layer Improvements

**Current:** Single `storage.py` with all tables and migrations
**Proposed:** Per-domain repository pattern

```python
stash_ai/repositories/
├── base.py                 # BaseRepository with connection management
├── scene_embeddings.py     # SceneEmbeddingRepository
├── frame_embeddings.py     # FrameEmbeddingRepository
├── preferences.py          # PreferenceRepository
├── taste_clusters.py       # TasteClusterRepository
└── migrations/             # Schema migrations by domain
    ├── v1_initial.py
    ├── v2_frame_embeddings.py
    └── v10_tag_coverage.py
```

**Repository Base:**
```python
class BaseRepository:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or self._default_db_path()
        self._conn: Optional[sqlite3.Connection] = None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        # Thread-safe connection management
        ...
```

**Pros:**
- Single responsibility per repository
- Easier to test with mocks
- Domain-specific queries
- Cleaner migrations

**Cons:**
- More files
- Cross-domain queries need coordination

---

### 6. Error Handling Standardization

**Current:** `except Exception:` in 15+ places
**Proposed:** Typed exception hierarchy

```python
# stash_ai/exceptions.py
class StashCopilotError(Exception):
    """Base exception for all plugin errors."""
    pass

class LLMError(StashCopilotError):
    """LLM provider errors."""
    pass

class EmbeddingError(StashCopilotError):
    """Embedding generation errors."""
    pass

class StorageError(StashCopilotError):
    """Database operation errors."""
    pass

class ConfigurationError(StashCopilotError):
    """Invalid configuration."""
    pass
```

**Usage:**
```python
# Instead of:
try:
    ...
except Exception as e:
    self.log(f"Error: {e}", "error")

# Use:
try:
    ...
except json.JSONDecodeError as e:
    raise LLMError(f"Invalid JSON response: {e}") from e
except requests.RequestException as e:
    raise LLMError(f"Network error: {e}") from e
```

**Pros:**
- Specific error handling
- Better debugging
- Type-safe exception handling
- Preserves stack traces

---

## Migration Strategy

### Phase 1: Backend Refactoring (Low Risk)

1. Create exception hierarchy
2. Extract utilities from tasks (no behavior change)
3. Add repository layer (wrap existing storage)
4. Create services on top of repositories

### Phase 2: Frontend Refactoring (Medium Risk)

1. Set up esbuild bundler
2. Extract shared utilities (`utils.js`)
3. Extract components (`cards.js`, `modals.js`)
4. Migrate features one at a time
5. Keep original file as output until migration complete

### Phase 3: Database Refactoring (Low Risk)

1. Extract repositories from storage.py
2. Keep storage.py as facade for compatibility
3. Migrate tasks to use repositories directly
4. Eventually deprecate storage.py facade

## Alternative Approaches Considered

### 1. Micro-Frontend Architecture

**Rejected:** Overkill for single-developer plugin. Web components would add complexity without proportional benefit.

### 2. Full ORM (SQLAlchemy)

**Rejected:** Heavy dependency for simple CRUD. Current raw SQLite is sufficient and faster.

### 3. Redux/MobX for State

**Rejected:** Plugin doesn't need reactive state management at this scale. Simple state objects are sufficient.

### 4. Complete Rewrite

**Rejected:** Too risky. Incremental refactoring preserves working functionality.

## Recommended Priority

| Change | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Exception hierarchy | Medium | Low | 1 |
| Utility consolidation | Low | Low | 2 |
| Service layer | High | Medium | 3 |
| Frontend modules | High | High | 4 |
| Repository pattern | Medium | Medium | 5 |
| CSS modules | Low | Low | 6 |

## Success Criteria

1. **No regression:** All 247 tests continue to pass
2. **Reduced duplication:** Utility functions consolidated
3. **Improved testability:** Service/repository layer enables mocking
4. **Maintainability:** Clear feature boundaries
5. **Documentation:** Updated architecture diagrams

## Decision Required

Please review and approve or suggest modifications to:

1. Frontend modularization approach (esbuild vs. stay monolithic)
2. Service layer adoption (full vs. incremental)
3. Repository pattern (separate repositories vs. enhanced storage.py)
4. Build tooling preferences (if any)
