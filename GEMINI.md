# GEMINI.md

This file provides foundational mandates and expert guidance for Gemini CLI when working in this repository. These instructions take precedence over general defaults.

## Architecture

**Source of Truth:** `docs/diagrams/architecture-post-cleanup.mmd`

> **Note:** Do not embed diagram code here. Always read the source file above to understand the current system architecture. When making structural changes, update the `.mmd` file, not this document.

## Core Mandates & Engineering Standards

### Python Environment & Tooling
- **UV First**: This project uses [uv](https://astral.sh/uv) for dependency management and execution.
  - Always use `uv sync` to install/update dependencies.
  - Use `uv run python ...` or `uv run pytest` for execution.
  - Avoid `pip` unless `uv` is unavailable or fails for a specific reason.
- **Strict Typing**: All Python code must be strictly typed.
  - Use `TypedDict` for complex dictionary structures.
  - Use `dataclasses` with type hints for internal state.
  - Ensure recursive typing for nested structures (e.g., `Dict[str, List[TypedDict]]`).
- **Python Version**: Target Python 3.10+ as specified in `pyproject.toml`.

### UI Interaction Feedback Principle
Every user interaction **MUST** produce instant, informative, and live-updating feedback.
1. **Instant**: Respond within ~16ms. Show a loading state (spinner, skeleton) immediately for async operations.
2. **Informative**: Tell the user *what* is happening (e.g., "Analyzing scene frames..." instead of just "Loading...").
3. **Live-updating**: Update the UI continuously for multi-step tasks (e.g., progress bars, status step counts).

### Performance & Resource Budgets
Maintain these performance targets. If a task exceeds its budget by >50%, it must be optimized before merge.
- **Embed Scenes**: < 5s per frame batch, < 3GB Peak RSS.
- **Scene Vision**: < 30s (excluding LLM wait), < 2GB Peak RSS.
- **Stats Summary**: < 10s, < 500MB Peak RSS.
- **Recommendations**: < 15s for top-N, < 2GB Peak RSS.
- **Startup**: < 3s (import + init), < 200MB Peak RSS.

---

## Architecture & Component Guidelines

### Entry Points
- **Backend**: `stash-copilot.py` (Plugin entry point). Uses `StashPlugin` base class and parses JSON from stdin.
- **Frontend**: `stash-copilot.js` (Sidebar tabs: Analyze, Similar, Recs, Tags).
- **CSS Prefixing**: All sidebar-specific CSS classes must use the `stash-copilot-sidebar-*` prefix.

### Stash Integration
- **StashInterface**: Use the `StashInterface` instance from the `stashapi` library. It is initialized in `StashPlugin.__init__` and passed to tasks. Do NOT create new connections.
- **Data Access**: Use the SQLite database via `StashInterface` for performance-critical queries (like stats) instead of the GraphQL endpoint when possible.
- **Assets Directory**: `assets/embedded_frames/` (1 FPS JPEGs). Only `standalone_embed.py` and `stash_ai/tasks/embed_scenes.py` should write here. All other tasks are read-only.

### LLM & VLM Patterns
- **Provider Registry**: Providers must use the `@register_provider` decorator pattern in `stash_ai/llm/providers/`.
- **Vision Support**: keyword-matched via `VISION_KEYWORDS` in each provider. Capabilities defined in `stash_ai/llm/model_caps.py`.

---

## Testing & Validation Protocol

### UI Testing with Playwright
- **Absolute Paths**: Always use absolute paths for screenshots (e.g., `~/.stash/plugins/stash-copilot/tests/screenshots/...`). Playwright MCP prepends `.playwright-mcp/` to relative paths, which breaks directory structure.
- **Log Verification**: After every UI interaction, check `~/.stash/stash.log` for errors (`error|exception|traceback|failed`).
- **Wait for Completion**: For long-running tasks (embedding, AI analysis), `tail -f` the log and wait for the "completion" entry before declaring success.

### Logging & Debugging
- **Log Format**: Plugin logs are JSON formatted: `{"output": "message", "level": "info"}`.
- **Filtering**: Filter logs using `grep -i "copilot\|stash_ai"`.
- **Debug Mode**: Enable verbose provider logging with `STASH_COPILOT_DEBUG=1`.

---

## Development Workflow
- **Branching**: `dev` is the default integration branch. Always branch from `dev` and merge back to `dev`.
- **Architecture Updates**: Any structural changes (modules, data flow) require updating the Mermaid diagram in `CLAUDE.md` and `docs/diagrams/architecture-post-cleanup.mmd`.
