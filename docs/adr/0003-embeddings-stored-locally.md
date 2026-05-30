# ADR-0003: All embeddings are computed and stored locally

**Status:** accepted

Every embedding (scene, frame, O-moment, performer, tag) is generated on the user's
own machine and persisted to a local SQLite store in the plugin's assets directory.
No embedding, frame, or library data is sent to a remote embedding or vector service.

## Why

- **Privacy.** The library is sensitive personal content and must never leave the
  user's machine for a third-party service.
- **Offline and zero running cost.** Local models (OpenCLIP / SigLIP) mean no API
  keys, no per-call cost, and full functionality without a network.

## Consequences

- Embedding generation is bound by **local hardware** (GPU/CPU, RAM) — hence the
  performance budgets documented in `CLAUDE.md`.
- There is **no cross-device sync**; each install builds its own store, so shared or
  multi-device profiles are not supported out of the box.
- This ADR governs **embeddings and the visual/library data they encode**. It does
  *not* forbid remote **LLM** providers (Ollama / OpenRouter / Anthropic) for
  vision/chat — sending a text prompt to an LLM the user has configured is a separate,
  opt-in choice.
