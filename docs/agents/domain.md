# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is **single-context** (one Python package + JS frontend, not a monorepo).

## Before exploring, read these

- **`CONTEXT.md`** at the repo root (the project's domain glossary), and
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved. (As of this setup, neither exists yet — that's expected.)

> Note: `CLAUDE.md` at the repo root already documents the architecture, LLM-provider
> pattern, recommendation scoring, performance budgets, and the scene-page UI. Read it
> alongside `CONTEXT.md` — it carries much of the domain context until `CONTEXT.md` is
> grown.

## File structure

Single-context layout:

```
/
├── CONTEXT.md          ← created lazily by /grill-with-docs
├── docs/adr/           ← created lazily, one ADR per decision
└── stash_ai/           ← the package
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (…) — but worth reopening because…_
