# Stash Copilot — Domain Glossary

The shared language of Stash Copilot: a StashApp plugin that reads the local Stash
library (scenes, tags, play history, markers) and uses **local** embeddings and LLMs
to surface recommendations, visual search, taste insights, and tagging help.

This file is a glossary only — no implementation detail. When a concept has competing
synonyms in the code, the canonical term is defined here and the rejected ones are
listed under _Avoid_. Architectural decisions live in `docs/adr/`.

## Library primitives

**Scene**:
A single video in the user's Stash library, with its metadata (title, date, rating, performers, tags). The atomic unit everything else hangs off.

**Tag**:
A categorical label attached to scenes (many-to-many), sourced from Stash.

**Marker**:
A timestamped annotation on a Scene, carrying a tag and a position in the video.
_Avoid_: bookmark, annotation

**O-marker**:
A Marker recording an orgasm at a specific timestamp. The strongest taste signal.

**O-count**:
The number of O-markers on a Scene.
_Avoid_: o_counter, orgasm count

**Funscript**:
An interactive haptics script (timestamped motion) paired with a Scene.

## Embeddings & visual search

**Embedding**:
A fixed-length vector capturing the visual and/or descriptive character of a Scene or Frame, used to measure similarity. Always computed and stored locally.

**Frame**:
A still image sampled from a Scene's video (roughly one per second) used for visual embedding and fine-grained search.
_Avoid_: thumbnail, still

**Composite Embedding**:
A Scene's single blended embedding — its visual character and its metadata character combined by a fixed weight. The default vector for scene-to-scene similarity.

**O-moment**:
A short window of video centred on an O-marker, embedded to capture what a user's "peak" moments look like.
_Avoid_: peak moment

**Embedding Model**:
The specific model that produces embeddings (e.g. OpenCLIP ViT-H-14, SigLIP). Several can coexist; an embedding is always scoped to the model that made it.

**Sprite sheet**:
A grid of small frame thumbnails for a Scene, assembled so a vision model can read a whole scene at once.

## Engagement & taste

**Engagement**:
What a user's behaviour on a Scene reveals about their taste — O-count, replays (views beyond the first), and rating. (Stash also records play time, but it is deliberately not scored — see Engagement Score.)
_Avoid_: play stats, interaction

**Engagement Score**:
The single number computed from a Scene's Engagement, ranking how much the user likes it. O-count weighs highest, then replays, then rating; **play time is excluded** because raw watch-hours bias toward long scenes. Computed in exactly one place — see ADR-0004.

**Engagement Profile**:
A single embedding — the engagement-weighted average of a user's most-engaged scenes — representing their taste as one point in embedding space. The basis for Discover and Rewatch.
_Avoid_: user preference profile, preference profile

**Taste Map**:
A 2-/3-D projection of a user's engaged scenes, grouped into labelled clusters, for visual exploration of their taste.

**Taste Cluster**:
One labelled group within the Taste Map — scenes that sit together in embedding space, with an auto-generated label and a share of total engagement.
_Avoid_: taste profile

**Recommendation**:
A ranked list of scenes for the user. Modes: **Discover** (unwatched, similar to the Engagement Profile), **Rewatch** (watched, ranked by engagement + similarity), **O-moments** (scenes whose peak moments resemble the user's), **Performer-preference** (scenes resembling favourite performers).

**Score** (disambiguation):
Never use "score" unqualified. **Engagement Score** = how much the user likes a scene (behavioural). **Similarity Score** = cosine closeness between two embeddings. **Combined Score** = a mode's blend of the two.
_Avoid_: bare "score"

## Providers

**Provider**:
A pluggable backend for an external capability — an **LLM Provider** (Ollama, OpenRouter, Anthropic, …) or an **Embedding Provider** (OpenCLIP, …). Reserve "provider" for the backend service; use **Embedding Model** for the specific model it serves.

## Out of scope

**Preference / Swipe / A-B comparison** — *out of scope (see ADR-0001)*:
Explicitly training taste by swiping or comparing scene pairs. Stash Copilot infers taste implicitly from Engagement instead; the `preferences/` subsystem is slated for removal. Do not build on it.
_Avoid_: preference model, preference profile, swipe, pairwise comparison (for taste). Note: the **performer-preference** recommendation mode is unrelated and is fine.
