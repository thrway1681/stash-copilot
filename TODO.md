# TODO

Future improvements to be implemented.

## Bugfix

- **Improve timestamp accuracy**: VLM timestamp extraction needs improvement.

- **Overly Broad Exception Handling**: 15 instances of `except Exception:` across LLM providers mask real errors. Replace with specific exception types (`json.JSONDecodeError`, `ValueError`, `requests.RequestException`).

## Backend/AI Features

- **Try Qwen3-VL-Embedding**: Evaluate `qwen3-vl-embedding` as an alternative to OpenCLIP for image embeddings. May provide better visual understanding for scene similarity and recommendations.

- **Performer Embeddings & Similar Performer Discovery**: Create embeddings for performers to enable visual similarity matching and external database integration.

  **Embedding Generation**:
  - Extract performer face/body embeddings from scene frames where they appear
  - Average embeddings across multiple scenes for robust performer representation
  - Store in `performer_embeddings` table with performer_id as key
  - Use existing OpenCLIP infrastructure or evaluate face-specific models (e.g., ArcFace, FaceNet)

  **Local Similarity Matching**:
  - Find scenes with visually similar performers (not just by name)
  - "Find performers who look like X" feature
  - Useful for finding uncredited/misidentified performers

  **StashDB/FansDB Integration**:
  - Query StashDB API (`https://stashdb.org/graphql`) for performer suggestions
  - Query FansDB API (`https://fansdb.cc/graphql`) as alternative source
  - Use local performer embedding to find visually similar performers in external databases
  - Workflow: Local performer → generate embedding → compare against StashDB performer images → suggest matches
  - Enable "Who is this performer?" feature for unidentified local performers
  - Support performer auto-tagging by matching against known performer database

  **API Integration Notes**:
  - StashDB requires API key for authenticated requests
  - FansDB may have different rate limits and authentication requirements
  - Cache external performer data locally to reduce API calls
  - Consider batch processing for bulk performer identification

- **Negative Embeddings/Exclusions**: "Find scenes like X but NOT like Y" - useful for refining recommendations.

- **Smart Playlists**: Auto-generate playlists based on embedding similarity, mood, or viewing patterns.

- **Duplicate Detection**: Use embeddings to find potential duplicate scenes (different files, same content).

## UI/UX Enhancements

- **Keyboard Navigation for Similar Modal**: Add keyboard shortcuts (Escape to close, arrow keys to navigate cards, Enter to open scene).

- **Dark/Light Theme Support**: Currently hardcoded dark theme; could respect Stash's theme setting.

- **Bulk Operations in Similar Modal**: Add checkboxes to select multiple similar scenes for bulk tagging, adding to playlists, etc.

- **Scene Comparison View**: Side-by-side view comparing the source scene with a selected similar scene (tags, performers, stats).

- **Similarity Score Breakdown**: Show what contributed to the similarity score (visual similarity %, metadata similarity %, tag overlap %).

## Integration Features

- **Export/Import Embeddings**: Backup and restore embedding database, share between instances.

- **Webhook Notifications**: Send notifications when analysis completes (Discord, Telegram, etc.).

## Performance

- **Embedding Cache Warming**: Background task to pre-compute embeddings for new scenes automatically.

- **Incremental Embedding Updates**: Only re-embed scenes that have changed since last run.

- **Embedding Space Visualization**: Create a visual representation of the embedding space to explore scene similarity clusters and relationships.

## Refactoring / Code Health

- **Add Recs Tab Filters**: Performer/tag exclusion filters are supported in backend (`RecommendationConfig`) but not exposed in Recs sidebar UI. Match Similar tab feature parity.
