# Gemini Batch API Integration — Design

**Date:** 2026-02-25
**Status:** Approved

## Overview

Add Gemini Batch API support as an alternative to real-time captioning. Batch mode offers **50% cost reduction** ($0.00064/frame vs $0.00127/frame) with up to 24-hour turnaround. Integrates into the existing caption dashboard with auto-polling and auto-collection.

## Cost Comparison

| Mode | Cost/frame | 208K frames | Time |
|---|---|---|---|
| Real-time | $0.00127 | ~$265 | 4-10 days (rate-limited) |
| **Batch** | ~$0.00064 | **~$133** | ≤24h per chunk |

Batch mode also bypasses RPD/RPM rate limits since Google processes requests on their infrastructure.

## Data Flow

```
User clicks "Submit Batch" in dashboard
  │
  ├─ 1. Frame Selection (reuses SmartFrameSelector)
  │    Select ≤20 frames/scene for pending scenes
  │    Skip frames with existing .txt files
  │    Group into chunks of ~25K frames (<2GB JSONL each)
  │
  ├─ 2. JSONL Build
  │    Each line: {"key": "s{sid}_f{idx}", "request": {generateContent with base64 image}}
  │    Save to assets/lora_dataset/batch_jobs/chunk-{N}.jsonl
  │
  ├─ 3. Upload + Submit
  │    Upload each .jsonl via Gemini File API
  │    Submit batch job per chunk via batchGenerateContent endpoint
  │    Save job metadata to batch_state.json
  │
  ├─ 4. Poll (dashboard auto-polls every 60s)
  │    GET job status from Gemini API
  │    Update batch_state.json
  │
  └─ 5. Auto-Collect (triggered when JOB_STATE_SUCCEEDED)
       Download result JSONL from Gemini
       Parse responses → write .txt caption files
       Update caption_progress.json + mark completed scenes
```

## Architecture

### New Module: `tools/dataset/batch_api.py`

Core functions:

```python
def prepare_batch_chunks(
    index: EmbeddingIndex,
    completed_scenes: set[int],
    images_dir: Path,
    frames_dir: Path,
    max_frames: int = 20,
    chunk_size: int = 25_000,
) -> list[Path]:
    """Select frames via SmartFrameSelector, build JSONL chunks.

    Each line in the JSONL:
    {"key": "s{scene_id}_f{frame_idx}", "request": {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": "<base64>"}},
            {"text": "<caption prompt>"}
        ]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 4096},
        "safetySettings": [...]
    }}

    Returns list of saved .jsonl file paths.
    """

def submit_batch_job(
    jsonl_path: Path,
    model: str,
    api_key: str,
) -> BatchJob:
    """Upload JSONL via Gemini File API, then submit batch job.

    Steps:
    1. Initiate resumable upload
    2. Upload file bytes
    3. POST to models/{model}:batchGenerateContent with file reference
    4. Return BatchJob with name, state, metadata
    """

def poll_batch_jobs(
    jobs: list[BatchJob],
    api_key: str,
) -> list[BatchJob]:
    """Check status of all active jobs via GET batches/{name}.

    Returns updated job list with current states.
    """

def collect_batch_results(
    job: BatchJob,
    images_dir: Path,
    api_key: str,
) -> CollectResult:
    """Download result file, parse responses, write .txt files.

    For each response line:
    1. Parse key to get scene_id and frame name
    2. Extract caption text from GenerateContentResponse
    3. Write to images/{key}.txt
    4. Track successes and errors

    Returns CollectResult with counts.
    """
```

### Type Definitions

```python
@dataclass
class BatchJob:
    name: str               # e.g. "batches/abc123"
    display_name: str       # e.g. "chunk-001"
    state: str              # JOB_STATE_PENDING | RUNNING | SUCCEEDED | FAILED | CANCELLED | EXPIRED
    submitted_at: str       # ISO timestamp
    frame_count: int        # frames in this chunk
    scene_ids: list[int]    # scenes covered by this chunk
    file_name: str | None   # uploaded file reference
    result_file: str | None # result file reference (set when succeeded)
    collected: bool         # whether results have been downloaded
    stats: CollectStats | None

@dataclass
class CollectStats:
    captions_written: int
    errors: int
    scenes_completed: int   # scenes where all selected frames now have .txt

@dataclass
class CollectResult:
    job_name: str
    stats: CollectStats
```

### State File: `assets/lora_dataset/batch_state.json`

```json
{
  "jobs": [
    {
      "name": "batches/abc123",
      "display_name": "chunk-001",
      "state": "JOB_STATE_RUNNING",
      "submitted_at": "2026-02-25T10:00:00Z",
      "frame_count": 25000,
      "scene_ids": [153, 154, "..."],
      "file_name": "files/xyz789",
      "result_file": null,
      "collected": false,
      "stats": null
    }
  ],
  "total_submitted": 208000,
  "total_collected": 0,
  "total_errors": 0
}
```

## API Endpoints

All use the Gemini Developer API (`generativelanguage.googleapis.com/v1beta`):

| Operation | Endpoint |
|---|---|
| Upload JSONL | `POST /upload/v1beta/files` (resumable) |
| Submit batch | `POST /v1beta/models/{model}:batchGenerateContent` |
| Poll status | `GET /v1beta/batches/{name}` |
| Download results | `GET /download/v1beta/{result_file}:download?alt=media` |
| Cancel job | `POST /v1beta/batches/{name}:cancel` |
| Delete job | `DELETE /v1beta/batches/{name}` |

Authentication: `x-goog-api-key` header or `key=` query param (existing Gemini API key).

## Dashboard Integration

### New Endpoints

| Path | Method | Description |
|---|---|---|
| `/api/batch/submit` | POST | Prepare chunks + upload + submit all pending frames |
| `/api/batch/status` | GET | Return batch_state.json contents |
| `/api/batch/collect` | POST | Collect results for a specific succeeded job |
| `/api/batch/collect-all` | POST | Collect all succeeded jobs |
| `/api/batch/cancel` | POST | Cancel a specific job |

### UI Components

**Batch Jobs section** in Pipeline Controls area:
- **"Submit Batch" button** — disabled when jobs are pending/running
- **Job status list** — one row per chunk: name, state badge, frame count, submitted time
- **Auto-poll** — every 60s via `/api/batch/status`
- **Auto-collect** — when poll detects `JOB_STATE_SUCCEEDED`, automatically POST to `/api/batch/collect`
- **Progress bar** — "3/7 jobs complete, 75,000 frames collected"
- **Cancel button** per job (when running/pending)

### State badges:
- `PENDING` → yellow
- `RUNNING` → blue pulsing
- `SUCCEEDED` → green (auto-collects)
- `FAILED` → red with error details
- `COLLECTED` → green checkmark (results written)

## Chunking Strategy

- **Chunk size:** ~25,000 frames per JSONL file
- Average JPEG frame: ~50KB → ~67KB base64
- Per-line JSON overhead: ~300 bytes
- **Estimated chunk file size:** ~1.7GB (under 2GB limit)
- For 208K total frames: **~8 chunks** submitted as separate batch jobs
- Each chunk processes independently — partial completion is useful

## JSONL Key Format

```
s{scene_id}_f{frame_filename_without_extension}
```

Example: `s1234_f0015` → maps to `s1234_f0015.txt` in images directory.

This matches the existing naming convention used by the real-time runner.

## Error Handling

- **Upload failure:** Retry once with exponential backoff, then report error
- **Job failure:** Mark in batch_state.json, show in dashboard. User can re-submit failed chunks
- **Partial results:** Parse available responses, write .txt for successes, log errors per frame
- **Expired jobs (>48h):** Mark as failed, frames remain uncaptioned for next submission
- **Network errors during collection:** Retry download, results are idempotent (same .txt overwrites)

## Scene Completion Logic

After collecting results, check each scene covered by the chunk:
1. List all frames that SmartFrameSelector would pick for the scene (at max_frames=20)
2. Check if all selected frames now have non-error .txt files
3. If yes → add scene to completed_scenes in caption_progress.json

This ensures scenes aren't marked complete until all their selected frames are captioned.

## Integration with Existing System

- **Frame selection:** Reuses `select_frames_for_scene()` from `frame_selector.py`
- **Caption prompt:** Reuses `PROMPT` from `constants.py`
- **Progress tracking:** Updates same `caption_progress.json`
- **Image naming:** Same `s{sid}_f{idx}.jpg/.txt` convention
- **Safety settings:** Same BLOCK_NONE configuration
- **No conflict with real-time runner:** Both check for existing .txt files before captioning
