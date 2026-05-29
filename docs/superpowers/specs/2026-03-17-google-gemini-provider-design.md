# Google (Gemini) LLM Provider — Design Spec

**Date:** 2026-03-17
**Status:** Approved

## Goal

Add a native `google` LLM provider that calls the Gemini REST API directly, allowing the user to use their Google API key without routing through OpenRouter. Full feature parity: text completion, streaming, vision, and tool calling.

## Decision Record

- **Direct REST API** over `google-genai` SDK — no new dependency, consistent with all other providers using raw `requests`.
- **API key auth only** (query param `?key=...`) — matches the user's existing `AIzaSy...` key. OAuth/Bearer support deferred.
- **Full-featured** (complete + stream + chat + vision + tools) — Gemini models support all of these natively, and the codebase uses all of them across different tasks.

## Files to Create

| File | Purpose |
|---|---|
| `stash_ai/llm/providers/google.py` | Provider implementation (~350 lines) |

## Files to Modify

| File | Change |
|---|---|
| `stash_ai/llm/providers/__init__.py` | Add `from .google import GoogleProvider` to class exports |
| `stash_ai/llm/__init__.py` | Add `google` to the module import block (triggers `@register_provider` decorator) — see [Wiring Details](#wiring-details) |
| `stash-copilot.yml` | Update `text_llm_provider` and `vision_llm_provider` description strings to include "google" |

## Files Unchanged

| File | Reason |
|---|---|
| `stash_ai/llm/model_caps.py` | Already has `gemini-1.5`, `gemini-2`, `gemini-3` capability entries |
| `stash_ai/config/legacy.py` | Provider string flows through generically — no changes needed |
| `stash_ai/llm/provider.py` | Registry is decorator-based — no changes needed |
| `stash_ai/llm/base.py` | Base class contract unchanged |

## Wiring Details

Both edits are required for the provider to register:

**`stash_ai/llm/providers/__init__.py`** — add class export:
```python
from .google import GoogleProvider
```

**`stash_ai/llm/__init__.py`** — add module import (triggers `@register_provider` decorator):
```python
from .providers import (
    anthropic,
    google,      # <-- add this
    ollama,
    openai,
    openrouter,
)
```

Without the `llm/__init__.py` import, the decorator never fires and `get_provider("google")` raises `ValueError`.

## Provider Class Design

### Registration & Class Structure

```python
@register_provider("google")
class GoogleProvider(BaseLLMProvider):
    HOSTED = True
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    VISION_KEYWORDS: ClassVar[set[str]] = {
        "gemini-1.5", "gemini-2", "gemini-3", "gemma-3",
    }

    TOOL_KEYWORDS: ClassVar[set[str]] = {
        "gemini-1.5", "gemini-2", "gemini-3",
    }
```

### Authentication

API key as query parameter on every request. No Bearer header.

```
POST {base_url}/models/{model}:generateContent?key={api_key}
POST {base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}
```

`base_url` defaults to `GEMINI_BASE_URL`, overridable via `config.base_url` (for Vertex AI or proxies). `__init__` validates API key presence.

### Vision & Tool Detection

Substring match on lowercased model name, same pattern as other providers:

- `supports_vision`: any keyword from `VISION_KEYWORDS` in model name
- `supports_tools`: any keyword from `TOOL_KEYWORDS` in model name

### Message Format Mapping

| Concept | OpenAI format | Gemini format |
|---|---|---|
| System prompt | `{"role": "system"}` message | Top-level `"systemInstruction": {"parts": [{"text": "..."}]}` |
| Text content | `{"type": "text", "text": "..."}` | `{"text": "..."}` part |
| Image | `image_url` with data URI | `{"inline_data": {"mime_type": "image/jpeg", "data": "<b64>"}}` part |
| Tool definition | `{"type": "function", "function": {...}}` | `{"function_declarations": [{...}]}` in `"tools"` array |
| Tool call (in response) | `tool_calls[].function.{name, arguments}` | `{"functionCall": {"name": "...", "args": {...}}}` part |
| Tool result (in request) | `{"role": "tool", "tool_call_id": ..., "content": ...}` | `{"role": "user", "parts": [{"functionResponse": {"name": "...", "response": {...}}}]}` |
| Role: assistant | `"assistant"` | `"model"` |
| Role: tool result | `"tool"` | `"user"` (Gemini only accepts `"user"` and `"model"` roles in request contents) |

### Method Specifications

#### `__init__(self, config: LLMConfig)`

- Store `api_key`, `model`, set `base_url` (use config value or `GEMINI_BASE_URL`)
- Create `requests.Session()`
- Raise `ValueError` if `api_key` is empty/None

#### `complete(self, prompt: str, **kwargs) -> str`

1. Build payload:
   - `systemInstruction` from `kwargs["system"]` if present
   - `contents`: single user message with parts: image `inline_data` parts (if vision model + images provided), then text part
   - `generationConfig`: `{"temperature": ..., "maxOutputTokens": ...}`
2. POST to `models/{model}:generateContent?key={api_key}`
3. Validate response via `_validate_response()`
4. Extract text from `candidates[0]["content"]["parts"]` — concatenate all `text` parts
5. Return concatenated text

#### `stream(self, prompt: str, **kwargs) -> Generator[str, None, None]`

1. Build same payload as `complete()` (no `stream` field needed — streaming is endpoint-based)
2. POST to `models/{model}:streamGenerateContent?alt=sse&key={api_key}` with `stream=True`
3. Parse SSE lines (`data: ` prefix)
4. For each chunk, extract text from `candidates[0]["content"]["parts"]`
5. Yield text content

#### `chat(self, messages: list[Message], tools=None, **kwargs) -> CompletionResult`

1. Convert messages via `_convert_messages()`:
   - Extract system messages → `systemInstruction`
   - Map `"assistant"` role → `"model"`, `"tool"` role → `"user"` (with `functionResponse` parts)
   - Convert image content to `inline_data` format
   - Convert tool call messages to `functionCall` parts
   - Convert tool result messages to `functionResponse` parts
2. Build tool definitions as `function_declarations` if tools provided
3. Include `generationConfig: {"temperature": ..., "maxOutputTokens": ...}` in payload (same as `complete()`)
4. POST to `models/{model}:generateContent?key={api_key}`
5. Validate response
6. Parse response parts:
   - `text` parts → `content` string
   - `functionCall` parts → `ToolCall` list (`args` mapped to `arguments`)
7. Determine `finish_reason`: if `functionCall` parts are present → `"tool_calls"`, otherwise map Gemini's `finishReason`: `"STOP"` → `"stop"`, `"MAX_TOKENS"` → `"length"`. Note: Gemini does not have a dedicated `"TOOL_CALLS"` finish reason — tool call detection is by inspecting response parts.
8. Return `CompletionResult`

#### `_validate_response(self, result: dict) -> dict`

Checks in order:
1. `"error"` key present → raise `RuntimeError` with error message
2. `"promptFeedback"` with `"blockReason"` → raise `RuntimeError("Gemini blocked the request: {reason}")`
3. `"candidates"` missing or empty → raise `RuntimeError` with response keys
4. `candidates[0]["finishReason"] == "SAFETY"` → raise `RuntimeError("Gemini stopped generation due to safety filters")`
5. Return `candidates[0]`

#### `_convert_messages(self, messages, images=None) -> tuple[list[dict], str | None]`

Returns `(gemini_contents, system_instruction)`. Same extraction pattern as Anthropic provider.

- System messages → concatenated into `system_instruction` string
- Images attached to first user message as `inline_data` parts (before text)
- Tool call assistant messages → `functionCall` parts
- Tool result messages → role `"user"`, `functionResponse` parts

#### `health_check(self) -> bool`

GET `models/{model}?key={api_key}` — returns model metadata if key and model are valid.

### Error Handling

| HTTP Status | Gemini Meaning | Mapped Error |
|---|---|---|
| 200 + empty candidates + `promptFeedback.blockReason` | Content blocked before generation | `RuntimeError("Gemini blocked the request: {reason}")` |
| 200 + `finishReason: "SAFETY"` | Generation stopped mid-response | `RuntimeError("Gemini stopped generation due to safety filters")` |
| 400 | Invalid request (bad model, too many tokens) | `RuntimeError("Gemini API error: {status} {detail}")` |
| 403 | Invalid API key or model not available | `RuntimeError("Gemini API error: {status} {detail}")` |
| 429 | Rate limit | `RuntimeError("Gemini API error: {status} {detail}")` |
| Timeout | Request exceeded 180s | `RuntimeError("Gemini request timed out...")` |

All errors extract the `message` field from Gemini's error JSON (`result["error"]["message"]`) when available.

### YAML Setting Updates

Update description text for `text_llm_provider` and `vision_llm_provider` in `stash-copilot.yml`:

```yaml
description: "Provider for text tasks. Options - ollama, openai, anthropic, openrouter, google."
```

## Out of Scope

- OAuth2 / Bearer token auth — deferred, API key covers current needs
- Vertex AI endpoint support — `base_url` override enables this manually if needed
- Gemini-specific features (grounding, code execution, Google Search) — not needed for current tasks
- New dependencies — uses `requests` only, like all other providers
