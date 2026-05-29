# Google (Gemini) LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `google` LLM provider that calls the Gemini REST API directly with full feature parity (complete, stream, chat, vision, tools).

**Architecture:** New provider class following the decorator-based registry pattern. Uses `requests.Session` for HTTP, API key as query parameter, Gemini `generateContent`/`streamGenerateContent` endpoints. Closest analog is the Anthropic provider (non-OpenAI API format with system message extraction).

**Tech Stack:** Python, `requests`, Gemini REST API v1beta

**Spec:** `docs/superpowers/specs/2026-03-17-google-gemini-provider-design.md`

---

### Task 1: Provider skeleton — init, registration, vision/tool detection

**Files:**
- Create: `stash_ai/llm/providers/google.py`
- Modify: `stash_ai/llm/providers/__init__.py`
- Modify: `stash_ai/llm/__init__.py`

- [ ] **Step 1: Create provider file with class skeleton**

Create `stash_ai/llm/providers/google.py`:

```python
"""Google Gemini LLM provider implementation."""

import json
from collections.abc import Generator
from typing import Any, ClassVar

import requests
from stashapi import log as stash_log

from ...config import LLMConfig
from ..base import BaseLLMProvider, CompletionResult, Message, ToolCall
from ..provider import register_provider


@register_provider("google")
class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini API provider.

    Supports Gemini 1.5, 2.x, and 3.x models via the REST API.
    Requires an API key from https://aistudio.google.com/apikey
    """

    HOSTED: bool = True

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    VISION_KEYWORDS: ClassVar[set[str]] = {
        "gemini-1.5",
        "gemini-2",
        "gemini-3",
        "gemma-3",
    }

    TOOL_KEYWORDS: ClassVar[set[str]] = {
        "gemini-1.5",
        "gemini-2",
        "gemini-3",
    }

    def __init__(self, config: LLMConfig):
        """Initialize Google Gemini provider."""
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or self.GEMINI_BASE_URL
        self._session = requests.Session()

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set it in plugin settings or pass via api_key."
            )

    @property
    def supports_vision(self) -> bool:
        """Check if the current model supports vision."""
        model_lower = self.model.lower()
        return any(kw in model_lower for kw in self.VISION_KEYWORDS)

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tool use."""
        model_lower = self.model.lower()
        return any(kw in model_lower for kw in self.TOOL_KEYWORDS)

    def _endpoint(self, action: str) -> str:
        """Build Gemini API endpoint URL with API key.

        Args:
            action: API action (e.g., "generateContent", "streamGenerateContent")

        Returns:
            Full URL with API key as query parameter
        """
        return f"{self.base_url}/models/{self.model}:{action}?key={self.api_key}"

    def _validate_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate Gemini response and return the first candidate.

        Gemini can return 200 with an error body, empty candidates (content
        blocked), or a SAFETY finish reason (generation stopped mid-response).

        Args:
            result: Parsed JSON response from Gemini

        Returns:
            The first candidate dict

        Raises:
            RuntimeError: If the response contains an error or is blocked
        """
        if "error" in result:
            error_msg = result["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            raise RuntimeError(f"Gemini API error: {error_msg}")

        # Content blocked before generation
        prompt_feedback = result.get("promptFeedback", {})
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini blocked the request: {block_reason}")

        candidates = result.get("candidates")
        if not candidates:
            stash_log.warning(
                f"[Google] Unexpected response (no candidates): "
                f"{json.dumps(result)[:500]}"
            )
            raise RuntimeError(
                f"Gemini returned unexpected response (no candidates). "
                f"Keys: {list(result.keys())}"
            )

        # Generation stopped by safety filters
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason == "SAFETY":
            raise RuntimeError(
                "Gemini stopped generation due to safety filters"
            )

        return candidates[0]

    def _build_generation_config(self, **kwargs: Any) -> dict[str, Any]:
        """Build generationConfig payload section.

        Args:
            **kwargs: Override temperature and max_tokens from caller

        Returns:
            generationConfig dict for Gemini API
        """
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
        }

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion using Gemini API."""
        raise NotImplementedError("Implemented in Task 2")

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Stream completion tokens from Gemini."""
        raise NotImplementedError("Implemented in Task 3")
        yield  # Make it a generator

    def health_check(self) -> bool:
        """Check if Gemini API is accessible and model exists."""
        try:
            response = self._session.get(
                f"{self.base_url}/models/{self.model}?key={self.api_key}",
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False
```

- [ ] **Step 2: Wire up registration in `__init__` files**

In `stash_ai/llm/providers/__init__.py`, add the import:

```python
from .google import GoogleProvider
```

And add `"GoogleProvider"` to `__all__`.

In `stash_ai/llm/__init__.py`, add `google` to the module import block:

```python
from .providers import (
    anthropic,
    google,
    ollama,
    openai,
    openrouter,
)
```

- [ ] **Step 3: Verify registration works**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.llm import get_provider; print('google registered OK')"`

Expected: prints `google registered OK` (no import errors)

- [ ] **Step 4: Commit**

```bash
git add stash_ai/llm/providers/google.py stash_ai/llm/providers/__init__.py stash_ai/llm/__init__.py
git commit -m "feat(llm): add Google Gemini provider skeleton with registration"
```

---

### Task 2: Implement `complete()` — text and vision completions

**Files:**
- Modify: `stash_ai/llm/providers/google.py` (replace `complete()` stub)

- [ ] **Step 1: Implement `complete()` method**

Replace the `complete()` stub in `google.py` with:

```python
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using Gemini API.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, images, etc.)

        Returns:
            The generated completion text

        Raises:
            RuntimeError: If the API request fails
        """
        # Build user message parts
        parts: list[dict[str, Any]] = []

        # Add images for vision models (inline_data format)
        images_sent = 0
        if kwargs.get("images"):
            images = kwargs["images"]
            if self.supports_vision:
                for img_b64 in images:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        }
                    )
                    images_sent += 1
            else:
                stash_log.warning(
                    f"[Google] {len(images)} images provided but model "
                    f"'{self.model}' not detected as vision model. "
                    f"Images NOT sent."
                )

        # Add text prompt
        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._build_generation_config(**kwargs),
        }

        # Add system instruction if provided
        if "system" in kwargs:
            payload["systemInstruction"] = {
                "parts": [{"text": kwargs["system"]}]
            }

        # Debug logging
        from ..model_caps import is_debug_logging_enabled

        if is_debug_logging_enabled():
            stash_log.debug(
                f"[Google] Model: {self.model}, Vision: {self.supports_vision}, "
                f"Images sent: {images_sent}"
            )

        try:
            response = self._session.post(
                self._endpoint("generateContent"),
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()
            candidate = self._validate_response(result)

            # Extract text from all parts
            text_parts = []
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])

            return "".join(text_parts) or ""

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(
                f"Gemini API error: {e}. {error_detail}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Gemini request timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e
```

- [ ] **Step 2: Verify text completion works**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.config import LLMConfig
from stash_ai.llm import get_provider
config = LLMConfig(provider='google', model='gemini-2.0-flash', api_key='$(grep GEMINI_API_KEY .env | cut -d= -f2)')
llm = get_provider(config)
print(llm.complete('Say hello in exactly 3 words'))
"`

Expected: A 3-word greeting from Gemini

- [ ] **Step 3: Commit**

```bash
git add stash_ai/llm/providers/google.py
git commit -m "feat(llm): implement Google provider complete() with vision support"
```

---

### Task 3: Implement `stream()` — streaming completions

**Files:**
- Modify: `stash_ai/llm/providers/google.py` (replace `stream()` stub)

- [ ] **Step 1: Implement `stream()` method**

Replace the `stream()` stub in `google.py` with:

```python
    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens from Gemini.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as they become available
        """
        # Build user message parts (same as complete)
        parts: list[dict[str, Any]] = []

        if kwargs.get("images") and self.supports_vision:
            for img_b64 in kwargs["images"]:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64,
                        }
                    }
                )

        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._build_generation_config(**kwargs),
        }

        if "system" in kwargs:
            payload["systemInstruction"] = {
                "parts": [{"text": kwargs["system"]}]
            }

        try:
            response = self._session.post(
                self._endpoint("streamGenerateContent") + "&alt=sse",
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=180,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            # Check for errors in stream chunks
                            if "error" in data:
                                err = data["error"]
                                if isinstance(err, dict):
                                    err = err.get("message", str(err))
                                raise RuntimeError(
                                    f"Gemini stream error: {err}"
                                )
                            candidates = data.get("candidates", [])
                            if candidates:
                                for part in (
                                    candidates[0]
                                    .get("content", {})
                                    .get("parts", [])
                                ):
                                    text = part.get("text")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Gemini stream timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini streaming failed: {e}") from e
```

- [ ] **Step 2: Verify streaming works**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.config import LLMConfig
from stash_ai.llm import get_provider
config = LLMConfig(provider='google', model='gemini-2.0-flash', api_key='$(grep GEMINI_API_KEY .env | cut -d= -f2)')
llm = get_provider(config)
for chunk in llm.stream('Count from 1 to 5'):
    print(chunk, end='', flush=True)
print()
"`

Expected: Numbers 1 through 5 printed incrementally

- [ ] **Step 3: Commit**

```bash
git add stash_ai/llm/providers/google.py
git commit -m "feat(llm): implement Google provider stream()"
```

---

### Task 4: Implement `_convert_messages()` and `chat()` — multi-turn with tools

**Files:**
- Modify: `stash_ai/llm/providers/google.py`

- [ ] **Step 1: Implement `_convert_messages()` method**

Add this method to `GoogleProvider`:

```python
    def _convert_messages(
        self,
        messages: list[Message],
        images: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Convert messages to Gemini format.

        Extracts system messages to a separate systemInstruction string.
        Maps roles: assistant -> model, tool -> user (with functionResponse).
        Attaches images to the first user message as inline_data parts.

        Args:
            messages: List of Message dicts
            images: Optional list of base64-encoded images

        Returns:
            Tuple of (Gemini contents list, system instruction or None)
        """
        gemini_contents: list[dict[str, Any]] = []
        system_instruction: str | None = None
        images_added = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Extract system prompt
            if role == "system":
                system_instruction = content
                continue

            # Tool result messages -> role "user" with functionResponse
            # NOTE: Gemini correlates tool results by function name (not by
            # opaque ID). chat() sets id=name on ToolCall, so tool_call_id
            # on the result Message carries the function name we need here.
            if role == "tool":
                # Parse content as JSON if possible for structured response
                response_data: Any
                if content:
                    try:
                        response_data = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        response_data = {"result": content}
                else:
                    response_data = {"result": ""}

                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("tool_call_id", "unknown"),
                                    "response": response_data,
                                }
                            }
                        ],
                    }
                )
                continue

            # Map roles
            gemini_role = "model" if role == "assistant" else "user"
            parts: list[dict[str, Any]] = []

            # Assistant messages with tool calls -> functionCall parts
            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                if content:
                    parts.append({"text": content})
                for tc in tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc["name"],
                                "args": tc["arguments"],
                            }
                        }
                    )
            elif role == "user" and images and not images_added:
                # First user message gets images (before text)
                for img in images:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img,
                            }
                        }
                    )
                if content:
                    parts.append({"text": content})
                images_added = True
            else:
                if content:
                    parts.append({"text": content})

            # Skip messages with empty parts (Gemini rejects empty text parts)
            if parts:
                gemini_contents.append({"role": gemini_role, "parts": parts})

        return gemini_contents, system_instruction
```

- [ ] **Step 2: Implement `chat()` method**

Add this method to `GoogleProvider`:

```python
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Chat completion with optional tool use.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters (images for vision models)

        Returns:
            CompletionResult with content and/or tool calls
        """
        images = kwargs.get("images")
        has_images = images and self.supports_vision

        gemini_contents, system_instruction = self._convert_messages(
            messages, images=images if has_images else None
        )

        payload: dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": self._build_generation_config(**kwargs),
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if tools and self.supports_tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"],
                        }
                        for tool in tools
                    ]
                }
            ]

        try:
            response = self._session.post(
                self._endpoint("generateContent"),
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()
            candidate = self._validate_response(result)

            # Parse response parts
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Gemini has no opaque call IDs — correlation is by
                    # function name. Set id=name so that tool_call_id on the
                    # subsequent tool-result Message carries the function name,
                    # which _convert_messages() needs for functionResponse.name.
                    tool_calls.append(
                        {
                            "id": fc.get("name", ""),
                            "name": fc.get("name", ""),
                            "arguments": fc.get("args", {}),
                        }
                    )

            content = "".join(text_parts) if text_parts else None

            # Determine finish reason from response parts, not finishReason field
            if tool_calls:
                finish_reason = "tool_calls"
            else:
                gemini_reason = candidate.get("finishReason", "STOP")
                if gemini_reason == "MAX_TOKENS":
                    finish_reason = "length"
                else:
                    finish_reason = "stop"

            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(
                f"Gemini API error: {e}. {error_detail}"
            ) from e
```

- [ ] **Step 3: Verify chat works (text only)**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.config import LLMConfig
from stash_ai.llm import get_provider
config = LLMConfig(provider='google', model='gemini-2.0-flash', api_key='$(grep GEMINI_API_KEY .env | cut -d= -f2)')
llm = get_provider(config)
result = llm.chat([
    {'role': 'system', 'content': 'You are a pirate.'},
    {'role': 'user', 'content': 'Say hello'},
])
print(f'Content: {result[\"content\"][:100]}')
print(f'Finish: {result[\"finish_reason\"]}')
print(f'Tools: {result[\"tool_calls\"]}')
"`

Expected: Pirate-themed greeting, finish_reason="stop", empty tool_calls

- [ ] **Step 4: Verify tool calling works**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.config import LLMConfig
from stash_ai.llm import get_provider
config = LLMConfig(provider='google', model='gemini-2.0-flash', api_key='$(grep GEMINI_API_KEY .env | cut -d= -f2)')
llm = get_provider(config)
tools = [{'name': 'get_weather', 'description': 'Get weather for a city', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}]
result = llm.chat([{'role': 'user', 'content': 'What is the weather in Tokyo?'}], tools=tools)
print(f'Content: {result[\"content\"]}')
print(f'Finish: {result[\"finish_reason\"]}')
print(f'Tools: {result[\"tool_calls\"]}')
"`

Expected: finish_reason="tool_calls", tool_calls with name="get_weather" and args={"city": "Tokyo"}

- [ ] **Step 5: Commit**

```bash
git add stash_ai/llm/providers/google.py
git commit -m "feat(llm): implement Google provider chat() with tool calling"
```

---

### Task 5: Update YAML settings and final integration test

**Files:**
- Modify: `stash-copilot.yml:11,29`

- [ ] **Step 1: Update YAML description strings**

In `stash-copilot.yml` line 11, change:
```
description: Provider for text tasks (stats, chat, tag suggestions). Options - ollama, openai, anthropic, openrouter. Default is ollama.
```
to:
```
description: Provider for text tasks (stats, chat, tag suggestions). Options - ollama, openai, anthropic, openrouter, google. Default is ollama.
```

In `stash-copilot.yml` line 29, change:
```
description: Provider for vision analysis. Options - ollama, openai, anthropic, openrouter. Leave empty to use Text LLM.
```
to:
```
description: Provider for vision analysis. Options - ollama, openai, anthropic, openrouter, google. Leave empty to use Text LLM.
```

- [ ] **Step 2: Run full import check**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.llm.provider import _PROVIDERS
print(f'Registered providers: {sorted(_PROVIDERS.keys())}')
assert 'google' in _PROVIDERS, 'Google provider not registered!'
print('All providers registered OK')
"`

Expected: `Registered providers: ['anthropic', 'google', 'ollama', 'openai', 'openrouter']`

- [ ] **Step 3: Run health check**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.config import LLMConfig
from stash_ai.llm import get_provider
config = LLMConfig(provider='google', model='gemini-2.0-flash', api_key='$(grep GEMINI_API_KEY .env | cut -d= -f2)')
llm = get_provider(config)
print(f'Health check: {llm.health_check()}')
print(f'Vision: {llm.supports_vision}')
print(f'Tools: {llm.supports_tools}')
print(f'Hosted: {llm.is_hosted}')
"`

Expected: All True

- [ ] **Step 4: Commit**

```bash
git add stash-copilot.yml
git commit -m "feat(config): add google to provider options in plugin settings"
```
