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
        # Always use Gemini's URL - ignore config.base_url (which defaults to Ollama)
        self.base_url = self.GEMINI_BASE_URL
        self._session = requests.Session()

        # Strip OpenRouter-style "google/" prefix from model name
        # so users can share the same model name across providers
        if self.model.startswith("google/"):
            self.model = self.model[len("google/") :]

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
                f"[Google] Unexpected response (no candidates): {json.dumps(result)[:500]}"
            )
            raise RuntimeError(
                f"Gemini returned unexpected response (no candidates). Keys: {list(result.keys())}"
            )

        # Generation stopped by safety filters
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason == "SAFETY":
            raise RuntimeError("Gemini stopped generation due to safety filters")

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
            payload["systemInstruction"] = {"parts": [{"text": kwargs["system"]}]}

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
            raise RuntimeError(f"Gemini API error: {e}. {error_detail}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Gemini request timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e

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
            payload["systemInstruction"] = {"parts": [{"text": kwargs["system"]}]}

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
                                raise RuntimeError(f"Gemini stream error: {err}")
                            candidates = data.get("candidates", [])
                            if candidates:
                                for part in candidates[0].get("content", {}).get("parts", []):
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
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

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
            raise RuntimeError(f"Gemini API error: {e}. {error_detail}") from e

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
