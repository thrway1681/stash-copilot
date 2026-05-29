"""OpenRouter LLM provider implementation."""

import json
from collections.abc import Generator
from typing import Any, cast

import requests
from stashapi import log as stash_log

from ...config import LLMConfig
from ..base import BaseLLMProvider, CompletionResult, Message, ToolCall
from ..provider import register_provider


@register_provider("openrouter")
class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider.

    OpenRouter provides unified access to many LLM providers including
    OpenAI, Anthropic, Google, Meta, and more.
    Requires an API key from https://openrouter.ai/
    """

    # Mark as hosted provider (has per-token/per-image costs)
    HOSTED: bool = True

    # Keywords that indicate vision support in model names
    VISION_KEYWORDS = {
        "vision",
        "4o",  # GPT-4o
        "claude-3",  # Claude 3.x models
        "claude-sonnet",  # Claude Sonnet 4.x (anthropic/claude-sonnet-4.5)
        "claude-opus",  # Claude Opus 4.x
        "gemini-pro-vision",
        "gemini-1.5",
        "gemini-2",  # Gemini 2.x models
        "gemini-3",  # Gemini 3.x models
        "grok",  # xAI Grok models (grok-2-vision, grok-4.1, etc.)
        "llava",
        "pixtral",
        "deepseek-vl",  # DeepSeek Vision-Language models
        "janus",  # DeepSeek Janus multimodal
        "kimi",  # Moonshot Kimi K2.5 multimodal
    }

    # Keywords that indicate tool support
    TOOL_KEYWORDS = {
        "gpt-4",
        "gpt-3.5",
        "claude-3",
        "claude-sonnet",
        "claude-opus",
        "gemini",
        "mistral-large",
        "grok",  # xAI Grok models support function calling
        "kimi",  # Moonshot Kimi K2.5 tool calling
    }

    # OpenRouter's fixed API endpoint
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: LLMConfig):
        """Initialize OpenRouter provider."""
        super().__init__(config)
        self.api_key = config.api_key
        # Always use OpenRouter's URL - ignore config.base_url (which may be set for Ollama)
        self.base_url = self.OPENROUTER_BASE_URL
        self._session = requests.Session()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set it in plugin settings or pass via api_key."
            )

    def _validate_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate OpenRouter response contains expected fields.

        OpenRouter sometimes returns 200 with an error body (e.g., content
        moderation, rate limits, model errors) instead of a proper choices array.

        Args:
            result: Parsed JSON response from OpenRouter

        Returns:
            The first choice's message dict

        Raises:
            RuntimeError: If the response is missing choices or contains an error
        """
        if "error" in result:
            error_msg = result["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            raise RuntimeError(f"OpenRouter API error: {error_msg}")

        if "choices" not in result or not result["choices"]:
            stash_log.warning(
                f"[OpenRouter] Unexpected response (no choices): {json.dumps(result)[:500]}"
            )
            raise RuntimeError(
                f"OpenRouter returned unexpected response (no choices). Keys: {list(result.keys())}"
            )

        return cast("dict[str, Any]", result["choices"][0])

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

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using OpenRouter API.

        Uses OpenAI-compatible format.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, images, etc.)

        Returns:
            The generated completion text

        Raises:
            RuntimeError: If the API request fails
        """
        messages: list[dict[str, Any]] = []

        # Add system prompt if provided
        if "system" in kwargs:
            messages.append({"role": "system", "content": kwargs["system"]})

        # Build user message content
        user_content: list[dict[str, Any]] = []

        # Add images for vision models (OpenAI format)
        images_sent = 0
        if kwargs.get("images"):
            images = kwargs["images"]
            if self.supports_vision:
                for img_b64 in images:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            },
                        }
                    )
                    images_sent += 1
            else:
                # Always warn if images not sent (this is important to know)
                stash_log.warning(
                    f"[OpenRouter] {len(images)} images provided but model '{self.model}' not detected as vision model. Images NOT sent."
                )

        # Add text prompt
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        # Debug logging (only when STASH_COPILOT_DEBUG=1)
        from ..model_caps import is_debug_logging_enabled

        if is_debug_logging_enabled():
            stash_log.debug(
                f"[OpenRouter] Model: {self.model}, Vision support: {self.supports_vision}, Images sent: {images_sent}, URL: {self.base_url}"
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/stashapp/stash",
                    "X-Title": "Stash Copilot",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()
            choice = self._validate_response(result)
            return choice["message"]["content"] or ""

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(f"OpenRouter API error: {e}. {error_detail}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "OpenRouter request timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter request failed: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens from OpenRouter.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as they become available
        """
        messages: list[dict[str, Any]] = []

        if "system" in kwargs:
            messages.append({"role": "system", "content": kwargs["system"]})

        user_content: list[dict[str, Any]] = []

        if "images" in kwargs and kwargs["images"] and self.supports_vision:
            for img_b64 in kwargs["images"]:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    }
                )

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/stashapp/stash",
                    "X-Title": "Stash Copilot",
                },
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
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" not in data or not data["choices"]:
                                if "error" in data:
                                    err = data["error"]
                                    if isinstance(err, dict):
                                        err = err.get("message", str(err))
                                    raise RuntimeError(f"OpenRouter stream error: {err}")
                                continue
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter streaming failed: {e}") from e

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
        # Check for images (for vision+tools models)
        images = kwargs.get("images")
        has_images = images and self.supports_vision

        openrouter_messages = self._convert_messages(
            messages, images=images if has_images else None
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openrouter_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools and self.supports_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/stashapp/stash",
                    "X-Title": "Stash Copilot",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()
            choice = self._validate_response(result)
            message = choice["message"]

            tool_calls: list[ToolCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            stash_log.warning(
                                f"[OpenRouter] Failed to parse tool arguments for "
                                f"{func.get('name')}: {raw_args!r}"
                            )
                            args = {}
                    else:
                        stash_log.warning(
                            f"[OpenRouter] Unexpected argument type "
                            f"{type(raw_args).__name__} for {func.get('name')}: {raw_args!r}"
                        )
                        args = {}
                    tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": args,
                        }
                    )

            finish_reason = choice.get("finish_reason", "stop")
            if finish_reason == "tool_calls" or tool_calls:
                finish_reason = "tool_calls"

            return {
                "content": message.get("content"),
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
            raise RuntimeError(f"OpenRouter API error: {e}. {error_detail}") from e

    def _convert_messages(
        self,
        messages: list[Message],
        images: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI/OpenRouter format.

        Args:
            messages: List of Message dicts
            images: Optional list of base64-encoded images (added to first user message)

        Returns:
            List of OpenRouter-formatted message dicts
        """
        openrouter_messages: list[dict[str, Any]] = []
        images_added = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            openrouter_msg: dict[str, Any] = {"role": role}

            # For user messages with images, use multimodal content format
            if role == "user" and images and not images_added:
                # Build multimodal content array
                # IMPORTANT: Images MUST come FIRST for proper VLM attention
                # If text comes first, models may generate responses based on the
                # text prompt without properly attending to the images
                content_parts: list[dict[str, Any]] = []
                for img in images:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                        }
                    )
                if content:
                    content_parts.append({"type": "text", "text": content})
                openrouter_msg["content"] = content_parts
                images_added = True
            elif content:
                openrouter_msg["content"] = content

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                openrouter_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]

            if role == "tool":
                openrouter_msg["tool_call_id"] = msg.get("tool_call_id", "")

            openrouter_messages.append(openrouter_msg)

        return openrouter_messages

    def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """
        List available models on OpenRouter.

        Returns:
            List of model info dictionaries
        """
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            models: list[dict[str, Any]] = data.get("data", [])
            return models
        except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError):
            return []
