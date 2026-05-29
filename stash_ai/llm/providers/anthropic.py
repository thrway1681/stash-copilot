"""Anthropic Claude LLM provider implementation."""

import json
from collections.abc import Generator
from typing import Any

import requests

from ...config import LLMConfig
from ..base import BaseLLMProvider, CompletionResult, Message, ToolCall
from ..provider import register_provider


@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    Supports Claude 3, Claude 3.5, and Claude 4 models.
    Requires an API key from https://console.anthropic.com/
    """

    # Mark as hosted provider (has per-token/per-image costs)
    HOSTED: bool = True

    # Anthropic API version
    API_VERSION = "2023-06-01"

    # All Claude 3+ models support vision
    VISION_MODELS = {
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-opus-4",
        "claude-sonnet-4",
    }

    # All Claude 3+ models support tools
    TOOL_MODELS = VISION_MODELS

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://api.anthropic.com"
        self._session = requests.Session()

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set it in plugin settings or pass via api_key."
            )

    @property
    def supports_vision(self) -> bool:
        """Check if the current model supports vision."""
        model_lower = self.model.lower()
        # All Claude 3+ models support vision
        return (
            any(model_lower.startswith(vm) for vm in self.VISION_MODELS)
            or "claude-3" in model_lower
            or "claude-opus" in model_lower
            or "claude-sonnet" in model_lower
        )

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tool use."""
        # Same as vision - all Claude 3+ models support tools
        return self.supports_vision

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using Anthropic Messages API.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, images, etc.)

        Returns:
            The generated completion text

        Raises:
            RuntimeError: If the API request fails
        """
        # Build message content
        content: list[dict[str, Any]] = []

        # Add images for vision models
        if "images" in kwargs and kwargs["images"] and self.supports_vision:
            images = kwargs["images"]
            for img_b64 in images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    }
                )

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": content}],
        }

        # Add system prompt if provided
        if "system" in kwargs:
            payload["system"] = kwargs["system"]

        # Add temperature if not default
        temperature = kwargs.get("temperature", self.temperature)
        if temperature != 1.0:  # Anthropic default is 1.0
            payload["temperature"] = temperature

        try:
            response = self._session.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
                timeout=180,  # Vision requests can take longer
            )
            response.raise_for_status()

            result = response.json()

            # Extract text from content blocks
            text_parts = []
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            return "".join(text_parts)

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(f"Anthropic API error: {e}. {error_detail}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Anthropic request timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Anthropic request failed: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens from Anthropic.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as they become available
        """
        content: list[dict[str, Any]] = []

        if "images" in kwargs and kwargs["images"] and self.supports_vision:
            for img_b64 in kwargs["images"]:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    }
                )

        content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }

        if "system" in kwargs:
            payload["system"] = kwargs["system"]

        temperature = kwargs.get("temperature", self.temperature)
        if temperature != 1.0:
            payload["temperature"] = temperature

        try:
            response = self._session.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
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
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type")

                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield text

                            elif event_type == "message_stop":
                                break

                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Anthropic streaming failed: {e}") from e

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
        # Check for images (for vision+tools models like claude-3)
        images = kwargs.get("images")
        has_images = images and self.supports_vision

        anthropic_messages, system_prompt = self._convert_messages(
            messages, images=images if has_images else None
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        temperature = kwargs.get("temperature", self.temperature)
        if temperature != 1.0:
            payload["temperature"] = temperature

        if tools and self.supports_tools:
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]

        try:
            response = self._session.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()

            # Parse content blocks
            text_parts = []
            tool_calls: list[ToolCall] = []

            for block in result.get("content", []):
                block_type = block.get("type")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        }
                    )

            content = "".join(text_parts) if text_parts else None
            stop_reason = result.get("stop_reason", "end_turn")

            if stop_reason == "tool_use" or tool_calls:
                finish_reason = "tool_calls"
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
            raise RuntimeError(f"Anthropic API error: {e}. {error_detail}") from e

    def _convert_messages(
        self,
        messages: list[Message],
        images: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Convert messages to Anthropic format.

        Args:
            messages: List of Message dicts
            images: Optional list of base64-encoded images (added to first user message)

        Returns:
            Tuple of (messages list, system prompt or None)
        """
        anthropic_messages: list[dict[str, Any]] = []
        system_prompt: str | None = None
        images_added = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Extract system prompt
            if role == "system":
                system_prompt = content
                continue

            # Map tool role to user role with tool_result
            if role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": content or "",
                            }
                        ],
                    }
                )
                continue

            anthropic_msg: dict[str, Any] = {"role": role}

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                # Assistant message with tool calls
                msg_content: list[dict[str, Any]] = []
                if content:
                    msg_content.append({"type": "text", "text": content})
                for tc in tool_calls:
                    msg_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
                anthropic_msg["content"] = msg_content
            elif role == "user" and images and not images_added:
                # User message with images - use multimodal content format
                msg_content_parts: list[dict[str, Any]] = []
                if content:
                    msg_content_parts.append({"type": "text", "text": content})
                for img in images:
                    msg_content_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img,
                            },
                        }
                    )
                anthropic_msg["content"] = msg_content_parts
                images_added = True
            else:
                anthropic_msg["content"] = content or ""

            anthropic_messages.append(anthropic_msg)

        return anthropic_messages, system_prompt

    def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Anthropic doesn't have a simple health endpoint, so we'll just
            # verify the API key format and assume it's valid
            return bool(self.api_key and len(self.api_key) > 20)
        except Exception:
            return False
