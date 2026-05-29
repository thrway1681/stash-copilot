"""OpenAI LLM provider implementation."""

import json
from collections.abc import Generator
from typing import Any

import requests

from ...config import LLMConfig
from ..base import BaseLLMProvider, CompletionResult, Message, ToolCall
from ..provider import register_provider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider for GPT models.

    Supports GPT-4, GPT-4 Vision, GPT-4o, and other OpenAI models.
    Requires an API key from https://platform.openai.com/
    """

    # Mark as hosted provider (has per-token/per-image costs)
    HOSTED: bool = True

    # Models known to support vision
    VISION_MODELS = {
        "gpt-4-vision-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
    }

    # Models known to support tool use
    TOOL_MODELS = {
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    }

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self._session = requests.Session()

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set it in plugin settings or pass via api_key."
            )

    @property
    def supports_vision(self) -> bool:
        """Check if the current model supports vision."""
        model_lower = self.model.lower()
        # Check exact match or if model starts with a known vision model prefix
        return (
            model_lower in self.VISION_MODELS
            or model_lower.startswith("gpt-4o")
            or model_lower.startswith("gpt-4-turbo")
            or model_lower.startswith("gpt-4-vision")
        )

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tool use."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(tm) for tm in self.TOOL_MODELS)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using OpenAI Chat API.

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

        # Add images for vision models
        if "images" in kwargs and kwargs["images"] and self.supports_vision:
            images = kwargs["images"]
            for img_b64 in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",  # Use low detail to reduce token usage
                        },
                    }
                )

        # Add text prompt
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

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
                },
                json=payload,
                timeout=180,  # Vision requests can take longer
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"] or ""

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(f"OpenAI API error: {e}. {error_detail}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "OpenAI request timed out. Try a smaller prompt or fewer images."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens from OpenAI.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as they become available
        """
        messages: list[dict[str, Any]] = []

        if "system" in kwargs:
            messages.append({"role": "system", "content": kwargs["system"]})

        # Build user message content
        user_content: list[dict[str, Any]] = []

        if "images" in kwargs and kwargs["images"] and self.supports_vision:
            for img_b64 in kwargs["images"]:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",
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
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI streaming failed: {e}") from e

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
        # Check for images (for vision+tools models like gpt-4o)
        images = kwargs.get("images")
        has_images = images and self.supports_vision

        openai_messages = self._convert_messages(messages, images=images if has_images else None)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
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
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]

            tool_calls: list[ToolCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": args,
                        }
                    )

            finish_reason = result["choices"][0].get("finish_reason", "stop")
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
                error_detail = e.response.json().get("error", {}).get("message", "")
            except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                pass
            raise RuntimeError(f"OpenAI API error: {e}. {error_detail}") from e

    def _convert_messages(
        self,
        messages: list[Message],
        images: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format.

        Args:
            messages: List of Message dicts
            images: Optional list of base64-encoded images (added to first user message)

        Returns:
            List of OpenAI-formatted message dicts
        """
        openai_messages: list[dict[str, Any]] = []
        images_added = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            openai_msg: dict[str, Any] = {"role": role}

            # For user messages with images, use multimodal content format
            if role == "user" and images and not images_added:
                # Build multimodal content array
                content_parts: list[dict[str, Any]] = []
                if content:
                    content_parts.append({"type": "text", "text": content})
                for img in images:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                        }
                    )
                openai_msg["content"] = content_parts
                images_added = True
            elif content:
                openai_msg["content"] = content

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                openai_msg["tool_calls"] = [
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
                openai_msg["tool_call_id"] = msg.get("tool_call_id", "")

            openai_messages.append(openai_msg)

        return openai_messages

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False
