"""Ollama LLM provider implementation."""

import json
import uuid
from collections.abc import Generator
from typing import Any

import requests
from stashapi import log as stash_log

from ...config import LLMConfig
from ..base import BaseLLMProvider, CompletionResult, Message, ToolCall
from ..provider import register_provider


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM inference.

    Ollama must be running locally (or at the specified base_url).
    See: https://ollama.ai/
    """

    # Models known to support vision (check by prefix/keyword)
    VISION_MODELS = {
        "llava",
        "llava-llama3",
        "bakllava",
        "moondream",
        "llama3.2-vision",
        "minicpm-v",
        "llava-phi3",
        "gemma2-vision",
        "qwen2-vl",
        "internvl2",
        "pixtral",
        "molmo",
        "granite3-vision",
    }
    # Keywords that indicate vision support
    VISION_KEYWORDS = {
        "vision",
        "llava",
        "minicpm-v",
        "-vl",
        "pixtral",
        "molmo",
        "joycaption",
        "gemma-3",
        "gemma3",  # Gemma 3 has native vision support
    }

    # Models known to support tool use
    TOOL_MODELS = {
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "qwen2.5",
        "qwen2.5-coder",
        "mistral",
        "mistral-nemo",
        "mistral-large",
        "command-r",
        "command-r-plus",
        "hermes3",
    }

    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        self.base_url = config.base_url
        self._session = requests.Session()

    @property
    def api_url(self) -> str:
        """Get the Ollama generate API endpoint URL."""
        return f"{self.base_url}/api/generate"

    @property
    def chat_url(self) -> str:
        """Get the Ollama chat API endpoint URL."""
        return f"{self.base_url}/api/chat"

    @property
    def supports_vision(self) -> bool:
        """Check if the current model supports vision."""
        model_name = self.model.lower().split(":")[0]
        # Check exact match first
        if model_name in self.VISION_MODELS:
            return True
        # Check for vision keywords in model name
        return any(kw in model_name for kw in self.VISION_KEYWORDS)

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tool use."""
        model_name = self.model.lower().split(":")[0]
        # Check if any known tool model is a prefix
        return any(model_name.startswith(tm) for tm in self.TOOL_MODELS)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion using Ollama.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, images, etc.)

        Returns:
            The generated completion text

        Raises:
            ConnectionError: If Ollama is not running
            RuntimeError: If the API request fails
        """
        # Check if we're doing a vision request
        images_provided = "images" in kwargs and kwargs["images"]
        has_images = images_provided and self.supports_vision

        # Debug logging (only when STASH_COPILOT_DEBUG=1)
        if images_provided:
            from ..model_caps import is_debug_logging_enabled

            if is_debug_logging_enabled():
                num_images = len(kwargs["images"])
                stash_log.debug(
                    f"[Ollama] Model: {self.model}, Vision support: {self.supports_vision}, Images provided: {num_images}, Images sent: {num_images if has_images else 0}"
                )

        # Use chat endpoint for vision requests (required by many newer models like llama3.2-vision)
        if has_images:
            return self._complete_vision(prompt, **kwargs)

        # Use chat endpoint when system prompt provided
        # The /api/generate endpoint doesn't handle system + options properly (returns 400)
        if "system" in kwargs:
            return self._complete_with_system(prompt, **kwargs)

        # Text-only without system prompt: use generate endpoint with raw mode
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,  # Safe since no system prompt - bypasses templating
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            response = self._session.post(
                self.api_url,
                json=payload,
                timeout=120,  # LLM generation can take time
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            return str(result.get("response", ""))

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError("Ollama request timed out. Try a smaller prompt or model.") from e
        except requests.exceptions.HTTPError as e:
            # Try to get Ollama's error message from response body
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", "")
            except (json.JSONDecodeError, ValueError):
                error_detail = response.text[:200] if response.text else ""

            if error_detail:
                raise RuntimeError(
                    f"Ollama API error for model '{self.model}': {error_detail}"
                ) from e
            raise RuntimeError(f"Ollama API error for model '{self.model}': {e}") from e

    def _complete_vision(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a vision completion using Ollama's chat endpoint.

        Many newer vision models (llama3.2-vision, gemma2-vision, etc.) require
        the chat endpoint instead of generate for multimodal requests.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, images, etc.)

        Returns:
            The generated completion text
        """
        images = kwargs.get("images", [])
        stash_log.debug(f"Sending {len(images)} images to Ollama VLM via chat endpoint")

        # Build messages for chat endpoint
        messages: list[dict[str, Any]] = []

        # Add system message if provided
        if "system" in kwargs:
            messages.append(
                {
                    "role": "system",
                    "content": kwargs["system"],
                }
            )

        # Add user message with images
        user_message: dict[str, Any] = {
            "role": "user",
            "content": prompt,
            "images": images,
        }
        messages.append(user_message)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            response = self._session.post(
                self.chat_url,
                json=payload,
                timeout=300,  # Vision requests can take longer
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            message: dict[str, Any] = result.get("message", {})
            return str(message.get("content", ""))

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Ollama vision request timed out. Try fewer images or a smaller model."
            ) from e
        except requests.exceptions.HTTPError as e:
            # Try to get Ollama's error message from response body
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", "")
            except (json.JSONDecodeError, ValueError):
                error_detail = response.text[:200] if response.text else ""

            if error_detail:
                raise RuntimeError(
                    f"Ollama API error for model '{self.model}': {error_detail}"
                ) from e
            raise RuntimeError(f"Ollama API error for model '{self.model}': {e}") from e

    def _complete_with_system(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a text completion using Ollama's chat endpoint with system prompt.

        The /api/generate endpoint doesn't handle system + options properly,
        so we use the chat endpoint when a system prompt is provided.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, temperature, max_tokens)

        Returns:
            The generated completion text
        """
        messages: list[dict[str, Any]] = []

        # Add system message
        if "system" in kwargs:
            messages.append(
                {
                    "role": "system",
                    "content": kwargs["system"],
                }
            )

        # Add user message
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            response = self._session.post(
                self.chat_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            message: dict[str, Any] = result.get("message", {})
            return str(message.get("content", ""))

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError("Ollama request timed out. Try a smaller prompt or model.") from e
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", "")
            except (json.JSONDecodeError, ValueError):
                error_detail = response.text[:200] if response.text else ""

            if error_detail:
                raise RuntimeError(
                    f"Ollama API error for model '{self.model}': {error_detail}"
                ) from e
            raise RuntimeError(f"Ollama API error for model '{self.model}': {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens from Ollama.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Generated tokens as they become available
        """
        # Check if we're doing a vision request
        has_images = "images" in kwargs and self.supports_vision

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        # Add system prompt if provided (incompatible with raw mode)
        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        elif not has_images:
            # Only use raw mode when no system prompt and no images
            payload["raw"] = True

        if has_images:
            payload["images"] = kwargs["images"]

        try:
            response = self._session.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Chat completion with optional tool use via Ollama's chat API.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool schemas
            **kwargs: Additional parameters (images for vision models)

        Returns:
            CompletionResult with content and/or tool calls
        """
        # Check for images (for vision+tools models like llama3.2-vision)
        images = kwargs.get("images")
        has_images = images and self.supports_vision

        # Convert messages to Ollama format, optionally adding images
        ollama_messages = self._convert_messages(messages, images=images if has_images else None)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        # Add tools if provided and model supports them
        if tools and self.supports_tools:
            payload["tools"] = self._convert_tools(tools)

        # Debug logging for vision+tools
        if has_images and images is not None:
            stash_log.debug(
                f"[Ollama] chat() with {len(images)} images and {len(tools) if tools else 0} tools"
            )

        try:
            response = self._session.post(
                self.chat_url,
                json=payload,
                timeout=180,  # Tool use may take longer
            )
            response.raise_for_status()

            result = response.json()
            message = result.get("message", {})

            # Check for tool calls
            tool_calls: list[ToolCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    raw_args = func.get("arguments", {})
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            stash_log.warning(
                                f"[Ollama] Failed to parse tool arguments for "
                                f"{func.get('name')}: {raw_args!r}"
                            )
                            args = {}
                    else:
                        stash_log.warning(
                            f"[Ollama] Unexpected argument type "
                            f"{type(raw_args).__name__} for {func.get('name')}: {raw_args!r}"
                        )
                        args = {}
                    tool_calls.append(
                        {
                            "id": str(uuid.uuid4()),
                            "name": func.get("name", ""),
                            "arguments": args,
                        }
                    )

            content = message.get("content")
            finish_reason = "tool_calls" if tool_calls else "stop"

            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError("Ollama chat request timed out.") from e
        except requests.exceptions.HTTPError as e:
            # Try to get Ollama's error message from response body
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", "")
            except (json.JSONDecodeError, ValueError):
                error_detail = response.text[:200] if response.text else ""

            if error_detail:
                raise RuntimeError(
                    f"Ollama API error for model '{self.model}': {error_detail}"
                ) from e
            raise RuntimeError(f"Ollama API error for model '{self.model}': {e}") from e

    def _convert_messages(
        self,
        messages: list[Message],
        images: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert messages to Ollama's chat format.

        Args:
            messages: List of Message dicts
            images: Optional list of base64-encoded images (added to first user message)

        Returns:
            List of Ollama-formatted message dicts
        """
        ollama_messages: list[dict[str, Any]] = []
        images_added = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            ollama_msg: dict[str, Any] = {"role": role}

            if content:
                ollama_msg["content"] = content

            # Add images to the first user message
            if role == "user" and images and not images_added:
                ollama_msg["images"] = images
                images_added = True

            # Handle tool calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                ollama_msg["tool_calls"] = [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                    }
                    for tc in tool_calls
                ]

            # Handle tool responses
            if role == "tool":
                ollama_msg["role"] = "tool"
                ollama_msg["content"] = content

            ollama_messages.append(ollama_msg)

        return ollama_messages

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert tool schemas to Ollama's format.

        Args:
            tools: List of tool schemas

        Returns:
            List of Ollama-formatted tool definitions
        """
        ollama_tools: list[dict[str, Any]] = []

        for tool in tools:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )

        return ollama_tools

    def list_models(self) -> list[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError):
            return []

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                return False

            # Check if our model is available
            models = self.list_models()
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False
