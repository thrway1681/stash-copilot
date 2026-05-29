"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, TypedDict

from ..config import LLMConfig


class ToolCall(TypedDict):
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(TypedDict, total=False):
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | None
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None  # For tool response messages


class CompletionResult(TypedDict):
    """Result from a completion with potential tool calls."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length"


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    # Whether this is a hosted (cloud) provider with cost/rate concerns
    # Override in subclasses for hosted providers (OpenAI, Anthropic, etc.)
    HOSTED: bool = False

    def __init__(self, config: LLMConfig):
        """
        Initialize the provider with configuration.

        Args:
            config: LLM configuration settings
        """
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated completion text
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Stream completion tokens for the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Yields:
            Generated tokens as they become available
        """
        pass

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
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResult with content and/or tool calls
        """
        # Default implementation: convert to simple prompt
        # Providers should override this for proper tool support
        prompt = self._messages_to_prompt(messages)
        content = self.complete(prompt, **kwargs)
        return {
            "content": content,
            "tool_calls": [],
            "finish_reason": "stop",
        }

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """
        Convert messages to a simple prompt string.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "tool":
                    parts.append(f"Tool result: {content}")
        return "\n\n".join(parts)

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """
        Whether this provider/model supports image input.

        Returns:
            True if vision is supported, False otherwise
        """
        pass

    @property
    def supports_tools(self) -> bool:
        """
        Whether this provider/model supports tool use.

        Returns:
            True if tools are supported, False otherwise
        """
        return False

    @property
    def is_hosted(self) -> bool:
        """
        Whether this is a hosted (cloud) provider with cost/rate concerns.

        Hosted providers may have per-token/per-image pricing, so the UI
        should warn users before sending many images.

        Returns:
            True if this is a hosted provider, False for local providers
        """
        return self.__class__.HOSTED

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    def health_check(self) -> bool:
        """
        Check if the provider is available and working.

        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            # Try a simple completion to verify connectivity
            result = self.complete("Say 'ok' and nothing else.")
            return len(result) > 0
        except Exception:
            return False
