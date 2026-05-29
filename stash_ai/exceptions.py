"""Custom exceptions for Stash Copilot.

This module provides a typed exception hierarchy for better error handling
and debugging. All exceptions inherit from StashCopilotError.

Exception Hierarchy:
    StashCopilotError (base)
    ├── LLMError (LLM provider issues)
    │   └── VisionError (vision analysis specific)
    ├── EmbeddingError (embedding generation/storage)
    ├── StorageError (database operations)
    ├── ConfigurationError (invalid settings)
    └── TaskError (task execution failures)
"""

from typing import Optional


class StashCopilotError(Exception):
    """Base exception for all plugin errors.

    All custom exceptions in the plugin should inherit from this class
    to allow catching all plugin-specific errors with a single except clause.
    """

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional additional context (e.g., stack trace, debug info)
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class LLMError(StashCopilotError):
    """LLM provider errors (network, API, parsing).

    Raised when communication with an LLM provider fails, including:
    - Network connectivity issues
    - API authentication failures
    - Rate limiting
    - Invalid response format
    - Model not found
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the LLM error.

        Args:
            message: Human-readable error message
            provider: Name of the LLM provider (e.g., "ollama", "openrouter")
            model: Model name that was being used
            details: Optional additional context
        """
        super().__init__(message, details)
        self.provider = provider
        self.model = model

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class VisionError(LLMError):
    """Vision analysis specific errors.

    Raised when vision analysis fails, including:
    - Image encoding failures
    - Model doesn't support vision
    - Image too large
    - Invalid image format
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        image_count: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the vision error.

        Args:
            message: Human-readable error message
            provider: Name of the LLM provider
            model: Model name that was being used
            image_count: Number of images that were being processed
            details: Optional additional context
        """
        super().__init__(message, provider, model, details)
        self.image_count = image_count


class EmbeddingError(StashCopilotError):
    """Embedding generation or storage errors.

    Raised when embedding operations fail, including:
    - Model loading failures
    - Image preprocessing errors
    - Dimension mismatches
    - Storage write failures
    """

    def __init__(
        self,
        message: str,
        model_key: Optional[str] = None,
        scene_id: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the embedding error.

        Args:
            message: Human-readable error message
            model_key: Embedding model identifier
            scene_id: Scene ID being processed (if applicable)
            details: Optional additional context
        """
        super().__init__(message, details)
        self.model_key = model_key
        self.scene_id = scene_id


class StorageError(StashCopilotError):
    """Database operation errors.

    Raised when database operations fail, including:
    - Connection failures
    - Schema migration errors
    - Query execution failures
    - Constraint violations
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the storage error.

        Args:
            message: Human-readable error message
            operation: Database operation that failed (e.g., "INSERT", "SELECT")
            table: Table name involved (if applicable)
            details: Optional additional context
        """
        super().__init__(message, details)
        self.operation = operation
        self.table = table


class ConfigurationError(StashCopilotError):
    """Invalid configuration or missing settings.

    Raised when configuration is invalid, including:
    - Missing required settings
    - Invalid setting values
    - Incompatible setting combinations
    - Environment variable issues
    """

    def __init__(
        self,
        message: str,
        setting: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the configuration error.

        Args:
            message: Human-readable error message
            setting: Name of the problematic setting
            expected: Expected value or format
            actual: Actual value received
            details: Optional additional context
        """
        super().__init__(message, details)
        self.setting = setting
        self.expected = expected
        self.actual = actual


class EroScriptsError(StashCopilotError):
    """EroScripts integration errors (auth, search, download).

    Raised when an operation against discuss.eroscripts.com cannot complete,
    including:
    - Missing or expired session cookie
    - Network failures
    - HTTP 4xx/5xx responses
    - Malformed search/topic responses
    - Download collisions or content sanity failures
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        operation: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the eroscripts error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code (if from an HTTP response)
            operation: Operation that failed (e.g., "search", "download")
            details: Optional additional context
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.operation = operation


class TaskError(StashCopilotError):
    """Task execution errors.

    Raised when a plugin task fails, including:
    - Task initialization failures
    - Processing errors
    - Timeout errors
    - Dependency failures
    """

    def __init__(
        self,
        message: str,
        task_name: Optional[str] = None,
        scene_id: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        """Initialize the task error.

        Args:
            message: Human-readable error message
            task_name: Name of the task that failed
            scene_id: Scene ID being processed (if applicable)
            details: Optional additional context
        """
        super().__init__(message, details)
        self.task_name = task_name
        self.scene_id = scene_id


__all__ = [
    "StashCopilotError",
    "LLMError",
    "VisionError",
    "EmbeddingError",
    "StorageError",
    "ConfigurationError",
    "EroScriptsError",
    "TaskError",
]
