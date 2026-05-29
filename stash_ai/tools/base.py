"""Base class for AI tools."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ..stash_client import StashClient


class ToolParameter(TypedDict):
    """Schema for a tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", "array"
    description: str
    required: bool
    enum: list[str] | None  # Optional list of allowed values


class ToolResult(TypedDict):
    """Result from a tool execution."""

    success: bool
    data: Any
    error: str | None


class BaseTool(ABC):
    """
    Abstract base class for AI-usable tools.

    Tools provide capabilities that LLMs can invoke to interact
    with the Stash database or perform other actions.
    """

    def __init__(self, stash: "StashClient"):
        """
        Initialize the tool with a Stash interface.

        Args:
            stash: StashClient instance for API calls
        """
        self.stash = stash
        # Plugin-level excluded tags (set by get_all_tools)
        self._excluded_tags: list[str] = []

    def set_excluded_tags(self, excluded_tags: list[str]) -> None:
        """
        Set the plugin-level excluded tags.

        These tags (and their children) will be automatically excluded
        from tag-related results. Set by get_all_tools() from plugin config.

        Args:
            excluded_tags: List of tag names to exclude
        """
        self._excluded_tags = [t.lower() for t in excluded_tags]

    def get_excluded_tags(self) -> list[str]:
        """Get the plugin-level excluded tags (lowercase)."""
        return self._excluded_tags

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this tool.

        Returns:
            Tool name (e.g., "query_performer_tags")
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.

        This is shown to the LLM to help it decide when to use the tool.

        Returns:
            Tool description
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """
        Schema for the tool's input parameters.

        Returns:
            List of parameter definitions
        """
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and data or error
        """
        pass

    def to_schema(self) -> dict[str, Any]:
        """
        Convert tool to a schema suitable for LLM tool use.

        Returns:
            Dictionary with name, description, and parameters schema
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param["type"],
                "description": param["description"],
            }
            if param.get("enum"):
                prop["enum"] = param["enum"]

            properties[param["name"]] = prop

            if param.get("required", False):
                required.append(param["name"])

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
