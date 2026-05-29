"""AI tools for interacting with Stash."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from .base import BaseTool, ToolParameter, ToolResult
from .content_detection import (
    CONTENT_DETECTORS,
    ContentDetector,
    FindContentTool,
    get_available_detectors,
)
from .database import (
    EnrichSceneResultsTool,
    QueryAllPerformersTool,
    # List all tools
    QueryAllTagsTool,
    # Phase 4 new tools
    QueryDuplicatesFindingTool,
    QueryFavoritesTool,
    QueryGroupProgressTool,
    QueryInteractiveContentTool,
    QueryLibraryStatsTool,
    QueryOHistoryTool,
    QueryPerformerCareerTimelineTool,
    QueryPerformerComparisonTool,
    QueryPerformerPairsTool,
    # Phase 2 new tools
    QueryPerformerProfileTool,
    # Phase 1 new tools
    QueryPerformersByAttributeTool,
    QueryPerformerTagsTool,
    QueryResumePointsTool,
    QuerySceneMarkersTool,
    QueryScenesByDateTool,
    # Compositional filtering tools
    QueryScenesByPerformerTool,
    QueryScenesByRatingTool,
    QueryScenesByTagTool,
    QueryStorageStatsTool,
    QueryStudioHierarchyTool,
    QueryStudioProfileTool,
    QueryTagCorrelationsTool,
    # Phase 3 new tools
    QueryTagHierarchyTool,
    QueryTagPerformersTool,
    QueryTagUsageOverTimeTool,
    QueryTopPerformerCommonTagsTool,
    QueryTopPerformersTool,
    QueryTopStudiosTool,
    QueryTopTagsTool,
    QueryUnwatchedContentTool,
    QueryViewingHistoryTool,
    QueryViewingStatsTool,
    QueryWatchingPatternsTool,
    RankScenesByEngagementTool,
)
from .embeddings import (
    FilterScenesByVisualContentTool,
    GetEmbeddingStatsTool,
    QuerySimilarScenesTool,
    SearchByTextTool,
)
from .vision import (
    FindSimilarFramesTool,
    GetFrameTimestampTool,
)

if TYPE_CHECKING:
    from ..embeddings.config import EmbeddingConfig
    from ..stash_client import StashClient

# Registry of available tools (basic tools that only need stash)
_TOOL_CLASSES: list[type[BaseTool]] = [
    # Original tools
    QueryPerformerTagsTool,
    QueryTagPerformersTool,
    QueryViewingStatsTool,
    # Core ranking tools
    QueryTopPerformersTool,
    QueryTopTagsTool,
    QueryTopStudiosTool,
    # Library statistics
    QueryLibraryStatsTool,
    # Analytics tools
    QueryWatchingPatternsTool,
    QueryTagCorrelationsTool,
    QueryTopPerformerCommonTagsTool,
    QueryPerformerPairsTool,
    # Specialized content tools
    QueryInteractiveContentTool,
    QueryUnwatchedContentTool,
    # Engagement ranking
    RankScenesByEngagementTool,
    # Phase 1 new tools: Performer, Date, Favorites, Resume, Rating
    QueryPerformersByAttributeTool,
    QueryScenesByDateTool,
    QueryFavoritesTool,
    QueryResumePointsTool,
    QueryScenesByRatingTool,
    # List all tools
    QueryAllTagsTool,
    QueryAllPerformersTool,
    # Phase 2 new tools
    QueryPerformerProfileTool,
    QueryStudioProfileTool,
    QueryGroupProgressTool,
    QueryViewingHistoryTool,
    QueryStorageStatsTool,
    # Phase 3 new tools
    QueryTagHierarchyTool,
    QueryStudioHierarchyTool,
    QuerySceneMarkersTool,
    QueryTagUsageOverTimeTool,
    QueryPerformerComparisonTool,
    # Phase 4 new tools
    QueryDuplicatesFindingTool,
    QueryOHistoryTool,
    QueryPerformerCareerTimelineTool,
    # Compositional filtering tools
    QueryScenesByPerformerTool,
    QueryScenesByTagTool,
    EnrichSceneResultsTool,
]


def get_all_tools(
    stash: "StashClient",
    embedding_config: Optional["EmbeddingConfig"] = None,
    excluded_tags: list[str] | None = None,
) -> list[BaseTool]:
    """
    Get instances of all registered tools.

    Args:
        stash: StashClient for database access
        embedding_config: Optional config for embedding-based tools (provides model_key)
        excluded_tags: Optional list of tag names to exclude from all tool results

    Returns:
        List of tool instances
    """
    tools: list[BaseTool] = [tool_cls(stash) for tool_cls in _TOOL_CLASSES]

    # Add embedding tools - they accept optional config for model_key scoping
    # QuerySimilarScenesTool works with or without config (defaults to siglip)
    tools.append(QuerySimilarScenesTool(stash, embedding_config))

    # GetEmbeddingStatsTool shows stats for current model (or siglip if no config)
    tools.append(GetEmbeddingStatsTool(stash, embedding_config))

    # Add SearchByTextTool only if embedding config is provided
    # (requires config to know which embedder to use for text queries)
    if embedding_config is not None:
        tools.append(SearchByTextTool(stash, embedding_config))
        tools.append(FilterScenesByVisualContentTool(stash, embedding_config))

    # Set excluded tags on all tools if provided
    if excluded_tags:
        for tool in tools:
            tool.set_excluded_tags(excluded_tags)

    return tools


def get_tools_schema(
    stash: "StashClient",
    embedding_config: Optional["EmbeddingConfig"] = None,
) -> list[dict[str, Any]]:
    """
    Get schemas for all tools (for LLM tool use).

    Args:
        stash: StashClient for database access
        embedding_config: Optional config for embedding-based tools

    Returns:
        List of tool schemas
    """
    return [tool.to_schema() for tool in get_all_tools(stash, embedding_config)]


__all__ = [
    # Base classes
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    # Original tools
    "QueryPerformerTagsTool",
    "QueryTagPerformersTool",
    "QueryViewingStatsTool",
    # Core ranking tools
    "QueryTopPerformersTool",
    "QueryTopTagsTool",
    "QueryTopStudiosTool",
    # Library statistics
    "QueryLibraryStatsTool",
    # Analytics tools
    "QueryWatchingPatternsTool",
    "QueryTagCorrelationsTool",
    "QueryTopPerformerCommonTagsTool",
    "QueryPerformerPairsTool",
    # Specialized content tools
    "QueryInteractiveContentTool",
    "QueryUnwatchedContentTool",
    # Engagement ranking
    "RankScenesByEngagementTool",
    # Phase 1 new tools
    "QueryPerformersByAttributeTool",
    "QueryScenesByDateTool",
    "QueryFavoritesTool",
    "QueryResumePointsTool",
    "QueryScenesByRatingTool",
    # List all tools
    "QueryAllTagsTool",
    "QueryAllPerformersTool",
    # Phase 2 new tools
    "QueryPerformerProfileTool",
    "QueryStudioProfileTool",
    "QueryGroupProgressTool",
    "QueryViewingHistoryTool",
    "QueryStorageStatsTool",
    # Phase 3 new tools
    "QueryTagHierarchyTool",
    "QueryStudioHierarchyTool",
    "QuerySceneMarkersTool",
    "QueryTagUsageOverTimeTool",
    "QueryPerformerComparisonTool",
    # Phase 4 new tools
    "QueryDuplicatesFindingTool",
    "QueryOHistoryTool",
    "QueryPerformerCareerTimelineTool",
    # Compositional filtering tools
    "QueryScenesByPerformerTool",
    "QueryScenesByTagTool",
    "EnrichSceneResultsTool",
    # Embedding tools
    "QuerySimilarScenesTool",
    "SearchByTextTool",
    "GetEmbeddingStatsTool",
    "FilterScenesByVisualContentTool",
    # Vision tools (for scene analysis)
    "GetFrameTimestampTool",
    "FindSimilarFramesTool",
    # Content detection tools
    "ContentDetector",
    "CONTENT_DETECTORS",
    "FindContentTool",
    "get_available_detectors",
    # Factory functions
    "get_all_tools",
    "get_tools_schema",
]
