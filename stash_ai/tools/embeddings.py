"""Embedding-based query tools for similarity search."""

import os
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np

from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from ..embeddings.storage import EmbeddingStorage, SimilarityResult
from .base import BaseTool, ToolParameter, ToolResult
from .database import get_readonly_connection, get_stash_db_path

if TYPE_CHECKING:
    from ..stash_client import StashClient


def escape_markdown(text: str) -> str:
    """
    Escape special markdown characters in text.

    Escapes underscores to prevent them from being interpreted as
    italic formatting when the text is used in markdown links.

    Args:
        text: Text that may contain markdown special characters

    Returns:
        Text with special characters escaped
    """
    # Escape underscores to prevent italic formatting
    # e.g., "file_name_here.mp4" -> "file\_name\_here.mp4"
    return text.replace("_", r"\_")


def get_scene_display_name(
    cursor: sqlite3.Cursor,
    scene_id: int,
    title: str | None,
    performers: list[str],
) -> str:
    """
    Get a human-readable display name for a scene.

    Priority:
    1. Scene title (if set)
    2. "Performer Name(s) - filename" (if performers exist)
    3. Just filename
    4. Fallback to "Scene {id}"

    Args:
        cursor: SQLite cursor for querying file info
        scene_id: The scene ID
        title: The scene title (may be None or empty)
        performers: List of performer names

    Returns:
        Human-readable display name for the scene
    """
    # If title is set, use it
    if title:
        return title

    # Get the filename from the files table
    cursor.execute(
        """
        SELECT f.basename
        FROM files f
        JOIN scenes_files sf ON f.id = sf.file_id
        WHERE sf.scene_id = ? AND sf."primary" = 1
        LIMIT 1
        """,
        (scene_id,),
    )
    file_row = cursor.fetchone()
    filename = file_row["basename"] if file_row else None

    # Build display name
    if performers and filename:
        performer_str = ", ".join(performers[:2])
        if len(performers) > 2:
            performer_str += f" +{len(performers) - 2}"
        return f"{performer_str} - {filename}"
    elif filename is not None:
        return str(filename)
    else:
        return f"Scene {scene_id}"


def get_scene_url(scene_id: int) -> str:
    """
    Get the Stash URL for a scene.

    Args:
        scene_id: The scene ID

    Returns:
        URL path to the scene (relative, works with any Stash host)
    """
    return f"/scenes/{scene_id}"


class QuerySimilarScenesTool(BaseTool):
    """
    Tool to find scenes similar to a given scene using embeddings.

    Requires embeddings to be pre-computed via the embed_scenes task.
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Initialize the similar scenes tool.

        Args:
            stash: StashClient instance
            embedding_config: Optional config for text queries (provides model_key)
        """
        super().__init__(stash)
        # Use model_key from config to query the correct embedding namespace
        self.model_key = embedding_config.model_key if embedding_config else "siglip"
        self.storage = EmbeddingStorage(model_key=self.model_key)
        self.embedding_config = embedding_config

    @property
    def name(self) -> str:
        return "query_similar_scenes"

    @property
    def description(self) -> str:
        return (
            "Find scenes similar to a given scene based on visual and metadata "
            "embeddings. Returns scenes ranked by similarity score. "
            "Results include 'name' (scene display name), 'url' (link to scene), "
            "and engagement data (view_count, o_count, engagement_score). "
            "Use engagement_score to identify 'favorites' - higher score means more engaged. "
            "When presenting results, use markdown links: [scene name](url)"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "scene_id",
                "type": "integer",
                "description": "The scene ID to find similar scenes for",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of similar scenes (default: 10)",
                "required": False,
                "enum": None,
            },
            {
                "name": "min_similarity",
                "type": "number",
                "description": "Minimum similarity score 0-1 (default: 0.5)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute similarity search."""
        scene_id: int = kwargs.get("scene_id", 0)
        limit: int = kwargs.get("limit", 10)
        min_similarity: float = kwargs.get("min_similarity", 0.5)

        if not scene_id:
            return {
                "success": False,
                "data": None,
                "error": "scene_id is required",
            }

        # Get the query scene's embedding
        query_record = self.storage.get_embedding(scene_id)
        if not query_record:
            return {
                "success": False,
                "data": None,
                "error": f"Scene {scene_id} has no embedding. Run embed_scenes task first.",
            }

        # Find similar scenes
        results = self.storage.find_similar(
            query_embedding=query_record["composite_embedding"],
            limit=limit + 1,  # +1 to exclude self
            exclude_scene_ids=[scene_id],
            min_similarity=min_similarity,
        )

        # Enrich with scene metadata from Stash DB
        enriched_results = self._enrich_results(results[:limit])

        # Build formatted results string for easy display
        if enriched_results:
            formatted_lines = [f"{i + 1}. {r['formatted']}" for i, r in enumerate(enriched_results)]
            formatted_results = "\n".join(formatted_lines)
        else:
            formatted_results = "No similar scenes found."

        return {
            "success": True,
            "data": {
                "query_scene_id": scene_id,
                "model_key": self.model_key,
                "similar_scenes": enriched_results,
                "count": len(enriched_results),
                "formatted_results": formatted_results,
            },
            "error": None,
        }

    def _enrich_results(
        self,
        results: list[SimilarityResult],
    ) -> list[dict[str, Any]]:
        """Add scene metadata and engagement data to similarity results.

        Engagement score uses replay_count instead of raw play_hours to avoid
        duration bias (longer scenes scoring higher unfairly).
        Formula: (o_count * 3) + (replay_count * 2)
        """
        db_path = get_stash_db_path()
        if not db_path.exists():
            return [
                {
                    "scene_id": r.scene_id,
                    "name": f"Scene {r.scene_id}",
                    "url": get_scene_url(r.scene_id),
                    "similarity": round(r.similarity, 4),
                    "view_count": 0,
                    "o_count": 0,
                    "replay_count": 0,
                    "engagement_score": 0.0,
                    "description_preview": (
                        r.visual_description[:200] + "..."
                        if r.visual_description and len(r.visual_description) > 200
                        else r.visual_description
                    ),
                }
                for r in results
            ]

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        enriched: list[dict[str, Any]] = []
        for r in results:
            # Get scene info with engagement data
            cursor.execute(
                """
                SELECT s.id, s.title, st.name as studio,
                       COALESCE(view_agg.view_count, 0) as view_count,
                       COALESCE(o_agg.o_count, 0) as o_count
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                WHERE s.id = ?
            """,
                (r.scene_id,),
            )

            row = cursor.fetchone()
            title = row["title"] if row else None
            view_count = row["view_count"] if row else 0
            o_count = row["o_count"] if row else 0

            # Calculate engagement score without duration bias
            # replay_count = views beyond the first one
            replay_count = max(view_count - 1, 0)
            engagement_score = (o_count * 20.0) + (replay_count * 2.0)

            # Get performers
            cursor.execute(
                """
                SELECT p.name FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                WHERE ps.scene_id = ?
            """,
                (r.scene_id,),
            )
            performers = [pr["name"] for pr in cursor.fetchall()]

            # Get display name (title, performer+filename, filename, or fallback)
            display_name = get_scene_display_name(cursor, r.scene_id, title, performers)
            url = get_scene_url(r.scene_id)
            similarity = round(r.similarity, 4)
            studio = row["studio"] if row else None

            # Build pre-formatted display text for consistent LLM output
            # Format: [Name](url) - Performers | Studio (score: X.XX, engagement: Y.Y)
            # Escape underscores in display name to prevent markdown italic formatting
            escaped_name = escape_markdown(display_name)
            parts = []
            if performers:
                parts.append(", ".join(escape_markdown(p) for p in performers))
            if studio:
                parts.append(escape_markdown(studio))
            meta = " | ".join(parts) if parts else ""

            if meta:
                formatted = f"[{escaped_name}]({url}) - {meta} (score: {similarity}, engagement: {round(engagement_score, 1)})"
            else:
                formatted = f"[{escaped_name}]({url}) (score: {similarity}, engagement: {round(engagement_score, 1)})"

            enriched.append(
                {
                    "scene_id": r.scene_id,
                    "name": display_name,
                    "url": url,
                    "similarity": similarity,
                    "studio": studio,
                    "performers": performers,
                    "view_count": view_count,
                    "o_count": o_count,
                    "replay_count": replay_count,
                    "engagement_score": round(engagement_score, 2),
                    "formatted": formatted,
                    "description_preview": (
                        r.visual_description[:200] + "..."
                        if r.visual_description and len(r.visual_description) > 200
                        else r.visual_description
                    ),
                }
            )

        conn.close()
        return enriched


class SearchByTextTool(BaseTool):
    """
    Tool to search scenes by natural language text query.

    Embeds the query text and finds scenes with similar embeddings.
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig,
    ) -> None:
        """
        Initialize the text search tool.

        Args:
            stash: StashClient instance
            embedding_config: Config for text embedding (also provides model_key)
        """
        super().__init__(stash)
        # Use model_key from config to search the correct embedding namespace
        self.model_key = embedding_config.model_key
        self.storage = EmbeddingStorage(model_key=self.model_key)
        self.embedding_config = embedding_config
        self._embedder: Any | None = None

    @property
    def embedder(self) -> Any:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.embedding_config)
        return self._embedder

    @property
    def name(self) -> str:
        return "search_scenes_by_text"

    @property
    def description(self) -> str:
        return (
            "Search for scenes using VISUAL DESCRIPTIONS ONLY. "
            "Finds scenes based on what appears in the video (actions, clothing, setting, positions, etc.). "
            "IMPORTANT: This tool uses visual embeddings and CANNOT search by performer names, "
            "studio names, or other metadata. For performer-based queries, use query_all_performers "
            "or query_performer_profile instead. "
            "Good queries: 'blonde in red lingerie', 'outdoor pool scene', 'POV angle'. "
            "Bad queries: 'Mia Malkova', 'Brazzers', performer names. "
            "Results include 'name' (scene display name), 'url' (link to scene), "
            "and engagement data (view_count, o_count, engagement_score). "
            "Use engagement_score to identify 'favorites' - higher score means more engaged. "
            "When presenting results, use markdown links: [scene name](url)"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "query",
                "type": "string",
                "description": "Natural language description of scenes to find",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute text-based scene search."""
        query: str = kwargs.get("query", "")
        limit: int = kwargs.get("limit", 10)

        if not query:
            return {
                "success": False,
                "data": None,
                "error": "query is required",
            }

        # Check if we have any embeddings
        stats = self.storage.get_stats()
        if stats["total_embeddings"] == 0:
            return {
                "success": False,
                "data": None,
                "error": "No scene embeddings found. Run embed_scenes task first.",
            }

        # Embed the query
        try:
            result = self.embedder.embed_text(query)
            query_embedding = result["embedding"]
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to embed query: {e!s}",
            }

        # Find similar scenes
        # Note: SigLIP text-to-image similarity scores are typically 0.01-0.10,
        # much lower than image-to-image scores. Use no threshold and let
        # the results speak for themselves based on relative ranking.
        results = self.storage.find_similar(
            query_embedding=query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        # Enrich with scene metadata
        enriched = self._enrich_results(results)

        # Build formatted results string for easy display
        if enriched:
            formatted_lines = [f"{i + 1}. {r['formatted']}" for i, r in enumerate(enriched)]
            formatted_results = "\n".join(formatted_lines)
        else:
            formatted_results = "No matching scenes found."

        return {
            "success": True,
            "data": {
                "query": query,
                "model_key": self.model_key,
                "results": enriched,
                "count": len(enriched),
                "formatted_results": formatted_results,
            },
            "error": None,
        }

    def _enrich_results(self, results: list[SimilarityResult]) -> list[dict[str, Any]]:
        """Add scene metadata and engagement data to results.

        Engagement score uses replay_count instead of raw play_hours to avoid
        duration bias (longer scenes scoring higher unfairly).
        Formula: (o_count * 3) + (replay_count * 2)
        """
        db_path = get_stash_db_path()
        if not db_path.exists():
            return [
                {
                    "scene_id": r.scene_id,
                    "name": f"Scene {r.scene_id}",
                    "url": get_scene_url(r.scene_id),
                    "similarity": round(r.similarity, 4),
                    "view_count": 0,
                    "o_count": 0,
                    "replay_count": 0,
                    "engagement_score": 0.0,
                }
                for r in results
            ]

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        enriched: list[dict[str, Any]] = []
        for r in results:
            # Get scene info with engagement data
            cursor.execute(
                """
                SELECT s.id, s.title, st.name as studio,
                       COALESCE(view_agg.view_count, 0) as view_count,
                       COALESCE(o_agg.o_count, 0) as o_count
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                WHERE s.id = ?
            """,
                (r.scene_id,),
            )

            row = cursor.fetchone()
            title = row["title"] if row else None
            view_count = row["view_count"] if row else 0
            o_count = row["o_count"] if row else 0

            # Calculate engagement score without duration bias
            # replay_count = views beyond the first one
            replay_count = max(view_count - 1, 0)
            engagement_score = (o_count * 20.0) + (replay_count * 2.0)

            cursor.execute(
                """
                SELECT p.name FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                WHERE ps.scene_id = ?
            """,
                (r.scene_id,),
            )
            performers = [pr["name"] for pr in cursor.fetchall()]

            # Get display name (title, performer+filename, filename, or fallback)
            display_name = get_scene_display_name(cursor, r.scene_id, title, performers)
            url = get_scene_url(r.scene_id)
            similarity = round(r.similarity, 4)
            studio = row["studio"] if row else None

            # Build pre-formatted display text for consistent LLM output
            # Format: [Name](url) - Performers | Studio (score: X.XX, engagement: Y.Y)
            # Escape underscores in display name to prevent markdown italic formatting
            escaped_name = escape_markdown(display_name)
            parts = []
            if performers:
                parts.append(", ".join(escape_markdown(p) for p in performers))
            if studio:
                parts.append(escape_markdown(studio))
            meta = " | ".join(parts) if parts else ""

            if meta:
                formatted = f"[{escaped_name}]({url}) - {meta} (score: {similarity}, engagement: {round(engagement_score, 1)})"
            else:
                formatted = f"[{escaped_name}]({url}) (score: {similarity}, engagement: {round(engagement_score, 1)})"

            enriched.append(
                {
                    "scene_id": r.scene_id,
                    "name": display_name,
                    "url": url,
                    "similarity": similarity,
                    "studio": studio,
                    "performers": performers,
                    "view_count": view_count,
                    "o_count": o_count,
                    "replay_count": replay_count,
                    "engagement_score": round(engagement_score, 2),
                    "formatted": formatted,
                }
            )

        conn.close()
        return enriched


class GetEmbeddingStatsTool(BaseTool):
    """Tool to get embedding database statistics."""

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Initialize the stats tool.

        Args:
            stash: StashClient instance
            embedding_config: Optional config (provides model_key for scoped stats)
        """
        super().__init__(stash)
        # Use model_key from config if provided, otherwise show siglip stats
        model_key = embedding_config.model_key if embedding_config else "siglip"
        self.storage = EmbeddingStorage(model_key=model_key)

    @property
    def name(self) -> str:
        return "get_embedding_stats"

    @property
    def description(self) -> str:
        return (
            "Get statistics about the scene embedding database. "
            "Shows stats for the current embedding model and all available models."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Get embedding stats."""
        stats = self.storage.get_stats()

        return {
            "success": True,
            "data": stats,
            "error": None,
        }


class FilterScenesByVisualContentTool(BaseTool):
    """
    Tool to filter scenes by visual/semantic content using text-to-image similarity.

    Enables queries like "wearing red lingerie", "outdoor pool setting", "POV camera angle".
    Uses existing OpenCLIP embeddings for fast similarity search.
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig | None = None,
        assets_dir: str | None = None,
    ) -> None:
        """
        Initialize the visual content filter tool.

        Args:
            stash: StashClient instance
            embedding_config: Config for embedding provider (required for text embedding)
            assets_dir: Path to assets directory for frame search index.
                        Derived from plugin root if not provided.
        """
        super().__init__(stash)
        if not embedding_config:
            raise ValueError("FilterScenesByVisualContentTool requires embedding_config")

        self.model_key = embedding_config.model_key
        self.storage = EmbeddingStorage(model_key=self.model_key)
        self.embedding_config = embedding_config

        if assets_dir is None:
            plugin_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            self._assets_dir = os.path.join(plugin_dir, "assets")
        else:
            self._assets_dir = assets_dir

    @property
    def name(self) -> str:
        return "filter_scenes_by_visual_content"

    @property
    def description(self) -> str:
        return (
            "Filter scenes by visual/semantic content description. "
            "Uses text-to-image similarity to find scenes matching the content query. "
            "Frame mode (default): Searches individual video frames via FAISS index for "
            "precise content matching at specific timestamps. Returns best_timestamp showing "
            "WHERE in the video the match was found. Falls back to scene mode if frame index not built. "
            "Scene mode: Uses scene-level composite embeddings (faster but less precise). "
            "Supports ranked scoring (top N matches), threshold filtering (all above threshold), "
            "or returning all matches. "
            "Returns scenes sorted by similarity score. "
            "Useful for finding scenes by clothing, setting, camera angle, actions, etc."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "scene_ids",
                "type": "array",
                "description": "List of scene IDs to filter (from previous tool like query_scenes_by_performer)",
                "required": True,
                "enum": None,
            },
            {
                "name": "content_query",
                "type": "string",
                "description": "Text description of visual content to search for (e.g., 'wearing red lingerie', 'outdoor pool')",
                "required": True,
                "enum": None,
            },
            {
                "name": "scoring_mode",
                "type": "string",
                "description": "Scoring mode: 'ranked' (top N), 'threshold' (all above min_similarity), 'all' (all scenes)",
                "required": False,
                "enum": ["ranked", "threshold", "all"],
            },
            {
                "name": "min_similarity",
                "type": "number",
                "description": "Minimum similarity threshold 0-1 (default: 0.3 for text-to-image matching)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum results to return for ranked mode. Consider input_scene_count and user intent (analysis = more, discovery = fewer).",
                "required": True,
                "enum": None,
            },
            {
                "name": "search_mode",
                "type": "string",
                "description": (
                    "Search granularity: 'frame' (default) searches individual video frames "
                    "via FAISS index for precise content matching at specific timestamps. "
                    "'scene' uses scene-level composite embeddings (faster but less precise). "
                    "Frame mode falls back to scene mode if frame search index not built."
                ),
                "required": False,
                "enum": ["frame", "scene"],
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute visual content filtering."""
        scene_ids: list[int] = kwargs.get("scene_ids", [])
        content_query: str = kwargs.get("content_query", "")
        scoring_mode: str = kwargs.get("scoring_mode", "ranked")
        min_similarity: float = kwargs.get("min_similarity", 0.3)
        limit: int | None = kwargs.get("limit")
        search_mode: str = kwargs.get("search_mode", "frame")

        if not scene_ids:
            return {
                "success": False,
                "data": None,
                "error": "scene_ids is required and must be a non-empty list",
            }

        if scoring_mode == "ranked" and (not limit or limit <= 0):
            return {
                "success": False,
                "data": None,
                "error": "limit parameter is required for ranked mode and must be > 0. Consider: comprehensive analysis = 50-100, discovery = 10-20.",
            }

        if not content_query or not content_query.strip():
            return {
                "success": False,
                "data": None,
                "error": "content_query is required and must be a non-empty string",
            }

        if scoring_mode not in ["ranked", "threshold", "all"]:
            return {
                "success": False,
                "data": None,
                "error": "scoring_mode must be 'ranked', 'threshold', or 'all'",
            }

        if search_mode not in ["frame", "scene"]:
            return {
                "success": False,
                "data": None,
                "error": "search_mode must be 'frame' or 'scene'",
            }

        try:
            # Embed the content query using text embedder
            embedder = get_embedding_provider(self.embedding_config)
            query_result = embedder.embed_text(content_query)
            query_embedding = np.array(query_result["embedding"], dtype=np.float32)

            # Branch by search mode
            if search_mode == "frame":
                return self._execute_frame_search(
                    scene_ids,
                    query_embedding,
                    scoring_mode,
                    min_similarity,
                    limit,
                    content_query,
                )
            else:
                return self._execute_scene_search(
                    scene_ids,
                    query_embedding,
                    scoring_mode,
                    min_similarity,
                    limit,
                )

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Error filtering scenes: {e!s}",
            }

    def _execute_scene_search(
        self,
        scene_ids: list[int],
        query_embedding: "np.ndarray[Any, np.dtype[np.float32]]",
        scoring_mode: str,
        min_similarity: float,
        limit: int | None,
    ) -> ToolResult:
        """Scene-level search using composite embeddings.

        Args:
            scene_ids: Scene IDs to search within
            query_embedding: Text query embedding vector
            scoring_mode: 'ranked', 'threshold', or 'all'
            min_similarity: Minimum similarity threshold
            limit: Max results for ranked mode

        Returns:
            ToolResult with matched scenes
        """
        scored_scenes: list[dict[str, Any]] = []
        scenes_with_embeddings = 0
        for scene_id in scene_ids:
            record = self.storage.get_embedding(scene_id)
            if not record:
                continue
            scenes_with_embeddings += 1

            # Always use visual embedding for content-based filtering
            scene_emb = np.array(record["visual_embedding"])

            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(query_embedding, scene_emb))

            if scoring_mode == "threshold" and similarity < min_similarity:
                continue

            scored_scenes.append(
                {
                    "scene_id": scene_id,
                    "similarity": similarity,
                    "visual_description": record.get("visual_description"),
                }
            )

        scored_scenes.sort(key=lambda x: x["similarity"], reverse=True)

        if scoring_mode == "ranked" and limit:
            scored_scenes = scored_scenes[:limit]

        if not scored_scenes:
            return {
                "success": True,
                "data": {
                    "results": [],
                    "count": 0,
                    "search_mode": "scene",
                    "input_scene_count": len(scene_ids),
                    "scenes_with_embeddings": scenes_with_embeddings,
                },
                "error": None,
            }

        similarity_results = [
            SimilarityResult(
                scene_id=int(s["scene_id"]),
                similarity=float(s["similarity"]),
                visual_description=str(s.get("visual_description"))
                if s.get("visual_description")
                else None,
            )
            for s in scored_scenes
        ]

        enriched = self._enrich_results(similarity_results)

        return {
            "success": True,
            "data": {
                "results": enriched,
                "count": len(enriched),
                "search_mode": "scene",
                "input_scene_count": len(scene_ids),
                "scenes_with_embeddings": scenes_with_embeddings,
            },
            "error": None,
        }

    def _execute_frame_search(
        self,
        scene_ids: list[int],
        query_embedding: "np.ndarray[Any, np.dtype[np.float32]]",
        scoring_mode: str,
        min_similarity: float,
        limit: int | None,
        content_query: str,
    ) -> ToolResult:
        """Frame-level search using FAISS index.

        Searches individual video frames for precise content matching.
        Falls back to scene-level search if frame index is not built.

        Args:
            scene_ids: Scene IDs to search within
            query_embedding: Text query embedding vector
            scoring_mode: 'ranked', 'threshold', or 'all'
            min_similarity: Minimum similarity threshold
            limit: Max results for ranked mode
            content_query: Original text query (for response metadata)

        Returns:
            ToolResult with matched scenes including frame-level info
        """
        from ..embeddings.frame_search import FrameSearchIndex, SceneMatch

        frame_index = FrameSearchIndex(
            assets_dir=self._assets_dir,
            model_key=self.model_key,
        )

        # Graceful fallback if index not built
        if not frame_index.exists:
            result = self._execute_scene_search(
                scene_ids,
                query_embedding,
                scoring_mode,
                min_similarity,
                limit,
            )
            if result["success"] and result["data"]:
                result["data"]["warning"] = (
                    "Frame search index not built. Fell back to scene-level search. "
                    "Run 'Build Frame Search Index' task for frame-level precision."
                )
                result["data"]["search_mode"] = "scene_fallback"
            return result

        # Search with enough coverage to find matches across candidate scenes
        top_k = max(10000, len(scene_ids) * 50)
        frame_matches = frame_index.search(query_embedding, top_k=top_k)

        # Filter to only the requested scene IDs
        scene_id_set = set(scene_ids)
        filtered = [m for m in frame_matches if m.scene_id in scene_id_set]

        # Aggregate to scene-level (best frame per scene)
        scene_matches: list[SceneMatch] = frame_index.aggregate_to_scenes(filtered)

        # Apply scoring mode
        if scoring_mode == "threshold":
            scene_matches = [m for m in scene_matches if m.similarity >= min_similarity]
        if scoring_mode == "ranked" and limit:
            scene_matches = scene_matches[:limit]

        if not scene_matches:
            return {
                "success": True,
                "data": {
                    "results": [],
                    "count": 0,
                    "search_mode": "frame",
                    "input_scene_count": len(scene_ids),
                    "scenes_with_embeddings": len(scene_id_set),
                },
                "error": None,
            }

        # Convert to SimilarityResult for _enrich_results
        similarity_results = [
            SimilarityResult(
                scene_id=m.scene_id,
                similarity=m.similarity,
            )
            for m in scene_matches
        ]

        enriched = self._enrich_results(similarity_results)

        # Post-enrich: add frame-level fields from SceneMatch data
        match_lookup = {m.scene_id: m for m in scene_matches}
        for item in enriched:
            match = match_lookup.get(item["scene_id"])
            if match:
                item["best_frame_index"] = match.best_frame_index
                item["best_timestamp"] = match.best_timestamp
                item["frame_path"] = (
                    f"embedded_frames/scene_{match.scene_id}/frame_{match.best_frame_index:04d}.jpg"
                )
                # Append timestamp to formatted string
                minutes = int(match.best_timestamp) // 60
                seconds = int(match.best_timestamp) % 60
                item["formatted"] += f" [best match at {minutes}:{seconds:02d}]"

        return {
            "success": True,
            "data": {
                "results": enriched,
                "count": len(enriched),
                "search_mode": "frame",
                "input_scene_count": len(scene_ids),
                "scenes_with_embeddings": len(scene_id_set),
            },
            "error": None,
        }

    def _enrich_results(self, results: list[SimilarityResult]) -> list[dict[str, Any]]:
        """Add scene metadata and engagement data to results."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return [
                {
                    "scene_id": r.scene_id,
                    "name": f"Scene {r.scene_id}",
                    "url": get_scene_url(r.scene_id),
                    "similarity": round(r.similarity, 4),
                    "view_count": 0,
                    "o_count": 0,
                    "replay_count": 0,
                    "engagement_score": 0.0,
                }
                for r in results
            ]

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        enriched: list[dict[str, Any]] = []
        for r in results:
            # Get scene info with engagement data
            cursor.execute(
                """
                SELECT s.id, s.title, st.name as studio,
                       COALESCE(view_agg.view_count, 0) as view_count,
                       COALESCE(o_agg.o_count, 0) as o_count
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                WHERE s.id = ?
            """,
                (r.scene_id,),
            )

            row = cursor.fetchone()
            title = row["title"] if row else None
            view_count = row["view_count"] if row else 0
            o_count = row["o_count"] if row else 0

            # Calculate engagement score
            replay_count = max(view_count - 1, 0)
            engagement_score = (o_count * 20.0) + (replay_count * 2.0)

            cursor.execute(
                """
                SELECT p.name FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                WHERE ps.scene_id = ?
            """,
                (r.scene_id,),
            )
            performers = [pr["name"] for pr in cursor.fetchall()]

            # Get display name
            display_name = get_scene_display_name(cursor, r.scene_id, title, performers)
            url = get_scene_url(r.scene_id)
            similarity = round(r.similarity, 4)
            studio = row["studio"] if row else None

            # Build formatted display text
            escaped_name = escape_markdown(display_name)
            parts = []
            if performers:
                parts.append(", ".join(escape_markdown(p) for p in performers))
            if studio:
                parts.append(escape_markdown(studio))
            meta = " | ".join(parts) if parts else ""

            if meta:
                formatted = f"[{escaped_name}]({url}) - {meta} (score: {similarity}, engagement: {round(engagement_score, 1)})"
            else:
                formatted = f"[{escaped_name}]({url}) (score: {similarity}, engagement: {round(engagement_score, 1)})"

            enriched.append(
                {
                    "scene_id": r.scene_id,
                    "name": display_name,
                    "url": url,
                    "similarity": similarity,
                    "studio": studio,
                    "performers": performers,
                    "view_count": view_count,
                    "o_count": o_count,
                    "replay_count": replay_count,
                    "engagement_score": round(engagement_score, 2),
                    "formatted": formatted,
                    "visual_description": r.visual_description,
                }
            )

        conn.close()
        return enriched
