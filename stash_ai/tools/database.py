"""Database query tools for Stash using direct SQLite access."""

import os
import sqlite3
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolResult


def get_stash_db_path() -> Path:
    """
    Get the path to the Stash SQLite database.

    Returns:
        Path to stash-go.sqlite
    """
    # Common locations for the Stash database
    # The database is typically in the Stash config directory
    possible_paths = [
        Path(os.environ["STASH_CONFIG_DIR"]) / "stash-go.sqlite"
        if os.environ.get("STASH_CONFIG_DIR")
        else None,
        Path.home() / ".stash" / "stash-go.sqlite",
        Path.cwd().parent / "stash-go.sqlite",  # Relative to plugin directory
        Path("/root/.stash/stash-go.sqlite"),
    ]

    for path in possible_paths:
        if path is None:
            continue
        try:
            if path.exists():
                return path
        except PermissionError:
            # Skip paths we can't access
            continue

    # Default fallback
    return Path.home() / ".stash" / "stash-go.sqlite"


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """
    Open a read-only connection to the SQLite database.

    This ensures no data can be modified or deleted, even if a
    malformed query is executed.

    Args:
        db_path: Path to the database file

    Returns:
        Read-only SQLite connection

    Raises:
        sqlite3.Error: If the connection fails
    """
    # Use URI mode with ?mode=ro to enforce read-only at the SQLite level
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_excluded_tag_ids_with_children(
    cursor: sqlite3.Cursor,
    excluded_tag_names: list[str],
) -> set[int]:
    """
    Get all tag IDs that should be excluded, including children of excluded parent tags.

    Uses recursive query to find all descendants of excluded tags via tags_relations.

    Args:
        cursor: Database cursor
        excluded_tag_names: List of tag names to exclude (case-insensitive)

    Returns:
        Set of tag IDs to exclude
    """
    if not excluded_tag_names:
        return set()

    try:
        # First, get IDs for directly excluded tags (case-insensitive match)
        placeholders = ", ".join("?" for _ in excluded_tag_names)
        cursor.execute(
            f"""
            SELECT id FROM tags
            WHERE LOWER(name) IN ({placeholders})
            """,
            [tag.lower() for tag in excluded_tag_names],
        )
        direct_ids = {row["id"] for row in cursor.fetchall()}

        if not direct_ids:
            return set()

        # Now recursively get all children of these tags
        # Using recursive CTE to traverse tag hierarchy
        id_placeholders = ", ".join("?" for _ in direct_ids)
        cursor.execute(
            f"""
            WITH RECURSIVE descendants AS (
                -- Base case: direct children of excluded tags
                SELECT child_id as id
                FROM tags_relations
                WHERE parent_id IN ({id_placeholders})

                UNION

                -- Recursive case: children of children
                SELECT tr.child_id
                FROM tags_relations tr
                JOIN descendants d ON tr.parent_id = d.id
            )
            SELECT id FROM descendants
            """,
            list(direct_ids),
        )

        child_ids = {row["id"] for row in cursor.fetchall()}

        # Combine direct and child IDs
        return direct_ids | child_ids

    except sqlite3.Error:
        # If anything fails, return just the direct matches
        return set()


class QueryPerformerTagsTool(BaseTool):
    """
    Tool to query tags associated with a performer using direct SQLite access.

    This tool finds all tags that appear in scenes featuring a specific
    performer, ranked by view frequency.
    """

    @property
    def name(self) -> str:
        return "query_performer_tags"

    @property
    def description(self) -> str:
        return (
            "Query the database to find all tags associated with a performer's scenes. "
            "Returns tags ranked by how often they appear in the performer's viewed content, "
            "weighted by view count. Also includes performer's directly assigned tags."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_name",
                "type": "string",
                "description": "The name of the performer to look up",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scene tags to return (default: 20)",
                "required": False,
                "enum": None,
            },
            {
                "name": "weighted_by_views",
                "type": "boolean",
                "description": "Weight tags by view count (default: True)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the performer tags query using direct SQLite access.

        Args:
            performer_name: Name of the performer
            limit: Max tags to return (default 20)
            weighted_by_views: Weight by view count (default True)

        Returns:
            ToolResult with performer info and associated tags
        """
        performer_name: str = kwargs.get("performer_name", "")
        limit: int = kwargs.get("limit", 20)
        weighted_by_views: bool = kwargs.get("weighted_by_views", True)

        if not performer_name:
            return {
                "success": False,
                "data": None,
                "error": "performer_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Step 1: Find the performer by name or alias
            performer = self._find_performer(cursor, performer_name)
            if not performer:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Performer '{performer_name}' not found",
                }

            performer_id = performer["id"]

            # Step 2: Get performer's direct tags (excluding excluded tags)
            direct_tags = self._get_performer_direct_tags(cursor, performer_id, excluded_ids)

            # Step 3: Get performer's aliases
            aliases = self._get_performer_aliases(cursor, performer_id)

            # Step 4: Get scene count for this performer
            scene_count = self._get_performer_scene_count(cursor, performer_id)

            # Step 5: Get tags from performer's scenes, weighted by views
            scene_tags = self._get_scene_tags_weighted(
                cursor, performer_id, weighted_by_views, limit, excluded_ids
            )

            conn.close()

            # Build result
            result_data: dict[str, Any] = {
                "performer": {
                    "id": performer_id,
                    "name": performer["name"],
                    "disambiguation": performer.get("disambiguation"),
                    "aliases": aliases,
                    "direct_tags": direct_tags,
                },
                "scene_count": scene_count,
                "scene_tags": scene_tags,
                "total_unique_scene_tags": len(scene_tags),
            }

            return {
                "success": True,
                "data": result_data,
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }

    def _find_performer(self, cursor: sqlite3.Cursor, name: str) -> dict[str, Any] | None:
        """
        Find a performer by name or alias.

        Args:
            cursor: Database cursor
            name: Performer name to search for

        Returns:
            Performer dict or None if not found
        """
        # First try exact name match
        cursor.execute(
            """
            SELECT id, name, disambiguation
            FROM performers
            WHERE LOWER(name) = LOWER(?)
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Try alias match
        cursor.execute(
            """
            SELECT p.id, p.name, p.disambiguation
            FROM performers p
            JOIN performer_aliases pa ON p.id = pa.performer_id
            WHERE LOWER(pa.alias) = LOWER(?)
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Try partial name match
        cursor.execute(
            """
            SELECT id, name, disambiguation
            FROM performers
            WHERE LOWER(name) LIKE LOWER(?)
            ORDER BY LENGTH(name)
            LIMIT 1
            """,
            (f"%{name}%",),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)

        return None

    def _get_performer_aliases(self, cursor: sqlite3.Cursor, performer_id: int) -> list[str]:
        """Get all aliases for a performer."""
        cursor.execute(
            """
            SELECT alias FROM performer_aliases
            WHERE performer_id = ?
            """,
            (performer_id,),
        )
        return [row["alias"] for row in cursor.fetchall()]

    def _get_performer_direct_tags(
        self, cursor: sqlite3.Cursor, performer_id: int, excluded_ids: set[int]
    ) -> list[dict[str, Any]]:
        """Get tags directly assigned to a performer."""
        # Build exclusion clause
        exclude_clause = ""
        params: list[Any] = [performer_id]
        if excluded_ids:
            placeholders = ",".join("?" * len(excluded_ids))
            exclude_clause = f"AND t.id NOT IN ({placeholders})"
            params.extend(list(excluded_ids))

        cursor.execute(
            f"""
            SELECT t.id, t.name
            FROM tags t
            JOIN performers_tags pt ON t.id = pt.tag_id
            WHERE pt.performer_id = ?
            {exclude_clause}
            ORDER BY t.name
            """,
            params,
        )
        return [{"id": row["id"], "name": row["name"]} for row in cursor.fetchall()]

    def _get_performer_scene_count(self, cursor: sqlite3.Cursor, performer_id: int) -> int:
        """Get the number of scenes featuring a performer."""
        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM performers_scenes
            WHERE performer_id = ?
            """,
            (performer_id,),
        )
        row = cursor.fetchone()
        return row["count"] if row else 0

    def _get_scene_tags_weighted(
        self,
        cursor: sqlite3.Cursor,
        performer_id: int,
        weighted: bool,
        limit: int,
        excluded_ids: set[int],
    ) -> list[dict[str, Any]]:
        """
        Get tags from performer's scenes, optionally weighted by view count.

        Uses scenes_view_dates to count views per scene.

        Args:
            cursor: Database cursor
            performer_id: The performer's ID
            weighted: Whether to weight by view count
            limit: Maximum tags to return
            excluded_ids: Set of tag IDs to exclude

        Returns:
            List of tag dicts with counts
        """
        # Build exclusion clause
        exclude_clause = ""
        base_params: list[Any] = [performer_id]
        if excluded_ids:
            placeholders = ",".join("?" * len(excluded_ids))
            exclude_clause = f"AND t.id NOT IN ({placeholders})"
            base_params.extend(list(excluded_ids))

        if weighted:
            # Get tags weighted by view count
            # Each entry in scenes_view_dates represents one view
            cursor.execute(
                f"""
                SELECT
                    t.id,
                    t.name,
                    SUM(COALESCE(view_counts.view_count, 1)) as weighted_count
                FROM performers_scenes ps
                JOIN scenes_tags st ON ps.scene_id = st.scene_id
                JOIN tags t ON st.tag_id = t.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates
                    GROUP BY scene_id
                ) view_counts ON ps.scene_id = view_counts.scene_id
                WHERE ps.performer_id = ?
                {exclude_clause}
                GROUP BY t.id, t.name
                ORDER BY weighted_count DESC
                LIMIT ?
                """,
                (*base_params, limit),
            )
        else:
            # Get tags by scene count only
            cursor.execute(
                f"""
                SELECT
                    t.id,
                    t.name,
                    COUNT(*) as weighted_count
                FROM performers_scenes ps
                JOIN scenes_tags st ON ps.scene_id = st.scene_id
                JOIN tags t ON st.tag_id = t.id
                WHERE ps.performer_id = ?
                {exclude_clause}
                GROUP BY t.id, t.name
                ORDER BY weighted_count DESC
                LIMIT ?
                """,
                (*base_params, limit),
            )

        return [
            {"id": row["id"], "name": row["name"], "count": row["weighted_count"]}
            for row in cursor.fetchall()
        ]


class QueryTagPerformersTool(BaseTool):
    """
    Tool to find performers associated with a specific tag.
    """

    @property
    def name(self) -> str:
        return "query_tag_performers"

    @property
    def description(self) -> str:
        return (
            "Query the database to find performers whose scenes have a specific tag. "
            "Returns performers ranked by how many of their scenes have that tag."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "tag_name",
                "type": "string",
                "description": "The name of the tag to look up",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of performers to return (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tag performers query."""
        tag_name: str = kwargs.get("tag_name", "")
        limit: int = kwargs.get("limit", 20)

        if not tag_name:
            return {
                "success": False,
                "data": None,
                "error": "tag_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find the tag
            cursor.execute(
                """
                SELECT id, name, description FROM tags
                WHERE LOWER(name) = LOWER(?)
                """,
                (tag_name,),
            )
            tag_row = cursor.fetchone()

            if not tag_row:
                # Try partial match
                cursor.execute(
                    """
                    SELECT id, name, description FROM tags
                    WHERE LOWER(name) LIKE LOWER(?)
                    ORDER BY LENGTH(name)
                    LIMIT 1
                    """,
                    (f"%{tag_name}%",),
                )
                tag_row = cursor.fetchone()

            if not tag_row:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Tag '{tag_name}' not found",
                }

            tag_id = tag_row["id"]

            # Get performers whose scenes have this tag
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    COUNT(DISTINCT ps.scene_id) as scene_count
                FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                JOIN scenes_tags st ON ps.scene_id = st.scene_id
                WHERE st.tag_id = ?
                GROUP BY p.id, p.name
                ORDER BY scene_count DESC
                LIMIT ?
                """,
                (tag_id, limit),
            )

            performers = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "scene_count": row["scene_count"],
                }
                for row in cursor.fetchall()
            ]

            conn.close()

            return {
                "success": True,
                "data": {
                    "tag": {
                        "id": tag_row["id"],
                        "name": tag_row["name"],
                        "description": tag_row["description"],
                    },
                    "performers": performers,
                    "total_performers": len(performers),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryViewingStatsTool(BaseTool):
    """
    Tool to get viewing statistics for the library.
    """

    @property
    def name(self) -> str:
        return "query_viewing_stats"

    @property
    def description(self) -> str:
        return (
            "Query viewing statistics from the database including most watched "
            "scenes, performers, tags, and viewing history patterns."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "stat_type",
                "type": "string",
                "description": "Type of stats to retrieve",
                "required": True,
                "enum": ["top_scenes", "top_performers", "top_tags", "recent_views"],
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
        """Execute the viewing stats query."""
        stat_type: str = kwargs.get("stat_type", "")
        limit: int = kwargs.get("limit", 10)

        if not stat_type:
            return {
                "success": False,
                "data": None,
                "error": "stat_type is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if stat_type == "top_scenes":
                data = self._get_top_scenes(cursor, limit)
            elif stat_type == "top_performers":
                data = self._get_top_performers(cursor, limit)
            elif stat_type == "top_tags":
                data = self._get_top_tags(cursor, limit)
            elif stat_type == "recent_views":
                data = self._get_recent_views(cursor, limit)
            else:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Unknown stat_type: {stat_type}",
                }

            conn.close()

            return {
                "success": True,
                "data": data,
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_top_scenes(self, cursor: sqlite3.Cursor, limit: int) -> dict[str, Any]:
        """Get most viewed scenes."""
        cursor.execute(
            """
            SELECT
                s.id,
                s.title,
                COUNT(svd.view_date) as view_count,
                s.play_duration
            FROM scenes s
            JOIN scenes_view_dates svd ON s.id = svd.scene_id
            GROUP BY s.id
            ORDER BY view_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        scenes = [
            {
                "id": row["id"],
                "title": row["title"] or f"Scene {row['id']}",
                "view_count": row["view_count"],
                "play_duration_hours": round(row["play_duration"] / 3600, 2)
                if row["play_duration"]
                else 0,
            }
            for row in cursor.fetchall()
        ]
        return {"top_scenes": scenes}

    def _get_top_performers(self, cursor: sqlite3.Cursor, limit: int) -> dict[str, Any]:
        """Get performers with most viewed content."""
        cursor.execute(
            """
            SELECT
                p.id,
                p.name,
                COUNT(svd.view_date) as total_views,
                COUNT(DISTINCT ps.scene_id) as scene_count
            FROM performers p
            JOIN performers_scenes ps ON p.id = ps.performer_id
            JOIN scenes_view_dates svd ON ps.scene_id = svd.scene_id
            GROUP BY p.id, p.name
            ORDER BY total_views DESC
            LIMIT ?
            """,
            (limit,),
        )
        performers = [
            {
                "id": row["id"],
                "name": row["name"],
                "total_views": row["total_views"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]
        return {"top_performers": performers}

    def _get_top_tags(self, cursor: sqlite3.Cursor, limit: int) -> dict[str, Any]:
        """Get tags from most viewed content."""
        cursor.execute(
            """
            SELECT
                t.id,
                t.name,
                COUNT(svd.view_date) as total_views,
                COUNT(DISTINCT st.scene_id) as scene_count
            FROM tags t
            JOIN scenes_tags st ON t.id = st.tag_id
            JOIN scenes_view_dates svd ON st.scene_id = svd.scene_id
            GROUP BY t.id, t.name
            ORDER BY total_views DESC
            LIMIT ?
            """,
            (limit,),
        )
        tags = [
            {
                "id": row["id"],
                "name": row["name"],
                "total_views": row["total_views"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]
        return {"top_tags": tags}

    def _get_recent_views(self, cursor: sqlite3.Cursor, limit: int) -> dict[str, Any]:
        """Get recently viewed scenes."""
        cursor.execute(
            """
            SELECT
                s.id,
                s.title,
                svd.view_date,
                GROUP_CONCAT(DISTINCT p.name) as performers
            FROM scenes_view_dates svd
            JOIN scenes s ON svd.scene_id = s.id
            LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
            LEFT JOIN performers p ON ps.performer_id = p.id
            GROUP BY svd.scene_id, svd.view_date
            ORDER BY svd.view_date DESC
            LIMIT ?
            """,
            (limit,),
        )
        views = [
            {
                "scene_id": row["id"],
                "title": row["title"] or f"Scene {row['id']}",
                "view_date": row["view_date"],
                "performers": row["performers"].split(",") if row["performers"] else [],
            }
            for row in cursor.fetchall()
        ]
        return {"recent_views": views}


class QueryTopPerformersTool(BaseTool):
    """
    Tool to get top performers ranked by various metrics.

    Queries the SQLite database directly for accurate results that match Stash UI.
    """

    @property
    def name(self) -> str:
        return "query_top_performers"

    @property
    def description(self) -> str:
        return (
            "Get top performers ranked by view count, scene count, o-count, or play duration. "
            "Default sorting by view_count matches Stash UI 'Play Count' sort."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "sort_by",
                "type": "string",
                "description": "Metric to sort by (default: view_count)",
                "required": False,
                "enum": ["view_count", "scene_count", "o_count", "play_duration"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of performers to return (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the top performers query."""
        sort_by: str = kwargs.get("sort_by", "view_count")
        limit: int = kwargs.get("limit", 10)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if sort_by == "view_count":
                performers = self._get_by_view_count(cursor, limit)
            elif sort_by == "scene_count":
                performers = self._get_by_scene_count(cursor, limit)
            elif sort_by == "o_count":
                performers = self._get_by_o_count(cursor, limit)
            elif sort_by == "play_duration":
                performers = self._get_by_play_duration(cursor, limit)
            else:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Unknown sort_by: {sort_by}",
                }

            conn.close()

            return {
                "success": True,
                "data": {
                    "performers": performers,
                    "sort_by": sort_by,
                    "count": len(performers),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_by_view_count(self, cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
        """Get performers by total view count (matches Stash UI Play Count sort)."""
        cursor.execute(
            """
            SELECT
                p.id,
                p.name,
                COUNT(svd.view_date) as view_count,
                COUNT(DISTINCT ps.scene_id) as scene_count
            FROM performers p
            JOIN performers_scenes ps ON p.id = ps.performer_id
            JOIN scenes_view_dates svd ON ps.scene_id = svd.scene_id
            GROUP BY p.id, p.name
            ORDER BY view_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "view_count": row["view_count"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_by_scene_count(self, cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
        """Get performers by scene count."""
        cursor.execute(
            """
            SELECT
                p.id,
                p.name,
                COUNT(ps.scene_id) as scene_count
            FROM performers p
            JOIN performers_scenes ps ON p.id = ps.performer_id
            GROUP BY p.id, p.name
            ORDER BY scene_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_by_o_count(self, cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
        """Get performers by o-count."""
        cursor.execute(
            """
            SELECT
                p.id,
                p.name,
                COUNT(sod.o_date) as o_count,
                COUNT(DISTINCT ps.scene_id) as scene_count
            FROM performers p
            JOIN performers_scenes ps ON p.id = ps.performer_id
            JOIN scenes_o_dates sod ON ps.scene_id = sod.scene_id
            GROUP BY p.id, p.name
            ORDER BY o_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "o_count": row["o_count"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_by_play_duration(self, cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
        """Get performers by total play duration."""
        cursor.execute(
            """
            SELECT
                p.id,
                p.name,
                SUM(s.play_duration) as total_play_duration,
                COUNT(DISTINCT ps.scene_id) as scene_count
            FROM performers p
            JOIN performers_scenes ps ON p.id = ps.performer_id
            JOIN scenes s ON ps.scene_id = s.id
            WHERE s.play_duration > 0
            GROUP BY p.id, p.name
            ORDER BY total_play_duration DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "play_duration_hours": round(row["total_play_duration"] / 3600, 2)
                if row["total_play_duration"]
                else 0,
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]


class QueryTopTagsTool(BaseTool):
    """
    Tool to get top tags ranked by various metrics.
    """

    @property
    def name(self) -> str:
        return "query_top_tags"

    @property
    def description(self) -> str:
        return (
            "Get top tags ranked by view count, scene count, or o-count. "
            "Supports excluding specific tags from results."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "sort_by",
                "type": "string",
                "description": "Metric to sort by (default: view_count)",
                "required": False,
                "enum": ["view_count", "scene_count", "o_count"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of tags to return (default: 10)",
                "required": False,
                "enum": None,
            },
            {
                "name": "exclude_tags",
                "type": "string",
                "description": "Comma-separated list of tag names to exclude",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the top tags query."""
        sort_by: str = kwargs.get("sort_by", "view_count")
        limit: int = kwargs.get("limit", 10)
        exclude_tags_str: str = kwargs.get("exclude_tags", "")

        # Merge user-specified exclusions with plugin-level exclusions
        user_exclude = (
            [t.strip().lower() for t in exclude_tags_str.split(",") if t.strip()]
            if exclude_tags_str
            else []
        )
        plugin_exclude = self.get_excluded_tags()  # Already lowercase
        exclude_tags = list(set(user_exclude + plugin_exclude))

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_ids = get_excluded_tag_ids_with_children(cursor, exclude_tags)

            if sort_by == "view_count":
                tags = self._get_by_view_count(cursor, limit, excluded_ids)
            elif sort_by == "scene_count":
                tags = self._get_by_scene_count(cursor, limit, excluded_ids)
            elif sort_by == "o_count":
                tags = self._get_by_o_count(cursor, limit, excluded_ids)
            else:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Unknown sort_by: {sort_by}",
                }

            conn.close()

            return {
                "success": True,
                "data": {
                    "tags": tags,
                    "sort_by": sort_by,
                    "count": len(tags),
                    "excluded": exclude_tags,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _build_exclude_clause_by_id(self, excluded_ids: set[int]) -> tuple[str, list[int]]:
        """Build SQL exclusion clause for tags using IDs."""
        if not excluded_ids:
            return "", []
        placeholders = ",".join("?" * len(excluded_ids))
        return f"AND t.id NOT IN ({placeholders})", list(excluded_ids)

    def _get_by_view_count(
        self, cursor: sqlite3.Cursor, limit: int, excluded_ids: set[int]
    ) -> list[dict[str, Any]]:
        """Get tags by total view count."""
        exclude_clause, exclude_params = self._build_exclude_clause_by_id(excluded_ids)
        cursor.execute(
            f"""
            SELECT
                t.id,
                t.name,
                COUNT(svd.view_date) as view_count,
                COUNT(DISTINCT st.scene_id) as scene_count
            FROM tags t
            JOIN scenes_tags st ON t.id = st.tag_id
            JOIN scenes_view_dates svd ON st.scene_id = svd.scene_id
            WHERE 1=1 {exclude_clause}
            GROUP BY t.id, t.name
            ORDER BY view_count DESC
            LIMIT ?
            """,
            (*exclude_params, limit),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "view_count": row["view_count"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_by_scene_count(
        self, cursor: sqlite3.Cursor, limit: int, excluded_ids: set[int]
    ) -> list[dict[str, Any]]:
        """Get tags by scene count."""
        exclude_clause, exclude_params = self._build_exclude_clause_by_id(excluded_ids)
        cursor.execute(
            f"""
            SELECT
                t.id,
                t.name,
                COUNT(st.scene_id) as scene_count
            FROM tags t
            JOIN scenes_tags st ON t.id = st.tag_id
            WHERE 1=1 {exclude_clause}
            GROUP BY t.id, t.name
            ORDER BY scene_count DESC
            LIMIT ?
            """,
            (*exclude_params, limit),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_by_o_count(
        self, cursor: sqlite3.Cursor, limit: int, excluded_ids: set[int]
    ) -> list[dict[str, Any]]:
        """Get tags by o-count."""
        exclude_clause, exclude_params = self._build_exclude_clause_by_id(excluded_ids)
        cursor.execute(
            f"""
            SELECT
                t.id,
                t.name,
                COUNT(sod.o_date) as o_count,
                COUNT(DISTINCT st.scene_id) as scene_count
            FROM tags t
            JOIN scenes_tags st ON t.id = st.tag_id
            JOIN scenes_o_dates sod ON st.scene_id = sod.scene_id
            WHERE 1=1 {exclude_clause}
            GROUP BY t.id, t.name
            ORDER BY o_count DESC
            LIMIT ?
            """,
            (*exclude_params, limit),
        )
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "o_count": row["o_count"],
                "scene_count": row["scene_count"],
            }
            for row in cursor.fetchall()
        ]


class QueryTopStudiosTool(BaseTool):
    """
    Tool to get top studios ranked by various metrics.
    """

    @property
    def name(self) -> str:
        return "query_top_studios"

    @property
    def description(self) -> str:
        return "Get top studios ranked by view count or scene count."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "sort_by",
                "type": "string",
                "description": "Metric to sort by (default: view_count)",
                "required": False,
                "enum": ["view_count", "scene_count"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of studios to return (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the top studios query."""
        sort_by: str = kwargs.get("sort_by", "view_count")
        limit: int = kwargs.get("limit", 10)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if sort_by == "view_count":
                cursor.execute(
                    """
                    SELECT
                        st.id,
                        st.name,
                        COUNT(svd.view_date) as view_count,
                        COUNT(DISTINCT s.id) as scene_count
                    FROM studios st
                    JOIN scenes s ON st.id = s.studio_id
                    JOIN scenes_view_dates svd ON s.id = svd.scene_id
                    GROUP BY st.id, st.name
                    ORDER BY view_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            else:  # scene_count
                cursor.execute(
                    """
                    SELECT
                        st.id,
                        st.name,
                        COUNT(s.id) as scene_count
                    FROM studios st
                    JOIN scenes s ON st.id = s.studio_id
                    GROUP BY st.id, st.name
                    ORDER BY scene_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            studios = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return {
                "success": True,
                "data": {
                    "studios": studios,
                    "sort_by": sort_by,
                    "count": len(studios),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryLibraryStatsTool(BaseTool):
    """
    Tool to get comprehensive library statistics.
    """

    @property
    def name(self) -> str:
        return "query_library_stats"

    @property
    def description(self) -> str:
        return (
            "Get comprehensive library statistics including scene counts, "
            "performer counts, viewing stats, and storage information."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []  # No parameters needed

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the library stats query."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get entity counts
            cursor.execute("SELECT COUNT(*) as count FROM scenes")
            scene_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM performers")
            performer_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM tags")
            tag_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM studios")
            studio_count = cursor.fetchone()["count"]

            # Get viewing stats
            cursor.execute("SELECT COUNT(DISTINCT scene_id) as count FROM scenes_view_dates")
            watched_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM scenes_view_dates")
            total_views = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM scenes_o_dates")
            total_o_count = cursor.fetchone()["count"]

            # Get duration and size
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(vf.duration), 0) / 3600.0 as total_hours,
                    COALESCE(SUM(f.size), 0) / (1024.0*1024.0*1024.0) as total_gb
                FROM scenes_files sf
                JOIN files f ON sf.file_id = f.id
                JOIN video_files vf ON f.id = vf.file_id
                WHERE sf."primary" = 1
                """
            )
            storage_row = cursor.fetchone()
            total_hours = storage_row["total_hours"] or 0
            total_gb = storage_row["total_gb"] or 0

            # Get play duration
            cursor.execute("SELECT COALESCE(SUM(play_duration), 0) / 3600.0 as hours FROM scenes")
            play_hours = cursor.fetchone()["hours"] or 0

            conn.close()

            avg_duration = (total_hours * 60 / scene_count) if scene_count > 0 else 0

            return {
                "success": True,
                "data": {
                    "scene_count": scene_count,
                    "performer_count": performer_count,
                    "tag_count": tag_count,
                    "studio_count": studio_count,
                    "watched_scene_count": watched_count,
                    "unwatched_scene_count": scene_count - watched_count,
                    "total_view_count": total_views,
                    "total_o_count": total_o_count,
                    "total_duration_hours": round(total_hours, 1),
                    "total_size_gb": round(total_gb, 1),
                    "total_play_duration_hours": round(play_hours, 1),
                    "average_scene_duration_minutes": round(avg_duration, 1),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryWatchingPatternsTool(BaseTool):
    """
    Tool to analyze viewing patterns.
    """

    @property
    def name(self) -> str:
        return "query_watching_patterns"

    @property
    def description(self) -> str:
        return (
            "Analyze viewing patterns including hourly distribution, "
            "daily distribution, monthly trends, and recent activity."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "pattern_type",
                "type": "string",
                "description": "Type of pattern to analyze",
                "required": True,
                "enum": ["hourly", "daily", "monthly", "recent_activity"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Limit for recent_activity (default: 30)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the watching patterns query."""
        pattern_type: str = kwargs.get("pattern_type", "")
        limit: int = kwargs.get("limit", 30)

        if not pattern_type:
            return {
                "success": False,
                "data": None,
                "error": "pattern_type is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if pattern_type == "hourly":
                data = self._get_hourly_pattern(cursor)
            elif pattern_type == "daily":
                data = self._get_daily_pattern(cursor)
            elif pattern_type == "monthly":
                data = self._get_monthly_pattern(cursor)
            elif pattern_type == "recent_activity":
                data = self._get_recent_activity(cursor, limit)
            else:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Unknown pattern_type: {pattern_type}",
                }

            conn.close()

            return {
                "success": True,
                "data": {
                    "pattern_type": pattern_type,
                    "data": data,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_hourly_pattern(self, cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        """Get viewing distribution by hour of day."""
        cursor.execute(
            """
            SELECT
                CAST(strftime('%H', view_date) AS INTEGER) as hour,
                COUNT(*) as view_count
            FROM scenes_view_dates
            GROUP BY hour
            ORDER BY hour
            """
        )
        return [{"hour": row["hour"], "view_count": row["view_count"]} for row in cursor.fetchall()]

    def _get_daily_pattern(self, cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        """Get viewing distribution by day of week."""
        day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        cursor.execute(
            """
            SELECT
                CAST(strftime('%w', view_date) AS INTEGER) as day_of_week,
                COUNT(*) as view_count
            FROM scenes_view_dates
            GROUP BY day_of_week
            ORDER BY day_of_week
            """
        )
        return [
            {
                "day_of_week": row["day_of_week"],
                "day_name": day_names[row["day_of_week"]],
                "view_count": row["view_count"],
            }
            for row in cursor.fetchall()
        ]

    def _get_monthly_pattern(self, cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        """Get viewing distribution by month."""
        cursor.execute(
            """
            SELECT
                strftime('%Y-%m', view_date) as month,
                COUNT(*) as view_count
            FROM scenes_view_dates
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
            """
        )
        return [
            {"month": row["month"], "view_count": row["view_count"]} for row in cursor.fetchall()
        ]

    def _get_recent_activity(self, cursor: sqlite3.Cursor, limit: int) -> list[dict[str, Any]]:
        """Get recent viewing activity by day."""
        cursor.execute(
            """
            SELECT
                date(view_date) as date,
                COUNT(*) as view_count
            FROM scenes_view_dates
            GROUP BY date
            ORDER BY date DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [{"date": row["date"], "view_count": row["view_count"]} for row in cursor.fetchall()]


class QueryTagCorrelationsTool(BaseTool):
    """
    Tool to find tags that commonly appear together.
    """

    @property
    def name(self) -> str:
        return "query_tag_correlations"

    @property
    def description(self) -> str:
        return "Find tags that commonly appear together with a given tag."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "tag_name",
                "type": "string",
                "description": "The tag to find correlations for",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of correlated tags (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tag correlations query."""
        tag_name: str = kwargs.get("tag_name", "")
        limit: int = kwargs.get("limit", 10)

        if not tag_name:
            return {
                "success": False,
                "data": None,
                "error": "tag_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Build exclusion clause
            exclude_clause = ""
            params: list[Any] = [tag_name]
            if excluded_ids:
                placeholders = ",".join("?" * len(excluded_ids))
                exclude_clause = f"AND t2.id NOT IN ({placeholders})"
                params.extend(list(excluded_ids))
            params.append(limit)

            # Find tags that co-occur with the given tag
            cursor.execute(
                f"""
                SELECT
                    t2.id,
                    t2.name,
                    COUNT(*) as co_occurrence_count
                FROM scenes_tags st1
                JOIN scenes_tags st2 ON st1.scene_id = st2.scene_id AND st1.tag_id != st2.tag_id
                JOIN tags t1 ON st1.tag_id = t1.id
                JOIN tags t2 ON st2.tag_id = t2.id
                WHERE LOWER(t1.name) = LOWER(?)
                {exclude_clause}
                GROUP BY t2.id, t2.name
                ORDER BY co_occurrence_count DESC
                LIMIT ?
                """,
                params,
            )

            correlations = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "co_occurrence_count": row["co_occurrence_count"],
                }
                for row in cursor.fetchall()
            ]

            conn.close()

            return {
                "success": True,
                "data": {
                    "source_tag": tag_name,
                    "correlated_tags": correlations,
                    "count": len(correlations),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryTopPerformerCommonTagsTool(BaseTool):
    """
    Tool to find tags that are common across top performers.

    This tool identifies shared themes/preferences among the user's
    most-watched performers by finding tags that appear in scenes
    featuring multiple top performers.
    """

    @property
    def name(self) -> str:
        return "query_top_performer_common_tags"

    @property
    def description(self) -> str:
        return (
            "Find tags that are common across top performers. "
            "Identifies shared themes/preferences among your most-watched performers."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "top_n_performers",
                "type": "integer",
                "description": "Number of top performers to analyze (default: 5)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to rank performers (default: view_count)",
                "required": False,
                "enum": ["view_count", "scene_count", "o_count"],
            },
            {
                "name": "min_performers",
                "type": "integer",
                "description": "Minimum performers a tag must appear with to be 'common' (default: 2)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of tags to return (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the top performer common tags query."""
        top_n_performers: int = kwargs.get("top_n_performers", 5)
        sort_by: str = kwargs.get("sort_by", "view_count")
        min_performers: int = kwargs.get("min_performers", 2)
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Step 1: Get top performers based on sort_by
            top_performers = self._get_top_performers(cursor, top_n_performers, sort_by)

            if not top_performers:
                conn.close()
                return {
                    "success": True,
                    "data": {
                        "top_performers": [],
                        "common_tags": [],
                        "settings": {
                            "top_n_performers": top_n_performers,
                            "sort_by": sort_by,
                            "min_performers": min_performers,
                        },
                    },
                    "error": None,
                }

            # Get performer IDs for the query
            performer_ids = [p["id"] for p in top_performers]
            performer_placeholders = ",".join("?" * len(performer_ids))

            # Build tag exclusion clause
            tag_exclude_clause = ""
            params: list[Any] = list(performer_ids)
            if excluded_ids:
                tag_placeholders = ",".join("?" * len(excluded_ids))
                tag_exclude_clause = f"AND t.id NOT IN ({tag_placeholders})"
                params.extend(list(excluded_ids))
            params.extend([min_performers, limit])

            # Step 2 & 3: Find tags common to multiple top performers
            cursor.execute(
                f"""
                WITH performer_tags AS (
                    SELECT DISTINCT
                        ps.performer_id,
                        p.name as performer_name,
                        t.id as tag_id,
                        t.name as tag_name
                    FROM performers_scenes ps
                    JOIN performers p ON ps.performer_id = p.id
                    JOIN scenes_tags st ON ps.scene_id = st.scene_id
                    JOIN tags t ON st.tag_id = t.id
                    WHERE ps.performer_id IN ({performer_placeholders})
                    {tag_exclude_clause}
                )
                SELECT
                    tag_id,
                    tag_name,
                    COUNT(DISTINCT performer_id) as performer_count,
                    GROUP_CONCAT(performer_name) as performers
                FROM performer_tags
                GROUP BY tag_id, tag_name
                HAVING performer_count >= ?
                ORDER BY performer_count DESC, tag_name ASC
                LIMIT ?
                """,
                params,
            )

            common_tags = [
                {
                    "id": row["tag_id"],
                    "name": row["tag_name"],
                    "performer_count": row["performer_count"],
                    "performers": row["performers"].split(",") if row["performers"] else [],
                }
                for row in cursor.fetchall()
            ]

            conn.close()

            return {
                "success": True,
                "data": {
                    "top_performers": top_performers,
                    "common_tags": common_tags,
                    "total_common_tags": len(common_tags),
                    "settings": {
                        "top_n_performers": top_n_performers,
                        "sort_by": sort_by,
                        "min_performers": min_performers,
                    },
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_top_performers(
        self,
        cursor: sqlite3.Cursor,
        limit: int,
        sort_by: str,
    ) -> list[dict[str, Any]]:
        """Get top performers by the specified metric."""
        if sort_by == "view_count":
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    COUNT(svd.view_date) as view_count
                FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                JOIN scenes_view_dates svd ON ps.scene_id = svd.scene_id
                GROUP BY p.id, p.name
                ORDER BY view_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        elif sort_by == "scene_count":
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    COUNT(ps.scene_id) as scene_count
                FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                GROUP BY p.id, p.name
                ORDER BY scene_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        elif sort_by == "o_count":
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    COUNT(sod.o_date) as o_count
                FROM performers p
                JOIN performers_scenes ps ON p.id = ps.performer_id
                JOIN scenes_o_dates sod ON ps.scene_id = sod.scene_id
                GROUP BY p.id, p.name
                ORDER BY o_count DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            return []

        return [dict(row) for row in cursor.fetchall()]


class QueryPerformerPairsTool(BaseTool):
    """
    Tool to find performers who frequently appear together.
    """

    @property
    def name(self) -> str:
        return "query_performer_pairs"

    @property
    def description(self) -> str:
        return (
            "Find performers who frequently appear together in scenes. "
            "Can find top pairs overall or co-stars for a specific performer."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_name",
                "type": "string",
                "description": "Find co-stars for this performer (optional, if not provided returns top pairs overall)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of pairs/co-stars (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performer pairs query."""
        performer_name: str = kwargs.get("performer_name", "")
        limit: int = kwargs.get("limit", 10)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if performer_name:
                # Find co-stars for specific performer
                cursor.execute(
                    """
                    SELECT
                        p2.id,
                        p2.name,
                        COUNT(*) as scenes_together
                    FROM performers_scenes ps1
                    JOIN performers_scenes ps2 ON ps1.scene_id = ps2.scene_id AND ps1.performer_id != ps2.performer_id
                    JOIN performers p1 ON ps1.performer_id = p1.id
                    JOIN performers p2 ON ps2.performer_id = p2.id
                    WHERE LOWER(p1.name) = LOWER(?)
                    GROUP BY p2.id, p2.name
                    ORDER BY scenes_together DESC
                    LIMIT ?
                    """,
                    (performer_name, limit),
                )
                results = [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "scenes_together": row["scenes_together"],
                    }
                    for row in cursor.fetchall()
                ]
                data = {
                    "performer": performer_name,
                    "co_stars": results,
                    "count": len(results),
                }
            else:
                # Find top pairs overall
                cursor.execute(
                    """
                    SELECT
                        p1.name as performer_1,
                        p2.name as performer_2,
                        COUNT(*) as scenes_together
                    FROM performers_scenes ps1
                    JOIN performers_scenes ps2 ON ps1.scene_id = ps2.scene_id AND ps1.performer_id < ps2.performer_id
                    JOIN performers p1 ON ps1.performer_id = p1.id
                    JOIN performers p2 ON ps2.performer_id = p2.id
                    GROUP BY p1.id, p2.id
                    ORDER BY scenes_together DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                results = [
                    {
                        "performer_1": row["performer_1"],
                        "performer_2": row["performer_2"],
                        "scenes_together": row["scenes_together"],
                    }
                    for row in cursor.fetchall()
                ]
                data = {
                    "top_pairs": results,
                    "count": len(results),
                }

            conn.close()

            return {
                "success": True,
                "data": data,
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryInteractiveContentTool(BaseTool):
    """
    Tool to find scenes with interactive/funscript support.
    """

    @property
    def name(self) -> str:
        return "query_interactive_content"

    @property
    def description(self) -> str:
        return "Find scenes that have interactive/funscript support."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the interactive content query."""
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    s.id,
                    s.title,
                    vf.interactive_speed,
                    GROUP_CONCAT(DISTINCT p.name) as performers
                FROM scenes s
                JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                JOIN video_files vf ON sf.file_id = vf.file_id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
                WHERE vf.interactive = 1
                GROUP BY s.id
                ORDER BY s.title
                LIMIT ?
                """,
                (limit,),
            )

            scenes = [
                {
                    "id": row["id"],
                    "title": row["title"] or f"Scene {row['id']}",
                    "interactive_speed": row["interactive_speed"],
                    "performers": row["performers"].split(",") if row["performers"] else [],
                }
                for row in cursor.fetchall()
            ]

            # Get total count
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM scenes s
                JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                JOIN video_files vf ON sf.file_id = vf.file_id
                WHERE vf.interactive = 1
                """
            )
            total_count = cursor.fetchone()["count"]

            conn.close()

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "returned_count": len(scenes),
                    "total_interactive_scenes": total_count,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryUnwatchedContentTool(BaseTool):
    """
    Tool to find unwatched content for recommendations.
    """

    @property
    def name(self) -> str:
        return "query_unwatched_content"

    @property
    def description(self) -> str:
        return "Find scenes that haven't been watched yet. Can filter by performer, tag, or studio."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "filter_type",
                "type": "string",
                "description": "Type of filter to apply",
                "required": False,
                "enum": ["all", "by_performer", "by_tag", "by_studio"],
            },
            {
                "name": "filter_value",
                "type": "string",
                "description": "Name of performer/tag/studio to filter by",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results",
                "required": False,
                "enum": ["newest", "oldest", "random", "duration"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the unwatched content query."""
        filter_type: str = kwargs.get("filter_type", "all")
        filter_value: str = kwargs.get("filter_value", "")
        sort_by: str = kwargs.get("sort_by", "newest")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build query based on filter type
            base_query = """
                SELECT DISTINCT
                    s.id,
                    s.title,
                    s.created_at,
                    GROUP_CONCAT(DISTINCT p.name) as performers
                FROM scenes s
                LEFT JOIN scenes_view_dates svd ON s.id = svd.scene_id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
            """

            where_clause = "WHERE svd.scene_id IS NULL"
            params: list[Any] = []

            if filter_type == "by_performer" and filter_value:
                base_query += """
                    JOIN performers_scenes ps_filter ON s.id = ps_filter.scene_id
                    JOIN performers p_filter ON ps_filter.performer_id = p_filter.id
                """
                where_clause += " AND LOWER(p_filter.name) = LOWER(?)"
                params.append(filter_value)
            elif filter_type == "by_tag" and filter_value:
                base_query += """
                    JOIN scenes_tags st_filter ON s.id = st_filter.scene_id
                    JOIN tags t_filter ON st_filter.tag_id = t_filter.id
                """
                where_clause += " AND LOWER(t_filter.name) = LOWER(?)"
                params.append(filter_value)
            elif filter_type == "by_studio" and filter_value:
                base_query += """
                    JOIN studios st_filter ON s.studio_id = st_filter.id
                """
                where_clause += " AND LOWER(st_filter.name) = LOWER(?)"
                params.append(filter_value)

            # Add GROUP BY
            group_clause = "GROUP BY s.id"

            # Build ORDER BY
            if sort_by == "newest":
                order_clause = "ORDER BY s.created_at DESC"
            elif sort_by == "oldest":
                order_clause = "ORDER BY s.created_at ASC"
            elif sort_by == "random":
                order_clause = "ORDER BY RANDOM()"
            else:  # duration
                order_clause = "ORDER BY s.id DESC"  # Fallback

            params.append(limit)
            full_query = f"{base_query} {where_clause} {group_clause} {order_clause} LIMIT ?"

            cursor.execute(full_query, params)

            scenes = [
                {
                    "id": row["id"],
                    "title": row["title"] or f"Scene {row['id']}",
                    "created_at": row["created_at"],
                    "performers": row["performers"].split(",") if row["performers"] else [],
                }
                for row in cursor.fetchall()
            ]

            # Get total unwatched count
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM scenes s
                LEFT JOIN scenes_view_dates svd ON s.id = svd.scene_id
                WHERE svd.scene_id IS NULL
                """
            )
            total_unwatched = cursor.fetchone()["count"]

            conn.close()

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "returned_count": len(scenes),
                    "total_unwatched": total_unwatched,
                    "filter_type": filter_type,
                    "filter_value": filter_value,
                    "sort_by": sort_by,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class RankScenesByEngagementTool(BaseTool):
    """
    Tool to rank scenes by engagement score with multiple scoring modes.

    Scoring modes:
    - favorites: (o_count * 3) + (replay_count * 2) - best for finding most-loved content
    - recent: favorites score with recency decay - best for current preferences
    - completion: completion_rate (play_duration / video_duration) - best for thoroughly watched
    - intensity: o_rate * view_count - best for consistently satisfying scenes

    No raw play_hours to avoid duration bias (longer scenes scoring higher unfairly).
    """

    @property
    def name(self) -> str:
        return "rank_scenes_by_engagement"

    @property
    def description(self) -> str:
        return (
            "Rank a list of scene IDs by engagement score. "
            "Scoring modes: 'favorites' (o_count * 3 + replay_count * 2), "
            "'recent' (favorites with recency decay), "
            "'completion' (play_duration / video_duration), "
            "'intensity' (o_rate * view_count). "
            "Default mode is 'favorites'. Use to find favorites from any scene list."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "scene_ids",
                "type": "array",
                "description": "List of scene IDs to rank by engagement",
                "required": True,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of results to return (default: all)",
                "required": False,
                "enum": None,
            },
            {
                "name": "scoring_mode",
                "type": "string",
                "description": "Scoring mode: favorites (default), recent, completion, intensity",
                "required": False,
                "enum": ["favorites", "recent", "completion", "intensity"],
            },
            {
                "name": "min_score",
                "type": "number",
                "description": "Minimum score to include (default: 0)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the scene ranking by engagement."""
        scene_ids: list[int] = kwargs.get("scene_ids", [])
        limit: int | None = kwargs.get("limit")
        scoring_mode: str = kwargs.get("scoring_mode", "favorites")
        min_score: float = kwargs.get("min_score", 0.0)

        if not scene_ids:
            return {
                "success": False,
                "data": None,
                "error": "scene_ids is required and must be non-empty",
            }

        valid_modes = {"favorites", "recent", "completion", "intensity"}
        if scoring_mode not in valid_modes:
            return {
                "success": False,
                "data": None,
                "error": f"Invalid scoring_mode '{scoring_mode}'. Valid: {valid_modes}",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build placeholders for IN clause
            placeholders = ",".join("?" * len(scene_ids))

            # Query engagement data for all provided scene IDs
            # Include video duration for completion rate calculation
            cursor.execute(
                f"""
                SELECT
                    s.id,
                    s.title,
                    s.play_duration,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    COALESCE(o_agg.o_count, 0) as o_count,
                    st.name as studio,
                    vf.duration as video_duration,
                    last_view.max_date as last_view_date
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN scenes_files fs ON s.id = fs.scene_id
                LEFT JOIN video_files vf ON fs.file_id = vf.file_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates
                    GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates
                    GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, MAX(view_date) as max_date
                    FROM scenes_view_dates
                    GROUP BY scene_id
                ) last_view ON s.id = last_view.scene_id
                WHERE s.id IN ({placeholders})
                """,
                scene_ids,
            )

            scenes: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                scene_id = row["id"]
                view_count = row["view_count"]
                o_count = row["o_count"]
                play_duration = row["play_duration"] or 0.0
                video_duration = row["video_duration"] or 0.0
                last_view_date = row["last_view_date"]

                # Calculate base metrics
                replay_count = max(view_count - 1, 0)
                o_rate = o_count / view_count if view_count > 0 else 0.0
                completion_rate = (
                    (play_duration / video_duration * 100.0) if video_duration > 0 else 0.0
                )
                # Cap completion rate at 500% to avoid extreme outliers
                completion_rate = min(completion_rate, 500.0)

                # Calculate recency decay (30-day half-life)
                recency_decay = 1.0
                if last_view_date:
                    try:
                        from datetime import datetime

                        last_dt = datetime.fromisoformat(last_view_date.replace("Z", "+00:00"))
                        days_ago = (datetime.now(last_dt.tzinfo) - last_dt).days
                        recency_decay = 0.5 ** (days_ago / 30.0)
                    except (ValueError, TypeError):
                        recency_decay = 1.0

                # Calculate score based on mode
                if scoring_mode == "favorites":
                    # (o_count * 20) + (replay_count * 2) - no duration bias
                    score = (o_count * 20.0) + (replay_count * 2.0)
                elif scoring_mode == "recent":
                    # Favorites with recency decay
                    base_score = (o_count * 20.0) + (replay_count * 2.0)
                    score = base_score * recency_decay
                elif scoring_mode == "completion":
                    # Completion rate (capped)
                    score = completion_rate
                elif scoring_mode == "intensity":
                    # O-rate * view_count - high intensity scenes
                    score = o_rate * view_count
                else:
                    score = 0.0

                # Skip if below minimum score
                if score < min_score:
                    continue

                # Get performers for this scene
                cursor.execute(
                    """
                    SELECT p.name FROM performers p
                    JOIN performers_scenes ps ON p.id = ps.performer_id
                    WHERE ps.scene_id = ?
                    """,
                    (scene_id,),
                )
                performers = [pr["name"] for pr in cursor.fetchall()]

                scenes.append(
                    {
                        "scene_id": scene_id,
                        "title": row["title"] or f"Scene {scene_id}",
                        "url": f"/scenes/{scene_id}",
                        "studio": row["studio"],
                        "performers": performers,
                        "view_count": view_count,
                        "o_count": o_count,
                        "replay_count": replay_count,
                        "o_rate": round(o_rate, 2),
                        "completion_rate": round(completion_rate, 1),
                        "recency_decay": round(recency_decay, 2),
                        "score": round(score, 2),
                        "scoring_mode": scoring_mode,
                    }
                )

            conn.close()

            # Sort by score descending
            scenes.sort(key=lambda x: x["score"], reverse=True)

            # Apply limit if specified
            if limit is not None and limit > 0:
                scenes = scenes[:limit]

            # Build formatted output with mode-appropriate labeling
            formatted_lines = []
            for i, s in enumerate(scenes):
                performers_str = ", ".join(s["performers"][:3]) if s["performers"] else "Unknown"
                if len(s["performers"]) > 3:
                    performers_str += f" +{len(s['performers']) - 3}"

                # Mode-specific detail
                if scoring_mode == "completion":
                    detail = f"completion: {s['completion_rate']:.0f}%"
                elif scoring_mode == "intensity":
                    detail = f"o-rate: {s['o_rate']:.2f}, views: {s['view_count']}"
                elif scoring_mode == "recent":
                    detail = f"score: {s['score']:.1f}, recency: {s['recency_decay']:.0%}"
                else:
                    detail = (
                        f"score: {s['score']:.1f}, o: {s['o_count']}, replays: {s['replay_count']}"
                    )

                formatted_lines.append(
                    f"{i + 1}. [{s['title']}]({s['url']}) - {performers_str} ({detail})"
                )

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "count": len(scenes),
                    "total_requested": len(scene_ids),
                    "scoring_mode": scoring_mode,
                    "min_score_filter": min_score,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No scenes meet the criteria.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryPerformersByAttributeTool(BaseTool):
    """
    Tool to find performers matching physical/demographic attributes.

    Enables queries like "Find blonde performers", "Show me performers from Japan",
    "Who are the tall brunettes in my library?"
    """

    @property
    def name(self) -> str:
        return "query_performers_by_attribute"

    @property
    def description(self) -> str:
        return (
            "Find performers matching physical or demographic attributes. "
            "Can filter by gender, hair color, ethnicity, country, height range, "
            "age range, tattoos, and piercings. Returns performers with key stats."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "gender",
                "type": "string",
                "description": "Filter by gender (e.g., 'female', 'male', 'transgender_female')",
                "required": False,
                "enum": None,
            },
            {
                "name": "hair_color",
                "type": "string",
                "description": "Filter by hair color (e.g., 'blonde', 'brunette', 'red', 'black')",
                "required": False,
                "enum": None,
            },
            {
                "name": "ethnicity",
                "type": "string",
                "description": "Filter by ethnicity (e.g., 'caucasian', 'asian', 'latina')",
                "required": False,
                "enum": None,
            },
            {
                "name": "country",
                "type": "string",
                "description": "Filter by country (e.g., 'USA', 'Japan', 'France')",
                "required": False,
                "enum": None,
            },
            {
                "name": "min_height",
                "type": "integer",
                "description": "Minimum height in cm",
                "required": False,
                "enum": None,
            },
            {
                "name": "max_height",
                "type": "integer",
                "description": "Maximum height in cm",
                "required": False,
                "enum": None,
            },
            {
                "name": "min_age",
                "type": "integer",
                "description": "Minimum age (calculated from birthdate)",
                "required": False,
                "enum": None,
            },
            {
                "name": "max_age",
                "type": "integer",
                "description": "Maximum age (calculated from birthdate)",
                "required": False,
                "enum": None,
            },
            {
                "name": "has_tattoos",
                "type": "boolean",
                "description": "Filter by tattoo presence (true = has tattoos, false = no tattoos)",
                "required": False,
                "enum": None,
            },
            {
                "name": "has_piercings",
                "type": "boolean",
                "description": "Filter by piercing presence (true = has piercings, false = no piercings)",
                "required": False,
                "enum": None,
            },
            {
                "name": "is_favorite",
                "type": "boolean",
                "description": "Filter by favorite status",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: scene_count)",
                "required": False,
                "enum": ["scene_count", "name", "view_count", "o_count"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of performers to return (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performers by attribute query."""
        gender: str | None = kwargs.get("gender")
        hair_color: str | None = kwargs.get("hair_color")
        ethnicity: str | None = kwargs.get("ethnicity")
        country: str | None = kwargs.get("country")
        min_height: int | None = kwargs.get("min_height")
        max_height: int | None = kwargs.get("max_height")
        min_age: int | None = kwargs.get("min_age")
        max_age: int | None = kwargs.get("max_age")
        has_tattoos: bool | None = kwargs.get("has_tattoos")
        has_piercings: bool | None = kwargs.get("has_piercings")
        is_favorite: bool | None = kwargs.get("is_favorite")
        sort_by: str = kwargs.get("sort_by", "scene_count")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build WHERE clause dynamically
            conditions: list[str] = []
            params: list[Any] = []

            if gender:
                conditions.append("LOWER(p.gender) LIKE LOWER(?)")
                params.append(f"%{gender}%")

            if hair_color:
                conditions.append("LOWER(p.hair_color) LIKE LOWER(?)")
                params.append(f"%{hair_color}%")

            if ethnicity:
                conditions.append("LOWER(p.ethnicity) LIKE LOWER(?)")
                params.append(f"%{ethnicity}%")

            if country:
                conditions.append("LOWER(p.country) LIKE LOWER(?)")
                params.append(f"%{country}%")

            if min_height is not None:
                conditions.append("p.height >= ?")
                params.append(min_height)

            if max_height is not None:
                conditions.append("p.height <= ?")
                params.append(max_height)

            if min_age is not None:
                # Calculate max birthdate for min age
                conditions.append("date(p.birthdate) <= date('now', '-' || ? || ' years')")
                params.append(min_age)

            if max_age is not None:
                # Calculate min birthdate for max age
                conditions.append("date(p.birthdate) >= date('now', '-' || ? || ' years')")
                params.append(max_age)

            if has_tattoos is not None:
                if has_tattoos:
                    conditions.append("p.tattoos IS NOT NULL AND p.tattoos != ''")
                else:
                    conditions.append("(p.tattoos IS NULL OR p.tattoos = '')")

            if has_piercings is not None:
                if has_piercings:
                    conditions.append("p.piercings IS NOT NULL AND p.piercings != ''")
                else:
                    conditions.append("(p.piercings IS NULL OR p.piercings = '')")

            if is_favorite is not None:
                conditions.append("p.favorite = ?")
                params.append(1 if is_favorite else 0)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Build ORDER BY clause
            if sort_by == "name":
                order_clause = "ORDER BY p.name ASC"
            elif sort_by == "view_count":
                order_clause = "ORDER BY view_count DESC"
            elif sort_by == "o_count":
                order_clause = "ORDER BY o_count DESC"
            else:  # scene_count (default)
                order_clause = "ORDER BY scene_count DESC"

            params.append(limit)

            cursor.execute(
                f"""
                SELECT
                    p.id,
                    p.name,
                    p.gender,
                    p.hair_color,
                    p.ethnicity,
                    p.country,
                    p.height,
                    p.birthdate,
                    p.favorite,
                    p.tattoos,
                    p.piercings,
                    COUNT(DISTINCT ps.scene_id) as scene_count,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    COALESCE(o_agg.o_count, 0) as o_count
                FROM performers p
                LEFT JOIN performers_scenes ps ON p.id = ps.performer_id
                LEFT JOIN (
                    SELECT ps2.performer_id, COUNT(svd.view_date) as view_count
                    FROM performers_scenes ps2
                    JOIN scenes_view_dates svd ON ps2.scene_id = svd.scene_id
                    GROUP BY ps2.performer_id
                ) view_agg ON p.id = view_agg.performer_id
                LEFT JOIN (
                    SELECT ps3.performer_id, COUNT(sod.o_date) as o_count
                    FROM performers_scenes ps3
                    JOIN scenes_o_dates sod ON ps3.scene_id = sod.scene_id
                    GROUP BY ps3.performer_id
                ) o_agg ON p.id = o_agg.performer_id
                {where_clause}
                GROUP BY p.id
                {order_clause}
                LIMIT ?
                """,
                params,
            )

            performers = []
            for row in cursor.fetchall():
                # Calculate age if birthdate exists
                age = None
                if row["birthdate"]:
                    try:
                        from datetime import datetime

                        birth_dt = datetime.fromisoformat(row["birthdate"])
                        today = datetime.now()
                        age = today.year - birth_dt.year
                        if (today.month, today.day) < (birth_dt.month, birth_dt.day):
                            age -= 1
                    except (ValueError, TypeError):
                        pass

                performers.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/performers/{row['id']}",
                        "gender": row["gender"],
                        "hair_color": row["hair_color"],
                        "ethnicity": row["ethnicity"],
                        "country": row["country"],
                        "height_cm": row["height"],
                        "age": age,
                        "favorite": bool(row["favorite"]),
                        "has_tattoos": bool(row["tattoos"]),
                        "has_piercings": bool(row["piercings"]),
                        "scene_count": row["scene_count"],
                        "view_count": row["view_count"],
                        "o_count": row["o_count"],
                    }
                )

            conn.close()

            # Build formatted output
            formatted_lines = []
            for p in performers:
                attrs = []
                if p["hair_color"]:
                    attrs.append(p["hair_color"])
                if p["ethnicity"]:
                    attrs.append(p["ethnicity"])
                if p["age"]:
                    attrs.append(f"{p['age']}yo")
                if p["height_cm"]:
                    attrs.append(f"{p['height_cm']}cm")
                attrs_str = ", ".join(attrs) if attrs else "no attributes"
                formatted_lines.append(
                    f"- [{p['name']}]({p['url']}) ({attrs_str}) - "
                    f"{p['scene_count']} scenes, {p['view_count']} views"
                )

            # Build filter summary
            filters_applied = []
            if gender:
                filters_applied.append(f"gender={gender}")
            if hair_color:
                filters_applied.append(f"hair={hair_color}")
            if ethnicity:
                filters_applied.append(f"ethnicity={ethnicity}")
            if country:
                filters_applied.append(f"country={country}")
            if min_height or max_height:
                h_range = f"{min_height or '?'}-{max_height or '?'}cm"
                filters_applied.append(f"height={h_range}")
            if min_age or max_age:
                a_range = f"{min_age or '?'}-{max_age or '?'}yo"
                filters_applied.append(f"age={a_range}")
            if has_tattoos is not None:
                filters_applied.append(f"tattoos={has_tattoos}")
            if has_piercings is not None:
                filters_applied.append(f"piercings={has_piercings}")
            if is_favorite is not None:
                filters_applied.append(f"favorite={is_favorite}")

            return {
                "success": True,
                "data": {
                    "performers": performers,
                    "count": len(performers),
                    "filters_applied": filters_applied,
                    "sort_by": sort_by,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No performers found matching the criteria.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryScenesByDateTool(BaseTool):
    """
    Tool to find scenes by release date, date added, or viewing date.

    Enables queries like "What scenes were released in 2023?",
    "What did I add to my library last month?",
    "Show me scenes I watched this week".
    """

    @property
    def name(self) -> str:
        return "query_scenes_by_date"

    @property
    def description(self) -> str:
        return (
            "Find scenes by release date, date added, or viewing date. "
            "Can filter by date range and sort by date or engagement. "
            "Useful for finding new content or browsing viewing history."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "start_date",
                "type": "string",
                "description": "Start date in YYYY-MM-DD format",
                "required": False,
                "enum": None,
            },
            {
                "name": "end_date",
                "type": "string",
                "description": "End date in YYYY-MM-DD format",
                "required": False,
                "enum": None,
            },
            {
                "name": "date_type",
                "type": "string",
                "description": "Which date to filter by (default: scene_date)",
                "required": False,
                "enum": ["scene_date", "created_at", "view_date"],
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: date_desc)",
                "required": False,
                "enum": ["date_asc", "date_desc", "engagement", "title"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes to return (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the scenes by date query."""
        start_date: str | None = kwargs.get("start_date")
        end_date: str | None = kwargs.get("end_date")
        date_type: str = kwargs.get("date_type", "scene_date")
        sort_by: str = kwargs.get("sort_by", "date_desc")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Determine which date column to use
            if date_type == "scene_date":
                date_column = "s.date"
                date_table = ""
            elif date_type == "created_at":
                date_column = "s.created_at"
                date_table = ""
            elif date_type == "view_date":
                date_column = "svd.view_date"
                date_table = "JOIN scenes_view_dates svd ON s.id = svd.scene_id"
            else:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Invalid date_type: {date_type}",
                }

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            if start_date:
                conditions.append(f"date({date_column}) >= date(?)")
                params.append(start_date)

            if end_date:
                conditions.append(f"date({date_column}) <= date(?)")
                params.append(end_date)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Build ORDER BY clause
            if sort_by == "date_asc":
                order_clause = f"ORDER BY {date_column} ASC"
            elif sort_by == "date_desc":
                order_clause = f"ORDER BY {date_column} DESC"
            elif sort_by == "engagement":
                order_clause = "ORDER BY engagement_score DESC"
            elif sort_by == "title":
                order_clause = "ORDER BY s.title ASC"
            else:
                order_clause = f"ORDER BY {date_column} DESC"

            params.append(limit)

            # Build query
            query = f"""
                SELECT
                    s.id,
                    s.title,
                    s.date as scene_date,
                    s.created_at,
                    st.name as studio,
                    GROUP_CONCAT(DISTINCT p.name) as performers,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    COALESCE(o_agg.o_count, 0) as o_count,
                    (COALESCE(o_agg.o_count, 0) * 20.0 +
                     GREATEST(COALESCE(view_agg.view_count, 0) - 1, 0) * 2.0) as engagement_score
                FROM scenes s
                {date_table}
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                {where_clause}
                GROUP BY s.id
                {order_clause}
                LIMIT ?
            """

            cursor.execute(query, params)

            scenes = []
            for row in cursor.fetchall():
                scenes.append(
                    {
                        "id": row["id"],
                        "title": row["title"] or f"Scene {row['id']}",
                        "url": f"/scenes/{row['id']}",
                        "scene_date": row["scene_date"],
                        "created_at": row["created_at"],
                        "studio": row["studio"],
                        "performers": row["performers"].split(",") if row["performers"] else [],
                        "view_count": row["view_count"],
                        "o_count": row["o_count"],
                        "engagement_score": round(row["engagement_score"], 1),
                    }
                )

            conn.close()

            # Build formatted output
            formatted_lines = []
            for s in scenes:
                performers_str = ", ".join(s["performers"][:2]) if s["performers"] else "Unknown"
                if len(s["performers"]) > 2:
                    performers_str += f" +{len(s['performers']) - 2}"

                date_str = s["scene_date"] or s["created_at"] or "No date"
                if date_str and "T" in date_str:
                    date_str = date_str.split("T")[0]

                formatted_lines.append(
                    f"- [{s['title']}]({s['url']}) - {date_str} - {performers_str}"
                )

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "count": len(scenes),
                    "date_type": date_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sort_by": sort_by,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No scenes found in the specified date range.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryFavoritesTool(BaseTool):
    """
    Tool to get all favorited items (performers, studios, tags).

    Enables queries like "Who are my favorite performers?",
    "Show my favorite studios", "What tags have I favorited?"
    """

    @property
    def name(self) -> str:
        return "query_favorites"

    @property
    def description(self) -> str:
        return (
            "Get all favorited items including performers, studios, and tags. "
            "Can filter by entity type and includes engagement statistics. "
            "Favorites are explicitly marked by the user in Stash."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "entity_type",
                "type": "string",
                "description": "Type of entity to get favorites for (default: all)",
                "required": False,
                "enum": ["all", "performers", "studios", "tags"],
            },
            {
                "name": "include_stats",
                "type": "boolean",
                "description": "Include engagement statistics (default: true)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number per category (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the favorites query."""
        entity_type: str = kwargs.get("entity_type", "all")
        include_stats: bool = kwargs.get("include_stats", True)
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            result_data: dict[str, Any] = {}

            # Get favorite performers
            if entity_type in ("all", "performers"):
                if include_stats:
                    cursor.execute(
                        """
                        SELECT
                            p.id,
                            p.name,
                            COUNT(DISTINCT ps.scene_id) as scene_count,
                            COALESCE(view_agg.view_count, 0) as view_count,
                            COALESCE(o_agg.o_count, 0) as o_count
                        FROM performers p
                        LEFT JOIN performers_scenes ps ON p.id = ps.performer_id
                        LEFT JOIN (
                            SELECT ps2.performer_id, COUNT(svd.view_date) as view_count
                            FROM performers_scenes ps2
                            JOIN scenes_view_dates svd ON ps2.scene_id = svd.scene_id
                            GROUP BY ps2.performer_id
                        ) view_agg ON p.id = view_agg.performer_id
                        LEFT JOIN (
                            SELECT ps3.performer_id, COUNT(sod.o_date) as o_count
                            FROM performers_scenes ps3
                            JOIN scenes_o_dates sod ON ps3.scene_id = sod.scene_id
                            GROUP BY ps3.performer_id
                        ) o_agg ON p.id = o_agg.performer_id
                        WHERE p.favorite = 1
                        GROUP BY p.id
                        ORDER BY view_count DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT p.id, p.name
                        FROM performers p
                        WHERE p.favorite = 1
                        ORDER BY p.name
                        LIMIT ?
                        """,
                        (limit,),
                    )

                performers = []
                for row in cursor.fetchall():
                    p = {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/performers/{row['id']}",
                    }
                    if include_stats:
                        p["scene_count"] = row["scene_count"]
                        p["view_count"] = row["view_count"]
                        p["o_count"] = row["o_count"]
                    performers.append(p)
                result_data["favorite_performers"] = performers

            # Get favorite studios
            if entity_type in ("all", "studios"):
                if include_stats:
                    cursor.execute(
                        """
                        SELECT
                            st.id,
                            st.name,
                            COUNT(DISTINCT s.id) as scene_count,
                            COALESCE(view_agg.view_count, 0) as view_count
                        FROM studios st
                        LEFT JOIN scenes s ON st.id = s.studio_id
                        LEFT JOIN (
                            SELECT s2.studio_id, COUNT(svd.view_date) as view_count
                            FROM scenes s2
                            JOIN scenes_view_dates svd ON s2.id = svd.scene_id
                            GROUP BY s2.studio_id
                        ) view_agg ON st.id = view_agg.studio_id
                        WHERE st.favorite = 1
                        GROUP BY st.id
                        ORDER BY view_count DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT st.id, st.name
                        FROM studios st
                        WHERE st.favorite = 1
                        ORDER BY st.name
                        LIMIT ?
                        """,
                        (limit,),
                    )

                studios = []
                for row in cursor.fetchall():
                    s = {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/studios/{row['id']}",
                    }
                    if include_stats:
                        s["scene_count"] = row["scene_count"]
                        s["view_count"] = row["view_count"]
                    studios.append(s)
                result_data["favorite_studios"] = studios

            # Get favorite tags
            if entity_type in ("all", "tags"):
                if include_stats:
                    cursor.execute(
                        """
                        SELECT
                            t.id,
                            t.name,
                            COUNT(DISTINCT st.scene_id) as scene_count,
                            COALESCE(view_agg.view_count, 0) as view_count
                        FROM tags t
                        LEFT JOIN scenes_tags st ON t.id = st.tag_id
                        LEFT JOIN (
                            SELECT st2.tag_id, COUNT(svd.view_date) as view_count
                            FROM scenes_tags st2
                            JOIN scenes_view_dates svd ON st2.scene_id = svd.scene_id
                            GROUP BY st2.tag_id
                        ) view_agg ON t.id = view_agg.tag_id
                        WHERE t.favorite = 1
                        GROUP BY t.id
                        ORDER BY view_count DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT t.id, t.name
                        FROM tags t
                        WHERE t.favorite = 1
                        ORDER BY t.name
                        LIMIT ?
                        """,
                        (limit,),
                    )

                tags = []
                for row in cursor.fetchall():
                    t = {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/tags/{row['id']}",
                    }
                    if include_stats:
                        t["scene_count"] = row["scene_count"]
                        t["view_count"] = row["view_count"]
                    tags.append(t)
                result_data["favorite_tags"] = tags

            conn.close()

            # Build formatted output
            formatted_sections = []

            if result_data.get("favorite_performers"):
                lines = ["**Favorite Performers:**"]
                for p in result_data["favorite_performers"]:
                    stats = (
                        f" ({p.get('scene_count', 0)} scenes, {p.get('view_count', 0)} views)"
                        if include_stats
                        else ""
                    )
                    lines.append(f"- [{p['name']}]({p['url']}){stats}")
                formatted_sections.append("\n".join(lines))

            if result_data.get("favorite_studios"):
                lines = ["**Favorite Studios:**"]
                for s in result_data["favorite_studios"]:
                    stats = f" ({s.get('scene_count', 0)} scenes)" if include_stats else ""
                    lines.append(f"- [{s['name']}]({s['url']}){stats}")
                formatted_sections.append("\n".join(lines))

            if result_data.get("favorite_tags"):
                lines = ["**Favorite Tags:**"]
                for t in result_data["favorite_tags"]:
                    stats = f" ({t.get('scene_count', 0)} scenes)" if include_stats else ""
                    lines.append(f"- [{t['name']}]({t['url']}){stats}")
                formatted_sections.append("\n".join(lines))

            # Count totals
            total_count = 0
            if "favorite_performers" in result_data:
                total_count += len(result_data["favorite_performers"])
            if "favorite_studios" in result_data:
                total_count += len(result_data["favorite_studios"])
            if "favorite_tags" in result_data:
                total_count += len(result_data["favorite_tags"])

            result_data["total_favorites"] = total_count
            result_data["entity_type"] = entity_type
            result_data["formatted_results"] = (
                "\n\n".join(formatted_sections) if formatted_sections else "No favorites found."
            )

            return {
                "success": True,
                "data": result_data,
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryResumePointsTool(BaseTool):
    """
    Tool to find scenes with resume points (partially watched content).

    Enables queries like "What scenes did I start but not finish?",
    "Show me my 'continue watching' list".
    """

    @property
    def name(self) -> str:
        return "query_resume_points"

    @property
    def description(self) -> str:
        return (
            "Find scenes with resume points (partially watched). "
            "These are scenes where playback was stopped before completion. "
            "Useful for finding your 'continue watching' list."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "min_progress",
                "type": "number",
                "description": "Minimum progress percentage (0.0-1.0, default: 0.05)",
                "required": False,
                "enum": None,
            },
            {
                "name": "max_progress",
                "type": "number",
                "description": "Maximum progress percentage (0.0-1.0, default: 0.95)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: resume_time)",
                "required": False,
                "enum": ["resume_time", "progress", "title"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the resume points query."""
        min_progress: float = kwargs.get("min_progress", 0.05)
        max_progress: float = kwargs.get("max_progress", 0.95)
        sort_by: str = kwargs.get("sort_by", "resume_time")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build ORDER BY clause
            if sort_by == "progress":
                order_clause = "ORDER BY progress DESC"
            elif sort_by == "title":
                order_clause = "ORDER BY s.title ASC"
            else:  # resume_time (most recently paused)
                order_clause = "ORDER BY s.resume_time DESC"

            cursor.execute(
                f"""
                SELECT
                    s.id,
                    s.title,
                    s.resume_time,
                    vf.duration as video_duration,
                    s.resume_time / NULLIF(vf.duration, 0) as progress,
                    st.name as studio,
                    GROUP_CONCAT(DISTINCT p.name) as performers
                FROM scenes s
                JOIN scenes_files fs ON s.id = fs.scene_id
                JOIN video_files vf ON fs.file_id = vf.file_id
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
                WHERE s.resume_time > 0
                  AND vf.duration > 0
                  AND (s.resume_time / vf.duration) >= ?
                  AND (s.resume_time / vf.duration) <= ?
                GROUP BY s.id
                {order_clause}
                LIMIT ?
                """,
                (min_progress, max_progress, limit),
            )

            scenes = []
            for row in cursor.fetchall():
                progress = row["progress"] or 0
                resume_time = row["resume_time"] or 0
                video_duration = row["video_duration"] or 0

                # Format time as HH:MM:SS
                resume_mins = int(resume_time // 60)
                resume_secs = int(resume_time % 60)
                resume_str = f"{resume_mins}:{resume_secs:02d}"

                duration_mins = int(video_duration // 60)
                duration_secs = int(video_duration % 60)
                duration_str = f"{duration_mins}:{duration_secs:02d}"

                scenes.append(
                    {
                        "id": row["id"],
                        "title": row["title"] or f"Scene {row['id']}",
                        "url": f"/scenes/{row['id']}",
                        "resume_time_seconds": resume_time,
                        "resume_time_formatted": resume_str,
                        "video_duration_seconds": video_duration,
                        "video_duration_formatted": duration_str,
                        "progress_percent": round(progress * 100, 1),
                        "studio": row["studio"],
                        "performers": row["performers"].split(",") if row["performers"] else [],
                    }
                )

            # Get total count of scenes with resume points
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM scenes s
                JOIN scenes_files fs ON s.id = fs.scene_id
                JOIN video_files vf ON fs.file_id = vf.file_id
                WHERE s.resume_time > 0
                  AND vf.duration > 0
                  AND (s.resume_time / vf.duration) >= ?
                  AND (s.resume_time / vf.duration) <= ?
                """,
                (min_progress, max_progress),
            )
            total_with_resume = cursor.fetchone()["count"]

            conn.close()

            # Build formatted output
            formatted_lines = []
            for s in scenes:
                performers_str = ", ".join(s["performers"][:2]) if s["performers"] else "Unknown"
                if len(s["performers"]) > 2:
                    performers_str += f" +{len(s['performers']) - 2}"

                formatted_lines.append(
                    f"- [{s['title']}]({s['url']}) - {s['progress_percent']:.0f}% "
                    f"({s['resume_time_formatted']}/{s['video_duration_formatted']}) - {performers_str}"
                )

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "count": len(scenes),
                    "total_with_resume_points": total_with_resume,
                    "min_progress": min_progress,
                    "max_progress": max_progress,
                    "sort_by": sort_by,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No partially watched scenes found.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryScenesByRatingTool(BaseTool):
    """
    Tool to find scenes by rating.

    Enables queries like "Show me my 5-star scenes",
    "What's my best-rated content?"
    """

    @property
    def name(self) -> str:
        return "query_scenes_by_rating"

    @property
    def description(self) -> str:
        return (
            "Find scenes by their rating (1-5 stars or 1-100 scale). "
            "Can filter by minimum rating and sort by rating or engagement. "
            "Ratings are explicitly set by the user in Stash."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "min_rating",
                "type": "integer",
                "description": "Minimum rating (1-100, where 100=5 stars). Default: 80 (4 stars)",
                "required": False,
                "enum": None,
            },
            {
                "name": "max_rating",
                "type": "integer",
                "description": "Maximum rating (1-100). Default: 100",
                "required": False,
                "enum": None,
            },
            {
                "name": "include_unrated",
                "type": "boolean",
                "description": "Include scenes without ratings (default: false)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: rating)",
                "required": False,
                "enum": ["rating", "engagement", "date", "title"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the scenes by rating query."""
        min_rating: int = kwargs.get("min_rating", 80)
        max_rating: int = kwargs.get("max_rating", 100)
        include_unrated: bool = kwargs.get("include_unrated", False)
        sort_by: str = kwargs.get("sort_by", "rating")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build WHERE clause
            if include_unrated:
                where_clause = "(s.rating IS NULL OR (s.rating >= ? AND s.rating <= ?))"
                params: list[Any] = [min_rating, max_rating]
            else:
                where_clause = "s.rating >= ? AND s.rating <= ?"
                params = [min_rating, max_rating]

            # Build ORDER BY clause
            if sort_by == "engagement":
                order_clause = "ORDER BY engagement_score DESC"
            elif sort_by == "date":
                order_clause = "ORDER BY s.date DESC NULLS LAST"
            elif sort_by == "title":
                order_clause = "ORDER BY s.title ASC"
            else:  # rating (default)
                order_clause = "ORDER BY s.rating DESC NULLS LAST"

            params.append(limit)

            cursor.execute(
                f"""
                SELECT
                    s.id,
                    s.title,
                    s.rating,
                    s.date,
                    st.name as studio,
                    GROUP_CONCAT(DISTINCT p.name) as performers,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    COALESCE(o_agg.o_count, 0) as o_count,
                    (COALESCE(o_agg.o_count, 0) * 20.0 +
                     GREATEST(COALESCE(view_agg.view_count, 0) - 1, 0) * 2.0) as engagement_score
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as o_count
                    FROM scenes_o_dates GROUP BY scene_id
                ) o_agg ON s.id = o_agg.scene_id
                WHERE {where_clause}
                GROUP BY s.id
                {order_clause}
                LIMIT ?
                """,
                params,
            )

            scenes = []
            for row in cursor.fetchall():
                rating = row["rating"]
                # Convert to 5-star scale
                stars = round(rating / 20, 1) if rating else None

                scenes.append(
                    {
                        "id": row["id"],
                        "title": row["title"] or f"Scene {row['id']}",
                        "url": f"/scenes/{row['id']}",
                        "rating_100": rating,
                        "stars": stars,
                        "date": row["date"],
                        "studio": row["studio"],
                        "performers": row["performers"].split(",") if row["performers"] else [],
                        "view_count": row["view_count"],
                        "o_count": row["o_count"],
                        "engagement_score": round(row["engagement_score"], 1),
                    }
                )

            # Get counts
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN rating >= 80 THEN 1 END) as rated_4_plus,
                    COUNT(CASE WHEN rating = 100 THEN 1 END) as rated_5
                FROM scenes
                WHERE rating IS NOT NULL
                """
            )
            counts = cursor.fetchone()

            conn.close()

            # Build formatted output
            formatted_lines = []
            for s in scenes:
                performers_str = ", ".join(s["performers"][:2]) if s["performers"] else "Unknown"
                if len(s["performers"]) > 2:
                    performers_str += f" +{len(s['performers']) - 2}"

                stars_str = f"{s['stars']}★" if s["stars"] else "No rating"
                formatted_lines.append(
                    f"- [{s['title']}]({s['url']}) - {stars_str} - {performers_str}"
                )

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "count": len(scenes),
                    "min_rating": min_rating,
                    "max_rating": max_rating,
                    "include_unrated": include_unrated,
                    "sort_by": sort_by,
                    "total_rated_scenes": counts["total"],
                    "scenes_4_stars_plus": counts["rated_4_plus"],
                    "scenes_5_stars": counts["rated_5"],
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No scenes found matching the rating criteria.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryAllTagsTool(BaseTool):
    """
    Tool to list all tags in the library.

    Enables queries like "What tags do I have?", "List all tags",
    "Show me available tags".
    """

    @property
    def name(self) -> str:
        return "query_all_tags"

    @property
    def description(self) -> str:
        return (
            "List all tags in the library with scene counts. "
            "Can filter by search term and sort by name or scene count. "
            "Useful for discovering available tags or auto-complete."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "search",
                "type": "string",
                "description": "Filter tags by name (case-insensitive partial match)",
                "required": False,
                "enum": None,
            },
            {
                "name": "favorites_only",
                "type": "boolean",
                "description": "Only show favorited tags (default: false)",
                "required": False,
                "enum": None,
            },
            {
                "name": "min_scene_count",
                "type": "integer",
                "description": "Minimum number of scenes with this tag (default: 0)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: scene_count)",
                "required": False,
                "enum": ["scene_count", "name", "view_count"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of tags to return (default: 50)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the all tags query."""
        search: str | None = kwargs.get("search")
        favorites_only: bool = kwargs.get("favorites_only", False)
        min_scene_count: int = kwargs.get("min_scene_count", 0)
        sort_by: str = kwargs.get("sort_by", "scene_count")
        limit: int = kwargs.get("limit", 50)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get plugin-level excluded tags including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            # Exclude configured tags
            if excluded_ids:
                placeholders = ",".join("?" * len(excluded_ids))
                conditions.append(f"t.id NOT IN ({placeholders})")
                params.extend(list(excluded_ids))

            if search:
                conditions.append("LOWER(t.name) LIKE LOWER(?)")
                params.append(f"%{search}%")

            if favorites_only:
                conditions.append("t.favorite = 1")

            # Build HAVING clause for min_scene_count
            having_clause = ""
            if min_scene_count > 0:
                having_clause = "HAVING scene_count >= ?"
                params.append(min_scene_count)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Build ORDER BY clause
            if sort_by == "name":
                order_clause = "ORDER BY t.name ASC"
            elif sort_by == "view_count":
                order_clause = "ORDER BY view_count DESC"
            else:  # scene_count (default)
                order_clause = "ORDER BY scene_count DESC"

            params.append(limit)

            cursor.execute(
                f"""
                SELECT
                    t.id,
                    t.name,
                    t.description,
                    t.favorite,
                    COUNT(DISTINCT st.scene_id) as scene_count,
                    COALESCE(view_agg.view_count, 0) as view_count
                FROM tags t
                LEFT JOIN scenes_tags st ON t.id = st.tag_id
                LEFT JOIN (
                    SELECT st2.tag_id, COUNT(svd.view_date) as view_count
                    FROM scenes_tags st2
                    JOIN scenes_view_dates svd ON st2.scene_id = svd.scene_id
                    GROUP BY st2.tag_id
                ) view_agg ON t.id = view_agg.tag_id
                {where_clause}
                GROUP BY t.id
                {having_clause}
                {order_clause}
                LIMIT ?
                """,
                params,
            )

            tags = []
            for row in cursor.fetchall():
                tags.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/tags/{row['id']}",
                        "description": row["description"],
                        "favorite": bool(row["favorite"]),
                        "scene_count": row["scene_count"],
                        "view_count": row["view_count"],
                    }
                )

            # Get total tag count
            cursor.execute("SELECT COUNT(*) as count FROM tags")
            total_tags = cursor.fetchone()["count"]

            conn.close()

            # Build formatted output
            formatted_lines = []
            for t in tags:
                fav_marker = "★ " if t["favorite"] else ""
                formatted_lines.append(
                    f"- {fav_marker}[{t['name']}]({t['url']}) ({t['scene_count']} scenes)"
                )

            return {
                "success": True,
                "data": {
                    "tags": tags,
                    "count": len(tags),
                    "total_tags_in_library": total_tags,
                    "search": search,
                    "favorites_only": favorites_only,
                    "sort_by": sort_by,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No tags found.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryAllPerformersTool(BaseTool):
    """
    Tool to list all performers in the library.

    Enables queries like "Who are all the performers?", "List all performers",
    "Show me performers in my library".
    """

    @property
    def name(self) -> str:
        return "query_all_performers"

    @property
    def description(self) -> str:
        return (
            "List all performers in the library with scene counts. "
            "Can filter by search term and sort by name or scene count. "
            "Useful for discovering available performers or auto-complete."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "search",
                "type": "string",
                "description": "Filter performers by name (case-insensitive partial match)",
                "required": False,
                "enum": None,
            },
            {
                "name": "favorites_only",
                "type": "boolean",
                "description": "Only show favorited performers (default: false)",
                "required": False,
                "enum": None,
            },
            {
                "name": "min_scene_count",
                "type": "integer",
                "description": "Minimum number of scenes with this performer (default: 0)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "How to sort results (default: scene_count)",
                "required": False,
                "enum": ["scene_count", "name", "view_count", "o_count"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of performers to return (default: 50)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the all performers query."""
        search: str | None = kwargs.get("search")
        favorites_only: bool = kwargs.get("favorites_only", False)
        min_scene_count: int = kwargs.get("min_scene_count", 0)
        sort_by: str = kwargs.get("sort_by", "scene_count")
        limit: int = kwargs.get("limit", 50)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            if search:
                conditions.append("LOWER(p.name) LIKE LOWER(?)")
                params.append(f"%{search}%")

            if favorites_only:
                conditions.append("p.favorite = 1")

            # Build HAVING clause for min_scene_count
            having_clause = ""
            if min_scene_count > 0:
                having_clause = "HAVING scene_count >= ?"
                params.append(min_scene_count)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Build ORDER BY clause
            if sort_by == "name":
                order_clause = "ORDER BY p.name ASC"
            elif sort_by == "view_count":
                order_clause = "ORDER BY view_count DESC"
            elif sort_by == "o_count":
                order_clause = "ORDER BY o_count DESC"
            else:  # scene_count (default)
                order_clause = "ORDER BY scene_count DESC"

            params.append(limit)

            cursor.execute(
                f"""
                SELECT
                    p.id,
                    p.name,
                    p.gender,
                    p.favorite,
                    COUNT(DISTINCT ps.scene_id) as scene_count,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    COALESCE(o_agg.o_count, 0) as o_count
                FROM performers p
                LEFT JOIN performers_scenes ps ON p.id = ps.performer_id
                LEFT JOIN (
                    SELECT ps2.performer_id, COUNT(svd.view_date) as view_count
                    FROM performers_scenes ps2
                    JOIN scenes_view_dates svd ON ps2.scene_id = svd.scene_id
                    GROUP BY ps2.performer_id
                ) view_agg ON p.id = view_agg.performer_id
                LEFT JOIN (
                    SELECT ps3.performer_id, COUNT(sod.o_date) as o_count
                    FROM performers_scenes ps3
                    JOIN scenes_o_dates sod ON ps3.scene_id = sod.scene_id
                    GROUP BY ps3.performer_id
                ) o_agg ON p.id = o_agg.performer_id
                {where_clause}
                GROUP BY p.id
                {having_clause}
                {order_clause}
                LIMIT ?
                """,
                params,
            )

            performers = []
            for row in cursor.fetchall():
                performers.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/performers/{row['id']}",
                        "gender": row["gender"],
                        "favorite": bool(row["favorite"]),
                        "scene_count": row["scene_count"],
                        "view_count": row["view_count"],
                        "o_count": row["o_count"],
                    }
                )

            # Get total performer count
            cursor.execute("SELECT COUNT(*) as count FROM performers")
            total_performers = cursor.fetchone()["count"]

            conn.close()

            # Build formatted output
            formatted_lines = []
            for p in performers:
                fav_marker = "★ " if p["favorite"] else ""
                gender_str = f" ({p['gender']})" if p["gender"] else ""
                formatted_lines.append(
                    f"- {fav_marker}[{p['name']}]({p['url']}){gender_str} - "
                    f"{p['scene_count']} scenes, {p['view_count']} views"
                )

            return {
                "success": True,
                "data": {
                    "performers": performers,
                    "count": len(performers),
                    "total_performers_in_library": total_performers,
                    "search": search,
                    "favorites_only": favorites_only,
                    "sort_by": sort_by,
                    "formatted_results": "\n".join(formatted_lines)
                    if formatted_lines
                    else "No performers found.",
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryPerformerProfileTool(BaseTool):
    """
    Tool to get detailed profile and statistics for a performer.

    Enables queries like "Tell me about performer X", "What are the stats for Y?",
    "Show me performer Z's profile".
    """

    @property
    def name(self) -> str:
        return "query_performer_profile"

    @property
    def description(self) -> str:
        return (
            "Get detailed profile and statistics for a specific performer. "
            "Returns profile data (gender, birthdate, ethnicity, country, hair_color, etc.), "
            "scene count, total play time, views, O count, most common tags, "
            "most common co-performers, and average scene rating."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_name",
                "type": "string",
                "description": "Name of the performer to look up (case-insensitive partial match)",
                "required": True,
                "enum": None,
            },
            {
                "name": "include_stats",
                "type": "boolean",
                "description": "Include engagement statistics (default: true)",
                "required": False,
                "enum": None,
            },
            {
                "name": "top_tags_limit",
                "type": "integer",
                "description": "Number of top tags to include (default: 10)",
                "required": False,
                "enum": None,
            },
            {
                "name": "top_coperformers_limit",
                "type": "integer",
                "description": "Number of top co-performers to include (default: 5)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performer profile query."""
        performer_name: str = kwargs.get("performer_name", "")
        include_stats: bool = kwargs.get("include_stats", True)
        top_tags_limit: int = kwargs.get("top_tags_limit", 10)
        top_coperformers_limit: int = kwargs.get("top_coperformers_limit", 5)

        if not performer_name:
            return {
                "success": False,
                "data": None,
                "error": "performer_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find the performer (fuzzy match)
            cursor.execute(
                """
                SELECT id, name FROM performers
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY
                    CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                    name
                LIMIT 1
                """,
                (f"%{performer_name}%", performer_name),
            )
            performer_row = cursor.fetchone()

            if not performer_row:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"No performer found matching '{performer_name}'",
                }

            performer_id = performer_row["id"]
            _performer_actual_name = performer_row["name"]  # Available for future use

            # Get full performer profile
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    p.disambiguation,
                    p.gender,
                    p.birthdate,
                    p.ethnicity,
                    p.country,
                    p.hair_color,
                    p.eye_color,
                    p.height,
                    p.weight,
                    p.measurements,
                    p.fake_tits,
                    p.tattoos,
                    p.piercings,
                    p.favorite,
                    p.details,
                    p.death_date,
                    p.career_length,
                    p.created_at,
                    p.updated_at
                FROM performers p
                WHERE p.id = ?
                """,
                (performer_id,),
            )
            profile_row = cursor.fetchone()

            # Calculate age if birthdate is available
            age = None
            if profile_row["birthdate"]:
                try:
                    from datetime import date

                    birth = date.fromisoformat(profile_row["birthdate"])
                    today = date.today()
                    age = (
                        today.year
                        - birth.year
                        - ((today.month, today.day) < (birth.month, birth.day))
                    )
                except (ValueError, TypeError):
                    pass

            profile = {
                "id": profile_row["id"],
                "name": profile_row["name"],
                "url": f"/performers/{profile_row['id']}",
                "disambiguation": profile_row["disambiguation"],
                "gender": profile_row["gender"],
                "birthdate": profile_row["birthdate"],
                "age": age,
                "ethnicity": profile_row["ethnicity"],
                "country": profile_row["country"],
                "hair_color": profile_row["hair_color"],
                "eye_color": profile_row["eye_color"],
                "height": profile_row["height"],
                "weight": profile_row["weight"],
                "measurements": profile_row["measurements"],
                "fake_tits": profile_row["fake_tits"],
                "tattoos": profile_row["tattoos"],
                "piercings": profile_row["piercings"],
                "favorite": bool(profile_row["favorite"]),
                "details": profile_row["details"],
                "death_date": profile_row["death_date"],
                "career_length": profile_row["career_length"],
            }

            # Get scene count
            cursor.execute(
                """
                SELECT COUNT(*) as scene_count
                FROM performers_scenes
                WHERE performer_id = ?
                """,
                (performer_id,),
            )
            profile["scene_count"] = cursor.fetchone()["scene_count"]

            # Get engagement stats if requested
            stats = {}
            if include_stats:
                # Total play time, views, O count
                cursor.execute(
                    """
                    SELECT
                        COALESCE(SUM(s.play_duration), 0) as total_play_seconds,
                        COALESCE(view_agg.view_count, 0) as view_count,
                        COALESCE(o_agg.o_count, 0) as o_count
                    FROM performers_scenes ps
                    JOIN scenes s ON ps.scene_id = s.id
                    LEFT JOIN (
                        SELECT ps2.performer_id, COUNT(svd.view_date) as view_count
                        FROM performers_scenes ps2
                        JOIN scenes_view_dates svd ON ps2.scene_id = svd.scene_id
                        WHERE ps2.performer_id = ?
                        GROUP BY ps2.performer_id
                    ) view_agg ON 1=1
                    LEFT JOIN (
                        SELECT ps3.performer_id, COUNT(sod.o_date) as o_count
                        FROM performers_scenes ps3
                        JOIN scenes_o_dates sod ON ps3.scene_id = sod.scene_id
                        WHERE ps3.performer_id = ?
                        GROUP BY ps3.performer_id
                    ) o_agg ON 1=1
                    WHERE ps.performer_id = ?
                    """,
                    (performer_id, performer_id, performer_id),
                )
                stats_row = cursor.fetchone()
                play_hours = (stats_row["total_play_seconds"] or 0) / 3600.0
                stats = {
                    "total_play_hours": round(play_hours, 2),
                    "view_count": stats_row["view_count"] or 0,
                    "o_count": stats_row["o_count"] or 0,
                }

                # Average scene rating
                cursor.execute(
                    """
                    SELECT AVG(CAST(s.rating AS FLOAT) / 20) as avg_rating
                    FROM performers_scenes ps
                    JOIN scenes s ON ps.scene_id = s.id
                    WHERE ps.performer_id = ? AND s.rating IS NOT NULL
                    """,
                    (performer_id,),
                )
                avg_rating_row = cursor.fetchone()
                stats["avg_rating"] = (
                    round(avg_rating_row["avg_rating"], 2) if avg_rating_row["avg_rating"] else None
                )

            # Get top tags for this performer
            cursor.execute(
                """
                SELECT
                    t.id,
                    t.name,
                    COUNT(*) as count
                FROM performers_scenes ps
                JOIN scenes_tags st ON ps.scene_id = st.scene_id
                JOIN tags t ON st.tag_id = t.id
                WHERE ps.performer_id = ?
                GROUP BY t.id
                ORDER BY count DESC
                LIMIT ?
                """,
                (performer_id, top_tags_limit),
            )
            top_tags = [
                {"id": row["id"], "name": row["name"], "count": row["count"]}
                for row in cursor.fetchall()
            ]

            # Get top co-performers
            cursor.execute(
                """
                SELECT
                    p2.id,
                    p2.name,
                    COUNT(*) as shared_scenes
                FROM performers_scenes ps1
                JOIN performers_scenes ps2 ON ps1.scene_id = ps2.scene_id
                JOIN performers p2 ON ps2.performer_id = p2.id
                WHERE ps1.performer_id = ? AND ps2.performer_id != ?
                GROUP BY p2.id
                ORDER BY shared_scenes DESC
                LIMIT ?
                """,
                (performer_id, performer_id, top_coperformers_limit),
            )
            top_coperformers = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "url": f"/performers/{row['id']}",
                    "shared_scenes": row["shared_scenes"],
                }
                for row in cursor.fetchall()
            ]

            conn.close()

            # Build formatted output
            formatted_lines = [f"## {profile['name']}"]
            if profile["favorite"]:
                formatted_lines[0] += " ★"

            # Basic info
            info_parts = []
            if profile["gender"]:
                info_parts.append(profile["gender"])
            if profile["age"]:
                info_parts.append(f"Age {profile['age']}")
            if profile["ethnicity"]:
                info_parts.append(profile["ethnicity"])
            if profile["country"]:
                info_parts.append(profile["country"])
            if info_parts:
                formatted_lines.append(f"**Info:** {', '.join(info_parts)}")

            # Physical
            physical_parts = []
            if profile["hair_color"]:
                physical_parts.append(f"{profile['hair_color']} hair")
            if profile["eye_color"]:
                physical_parts.append(f"{profile['eye_color']} eyes")
            if profile["height"]:
                physical_parts.append(f"{profile['height']}cm")
            if profile["measurements"]:
                physical_parts.append(profile["measurements"])
            if physical_parts:
                formatted_lines.append(f"**Physical:** {', '.join(physical_parts)}")

            formatted_lines.append(f"**Scenes:** {profile['scene_count']}")

            if include_stats and stats:
                formatted_lines.append(
                    f"**Engagement:** {stats['view_count']} views, {stats['o_count']} O's, "
                    f"{stats['total_play_hours']} hours watched"
                )
                if stats["avg_rating"]:
                    formatted_lines.append(f"**Avg Rating:** {stats['avg_rating']:.1f}★")

            if top_tags:
                tags_str = ", ".join([f"{t['name']} ({t['count']})" for t in top_tags[:5]])
                formatted_lines.append(f"**Top Tags:** {tags_str}")

            if top_coperformers:
                coperf_str = ", ".join(
                    [f"[{c['name']}]({c['url']}) ({c['shared_scenes']})" for c in top_coperformers]
                )
                formatted_lines.append(f"**Frequent Co-performers:** {coperf_str}")

            return {
                "success": True,
                "data": {
                    "profile": profile,
                    "stats": stats if include_stats else None,
                    "top_tags": top_tags,
                    "top_coperformers": top_coperformers,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryStudioProfileTool(BaseTool):
    """
    Tool to get detailed profile and statistics for a studio.

    Enables queries like "Tell me about studio X", "What are the stats for studio Y?",
    "Show me studio Z's profile".
    """

    @property
    def name(self) -> str:
        return "query_studio_profile"

    @property
    def description(self) -> str:
        return (
            "Get detailed profile and statistics for a specific studio. "
            "Returns scene count, total duration, average rating, "
            "top performers, top tags, date range (earliest/latest scene), "
            "engagement statistics, and parent/child studio relationships."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "studio_name",
                "type": "string",
                "description": "Name of the studio to look up (case-insensitive partial match)",
                "required": True,
                "enum": None,
            },
            {
                "name": "include_performers",
                "type": "boolean",
                "description": "Include top performers for this studio (default: true)",
                "required": False,
                "enum": None,
            },
            {
                "name": "top_performers_limit",
                "type": "integer",
                "description": "Number of top performers to include (default: 10)",
                "required": False,
                "enum": None,
            },
            {
                "name": "top_tags_limit",
                "type": "integer",
                "description": "Number of top tags to include (default: 10)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the studio profile query."""
        studio_name: str = kwargs.get("studio_name", "")
        include_performers: bool = kwargs.get("include_performers", True)
        top_performers_limit: int = kwargs.get("top_performers_limit", 10)
        top_tags_limit: int = kwargs.get("top_tags_limit", 10)

        if not studio_name:
            return {
                "success": False,
                "data": None,
                "error": "studio_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find the studio (fuzzy match)
            cursor.execute(
                """
                SELECT id, name FROM studios
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY
                    CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                    name
                LIMIT 1
                """,
                (f"%{studio_name}%", studio_name),
            )
            studio_row = cursor.fetchone()

            if not studio_row:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"No studio found matching '{studio_name}'",
                }

            studio_id = studio_row["id"]

            # Get full studio profile with stats
            cursor.execute(
                """
                SELECT
                    st.id,
                    st.name,
                    st.url,
                    st.parent_id,
                    st.rating,
                    st.favorite,
                    st.details,
                    parent_st.name as parent_name,
                    COUNT(DISTINCT s.id) as scene_count,
                    COALESCE(SUM(vf.duration), 0) as total_duration_seconds,
                    MIN(s.date) as earliest_scene_date,
                    MAX(s.date) as latest_scene_date,
                    AVG(CAST(s.rating AS FLOAT) / 20) as avg_rating,
                    COALESCE(SUM(s.play_duration), 0) as total_play_seconds
                FROM studios st
                LEFT JOIN studios parent_st ON st.parent_id = parent_st.id
                LEFT JOIN scenes s ON s.studio_id = st.id
                LEFT JOIN scenes_files fs ON s.id = fs.scene_id
                LEFT JOIN video_files vf ON fs.file_id = vf.file_id
                WHERE st.id = ?
                GROUP BY st.id
                """,
                (studio_id,),
            )
            profile_row = cursor.fetchone()

            total_hours = (profile_row["total_duration_seconds"] or 0) / 3600.0
            play_hours = (profile_row["total_play_seconds"] or 0) / 3600.0

            profile = {
                "id": profile_row["id"],
                "name": profile_row["name"],
                "url": f"/studios/{profile_row['id']}",
                "website_url": profile_row["url"],
                "parent_id": profile_row["parent_id"],
                "parent_name": profile_row["parent_name"],
                "rating": round(profile_row["rating"] / 20, 1) if profile_row["rating"] else None,
                "favorite": bool(profile_row["favorite"]),
                "details": profile_row["details"],
                "scene_count": profile_row["scene_count"],
                "total_hours": round(total_hours, 2),
                "earliest_scene_date": profile_row["earliest_scene_date"],
                "latest_scene_date": profile_row["latest_scene_date"],
                "avg_rating": round(profile_row["avg_rating"], 2)
                if profile_row["avg_rating"]
                else None,
            }

            # Get engagement stats (views and O's for this studio's scenes)
            cursor.execute(
                """
                SELECT
                    COALESCE(COUNT(DISTINCT svd.view_date), 0) as view_count,
                    COALESCE(COUNT(DISTINCT sod.o_date), 0) as o_count
                FROM scenes s
                LEFT JOIN scenes_view_dates svd ON s.id = svd.scene_id
                LEFT JOIN scenes_o_dates sod ON s.id = sod.scene_id
                WHERE s.studio_id = ?
                """,
                (studio_id,),
            )
            engagement_row = cursor.fetchone()
            stats = {
                "view_count": engagement_row["view_count"],
                "o_count": engagement_row["o_count"],
                "total_play_hours": round(play_hours, 2),
            }

            # Get child studios
            cursor.execute(
                """
                SELECT id, name, (SELECT COUNT(*) FROM scenes WHERE studio_id = studios.id) as scene_count
                FROM studios
                WHERE parent_id = ?
                ORDER BY scene_count DESC
                """,
                (studio_id,),
            )
            child_studios = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "url": f"/studios/{row['id']}",
                    "scene_count": row["scene_count"],
                }
                for row in cursor.fetchall()
            ]

            # Get top performers if requested
            top_performers = []
            if include_performers:
                cursor.execute(
                    """
                    SELECT
                        p.id,
                        p.name,
                        COUNT(*) as scene_count
                    FROM scenes s
                    JOIN performers_scenes ps ON s.id = ps.scene_id
                    JOIN performers p ON ps.performer_id = p.id
                    WHERE s.studio_id = ?
                    GROUP BY p.id
                    ORDER BY scene_count DESC
                    LIMIT ?
                    """,
                    (studio_id, top_performers_limit),
                )
                top_performers = [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "url": f"/performers/{row['id']}",
                        "scene_count": row["scene_count"],
                    }
                    for row in cursor.fetchall()
                ]

            # Get top tags
            cursor.execute(
                """
                SELECT
                    t.id,
                    t.name,
                    COUNT(*) as count
                FROM scenes s
                JOIN scenes_tags st ON s.id = st.scene_id
                JOIN tags t ON st.tag_id = t.id
                WHERE s.studio_id = ?
                GROUP BY t.id
                ORDER BY count DESC
                LIMIT ?
                """,
                (studio_id, top_tags_limit),
            )
            top_tags = [
                {"id": row["id"], "name": row["name"], "count": row["count"]}
                for row in cursor.fetchall()
            ]

            conn.close()

            # Build formatted output
            formatted_lines = [f"## {profile['name']}"]
            if profile["favorite"]:
                formatted_lines[0] += " ★"

            if profile["parent_name"]:
                formatted_lines.append(f"**Parent Studio:** {profile['parent_name']}")

            formatted_lines.append(
                f"**Scenes:** {profile['scene_count']} ({profile['total_hours']} hours of content)"
            )

            if profile["earliest_scene_date"] and profile["latest_scene_date"]:
                formatted_lines.append(
                    f"**Date Range:** {profile['earliest_scene_date']} to {profile['latest_scene_date']}"
                )

            if profile["avg_rating"]:
                formatted_lines.append(f"**Avg Rating:** {profile['avg_rating']:.1f}★")

            formatted_lines.append(
                f"**Engagement:** {stats['view_count']} views, {stats['o_count']} O's, "
                f"{stats['total_play_hours']} hours watched"
            )

            if child_studios:
                child_str = ", ".join(
                    [f"[{c['name']}]({c['url']}) ({c['scene_count']})" for c in child_studios[:5]]
                )
                formatted_lines.append(f"**Sub-studios:** {child_str}")

            if top_performers:
                perf_str = ", ".join(
                    [f"[{p['name']}]({p['url']}) ({p['scene_count']})" for p in top_performers[:5]]
                )
                formatted_lines.append(f"**Top Performers:** {perf_str}")

            if top_tags:
                tags_str = ", ".join([f"{t['name']} ({t['count']})" for t in top_tags[:5]])
                formatted_lines.append(f"**Top Tags:** {tags_str}")

            return {
                "success": True,
                "data": {
                    "profile": profile,
                    "stats": stats,
                    "child_studios": child_studios,
                    "top_performers": top_performers if include_performers else None,
                    "top_tags": top_tags,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryGroupProgressTool(BaseTool):
    """
    Tool to track completion progress for a group/series.

    Enables queries like "How much of series X have I watched?",
    "What's my progress on group Y?", "What's the next episode I should watch?"
    """

    @property
    def name(self) -> str:
        return "query_group_progress"

    @property
    def description(self) -> str:
        return (
            "Track completion progress for a group/series. "
            "Returns total scenes, watched vs unwatched count, "
            "completion percentage, scene-by-scene breakdown, "
            "and the next unwatched scene."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "group_name",
                "type": "string",
                "description": "Name of the group/series to look up (case-insensitive partial match)",
                "required": True,
                "enum": None,
            },
            {
                "name": "include_scenes",
                "type": "boolean",
                "description": "Include full scene-by-scene breakdown (default: true)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the group progress query."""
        group_name: str = kwargs.get("group_name", "")
        include_scenes: bool = kwargs.get("include_scenes", True)

        if not group_name:
            return {
                "success": False,
                "data": None,
                "error": "group_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find the group (fuzzy match)
            cursor.execute(
                """
                SELECT id, name FROM groups
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY
                    CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                    name
                LIMIT 1
                """,
                (f"%{group_name}%", group_name),
            )
            group_row = cursor.fetchone()

            if not group_row:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"No group found matching '{group_name}'",
                }

            group_id = group_row["id"]
            group_actual_name = group_row["name"]

            # Get group info with scene progress
            cursor.execute(
                """
                SELECT
                    g.id,
                    g.name,
                    g.date,
                    g.director,
                    st.name as studio_name,
                    st.id as studio_id
                FROM groups g
                LEFT JOIN studios st ON g.studio_id = st.id
                WHERE g.id = ?
                """,
                (group_id,),
            )
            group_info = cursor.fetchone()

            # Get all scenes in the group with watch status
            cursor.execute(
                """
                SELECT
                    s.id,
                    s.title,
                    s.date,
                    gs.scene_index,
                    s.play_duration,
                    vf.duration as file_duration,
                    COALESCE(view_agg.view_count, 0) as view_count,
                    CASE
                        WHEN view_agg.view_count > 0 THEN 1
                        WHEN s.play_duration > 0 AND vf.duration > 0 AND s.play_duration >= (vf.duration * 0.5) THEN 1
                        ELSE 0
                    END as watched
                FROM groups_scenes gs
                JOIN scenes s ON gs.scene_id = s.id
                LEFT JOIN scenes_files fs ON s.id = fs.scene_id
                LEFT JOIN video_files vf ON fs.file_id = vf.file_id
                LEFT JOIN (
                    SELECT scene_id, COUNT(*) as view_count
                    FROM scenes_view_dates
                    GROUP BY scene_id
                ) view_agg ON s.id = view_agg.scene_id
                WHERE gs.group_id = ?
                ORDER BY gs.scene_index ASC, s.date ASC, s.title ASC
                """,
                (group_id,),
            )

            scenes = []
            watched_count = 0
            next_unwatched = None

            for row in cursor.fetchall():
                scene = {
                    "id": row["id"],
                    "title": row["title"],
                    "url": f"/scenes/{row['id']}",
                    "date": row["date"],
                    "scene_index": row["scene_index"],
                    "view_count": row["view_count"],
                    "watched": bool(row["watched"]),
                }
                scenes.append(scene)

                if row["watched"]:
                    watched_count += 1
                elif next_unwatched is None:
                    next_unwatched = scene

            conn.close()

            total_scenes = len(scenes)
            completion_pct = (watched_count / total_scenes * 100) if total_scenes > 0 else 0

            progress = {
                "group_id": group_id,
                "group_name": group_actual_name,
                "url": f"/groups/{group_id}",
                "date": group_info["date"],
                "director": group_info["director"],
                "studio_name": group_info["studio_name"],
                "studio_id": group_info["studio_id"],
                "total_scenes": total_scenes,
                "watched_count": watched_count,
                "unwatched_count": total_scenes - watched_count,
                "completion_percentage": round(completion_pct, 1),
                "next_unwatched": next_unwatched,
            }

            # Build formatted output
            formatted_lines = [f"## {group_actual_name}"]
            if group_info["studio_name"]:
                formatted_lines.append(f"**Studio:** {group_info['studio_name']}")
            if group_info["director"]:
                formatted_lines.append(f"**Director:** {group_info['director']}")

            formatted_lines.append(
                f"**Progress:** {watched_count}/{total_scenes} scenes ({completion_pct:.0f}%)"
            )

            if next_unwatched:
                formatted_lines.append(
                    f"**Next to Watch:** [{next_unwatched['title']}]({next_unwatched['url']})"
                )

            if include_scenes and scenes:
                formatted_lines.append("\n**Scenes:**")
                for i, scene in enumerate(scenes, 1):
                    status = "✓" if scene["watched"] else "○"
                    idx_str = f"#{scene['scene_index']}" if scene["scene_index"] else f"#{i}"
                    formatted_lines.append(
                        f"  {status} {idx_str}: [{scene['title']}]({scene['url']})"
                    )

            return {
                "success": True,
                "data": {
                    "progress": progress,
                    "scenes": scenes if include_scenes else None,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryViewingHistoryTool(BaseTool):
    """
    Tool to get detailed viewing history with timestamps.

    Enables queries like "What did I watch yesterday?",
    "Show my viewing history for last week", "What scenes have I watched recently?"
    """

    @property
    def name(self) -> str:
        return "query_viewing_history"

    @property
    def description(self) -> str:
        return (
            "Get detailed viewing history with timestamps. "
            "Returns a chronological list of viewed scenes. "
            "Can filter by date range, performer, or tag."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "start_date",
                "type": "string",
                "description": "Start date for history (YYYY-MM-DD format)",
                "required": False,
                "enum": None,
            },
            {
                "name": "end_date",
                "type": "string",
                "description": "End date for history (YYYY-MM-DD format)",
                "required": False,
                "enum": None,
            },
            {
                "name": "performer_name",
                "type": "string",
                "description": "Filter by performer name",
                "required": False,
                "enum": None,
            },
            {
                "name": "tag_name",
                "type": "string",
                "description": "Filter by tag name",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of results (default: 50)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the viewing history query."""
        start_date: str | None = kwargs.get("start_date")
        end_date: str | None = kwargs.get("end_date")
        performer_name: str | None = kwargs.get("performer_name")
        tag_name: str | None = kwargs.get("tag_name")
        limit: int = kwargs.get("limit", 50)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build query with filters
            conditions: list[str] = []
            params: list[Any] = []

            if start_date:
                conditions.append("DATE(svd.view_date) >= ?")
                params.append(start_date)

            if end_date:
                conditions.append("DATE(svd.view_date) <= ?")
                params.append(end_date)

            if performer_name:
                conditions.append(
                    "EXISTS (SELECT 1 FROM performers_scenes ps "
                    "JOIN performers p ON ps.performer_id = p.id "
                    "WHERE ps.scene_id = s.id AND LOWER(p.name) LIKE LOWER(?))"
                )
                params.append(f"%{performer_name}%")

            if tag_name:
                conditions.append(
                    "EXISTS (SELECT 1 FROM scenes_tags st "
                    "JOIN tags t ON st.tag_id = t.id "
                    "WHERE st.scene_id = s.id AND LOWER(t.name) LIKE LOWER(?))"
                )
                params.append(f"%{tag_name}%")

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            params.append(limit)

            cursor.execute(
                f"""
                SELECT
                    svd.view_date,
                    s.id as scene_id,
                    s.title,
                    st.name as studio_name,
                    GROUP_CONCAT(DISTINCT p.name) as performers
                FROM scenes_view_dates svd
                JOIN scenes s ON svd.scene_id = s.id
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN performers_scenes ps ON s.id = ps.scene_id
                LEFT JOIN performers p ON ps.performer_id = p.id
                {where_clause}
                GROUP BY svd.view_date, s.id
                ORDER BY svd.view_date DESC
                LIMIT ?
                """,
                params,
            )

            views = []
            for row in cursor.fetchall():
                performers_list = row["performers"].split(",") if row["performers"] else []
                views.append(
                    {
                        "view_date": row["view_date"],
                        "scene_id": row["scene_id"],
                        "title": row["title"],
                        "url": f"/scenes/{row['scene_id']}",
                        "studio": row["studio_name"],
                        "performers": performers_list,
                    }
                )

            # Get summary stats
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_views,
                    COUNT(DISTINCT svd.scene_id) as unique_scenes,
                    MIN(svd.view_date) as earliest,
                    MAX(svd.view_date) as latest
                FROM scenes_view_dates svd
                JOIN scenes s ON svd.scene_id = s.id
                {where_clause}
                """,
                params[:-1],  # Remove limit param
            )
            summary_row = cursor.fetchone()
            summary = {
                "total_views": summary_row["total_views"],
                "unique_scenes": summary_row["unique_scenes"],
                "date_range": {
                    "earliest": summary_row["earliest"],
                    "latest": summary_row["latest"],
                },
            }

            conn.close()

            # Build formatted output
            formatted_lines = ["## Viewing History"]
            if start_date or end_date:
                date_range_str = f"{start_date or 'beginning'} to {end_date or 'now'}"
                formatted_lines.append(f"**Date Range:** {date_range_str}")
            if performer_name:
                formatted_lines.append(f"**Performer:** {performer_name}")
            if tag_name:
                formatted_lines.append(f"**Tag:** {tag_name}")

            formatted_lines.append(
                f"**Summary:** {summary['total_views']} views of {summary['unique_scenes']} unique scenes"
            )

            if views:
                formatted_lines.append("\n**Recent Views:**")
                for view in views[:20]:
                    performers_str = (
                        ", ".join(view["performers"][:2]) if view["performers"] else "Unknown"
                    )
                    if len(view["performers"]) > 2:
                        performers_str += f" +{len(view['performers']) - 2}"
                    formatted_lines.append(
                        f"- {view['view_date'][:10]}: [{view['title']}]({view['url']}) - {performers_str}"
                    )

            return {
                "success": True,
                "data": {
                    "views": views,
                    "summary": summary,
                    "filters": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "performer_name": performer_name,
                        "tag_name": tag_name,
                    },
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryStorageStatsTool(BaseTool):
    """
    Tool to analyze storage usage by various dimensions.

    Enables queries like "How much storage does each studio use?",
    "Which performers have the most GB?", "Show storage by video resolution".
    """

    @property
    def name(self) -> str:
        return "query_storage_stats"

    @property
    def description(self) -> str:
        return (
            "Analyze storage usage by various dimensions. "
            "Returns total size, file count, and average file size. "
            "Can group by studio, performer, tag, resolution, or codec."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "group_by",
                "type": "string",
                "description": "Dimension to group storage stats by (default: studio)",
                "required": False,
                "enum": ["studio", "performer", "tag", "resolution", "codec"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of results (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the storage stats query."""
        group_by: str = kwargs.get("group_by", "studio")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get total library stats first
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT s.id) as total_scenes,
                    COUNT(DISTINCT f.id) as total_files,
                    COALESCE(SUM(f.size), 0) as total_bytes
                FROM scenes s
                LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                LEFT JOIN files f ON sf.file_id = f.id
                """
            )
            total_row = cursor.fetchone()
            total_gb = (total_row["total_bytes"] or 0) / (1024**3)

            # Build group-specific query
            if group_by == "studio":
                cursor.execute(
                    """
                    SELECT
                        st.id as group_id,
                        st.name as group_name,
                        COUNT(DISTINCT s.id) as scene_count,
                        COUNT(DISTINCT f.id) as file_count,
                        COALESCE(SUM(f.size), 0) as total_bytes
                    FROM scenes s
                    LEFT JOIN studios st ON s.studio_id = st.id
                    LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                    LEFT JOIN files f ON sf.file_id = f.id
                    GROUP BY st.id
                    ORDER BY total_bytes DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            elif group_by == "performer":
                cursor.execute(
                    """
                    SELECT
                        p.id as group_id,
                        p.name as group_name,
                        COUNT(DISTINCT s.id) as scene_count,
                        COUNT(DISTINCT f.id) as file_count,
                        COALESCE(SUM(f.size), 0) as total_bytes
                    FROM performers p
                    JOIN performers_scenes ps ON p.id = ps.performer_id
                    JOIN scenes s ON ps.scene_id = s.id
                    LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                    LEFT JOIN files f ON sf.file_id = f.id
                    GROUP BY p.id
                    ORDER BY total_bytes DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            elif group_by == "tag":
                cursor.execute(
                    """
                    SELECT
                        t.id as group_id,
                        t.name as group_name,
                        COUNT(DISTINCT s.id) as scene_count,
                        COUNT(DISTINCT f.id) as file_count,
                        COALESCE(SUM(f.size), 0) as total_bytes
                    FROM tags t
                    JOIN scenes_tags st ON t.id = st.tag_id
                    JOIN scenes s ON st.scene_id = s.id
                    LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                    LEFT JOIN files f ON sf.file_id = f.id
                    GROUP BY t.id
                    ORDER BY total_bytes DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            elif group_by == "resolution":
                cursor.execute(
                    """
                    SELECT
                        CASE
                            WHEN vf.height >= 2160 THEN '4K (2160p+)'
                            WHEN vf.height >= 1080 THEN '1080p'
                            WHEN vf.height >= 720 THEN '720p'
                            WHEN vf.height >= 480 THEN '480p'
                            ELSE 'Lower'
                        END as group_name,
                        NULL as group_id,
                        COUNT(DISTINCT s.id) as scene_count,
                        COUNT(DISTINCT f.id) as file_count,
                        COALESCE(SUM(f.size), 0) as total_bytes
                    FROM scenes s
                    LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                    LEFT JOIN files f ON sf.file_id = f.id
                    LEFT JOIN video_files vf ON f.id = vf.file_id
                    GROUP BY group_name
                    ORDER BY total_bytes DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            elif group_by == "codec":
                cursor.execute(
                    """
                    SELECT
                        COALESCE(vf.video_codec, 'Unknown') as group_name,
                        NULL as group_id,
                        COUNT(DISTINCT s.id) as scene_count,
                        COUNT(DISTINCT f.id) as file_count,
                        COALESCE(SUM(f.size), 0) as total_bytes
                    FROM scenes s
                    LEFT JOIN scenes_files sf ON s.id = sf.scene_id
                    LEFT JOIN files f ON sf.file_id = f.id
                    LEFT JOIN video_files vf ON f.id = vf.file_id
                    GROUP BY group_name
                    ORDER BY total_bytes DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            else:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Invalid group_by value: {group_by}",
                }

            results = []
            for row in cursor.fetchall():
                total_bytes = row["total_bytes"] or 0
                gb = total_bytes / (1024**3)
                avg_gb = gb / row["file_count"] if row["file_count"] > 0 else 0
                results.append(
                    {
                        "group_id": row["group_id"],
                        "group_name": row["group_name"] or "(No " + group_by + ")",
                        "scene_count": row["scene_count"],
                        "file_count": row["file_count"],
                        "total_gb": round(gb, 2),
                        "avg_file_gb": round(avg_gb, 3),
                        "percentage_of_library": round(
                            (gb / total_gb * 100) if total_gb > 0 else 0, 1
                        ),
                    }
                )

            conn.close()

            # Build formatted output
            formatted_lines = [f"## Storage by {group_by.title()}"]
            formatted_lines.append(
                f"**Total Library:** {total_row['total_scenes']} scenes, {round(total_gb, 2)} GB"
            )

            if results:
                formatted_lines.append(f"\n**Top {len(results)} by storage:**")
                for r in results:
                    formatted_lines.append(
                        f"- {r['group_name']}: {r['total_gb']} GB ({r['percentage_of_library']}%) - "
                        f"{r['scene_count']} scenes"
                    )

            return {
                "success": True,
                "data": {
                    "results": results,
                    "group_by": group_by,
                    "library_total": {
                        "scenes": total_row["total_scenes"],
                        "files": total_row["total_files"],
                        "total_gb": round(total_gb, 2),
                    },
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


# =============================================================================
# Phase 3: Higher Complexity Tools
# =============================================================================


class QueryTagHierarchyTool(BaseTool):
    """
    Tool to explore tag parent/child relationships.

    Uses the tags_relations table to traverse tag hierarchies.
    Enables queries like "What sub-tags does 'oral' have?",
    "Show me the tag hierarchy for 'position'".
    """

    @property
    def name(self) -> str:
        return "query_tag_hierarchy"

    @property
    def description(self) -> str:
        return (
            "Explore tag parent/child relationships. "
            "Returns tag hierarchy tree with scene counts at each level. "
            "Can show parents, children, or both directions."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "tag_name",
                "type": "string",
                "description": "Tag name to get hierarchy for (optional - omit for overview)",
                "required": False,
                "enum": None,
            },
            {
                "name": "direction",
                "type": "string",
                "description": "Which direction to traverse (default: both)",
                "required": False,
                "enum": ["parents", "children", "both"],
            },
            {
                "name": "depth",
                "type": "integer",
                "description": "How deep to traverse the hierarchy (default: 3)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tag hierarchy query."""
        tag_name: str | None = kwargs.get("tag_name")
        direction: str = kwargs.get("direction", "both")
        depth: int = kwargs.get("depth", 3)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            if tag_name:
                # Build exclusion clause
                exclude_clause = ""
                exclude_params: list[Any] = []
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND id NOT IN ({placeholders})"
                    exclude_params = list(excluded_ids)

                # Find the specific tag (excluding excluded tags)
                cursor.execute(
                    f"""
                    SELECT id, name FROM tags
                    WHERE LOWER(name) LIKE LOWER(?)
                    {exclude_clause}
                    ORDER BY
                        CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                        name
                    LIMIT 1
                    """,
                    (f"%{tag_name}%", *exclude_params, tag_name),
                )
                tag_row = cursor.fetchone()

                if not tag_row:
                    conn.close()
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Tag '{tag_name}' not found",
                    }

                tag_id = tag_row["id"]
                tag_name_found = tag_row["name"]

                # Get scene count for the tag
                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM scenes_tags WHERE tag_id = ?",
                    (tag_id,),
                )
                scene_count = cursor.fetchone()["cnt"]

                result = {
                    "tag_id": tag_id,
                    "tag_name": tag_name_found,
                    "scene_count": scene_count,
                    "parents": [],
                    "children": [],
                }

                # Get parents (tags this tag is a child of)
                if direction in ("parents", "both"):
                    result["parents"] = self._get_ancestors(cursor, tag_id, depth, [], excluded_ids)

                # Get children (tags that are children of this tag)
                if direction in ("children", "both"):
                    result["children"] = self._get_descendants(cursor, tag_id, depth, excluded_ids)

                conn.close()

                # Build formatted output
                formatted_lines = [f"## Tag Hierarchy: {tag_name_found}"]
                formatted_lines.append(f"**Scene count:** {scene_count}")

                if result["parents"]:
                    formatted_lines.append("\n**Parent tags:**")
                    self._format_ancestors(result["parents"], formatted_lines, 0)

                if result["children"]:
                    formatted_lines.append("\n**Child tags:**")
                    self._format_descendants(result["children"], formatted_lines, 0)

                return {
                    "success": True,
                    "data": {
                        **result,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

            else:
                # No tag specified - show overview of tag hierarchy
                # Find root tags (tags with no parents) that have children
                # Build exclusion clause for overview
                exclude_clause = ""
                exclude_params_overview: list[Any] = []
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND t.id NOT IN ({placeholders})"
                    exclude_params_overview = list(excluded_ids)

                cursor.execute(
                    f"""
                    SELECT t.id, t.name,
                           (SELECT COUNT(*) FROM scenes_tags WHERE tag_id = t.id) as scene_count,
                           (SELECT COUNT(*) FROM tags_relations WHERE parent_id = t.id) as child_count
                    FROM tags t
                    WHERE t.id NOT IN (SELECT child_id FROM tags_relations)
                    AND t.id IN (SELECT parent_id FROM tags_relations)
                    {exclude_clause}
                    ORDER BY child_count DESC, scene_count DESC
                    LIMIT 20
                    """,
                    exclude_params_overview,
                )

                root_tags = []
                for row in cursor.fetchall():
                    root_tags.append(
                        {
                            "tag_id": row["id"],
                            "tag_name": row["name"],
                            "scene_count": row["scene_count"],
                            "child_count": row["child_count"],
                        }
                    )

                # Get total stats
                cursor.execute("SELECT COUNT(*) as cnt FROM tags_relations")
                total_relations = cursor.fetchone()["cnt"]

                cursor.execute(
                    """
                    SELECT COUNT(DISTINCT child_id) as cnt FROM tags_relations
                    """
                )
                tags_with_parents = cursor.fetchone()["cnt"]

                conn.close()

                formatted_lines = ["## Tag Hierarchy Overview"]
                formatted_lines.append(f"**Total relationships:** {total_relations}")
                formatted_lines.append(f"**Tags with parents:** {tags_with_parents}")
                formatted_lines.append("\n**Root tags (no parents, have children):**")

                for t in root_tags:
                    formatted_lines.append(
                        f"- {t['tag_name']}: {t['child_count']} children, {t['scene_count']} scenes"
                    )

                return {
                    "success": True,
                    "data": {
                        "root_tags": root_tags,
                        "total_relationships": total_relations,
                        "tags_with_parents": tags_with_parents,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_ancestors(
        self,
        cursor: sqlite3.Cursor,
        tag_id: int,
        depth: int,
        visited: list[int],
        excluded_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Recursively get parent tags."""
        if depth <= 0 or tag_id in visited:
            return []

        visited.append(tag_id)

        # Build exclusion clause
        exclude_clause = ""
        params: list[Any] = [tag_id]
        if excluded_ids:
            placeholders = ",".join("?" * len(excluded_ids))
            exclude_clause = f"AND t.id NOT IN ({placeholders})"
            params.extend(list(excluded_ids))

        cursor.execute(
            f"""
            SELECT t.id, t.name,
                   (SELECT COUNT(*) FROM scenes_tags WHERE tag_id = t.id) as scene_count
            FROM tags t
            JOIN tags_relations tr ON t.id = tr.parent_id
            WHERE tr.child_id = ?
            {exclude_clause}
            ORDER BY scene_count DESC
            """,
            params,
        )

        parents = []
        for row in cursor.fetchall():
            parent = {
                "tag_id": row["id"],
                "tag_name": row["name"],
                "scene_count": row["scene_count"],
                "parents": self._get_ancestors(
                    cursor, row["id"], depth - 1, visited.copy(), excluded_ids
                ),
            }
            parents.append(parent)

        return parents

    def _get_descendants(
        self,
        cursor: sqlite3.Cursor,
        tag_id: int,
        depth: int,
        excluded_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Recursively get child tags."""
        if depth <= 0:
            return []

        # Build exclusion clause
        exclude_clause = ""
        params: list[Any] = [tag_id]
        if excluded_ids:
            placeholders = ",".join("?" * len(excluded_ids))
            exclude_clause = f"AND t.id NOT IN ({placeholders})"
            params.extend(list(excluded_ids))

        cursor.execute(
            f"""
            SELECT t.id, t.name,
                   (SELECT COUNT(*) FROM scenes_tags WHERE tag_id = t.id) as scene_count
            FROM tags t
            JOIN tags_relations tr ON t.id = tr.child_id
            WHERE tr.parent_id = ?
            {exclude_clause}
            ORDER BY scene_count DESC
            """,
            params,
        )

        children = []
        for row in cursor.fetchall():
            child = {
                "tag_id": row["id"],
                "tag_name": row["name"],
                "scene_count": row["scene_count"],
                "children": self._get_descendants(cursor, row["id"], depth - 1, excluded_ids),
            }
            children.append(child)

        return children

    def _format_ancestors(
        self,
        ancestors: list[dict[str, Any]],
        lines: list[str],
        level: int,
    ) -> None:
        """Format ancestor tags for display."""
        indent = "  " * level
        for a in ancestors:
            lines.append(f"{indent}↑ {a['tag_name']} ({a['scene_count']} scenes)")
            if a.get("parents"):
                self._format_ancestors(a["parents"], lines, level + 1)

    def _format_descendants(
        self,
        descendants: list[dict[str, Any]],
        lines: list[str],
        level: int,
    ) -> None:
        """Format descendant tags for display."""
        indent = "  " * level
        for d in descendants:
            lines.append(f"{indent}↓ {d['tag_name']} ({d['scene_count']} scenes)")
            if d.get("children"):
                self._format_descendants(d["children"], lines, level + 1)


class QueryStudioHierarchyTool(BaseTool):
    """
    Tool to explore studio parent/child relationships.

    Uses the studios.parent_id field to traverse studio hierarchies.
    Enables queries like "Show me studio X's sub-labels",
    "What studios are under parent Y?".
    """

    @property
    def name(self) -> str:
        return "query_studio_hierarchy"

    @property
    def description(self) -> str:
        return (
            "Explore studio parent/child relationships. "
            "Returns studio hierarchy tree with scene counts. "
            "Studios can have parent studios (networks) and child studios (sub-labels)."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "studio_name",
                "type": "string",
                "description": "Studio name to get hierarchy for (optional - omit for overview)",
                "required": False,
                "enum": None,
            },
            {
                "name": "include_stats",
                "type": "boolean",
                "description": "Include scene counts and engagement stats (default: true)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the studio hierarchy query."""
        studio_name: str | None = kwargs.get("studio_name")
        _include_stats: bool = kwargs.get("include_stats", True)  # TODO: Use for detailed stats

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            if studio_name:
                # Find the specific studio
                cursor.execute(
                    """
                    SELECT id, name, parent_id FROM studios
                    WHERE LOWER(name) LIKE LOWER(?)
                    ORDER BY
                        CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                        name
                    LIMIT 1
                    """,
                    (f"%{studio_name}%", studio_name),
                )
                studio_row = cursor.fetchone()

                if not studio_row:
                    conn.close()
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Studio '{studio_name}' not found",
                    }

                studio_id = studio_row["id"]
                studio_name_found = studio_row["name"]
                parent_id = studio_row["parent_id"]

                # Get scene count
                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM scenes WHERE studio_id = ?",
                    (studio_id,),
                )
                scene_count = cursor.fetchone()["cnt"]

                result = {
                    "studio_id": studio_id,
                    "studio_name": studio_name_found,
                    "scene_count": scene_count,
                    "parent": None,
                    "children": [],
                }

                # Get parent chain
                if parent_id:
                    result["parent"] = self._get_parent_chain(cursor, parent_id)

                # Get child studios
                result["children"] = self._get_child_studios(cursor, studio_id)

                # Calculate total scenes including children
                total_scenes = scene_count
                for child in result["children"]:
                    total_scenes += self._count_subtree_scenes(child)

                result["total_scenes_with_children"] = total_scenes

                conn.close()

                # Build formatted output
                formatted_lines = [f"## Studio Hierarchy: {studio_name_found}"]
                formatted_lines.append(f"**Direct scenes:** {scene_count}")
                formatted_lines.append(f"**Total (with sub-studios):** {total_scenes}")

                if result["parent"]:
                    formatted_lines.append("\n**Parent studios:**")
                    self._format_parent_chain(result["parent"], formatted_lines, 0)

                if result["children"]:
                    formatted_lines.append("\n**Sub-studios:**")
                    self._format_children(result["children"], formatted_lines, 0)

                return {
                    "success": True,
                    "data": {
                        **result,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

            else:
                # No studio specified - show overview
                # Find root studios (networks) with children
                cursor.execute(
                    """
                    SELECT s.id, s.name,
                           (SELECT COUNT(*) FROM scenes WHERE studio_id = s.id) as scene_count,
                           (SELECT COUNT(*) FROM studios WHERE parent_id = s.id) as child_count
                    FROM studios s
                    WHERE s.parent_id IS NULL
                    AND s.id IN (SELECT parent_id FROM studios WHERE parent_id IS NOT NULL)
                    ORDER BY child_count DESC, scene_count DESC
                    LIMIT 20
                    """
                )

                networks = []
                for row in cursor.fetchall():
                    networks.append(
                        {
                            "studio_id": row["id"],
                            "studio_name": row["name"],
                            "scene_count": row["scene_count"],
                            "child_count": row["child_count"],
                        }
                    )

                # Get stats
                cursor.execute("SELECT COUNT(*) as cnt FROM studios WHERE parent_id IS NOT NULL")
                studios_with_parent = cursor.fetchone()["cnt"]

                cursor.execute("SELECT COUNT(*) as cnt FROM studios")
                total_studios = cursor.fetchone()["cnt"]

                conn.close()

                formatted_lines = ["## Studio Hierarchy Overview"]
                formatted_lines.append(f"**Total studios:** {total_studios}")
                formatted_lines.append(f"**Studios with parents:** {studios_with_parent}")
                formatted_lines.append("\n**Networks (top-level with sub-studios):**")

                for n in networks:
                    formatted_lines.append(
                        f"- {n['studio_name']}: {n['child_count']} sub-studios, "
                        f"{n['scene_count']} direct scenes"
                    )

                return {
                    "success": True,
                    "data": {
                        "networks": networks,
                        "total_studios": total_studios,
                        "studios_with_parent": studios_with_parent,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_parent_chain(self, cursor: sqlite3.Cursor, studio_id: int) -> dict[str, Any] | None:
        """Get the parent studio and its ancestors."""
        cursor.execute(
            """
            SELECT id, name, parent_id,
                   (SELECT COUNT(*) FROM scenes WHERE studio_id = studios.id) as scene_count
            FROM studios WHERE id = ?
            """,
            (studio_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        parent = {
            "studio_id": row["id"],
            "studio_name": row["name"],
            "scene_count": row["scene_count"],
            "parent": None,
        }

        if row["parent_id"]:
            parent["parent"] = self._get_parent_chain(cursor, row["parent_id"])

        return parent

    def _get_child_studios(self, cursor: sqlite3.Cursor, studio_id: int) -> list[dict[str, Any]]:
        """Get all child studios recursively."""
        cursor.execute(
            """
            SELECT id, name,
                   (SELECT COUNT(*) FROM scenes WHERE studio_id = studios.id) as scene_count
            FROM studios WHERE parent_id = ?
            ORDER BY scene_count DESC
            """,
            (studio_id,),
        )

        children = []
        for row in cursor.fetchall():
            child = {
                "studio_id": row["id"],
                "studio_name": row["name"],
                "scene_count": row["scene_count"],
                "children": self._get_child_studios(cursor, row["id"]),
            }
            children.append(child)

        return children

    def _count_subtree_scenes(self, node: dict[str, Any]) -> int:
        """Count scenes in a studio subtree."""
        total: int = int(node["scene_count"])
        for child in node.get("children", []):
            total += self._count_subtree_scenes(child)
        return total

    def _format_parent_chain(
        self,
        parent: dict[str, Any],
        lines: list[str],
        level: int,
    ) -> None:
        """Format parent chain for display."""
        indent = "  " * level
        lines.append(f"{indent}↑ {parent['studio_name']} ({parent['scene_count']} scenes)")
        if parent.get("parent"):
            self._format_parent_chain(parent["parent"], lines, level + 1)

    def _format_children(
        self,
        children: list[dict[str, Any]],
        lines: list[str],
        level: int,
    ) -> None:
        """Format child studios for display."""
        indent = "  " * level
        for c in children:
            lines.append(f"{indent}↓ {c['studio_name']} ({c['scene_count']} scenes)")
            if c.get("children"):
                self._format_children(c["children"], lines, level + 1)


class QuerySceneMarkersTool(BaseTool):
    """
    Tool to find and analyze scene markers (timestamps).

    Enables queries like "What markers exist in scene 123?",
    "Find all scenes with 'blowjob' markers",
    "What are the most common marker tags?".
    """

    @property
    def name(self) -> str:
        return "query_scene_markers"

    @property
    def description(self) -> str:
        return (
            "Find and analyze scene markers (timestamps with tags). "
            "Can get markers for a specific scene, find scenes with a specific marker tag, "
            "or show most common marker tags. Markers have start/end times and titles."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "scene_id",
                "type": "integer",
                "description": "Scene ID to get markers for (optional)",
                "required": False,
                "enum": None,
            },
            {
                "name": "tag_name",
                "type": "string",
                "description": "Find markers with this tag (optional)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum results (default: 50)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the scene markers query."""
        scene_id: int | None = kwargs.get("scene_id")
        tag_name: str | None = kwargs.get("tag_name")
        limit: int = kwargs.get("limit", 50)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            if scene_id:
                # Get markers for a specific scene (excluding markers with excluded tags)
                exclude_clause = ""
                params: list[Any] = [scene_id]
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND (t.id IS NULL OR t.id NOT IN ({placeholders}))"
                    params.extend(list(excluded_ids))

                cursor.execute(
                    f"""
                    SELECT sm.id, sm.title, sm.seconds, sm.end_seconds,
                           t.id as tag_id, t.name as tag_name
                    FROM scene_markers sm
                    LEFT JOIN tags t ON sm.primary_tag_id = t.id
                    WHERE sm.scene_id = ?
                    {exclude_clause}
                    ORDER BY sm.seconds
                    """,
                    params,
                )

                markers = []
                for row in cursor.fetchall():
                    start_secs = row["seconds"] or 0
                    end_secs = row["end_seconds"]
                    duration = (end_secs - start_secs) if end_secs else None

                    markers.append(
                        {
                            "marker_id": row["id"],
                            "title": row["title"],
                            "tag_id": row["tag_id"],
                            "tag_name": row["tag_name"],
                            "start_time": self._format_time(start_secs),
                            "start_seconds": start_secs,
                            "end_time": self._format_time(end_secs) if end_secs else None,
                            "end_seconds": end_secs,
                            "duration_seconds": duration,
                        }
                    )

                # Get scene info
                cursor.execute(
                    """
                    SELECT s.title, st.name as studio
                    FROM scenes s
                    LEFT JOIN studios st ON s.studio_id = st.id
                    WHERE s.id = ?
                    """,
                    (scene_id,),
                )
                scene_row = cursor.fetchone()

                conn.close()

                formatted_lines = [f"## Markers for Scene {scene_id}"]
                if scene_row:
                    formatted_lines.append(f"**Scene:** {scene_row['title'] or 'Untitled'}")
                formatted_lines.append(f"**Total markers:** {len(markers)}")

                if markers:
                    formatted_lines.append("\n**Timeline:**")
                    for m in markers:
                        time_str = m["start_time"]
                        if m["end_time"]:
                            time_str += f" - {m['end_time']}"
                        tag_str = f" [{m['tag_name']}]" if m["tag_name"] else ""
                        title_str = f": {m['title']}" if m["title"] else ""
                        formatted_lines.append(f"- {time_str}{tag_str}{title_str}")

                return {
                    "success": True,
                    "data": {
                        "scene_id": scene_id,
                        "scene_title": scene_row["title"] if scene_row else None,
                        "markers": markers,
                        "total_markers": len(markers),
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

            elif tag_name:
                # Find scenes with markers having this tag (excluding excluded tags)
                exclude_clause = ""
                tag_params: list[Any] = [f"%{tag_name}%"]
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND id NOT IN ({placeholders})"
                    tag_params.extend(list(excluded_ids))
                tag_params.append(tag_name)

                cursor.execute(
                    f"""
                    SELECT id, name FROM tags
                    WHERE LOWER(name) LIKE LOWER(?)
                    {exclude_clause}
                    ORDER BY
                        CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                        name
                    LIMIT 1
                    """,
                    tag_params,
                )
                tag_row = cursor.fetchone()

                if not tag_row:
                    conn.close()
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Tag '{tag_name}' not found",
                    }

                tag_id = tag_row["id"]
                tag_name_found = tag_row["name"]

                cursor.execute(
                    """
                    SELECT sm.scene_id, s.title as scene_title, st.name as studio,
                           sm.seconds, sm.end_seconds, sm.title as marker_title,
                           COUNT(*) OVER () as total_count
                    FROM scene_markers sm
                    JOIN scenes s ON sm.scene_id = s.id
                    LEFT JOIN studios st ON s.studio_id = st.id
                    WHERE sm.primary_tag_id = ?
                    ORDER BY sm.scene_id, sm.seconds
                    LIMIT ?
                    """,
                    (tag_id, limit),
                )

                results = []
                total_count = 0
                for row in cursor.fetchall():
                    total_count = row["total_count"]
                    results.append(
                        {
                            "scene_id": row["scene_id"],
                            "scene_title": row["scene_title"],
                            "studio": row["studio"],
                            "start_time": self._format_time(row["seconds"]),
                            "end_time": self._format_time(row["end_seconds"])
                            if row["end_seconds"]
                            else None,
                            "marker_title": row["marker_title"],
                        }
                    )

                conn.close()

                formatted_lines = [f"## Scenes with '{tag_name_found}' markers"]
                formatted_lines.append(f"**Total markers found:** {total_count}")

                if results:
                    formatted_lines.append(f"\n**Results (showing {len(results)}):**")
                    current_scene = None
                    for r in results:
                        if r["scene_id"] != current_scene:
                            current_scene = r["scene_id"]
                            scene_title = r["scene_title"] or f"Scene {r['scene_id']}"
                            formatted_lines.append(
                                f"\n**{scene_title}** ({r['studio'] or 'Unknown'})"
                            )
                        time_str = r["start_time"]
                        if r["end_time"]:
                            time_str += f" - {r['end_time']}"
                        formatted_lines.append(
                            f"  - {time_str}: {r['marker_title'] or '(no title)'}"
                        )

                return {
                    "success": True,
                    "data": {
                        "tag_name": tag_name_found,
                        "tag_id": tag_id,
                        "markers": results,
                        "total_count": total_count,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

            else:
                # Show most common marker tags (excluding excluded)
                exclude_clause = ""
                stats_params: list[Any] = []
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"WHERE t.id NOT IN ({placeholders})"
                    stats_params = list(excluded_ids)
                stats_params.append(limit)

                cursor.execute(
                    f"""
                    SELECT t.id, t.name,
                           COUNT(*) as marker_count,
                           COUNT(DISTINCT sm.scene_id) as scene_count
                    FROM scene_markers sm
                    JOIN tags t ON sm.primary_tag_id = t.id
                    {exclude_clause}
                    GROUP BY t.id
                    ORDER BY marker_count DESC
                    LIMIT ?
                    """,
                    stats_params,
                )

                top_tags = []
                for row in cursor.fetchall():
                    top_tags.append(
                        {
                            "tag_id": row["id"],
                            "tag_name": row["name"],
                            "marker_count": row["marker_count"],
                            "scene_count": row["scene_count"],
                        }
                    )

                # Get total marker stats
                cursor.execute("SELECT COUNT(*) as cnt FROM scene_markers")
                total_markers = cursor.fetchone()["cnt"]

                cursor.execute("SELECT COUNT(DISTINCT scene_id) as cnt FROM scene_markers")
                scenes_with_markers = cursor.fetchone()["cnt"]

                conn.close()

                formatted_lines = ["## Scene Marker Statistics"]
                formatted_lines.append(f"**Total markers:** {total_markers}")
                formatted_lines.append(f"**Scenes with markers:** {scenes_with_markers}")
                formatted_lines.append("\n**Most common marker tags:**")

                for t in top_tags:
                    formatted_lines.append(
                        f"- {t['tag_name']}: {t['marker_count']} markers in {t['scene_count']} scenes"
                    )

                return {
                    "success": True,
                    "data": {
                        "top_tags": top_tags,
                        "total_markers": total_markers,
                        "scenes_with_markers": scenes_with_markers,
                        "formatted_results": "\n".join(formatted_lines),
                    },
                    "error": None,
                }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _format_time(self, seconds: float | None) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds is None:
            return "0:00"
        total_secs = int(seconds)
        hours = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60
        if hours > 0:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"


class QueryTagUsageOverTimeTool(BaseTool):
    """
    Tool to analyze tag trends over time based on viewing history.

    Enables queries like "What tags are trending in my viewing?",
    "How has my taste changed over time?".
    """

    @property
    def name(self) -> str:
        return "query_tag_usage_over_time"

    @property
    def description(self) -> str:
        return (
            "Analyze tag trends over time based on viewing history. "
            "Shows how tag preferences have changed over months/years. "
            "Can track specific tags or show top trending tags."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "tag_names",
                "type": "array",
                "description": "Specific tags to track (optional - defaults to top 10)",
                "required": False,
                "enum": None,
            },
            {
                "name": "group_by",
                "type": "string",
                "description": "Time period grouping (default: month)",
                "required": False,
                "enum": ["month", "quarter", "year"],
            },
            {
                "name": "metric",
                "type": "string",
                "description": "What to measure (default: views)",
                "required": False,
                "enum": ["views", "o_count", "scene_count"],
            },
            {
                "name": "limit_periods",
                "type": "integer",
                "description": "Maximum number of time periods to show (default: 12)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tag usage over time query."""
        tag_names: list[str] | None = kwargs.get("tag_names")
        group_by: str = kwargs.get("group_by", "month")
        metric: str = kwargs.get("metric", "views")
        limit_periods: int = kwargs.get("limit_periods", 12)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tag IDs including children
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Build the date format for grouping
            if group_by == "month":
                date_format = "%Y-%m"
            elif group_by == "quarter":
                # SQLite doesn't have native quarter, we'll handle in Python
                date_format = "%Y-%m"
            else:  # year
                date_format = "%Y"

            # Determine which tags to track
            if tag_names:
                # Find the specified tags (excluding excluded)
                placeholders = ",".join("?" * len(tag_names))
                params: list[Any] = [t.lower() for t in tag_names]

                exclude_clause = ""
                if excluded_ids:
                    id_placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND id NOT IN ({id_placeholders})"
                    params.extend(list(excluded_ids))

                cursor.execute(
                    f"""
                    SELECT id, name FROM tags
                    WHERE LOWER(name) IN ({placeholders})
                    {exclude_clause}
                    """,
                    params,
                )
                tracked_tags = {row["id"]: row["name"] for row in cursor.fetchall()}

                if not tracked_tags:
                    conn.close()
                    return {
                        "success": False,
                        "data": None,
                        "error": "None of the specified tags found",
                    }
            else:
                # Get top 10 tags by views (excluding excluded)
                exclude_clause = ""
                params = []
                if excluded_ids:
                    placeholders = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"WHERE t.id NOT IN ({placeholders})"
                    params = list(excluded_ids)

                cursor.execute(
                    f"""
                    SELECT t.id, t.name, COUNT(*) as view_count
                    FROM scenes_view_dates svd
                    JOIN scenes_tags st ON svd.scene_id = st.scene_id
                    JOIN tags t ON st.tag_id = t.id
                    {exclude_clause}
                    GROUP BY t.id
                    ORDER BY view_count DESC
                    LIMIT 10
                    """,
                    params,
                )
                tracked_tags = {row["id"]: row["name"] for row in cursor.fetchall()}

            if not tracked_tags:
                conn.close()
                return {
                    "success": True,
                    "data": {
                        "message": "No viewing data found to analyze trends",
                        "formatted_results": "No viewing data found to analyze trends.",
                    },
                    "error": None,
                }

            # Build time series data
            tag_ids = list(tracked_tags.keys())
            placeholders = ",".join("?" * len(tag_ids))

            if metric == "views":
                cursor.execute(
                    f"""
                    SELECT strftime('{date_format}', svd.view_date) as period,
                           st.tag_id,
                           COUNT(*) as value
                    FROM scenes_view_dates svd
                    JOIN scenes_tags st ON svd.scene_id = st.scene_id
                    WHERE st.tag_id IN ({placeholders})
                    GROUP BY period, st.tag_id
                    ORDER BY period DESC
                    """,
                    tag_ids,
                )
            elif metric == "o_count":
                cursor.execute(
                    f"""
                    SELECT strftime('{date_format}', sod.o_date) as period,
                           st.tag_id,
                           COUNT(*) as value
                    FROM scenes_o_dates sod
                    JOIN scenes_tags st ON sod.scene_id = st.scene_id
                    WHERE st.tag_id IN ({placeholders})
                    GROUP BY period, st.tag_id
                    ORDER BY period DESC
                    """,
                    tag_ids,
                )
            else:  # scene_count based on scene date
                cursor.execute(
                    f"""
                    SELECT strftime('{date_format}', s.date) as period,
                           st.tag_id,
                           COUNT(DISTINCT s.id) as value
                    FROM scenes s
                    JOIN scenes_tags st ON s.id = st.scene_id
                    WHERE st.tag_id IN ({placeholders})
                    AND s.date IS NOT NULL
                    GROUP BY period, st.tag_id
                    ORDER BY period DESC
                    """,
                    tag_ids,
                )

            # Organize data by period and tag
            periods_data: dict[str, dict[int, int]] = {}
            for row in cursor.fetchall():
                period = row["period"]
                if period is None:
                    continue

                # Handle quarter grouping
                if group_by == "quarter":
                    year, month = period.split("-")
                    quarter = (int(month) - 1) // 3 + 1
                    period = f"{year}-Q{quarter}"

                if period not in periods_data:
                    periods_data[period] = {}
                periods_data[period][row["tag_id"]] = row["value"]

            conn.close()

            # Sort periods and limit
            sorted_periods = sorted(periods_data.keys(), reverse=True)[:limit_periods]
            sorted_periods.reverse()  # Oldest first for display

            # Build result structure
            time_series: list[dict[str, Any]] = []
            for period in sorted_periods:
                period_entry: dict[str, Any] = {"period": period, "tags": {}}
                for tag_id, tag_name in tracked_tags.items():
                    period_entry["tags"][tag_name] = periods_data.get(period, {}).get(tag_id, 0)
                time_series.append(period_entry)

            # Calculate trends (compare latest period to previous)
            trends: dict[str, str] = {}
            if len(time_series) >= 2:
                latest: dict[str, int] = time_series[-1]["tags"]
                previous: dict[str, int] = time_series[-2]["tags"]
                for tag_name in tracked_tags.values():
                    curr = latest.get(tag_name, 0)
                    prev = previous.get(tag_name, 0)
                    if prev > 0:
                        change_pct = ((curr - prev) / prev) * 100
                        if change_pct > 10:
                            trends[tag_name] = "↑ trending up"
                        elif change_pct < -10:
                            trends[tag_name] = "↓ trending down"
                        else:
                            trends[tag_name] = "→ stable"
                    elif curr > 0:
                        trends[tag_name] = "↑ new activity"
                    else:
                        trends[tag_name] = "→ no activity"

            # Build formatted output
            formatted_lines = [f"## Tag Trends by {group_by.title()}"]
            formatted_lines.append(f"**Metric:** {metric}")
            formatted_lines.append(f"**Tags tracked:** {', '.join(tracked_tags.values())}")

            if trends:
                formatted_lines.append("\n**Recent trends:**")
                for tag_name, trend in sorted(trends.items(), key=lambda x: x[1]):
                    formatted_lines.append(f"- {tag_name}: {trend}")

            if time_series:
                formatted_lines.append(f"\n**Time series (last {len(time_series)} periods):**")
                for entry in time_series:
                    entry_tags: dict[str, int] = entry["tags"]
                    tag_values = [f"{k}: {v}" for k, v in entry_tags.items() if v > 0]
                    if tag_values:
                        formatted_lines.append(f"- {entry['period']}: {', '.join(tag_values)}")

            return {
                "success": True,
                "data": {
                    "tags": list(tracked_tags.values()),
                    "metric": metric,
                    "group_by": group_by,
                    "time_series": time_series,
                    "trends": trends,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryPerformerComparisonTool(BaseTool):
    """
    Tool to compare two or more performers side-by-side.

    Enables queries like "Compare performer A and performer B",
    "Who has more scenes, X or Y?".
    """

    @property
    def name(self) -> str:
        return "query_performer_comparison"

    @property
    def description(self) -> str:
        return (
            "Compare two or more performers side-by-side. "
            "Shows scene counts, engagement stats, shared scenes, shared tags. "
            "Useful for answering 'who has more' type questions."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_names",
                "type": "array",
                "description": "List of performer names to compare (2-5 performers)",
                "required": True,
                "enum": None,
            },
            {
                "name": "metrics",
                "type": "array",
                "description": "Metrics to compare (default: all). Options: scene_count, views, o_count, tags",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performer comparison query."""
        performer_names: list[str] = kwargs.get("performer_names", [])
        _metrics: list[str] | None = kwargs.get("metrics")  # TODO: Filter by metrics

        if not performer_names or len(performer_names) < 2:
            return {
                "success": False,
                "data": None,
                "error": "At least 2 performer names are required for comparison",
            }

        if len(performer_names) > 5:
            performer_names = performer_names[:5]

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find all performers
            performers: list[dict[str, Any]] = []
            not_found: list[str] = []

            for name in performer_names:
                cursor.execute(
                    """
                    SELECT id, name FROM performers
                    WHERE LOWER(name) LIKE LOWER(?)
                    ORDER BY
                        CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                        name
                    LIMIT 1
                    """,
                    (f"%{name}%", name),
                )
                row = cursor.fetchone()
                if row:
                    performers.append({"id": row["id"], "name": row["name"]})
                else:
                    not_found.append(name)

            if len(performers) < 2:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Need at least 2 performers. Not found: {', '.join(not_found)}",
                }

            # Gather stats for each performer
            comparison = []
            all_scene_ids: dict[int, set[int]] = {}  # performer_id -> set of scene_ids
            all_tag_ids: dict[int, set[int]] = {}  # performer_id -> set of tag_ids

            for p in performers:
                performer_id = p["id"]

                # Scene count
                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM performers_scenes WHERE performer_id = ?",
                    (performer_id,),
                )
                scene_count = cursor.fetchone()["cnt"]

                # Get scene IDs for shared scenes calculation
                cursor.execute(
                    "SELECT scene_id FROM performers_scenes WHERE performer_id = ?",
                    (performer_id,),
                )
                scene_ids = {row["scene_id"] for row in cursor.fetchall()}
                all_scene_ids[performer_id] = scene_ids

                # Views and O count
                cursor.execute(
                    """
                    SELECT
                        COALESCE(SUM(view_agg.view_count), 0) as total_views,
                        COALESCE(SUM(o_agg.o_count), 0) as total_o
                    FROM performers_scenes ps
                    LEFT JOIN (
                        SELECT scene_id, COUNT(*) as view_count
                        FROM scenes_view_dates GROUP BY scene_id
                    ) view_agg ON ps.scene_id = view_agg.scene_id
                    LEFT JOIN (
                        SELECT scene_id, COUNT(*) as o_count
                        FROM scenes_o_dates GROUP BY scene_id
                    ) o_agg ON ps.scene_id = o_agg.scene_id
                    WHERE ps.performer_id = ?
                    """,
                    (performer_id,),
                )
                stats_row = cursor.fetchone()
                total_views = stats_row["total_views"] or 0
                total_o = stats_row["total_o"] or 0

                # Top tags
                cursor.execute(
                    """
                    SELECT t.id, t.name, COUNT(*) as cnt
                    FROM scenes_tags st
                    JOIN performers_scenes ps ON st.scene_id = ps.scene_id
                    JOIN tags t ON st.tag_id = t.id
                    WHERE ps.performer_id = ?
                    GROUP BY t.id
                    ORDER BY cnt DESC
                    LIMIT 5
                    """,
                    (performer_id,),
                )
                top_tags = [
                    {"id": r["id"], "name": r["name"], "count": r["cnt"]} for r in cursor.fetchall()
                ]
                all_tag_ids[performer_id] = {t["id"] for t in top_tags}

                comparison.append(
                    {
                        "performer_id": performer_id,
                        "performer_name": p["name"],
                        "scene_count": scene_count,
                        "total_views": total_views,
                        "total_o": total_o,
                        "top_tags": top_tags,
                    }
                )

            # Find shared scenes (scenes with multiple compared performers)
            shared_scenes = []
            performer_ids = [p["id"] for p in performers]
            common: set[int] = set()
            if len(performer_ids) >= 2:
                # Find scenes that have at least 2 of the compared performers
                scene_sets = [all_scene_ids[pid] for pid in performer_ids]
                # Start with first performer's scenes
                common = scene_sets[0]
                for other_set in scene_sets[1:]:
                    common = common & other_set
                    if not common:
                        break

                if common:
                    # Get details of shared scenes
                    common_list = list(common)[:10]
                    placeholders = ",".join("?" * len(common_list))
                    cursor.execute(
                        f"""
                        SELECT s.id, s.title, st.name as studio
                        FROM scenes s
                        LEFT JOIN studios st ON s.studio_id = st.id
                        WHERE s.id IN ({placeholders})
                        """,
                        common_list,
                    )
                    for row in cursor.fetchall():
                        shared_scenes.append(
                            {
                                "scene_id": row["id"],
                                "title": row["title"],
                                "studio": row["studio"],
                            }
                        )

            # Find shared tags
            tag_sets = [all_tag_ids[pid] for pid in performer_ids]
            common_tags: set[int] = tag_sets[0] if tag_sets else set()
            for other_set in tag_sets[1:]:
                common_tags = common_tags & other_set

            shared_tags = []
            if common_tags:
                placeholders = ",".join("?" * len(common_tags))
                cursor.execute(
                    f"SELECT id, name FROM tags WHERE id IN ({placeholders})",
                    list(common_tags),
                )
                shared_tags = [row["name"] for row in cursor.fetchall()]

            conn.close()

            # Build formatted output
            formatted_lines = ["## Performer Comparison"]

            # Comparison table
            formatted_lines.append(
                "\n| Metric | " + " | ".join(p["performer_name"] for p in comparison) + " |"
            )
            formatted_lines.append("|" + "|".join(["---"] * (len(comparison) + 1)) + "|")
            formatted_lines.append(
                "| Scenes | " + " | ".join(str(p["scene_count"]) for p in comparison) + " |"
            )
            formatted_lines.append(
                "| Total Views | " + " | ".join(str(p["total_views"]) for p in comparison) + " |"
            )
            formatted_lines.append(
                "| Total O | " + " | ".join(str(p["total_o"]) for p in comparison) + " |"
            )

            # Winner for each metric
            formatted_lines.append("\n**Leaders:**")
            max_scenes = max(comparison, key=lambda x: x["scene_count"])
            max_views = max(comparison, key=lambda x: x["total_views"])
            max_o = max(comparison, key=lambda x: x["total_o"])
            formatted_lines.append(
                f"- Most scenes: {max_scenes['performer_name']} ({max_scenes['scene_count']})"
            )
            formatted_lines.append(
                f"- Most views: {max_views['performer_name']} ({max_views['total_views']})"
            )
            formatted_lines.append(f"- Most O: {max_o['performer_name']} ({max_o['total_o']})")

            if shared_scenes:
                formatted_lines.append(f"\n**Shared scenes ({len(shared_scenes)}):**")
                for s in shared_scenes[:5]:
                    scene_title = s["title"] or f"Scene {s['scene_id']}"
                    formatted_lines.append(f"- {scene_title}")

            if shared_tags:
                formatted_lines.append(f"\n**Shared tags:** {', '.join(shared_tags)}")

            if not_found:
                formatted_lines.append(f"\n*Note: Not found: {', '.join(not_found)}*")

            return {
                "success": True,
                "data": {
                    "performers": comparison,
                    "shared_scenes": shared_scenes,
                    "shared_scene_count": len(common),
                    "shared_tags": shared_tags,
                    "not_found": not_found,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


# =============================================================================
# Phase 4: Advanced Tools
# =============================================================================


class QueryDuplicatesFindingTool(BaseTool):
    """
    Tool to find potential duplicate files using fingerprints.

    Enables queries like "Find duplicate files", "Are there any redundant scenes?".
    Uses the files_fingerprints table to match content.
    """

    @property
    def name(self) -> str:
        return "query_duplicates"

    @property
    def description(self) -> str:
        return (
            "Find potential duplicate files based on fingerprints, file size, or duration. "
            "Returns groups of scenes that may be duplicates with match confidence. "
            "Useful for identifying redundant content and freeing storage."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "method",
                "type": "string",
                "description": "Detection method (default: fingerprint)",
                "required": False,
                "enum": ["fingerprint", "size", "duration"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of duplicate groups to return (default: 20)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the duplicates finding query."""
        method: str = kwargs.get("method", "fingerprint")
        limit: int = kwargs.get("limit", 20)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            duplicate_groups: list[dict[str, Any]] = []

            if method == "fingerprint":
                # Find scenes with matching fingerprints
                cursor.execute(
                    """
                    SELECT fp.fingerprint, fp.type, COUNT(*) as match_count,
                           GROUP_CONCAT(DISTINCT fs.scene_id) as scene_ids
                    FROM files_fingerprints fp
                    JOIN files f ON fp.file_id = f.id
                    JOIN scenes_files fs ON f.id = fs.file_id
                    WHERE fp.fingerprint IS NOT NULL AND fp.fingerprint != ''
                    GROUP BY fp.fingerprint, fp.type
                    HAVING match_count > 1
                    ORDER BY match_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                for row in cursor.fetchall():
                    scene_ids = [int(sid) for sid in row["scene_ids"].split(",")]
                    scenes = self._get_scene_details(cursor, scene_ids)
                    duplicate_groups.append(
                        {
                            "match_type": f"fingerprint ({row['type']})",
                            "confidence": "high",
                            "scene_count": row["match_count"],
                            "scenes": scenes,
                        }
                    )

            elif method == "size":
                # Find scenes with identical file sizes
                cursor.execute(
                    """
                    SELECT f.size, COUNT(DISTINCT fs.scene_id) as match_count,
                           GROUP_CONCAT(DISTINCT fs.scene_id) as scene_ids
                    FROM files f
                    JOIN scenes_files fs ON f.id = fs.file_id
                    WHERE f.size > 0
                    GROUP BY f.size
                    HAVING match_count > 1
                    ORDER BY f.size DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                for row in cursor.fetchall():
                    scene_ids = [int(sid) for sid in row["scene_ids"].split(",")]
                    scenes = self._get_scene_details(cursor, scene_ids)
                    size_mb = row["size"] / (1024 * 1024)
                    duplicate_groups.append(
                        {
                            "match_type": f"file size ({size_mb:.1f} MB)",
                            "confidence": "medium",
                            "scene_count": row["match_count"],
                            "scenes": scenes,
                        }
                    )

            elif method == "duration":
                # Find scenes with identical durations (within 1 second)
                cursor.execute(
                    """
                    SELECT ROUND(vf.duration) as rounded_duration,
                           COUNT(DISTINCT fs.scene_id) as match_count,
                           GROUP_CONCAT(DISTINCT fs.scene_id) as scene_ids
                    FROM video_files vf
                    JOIN scenes_files fs ON vf.file_id = fs.file_id
                    WHERE vf.duration > 60
                    GROUP BY rounded_duration
                    HAVING match_count > 1
                    ORDER BY match_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                for row in cursor.fetchall():
                    scene_ids = [int(sid) for sid in row["scene_ids"].split(",")]
                    scenes = self._get_scene_details(cursor, scene_ids)
                    duration_mins = row["rounded_duration"] / 60
                    duplicate_groups.append(
                        {
                            "match_type": f"duration ({duration_mins:.1f} min)",
                            "confidence": "low",
                            "scene_count": row["match_count"],
                            "scenes": scenes,
                        }
                    )

            conn.close()

            # Calculate totals
            total_groups = len(duplicate_groups)
            total_potential_duplicates = sum(g["scene_count"] for g in duplicate_groups)

            # Format output
            formatted_lines = [f"## Potential Duplicates ({method} method)"]
            formatted_lines.append(f"**Groups found:** {total_groups}")
            formatted_lines.append(f"**Total potential duplicates:** {total_potential_duplicates}")

            for i, group in enumerate(duplicate_groups[:10], 1):
                formatted_lines.append(
                    f"\n### Group {i} ({group['match_type']}, {group['confidence']} confidence)"
                )
                for s in group["scenes"]:
                    title = s.get("title") or f"Scene {s['scene_id']}"
                    size = s.get("size_mb", 0)
                    formatted_lines.append(f"- [{title}](/scenes/{s['scene_id']}) ({size:.1f} MB)")

            return {
                "success": True,
                "data": {
                    "method": method,
                    "duplicate_groups": duplicate_groups,
                    "total_groups": total_groups,
                    "total_potential_duplicates": total_potential_duplicates,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }

    def _get_scene_details(
        self, cursor: sqlite3.Cursor, scene_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Get details for a list of scene IDs."""
        scenes = []
        for scene_id in scene_ids:
            cursor.execute(
                """
                SELECT s.id, s.title, st.name as studio,
                       f.size, vf.duration
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN scenes_files fs ON s.id = fs.scene_id
                LEFT JOIN files f ON fs.file_id = f.id
                LEFT JOIN video_files vf ON f.id = vf.file_id
                WHERE s.id = ?
                """,
                (scene_id,),
            )
            row = cursor.fetchone()
            if row:
                scenes.append(
                    {
                        "scene_id": scene_id,
                        "title": row["title"],
                        "studio": row["studio"],
                        "size_mb": (row["size"] or 0) / (1024 * 1024),
                        "duration_mins": (row["duration"] or 0) / 60,
                    }
                )
        return scenes


class QueryOHistoryTool(BaseTool):
    """
    Tool to analyze O event patterns.

    Enables queries like "When do I typically O?", "What scenes have I O'd to this month?".
    Uses the scenes_o_dates table for pattern analysis.
    """

    @property
    def name(self) -> str:
        return "query_o_history"

    @property
    def description(self) -> str:
        return (
            "Analyze O event patterns and history. "
            "Shows O events over time, scenes with multiple O events, "
            "and patterns (time of day, day of week). "
            "Useful for understanding engagement patterns."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "start_date",
                "type": "string",
                "description": "Start date filter (YYYY-MM-DD format)",
                "required": False,
                "enum": None,
            },
            {
                "name": "end_date",
                "type": "string",
                "description": "End date filter (YYYY-MM-DD format)",
                "required": False,
                "enum": None,
            },
            {
                "name": "group_by",
                "type": "string",
                "description": "Time period grouping (default: day)",
                "required": False,
                "enum": ["day", "week", "month", "hour", "dayofweek"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum results (default: 30)",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the O history query."""
        start_date: str | None = kwargs.get("start_date")
        end_date: str | None = kwargs.get("end_date")
        group_by: str = kwargs.get("group_by", "day")
        limit: int = kwargs.get("limit", 30)

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build date filter
            date_conditions: list[str] = []
            date_params: list[Any] = []
            if start_date:
                date_conditions.append("DATE(o_date) >= ?")
                date_params.append(start_date)
            if end_date:
                date_conditions.append("DATE(o_date) <= ?")
                date_params.append(end_date)

            date_where = "WHERE " + " AND ".join(date_conditions) if date_conditions else ""

            # Get O events grouped by time period
            if group_by == "hour":
                date_format = "strftime('%H', o_date)"
                label = "hour"
            elif group_by == "dayofweek":
                date_format = "strftime('%w', o_date)"
                label = "day_of_week"
            elif group_by == "week":
                date_format = "strftime('%Y-W%W', o_date)"
                label = "week"
            elif group_by == "month":
                date_format = "strftime('%Y-%m', o_date)"
                label = "month"
            else:  # day
                date_format = "DATE(o_date)"
                label = "date"

            cursor.execute(
                f"""
                SELECT {date_format} as period, COUNT(*) as o_count
                FROM scenes_o_dates
                {date_where}
                GROUP BY period
                ORDER BY period DESC
                LIMIT ?
                """,
                (*date_params, limit),
            )

            time_series = []
            for row in cursor.fetchall():
                period_val = row["period"]
                # Convert day of week number to name
                if group_by == "dayofweek" and period_val is not None:
                    days = [
                        "Sunday",
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                    ]
                    try:
                        period_val = days[int(period_val)]
                    except (ValueError, IndexError):
                        pass
                elif group_by == "hour" and period_val is not None:
                    period_val = f"{period_val}:00"

                time_series.append(
                    {
                        label: period_val,
                        "o_count": row["o_count"],
                    }
                )

            # Get top scenes by O count
            cursor.execute(
                f"""
                SELECT sod.scene_id, s.title, st.name as studio,
                       COUNT(*) as o_count,
                       MAX(DATE(sod.o_date)) as last_o_date
                FROM scenes_o_dates sod
                JOIN scenes s ON sod.scene_id = s.id
                LEFT JOIN studios st ON s.studio_id = st.id
                {date_where.replace("o_date", "sod.o_date")}
                GROUP BY sod.scene_id
                ORDER BY o_count DESC
                LIMIT 10
                """,
                date_params,
            )

            top_scenes = []
            for row in cursor.fetchall():
                top_scenes.append(
                    {
                        "scene_id": row["scene_id"],
                        "title": row["title"],
                        "studio": row["studio"],
                        "o_count": row["o_count"],
                        "last_o_date": row["last_o_date"],
                    }
                )

            # Get total stats
            cursor.execute(
                f"""
                SELECT COUNT(*) as total_o,
                       COUNT(DISTINCT scene_id) as unique_scenes
                FROM scenes_o_dates
                {date_where}
                """,
                date_params,
            )
            stats_row = cursor.fetchone()
            total_o = stats_row["total_o"]
            unique_scenes = stats_row["unique_scenes"]

            conn.close()

            # Format output
            formatted_lines = ["## O Event History"]
            formatted_lines.append(f"**Total O events:** {total_o}")
            formatted_lines.append(f"**Unique scenes:** {unique_scenes}")
            if start_date or end_date:
                date_range = f"{start_date or 'beginning'} to {end_date or 'now'}"
                formatted_lines.append(f"**Date range:** {date_range}")

            if time_series:
                formatted_lines.append(f"\n**O events by {label}:**")
                for ts in time_series[:15]:
                    formatted_lines.append(f"- {ts[label]}: {ts['o_count']} O events")

            if top_scenes:
                formatted_lines.append("\n**Top scenes by O count:**")
                for s in top_scenes[:5]:
                    title = s["title"] or f"Scene {s['scene_id']}"
                    formatted_lines.append(
                        f"- [{title}](/scenes/{s['scene_id']}): {s['o_count']} O events"
                    )

            return {
                "success": True,
                "data": {
                    "group_by": group_by,
                    "time_series": time_series,
                    "top_scenes": top_scenes,
                    "total_o": total_o,
                    "unique_scenes": unique_scenes,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                    },
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryPerformerCareerTimelineTool(BaseTool):
    """
    Tool to analyze a performer's content over time.

    Enables queries like "Show me X's career timeline", "When was Y most active?".
    Tracks scenes by date with studios and tags.
    """

    @property
    def name(self) -> str:
        return "query_performer_career_timeline"

    @property
    def description(self) -> str:
        return (
            "Analyze a performer's content over time. "
            "Shows timeline of scenes by date, studios worked with, "
            "tag evolution, and career length. "
            "Useful for understanding a performer's history in your library."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_name",
                "type": "string",
                "description": "Performer name to analyze",
                "required": True,
                "enum": None,
            },
            {
                "name": "group_by",
                "type": "string",
                "description": "Time period grouping (default: year)",
                "required": False,
                "enum": ["year", "month", "quarter"],
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performer career timeline query."""
        performer_name: str = kwargs.get("performer_name", "")
        group_by: str = kwargs.get("group_by", "year")

        if not performer_name:
            return {
                "success": False,
                "data": None,
                "error": "performer_name is required",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Find the performer
            cursor.execute(
                """
                SELECT id, name, birthdate FROM performers
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY
                    CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
                    name
                LIMIT 1
                """,
                (f"%{performer_name}%", performer_name),
            )
            performer_row = cursor.fetchone()

            if not performer_row:
                conn.close()
                return {
                    "success": False,
                    "data": None,
                    "error": f"Performer '{performer_name}' not found",
                }

            performer_id = performer_row["id"]
            performer_name_found = performer_row["name"]
            birthdate = performer_row["birthdate"]

            # Build date format for grouping
            if group_by == "month":
                date_format = "strftime('%Y-%m', s.date)"
            elif group_by == "quarter":
                date_format = "strftime('%Y', s.date) || '-Q' || ((CAST(strftime('%m', s.date) AS INTEGER) - 1) / 3 + 1)"
            else:  # year
                date_format = "strftime('%Y', s.date)"

            # Get scenes by time period
            cursor.execute(
                f"""
                SELECT {date_format} as period,
                       COUNT(*) as scene_count,
                       GROUP_CONCAT(DISTINCT st.name) as studios
                FROM performers_scenes ps
                JOIN scenes s ON ps.scene_id = s.id
                LEFT JOIN studios st ON s.studio_id = st.id
                WHERE ps.performer_id = ?
                AND s.date IS NOT NULL AND s.date != ''
                GROUP BY period
                ORDER BY period
                """,
                (performer_id,),
            )

            timeline = []
            for row in cursor.fetchall():
                if row["period"]:
                    studios = row["studios"].split(",") if row["studios"] else []
                    timeline.append(
                        {
                            "period": row["period"],
                            "scene_count": row["scene_count"],
                            "studios": studios[:5],  # Limit studios shown
                        }
                    )

            # Get earliest and latest scene dates
            cursor.execute(
                """
                SELECT MIN(s.date) as earliest, MAX(s.date) as latest
                FROM performers_scenes ps
                JOIN scenes s ON ps.scene_id = s.id
                WHERE ps.performer_id = ?
                AND s.date IS NOT NULL AND s.date != ''
                """,
                (performer_id,),
            )
            dates_row = cursor.fetchone()
            earliest_date = dates_row["earliest"]
            latest_date = dates_row["latest"]

            # Get total scene count
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM performers_scenes WHERE performer_id = ?",
                (performer_id,),
            )
            total_scenes = cursor.fetchone()["cnt"]

            # Get top studios over career
            cursor.execute(
                """
                SELECT st.name, COUNT(*) as scene_count
                FROM performers_scenes ps
                JOIN scenes s ON ps.scene_id = s.id
                JOIN studios st ON s.studio_id = st.id
                WHERE ps.performer_id = ?
                GROUP BY st.id
                ORDER BY scene_count DESC
                LIMIT 5
                """,
                (performer_id,),
            )
            top_studios = [
                {"name": row["name"], "scene_count": row["scene_count"]}
                for row in cursor.fetchall()
            ]

            # Get excluded tag IDs for tag query
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # Get top tags over career (excluding excluded)
            exclude_clause = ""
            tag_params: list[Any] = [performer_id]
            if excluded_ids:
                placeholders = ",".join("?" * len(excluded_ids))
                exclude_clause = f"AND t.id NOT IN ({placeholders})"
                tag_params.extend(list(excluded_ids))

            cursor.execute(
                f"""
                SELECT t.name, COUNT(*) as usage_count
                FROM performers_scenes ps
                JOIN scenes_tags stg ON ps.scene_id = stg.scene_id
                JOIN tags t ON stg.tag_id = t.id
                WHERE ps.performer_id = ?
                {exclude_clause}
                GROUP BY t.id
                ORDER BY usage_count DESC
                LIMIT 10
                """,
                tag_params,
            )
            top_tags = [
                {"name": row["name"], "usage_count": row["usage_count"]}
                for row in cursor.fetchall()
            ]

            conn.close()

            # Calculate career span
            career_span = None
            if earliest_date and latest_date:
                try:
                    from datetime import datetime

                    earliest = datetime.strptime(earliest_date[:10], "%Y-%m-%d")
                    latest = datetime.strptime(latest_date[:10], "%Y-%m-%d")
                    career_days = (latest - earliest).days
                    career_years = career_days / 365.25
                    career_span = f"{career_years:.1f} years ({career_days} days)"
                except (ValueError, TypeError):
                    pass

            # Format output
            formatted_lines = [f"## Career Timeline: {performer_name_found}"]
            formatted_lines.append(f"**Total scenes in library:** {total_scenes}")
            if earliest_date and latest_date:
                formatted_lines.append(
                    f"**Date range:** {earliest_date[:10]} to {latest_date[:10]}"
                )
            if career_span:
                formatted_lines.append(f"**Career span (in library):** {career_span}")

            if timeline:
                formatted_lines.append(f"\n**Scenes by {group_by}:**")
                for t in timeline:
                    studios_str = f" ({', '.join(t['studios'][:3])})" if t["studios"] else ""
                    formatted_lines.append(
                        f"- {t['period']}: {t['scene_count']} scenes{studios_str}"
                    )

            if top_studios:
                formatted_lines.append("\n**Top studios:**")
                for s in top_studios:
                    formatted_lines.append(f"- {s['name']}: {s['scene_count']} scenes")

            if top_tags:
                formatted_lines.append("\n**Most common tags:**")
                tag_list = ", ".join(t["name"] for t in top_tags[:8])
                formatted_lines.append(tag_list)

            return {
                "success": True,
                "data": {
                    "performer_id": performer_id,
                    "performer_name": performer_name_found,
                    "birthdate": birthdate,
                    "total_scenes": total_scenes,
                    "earliest_date": earliest_date,
                    "latest_date": latest_date,
                    "career_span": career_span,
                    "timeline": timeline,
                    "top_studios": top_studios,
                    "top_tags": top_tags,
                    "group_by": group_by,
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryScenesByPerformerTool(BaseTool):
    """
    Tool to retrieve scenes featuring specific performer(s).

    Enables filtering by performer name(s) with AND/OR logic.
    Returns scene IDs only for efficiency (use with EnrichSceneResultsTool for metadata).
    """

    @property
    def name(self) -> str:
        return "query_scenes_by_performer"

    @property
    def description(self) -> str:
        return (
            "Retrieve scenes featuring specific performer(s). "
            "Supports multiple performers with 'any' (OR) or 'all' (AND) matching. "
            "Returns scene IDs efficiently for further filtering or enrichment. "
            "Useful for finding all scenes from a specific performer or combination of performers."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "performer_names",
                "type": "array",
                "description": "List of performer names to search for (case-insensitive)",
                "required": True,
                "enum": None,
            },
            {
                "name": "match_mode",
                "type": "string",
                "description": "Matching mode: 'any' (scenes with ANY performer) or 'all' (scenes with ALL performers)",
                "required": False,
                "enum": ["any", "all"],
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes to return. Use query_performer_profile first to get scene_count.",
                "required": True,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the performer scene query."""
        performer_names: list[str] = kwargs.get("performer_names", [])
        match_mode: str = kwargs.get("match_mode", "any")
        limit: int | None = kwargs.get("limit")

        if not performer_names:
            return {
                "success": False,
                "data": None,
                "error": "performer_names is required and must be a non-empty list",
            }

        if not limit or limit <= 0:
            return {
                "success": False,
                "data": None,
                "error": "limit parameter is required and must be > 0. Use query_performer_profile to get scene_count first.",
            }

        if match_mode not in ["any", "all"]:
            return {
                "success": False,
                "data": None,
                "error": "match_mode must be 'any' or 'all'",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # First, find matching performers (case-insensitive)
            placeholders = ",".join("?" * len(performer_names))
            cursor.execute(
                f"""
                SELECT id, name
                FROM performers
                WHERE LOWER(name) IN ({placeholders})
                """,
                [name.lower() for name in performer_names],
            )

            matched_performers = []
            matched_ids = []
            for row in cursor.fetchall():
                matched_performers.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                    }
                )
                matched_ids.append(row["id"])

            # Track performers not found
            matched_names_lower = {p["name"].lower() for p in matched_performers}
            not_found = [
                name for name in performer_names if name.lower() not in matched_names_lower
            ]

            if not matched_ids:
                conn.close()
                return {
                    "success": True,
                    "data": {
                        "scene_ids": [],
                        "count": 0,
                        "matched_performers": [],
                        "not_found": not_found,
                    },
                    "error": None,
                }

            # Query scenes based on match mode
            if match_mode == "any":
                # OR logic: scenes with ANY of the performers
                id_placeholders = ",".join("?" * len(matched_ids))
                cursor.execute(
                    f"""
                    SELECT DISTINCT s.id
                    FROM scenes s
                    JOIN performers_scenes ps ON s.id = ps.scene_id
                    WHERE ps.performer_id IN ({id_placeholders})
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    [*matched_ids, limit],
                )
            else:  # match_mode == "all"
                # AND logic: scenes with ALL of the performers
                id_placeholders = ",".join("?" * len(matched_ids))
                cursor.execute(
                    f"""
                    SELECT s.id
                    FROM scenes s
                    JOIN performers_scenes ps ON s.id = ps.scene_id
                    WHERE ps.performer_id IN ({id_placeholders})
                    GROUP BY s.id
                    HAVING COUNT(DISTINCT ps.performer_id) = ?
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    [*matched_ids, len(matched_ids), limit],
                )

            scene_ids = [row["id"] for row in cursor.fetchall()]

            # Get scene count for each matched performer
            for performer in matched_performers:
                cursor.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM performers_scenes
                    WHERE performer_id = ?
                    """,
                    (performer["id"],),
                )
                performer["scene_count"] = cursor.fetchone()["cnt"]

            conn.close()

            return {
                "success": True,
                "data": {
                    "scene_ids": scene_ids,
                    "count": len(scene_ids),
                    "matched_performers": matched_performers,
                    "not_found": not_found,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class QueryScenesByTagTool(BaseTool):
    """
    Tool to retrieve scenes with specific tag(s).

    Enables filtering by tag name(s) with AND/OR logic and optional tag hierarchy.
    Returns scene IDs only for efficiency (use with EnrichSceneResultsTool for metadata).
    """

    @property
    def name(self) -> str:
        return "query_scenes_by_tag"

    @property
    def description(self) -> str:
        return (
            "Retrieve scenes with specific tag(s). "
            "Supports multiple tags with 'any' (OR) or 'all' (AND) matching. "
            "Can optionally include child tags in the hierarchy. "
            "Returns scene IDs efficiently for further filtering or enrichment. "
            "Respects plugin-level excluded tags. "
            "Useful for finding scenes by content type, actions, or categories."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "tag_names",
                "type": "array",
                "description": "List of tag names to search for (case-insensitive)",
                "required": True,
                "enum": None,
            },
            {
                "name": "match_mode",
                "type": "string",
                "description": "Matching mode: 'any' (scenes with ANY tag) or 'all' (scenes with ALL tags)",
                "required": False,
                "enum": ["any", "all"],
            },
            {
                "name": "include_child_tags",
                "type": "boolean",
                "description": "Include child tags in tag hierarchy (default: true)",
                "required": False,
                "enum": None,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of scenes to return. Use query_all_tags to check tag scene_count first.",
                "required": True,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tag scene query."""
        tag_names: list[str] = kwargs.get("tag_names", [])
        match_mode: str = kwargs.get("match_mode", "any")
        include_child_tags: bool = kwargs.get("include_child_tags", True)
        limit: int | None = kwargs.get("limit")

        if not tag_names:
            return {
                "success": False,
                "data": None,
                "error": "tag_names is required and must be a non-empty list",
            }

        if not limit or limit <= 0:
            return {
                "success": False,
                "data": None,
                "error": "limit parameter is required and must be > 0. Use query_all_tags to get scene_count for each tag first.",
            }

        if match_mode not in ["any", "all"]:
            return {
                "success": False,
                "data": None,
                "error": "match_mode must be 'any' or 'all'",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get excluded tags from plugin settings
            excluded_tags = self.get_excluded_tags()
            excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

            # First, find matching tags (case-insensitive)
            placeholders = ",".join("?" * len(tag_names))
            cursor.execute(
                f"""
                SELECT id, name
                FROM tags
                WHERE LOWER(name) IN ({placeholders})
                """,
                [name.lower() for name in tag_names],
            )

            matched_tags = []
            matched_ids = set()
            for row in cursor.fetchall():
                tag_id = row["id"]
                tag_name = row["name"]
                matched_tags.append(
                    {
                        "id": tag_id,
                        "name": tag_name,
                    }
                )
                matched_ids.add(tag_id)

            # Track tags not found
            matched_names_lower = {t["name"].lower() for t in matched_tags}
            not_found = [name for name in tag_names if name.lower() not in matched_names_lower]

            if not matched_ids:
                conn.close()
                return {
                    "success": True,
                    "data": {
                        "scene_ids": [],
                        "count": 0,
                        "matched_tags": [],
                        "not_found": not_found,
                    },
                    "error": None,
                }

            # If include_child_tags is enabled, find all descendants
            all_tag_ids = set(matched_ids)
            if include_child_tags:
                for tag_id in matched_ids:
                    cursor.execute(
                        """
                        WITH RECURSIVE descendants AS (
                            SELECT child_id as id
                            FROM tags_relations
                            WHERE parent_id = ?

                            UNION

                            SELECT tr.child_id
                            FROM tags_relations tr
                            JOIN descendants d ON tr.parent_id = d.id
                        )
                        SELECT id FROM descendants
                        """,
                        (tag_id,),
                    )
                    child_ids = {row["id"] for row in cursor.fetchall()}
                    all_tag_ids.update(child_ids)

            # Remove excluded tags
            all_tag_ids = all_tag_ids - excluded_ids

            if not all_tag_ids:
                conn.close()
                return {
                    "success": True,
                    "data": {
                        "scene_ids": [],
                        "count": 0,
                        "matched_tags": matched_tags,
                        "not_found": not_found,
                    },
                    "error": None,
                }

            # Query scenes based on match mode
            if match_mode == "any":
                # OR logic: scenes with ANY of the tags
                id_placeholders = ",".join("?" * len(all_tag_ids))
                cursor.execute(
                    f"""
                    SELECT DISTINCT s.id
                    FROM scenes s
                    JOIN scenes_tags st ON s.id = st.scene_id
                    WHERE st.tag_id IN ({id_placeholders})
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    [*all_tag_ids, limit],
                )
            else:  # match_mode == "all"
                # AND logic: scenes with ALL of the original matched tags
                # Note: We use matched_ids (not all_tag_ids) for AND logic
                # to ensure scenes have ALL requested tags, not just descendants
                id_placeholders = ",".join("?" * len(matched_ids))
                cursor.execute(
                    f"""
                    SELECT s.id
                    FROM scenes s
                    JOIN scenes_tags st ON s.id = st.scene_id
                    WHERE st.tag_id IN ({id_placeholders})
                    GROUP BY s.id
                    HAVING COUNT(DISTINCT st.tag_id) = ?
                    ORDER BY s.id DESC
                    LIMIT ?
                    """,
                    [*matched_ids, len(matched_ids), limit],
                )

            scene_ids = [row["id"] for row in cursor.fetchall()]

            # Get scene count for each matched tag
            for tag in matched_tags:
                cursor.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM scenes_tags
                    WHERE tag_id = ?
                    """,
                    (tag["id"],),
                )
                tag["scene_count"] = cursor.fetchone()["cnt"]

            conn.close()

            return {
                "success": True,
                "data": {
                    "scene_ids": scene_ids,
                    "count": len(scene_ids),
                    "matched_tags": matched_tags,
                    "not_found": not_found,
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }


class EnrichSceneResultsTool(BaseTool):
    """
    Tool to enrich scene IDs with full metadata.

    Adds titles, performers, tags, studio, ratings, engagement stats, and duration.
    Preserves input order by default (maintains similarity/engagement ranking).
    """

    @property
    def name(self) -> str:
        return "enrich_scene_results"

    @property
    def description(self) -> str:
        return (
            "Add full metadata to a list of scene IDs. "
            "Returns complete scene information including title, performers, tags, "
            "studio, rating, view count, O count, engagement score, and duration. "
            "Preserves input order by default to maintain ranking from previous tools. "
            "Useful as final step after filtering to get presentation-ready results."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "scene_ids",
                "type": "array",
                "description": "List of scene IDs to enrich with metadata",
                "required": True,
                "enum": None,
            },
            {
                "name": "include_fields",
                "type": "array",
                "description": "Fields to include: performers, tags, studio, rating, engagement (default: all)",
                "required": False,
                "enum": None,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "Sort order: 'input_order' (preserve input), 'title', 'date', 'rating' (default: input_order)",
                "required": False,
                "enum": ["input_order", "title", "date", "rating"],
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the scene enrichment."""
        scene_ids: list[int] = kwargs.get("scene_ids", [])
        include_fields: list[str] = kwargs.get(
            "include_fields", ["performers", "tags", "studio", "rating", "engagement"]
        )
        sort_by: str = kwargs.get("sort_by", "input_order")

        if not scene_ids:
            return {
                "success": False,
                "data": None,
                "error": "scene_ids is required and must be a non-empty list",
            }

        db_path = get_stash_db_path()
        if not db_path.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Stash database not found at {db_path}",
            }

        # Engagement counts, scoring, and the rating-to-stars conversion all come
        # from the single EngagementCalculator (ADR-0004 canonical formula). This
        # tool no longer carries its own engagement query or formula. Lazy import
        # avoids the engagement.py -> tools.database import cycle.
        from ..recommendations.engagement import EngagementCalculator
        from ..recommendations.types import EngagementScoringMethod

        calculator = EngagementCalculator()
        engagement_data = calculator.get_engagement(scene_ids)

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Get basic scene metadata (engagement counts come from get_engagement above)
            placeholders = ",".join("?" * len(scene_ids))
            cursor.execute(
                f"""
                SELECT s.id, s.title, s.date, s.created_at,
                       st.id as studio_id, st.name as studio_name,
                       vf.duration
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                LEFT JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                LEFT JOIN files f ON sf.file_id = f.id
                LEFT JOIN video_files vf ON f.id = vf.file_id
                WHERE s.id IN ({placeholders})
                """,
                scene_ids,
            )

            scenes = []
            scene_map = {}
            for row in cursor.fetchall():
                scene_id = row["id"]

                engagement = engagement_data.get(scene_id)
                if engagement is not None:
                    view_count = engagement["view_count"]
                    o_count = engagement["o_count"]
                    rating100 = engagement["rating"]
                    engagement_score = calculator.calculate_score(
                        engagement, EngagementScoringMethod.BASE_WEIGHTED
                    ).raw_score
                else:
                    view_count = 0
                    o_count = 0
                    rating100 = None
                    engagement_score = 0.0

                replay_count = max(view_count - 1, 0)  # Replays = views beyond first
                # rating100 (0-100) -> 0-5 star scale; unrated stays None (no penalty)
                stars = rating100 / 20.0 if rating100 else None

                scene_data = {
                    "scene_id": scene_id,
                    "title": row["title"],
                    "date": row["date"],
                    "created_at": row["created_at"],
                    "duration_seconds": row["duration"],
                }

                if "rating" in include_fields:
                    scene_data["rating"] = stars  # 0-5 star scale (rating100 / 20)

                if "engagement" in include_fields:
                    scene_data["view_count"] = view_count
                    scene_data["o_count"] = o_count
                    scene_data["replay_count"] = replay_count
                    scene_data["engagement_score"] = round(engagement_score, 2)

                if "studio" in include_fields and row["studio_id"]:
                    scene_data["studio"] = {
                        "id": row["studio_id"],
                        "name": row["studio_name"],
                    }

                scenes.append(scene_data)
                scene_map[scene_id] = scene_data

            # Get performers for scenes if requested
            if "performers" in include_fields and scenes:
                cursor.execute(
                    f"""
                    SELECT ps.scene_id, p.id, p.name
                    FROM performers_scenes ps
                    JOIN performers p ON ps.performer_id = p.id
                    WHERE ps.scene_id IN ({placeholders})
                    ORDER BY ps.scene_id, p.name
                    """,
                    scene_ids,
                )

                for row in cursor.fetchall():
                    scene_id = row["scene_id"]
                    if scene_id in scene_map:
                        if "performers" not in scene_map[scene_id]:
                            scene_map[scene_id]["performers"] = []
                        scene_map[scene_id]["performers"].append(
                            {
                                "id": row["id"],
                                "name": row["name"],
                            }
                        )

            # Get tags for scenes if requested (excluding plugin-level excluded tags)
            if "tags" in include_fields and scenes:
                excluded_tags = self.get_excluded_tags()
                excluded_ids = get_excluded_tag_ids_with_children(cursor, excluded_tags)

                exclude_clause = ""
                tag_params = list(scene_ids)
                if excluded_ids:
                    placeholders_excluded = ",".join("?" * len(excluded_ids))
                    exclude_clause = f"AND t.id NOT IN ({placeholders_excluded})"
                    tag_params.extend(list(excluded_ids))

                cursor.execute(
                    f"""
                    SELECT st.scene_id, t.id, t.name
                    FROM scenes_tags st
                    JOIN tags t ON st.tag_id = t.id
                    WHERE st.scene_id IN ({placeholders})
                    {exclude_clause}
                    ORDER BY st.scene_id, t.name
                    """,
                    tag_params,
                )

                for row in cursor.fetchall():
                    scene_id = row["scene_id"]
                    if scene_id in scene_map:
                        if "tags" not in scene_map[scene_id]:
                            scene_map[scene_id]["tags"] = []
                        scene_map[scene_id]["tags"].append(
                            {
                                "id": row["id"],
                                "name": row["name"],
                            }
                        )

            conn.close()

            # Sort scenes if requested
            if sort_by == "title":
                scenes.sort(key=lambda x: (x["title"] or "").lower())
            elif sort_by == "date":
                scenes.sort(key=lambda x: x["date"] or "", reverse=True)
            elif sort_by == "rating":
                scenes.sort(key=lambda x: x.get("rating", 0) or 0, reverse=True)
            else:  # input_order
                # Preserve input order by sorting by position in scene_ids
                id_to_position = {sid: i for i, sid in enumerate(scene_ids)}
                scenes.sort(key=lambda x: id_to_position.get(x["scene_id"], 999999))

            # Format results for LLM display
            formatted_lines = []
            for i, scene in enumerate(scenes, 1):
                title = scene.get("title") or f"Scene {scene['scene_id']}"
                parts = [f"{i}. [{title}](/scenes/{scene['scene_id']})"]

                if scene.get("performers"):
                    performer_names = [p["name"] for p in scene["performers"]]
                    parts.append(f"Performers: {', '.join(performer_names)}")

                if "studio" in scene:
                    parts.append(f"Studio: {scene['studio']['name']}")

                if "rating" in scene and scene.get("rating"):
                    parts.append(f"Rating: {scene['rating']:.1f}⭐")

                if "engagement_score" in scene:
                    parts.append(
                        f"Engagement: {scene['engagement_score']:.1f} (👁 {scene['view_count']}, 🔥 {scene['o_count']})"
                    )

                formatted_lines.append(" | ".join(parts))

                if scene.get("tags"):
                    tag_names = [t["name"] for t in scene["tags"][:10]]
                    formatted_lines.append(f"   Tags: {', '.join(tag_names)}")

            return {
                "success": True,
                "data": {
                    "scenes": scenes,
                    "count": len(scenes),
                    "formatted_results": "\n".join(formatted_lines),
                },
                "error": None,
            }

        except sqlite3.Error as e:
            return {
                "success": False,
                "data": None,
                "error": f"Database error: {e!s}",
            }
