"""Statistics aggregation logic for library analysis using SQLite."""

import sqlite3
from typing import TYPE_CHECKING, Any

from ..tools.database import get_readonly_connection, get_stash_db_path

if TYPE_CHECKING:
    from ..stash_client import StashClient


class LibraryStatsAggregator:
    """
    Aggregates library statistics for LLM analysis.

    Collects and processes data from Stash SQLite database directly
    for accurate statistics that match Stash UI displays.
    """

    def __init__(self, stash: "StashClient", excluded_tags: list[str] | None = None):
        """
        Initialize the aggregator.

        Args:
            stash: StashClient instance (used for config path)
            excluded_tags: Optional list of tag names to exclude from analysis
        """
        self.stash = stash
        self.excluded_tags = {tag.lower() for tag in (excluded_tags or [])}

    def _get_connection(self) -> sqlite3.Connection:
        """Get a readonly SQLite connection."""
        db_path = get_stash_db_path()
        return get_readonly_connection(db_path)

    def get_basic_counts(self) -> dict[str, Any]:
        """
        Get basic library counts directly from SQLite.

        Returns:
            Dictionary with scene, performer, tag, studio counts and totals
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get counts
            cursor.execute("SELECT COUNT(*) FROM scenes")
            scene_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM performers")
            performer_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tags")
            tag_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM studios")
            studio_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM groups")
            movie_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM galleries")
            gallery_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]

            # Get duration and size from video files
            cursor.execute("""
                SELECT
                    COALESCE(SUM(vf.duration), 0) as total_duration,
                    COALESCE(SUM(f.size), 0) as total_size
                FROM scenes_files sf
                JOIN files f ON sf.file_id = f.id
                JOIN video_files vf ON f.id = vf.file_id
                WHERE sf."primary" = 1
            """)
            duration_size = cursor.fetchone()
            total_duration_seconds = duration_size[0] or 0
            total_size_bytes = duration_size[1] or 0

            # Get play statistics
            cursor.execute("SELECT COUNT(*) FROM scenes_view_dates")
            total_play_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM scenes_o_dates")
            total_o_count = cursor.fetchone()[0]

            # Get total play duration from scenes
            cursor.execute("SELECT COALESCE(SUM(play_duration), 0) FROM scenes")
            total_play_duration = cursor.fetchone()[0]

            return {
                "scene_count": scene_count,
                "performer_count": performer_count,
                "tag_count": tag_count,
                "studio_count": studio_count,
                "movie_count": movie_count,
                "gallery_count": gallery_count,
                "image_count": image_count,
                "total_duration_hours": round(total_duration_seconds / 3600, 1),
                "total_size_gb": round(total_size_bytes / (1024**3), 1),
                "total_play_count": total_play_count,
                "total_o_count": total_o_count,
                "total_play_duration_hours": round(total_play_duration / 3600, 1),
            }
        finally:
            conn.close()

    def get_viewing_stats(self) -> dict[str, Any]:
        """
        Get viewing statistics from SQLite.

        Returns:
            Dictionary with watch counts, times, and percentages
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Total scenes
            cursor.execute("SELECT COUNT(*) FROM scenes")
            total_scenes = cursor.fetchone()[0]

            # Watched scenes (distinct scenes that have view records)
            cursor.execute("SELECT COUNT(DISTINCT scene_id) FROM scenes_view_dates")
            watched_count = cursor.fetchone()[0]

            # Total plays (view events)
            cursor.execute("SELECT COUNT(*) FROM scenes_view_dates")
            total_plays = cursor.fetchone()[0]

            # Total play duration
            cursor.execute("SELECT COALESCE(SUM(play_duration), 0) FROM scenes")
            total_play_duration = cursor.fetchone()[0]

            watched_percent = (watched_count / total_scenes * 100) if total_scenes > 0 else 0
            avg_plays_per_watched = (total_plays / watched_count) if watched_count > 0 else 0

            return {
                "watched_count": watched_count,
                "watched_percent": round(watched_percent, 1),
                "unwatched_count": total_scenes - watched_count,
                "total_plays": total_plays,
                "avg_plays_per_watched": round(avg_plays_per_watched, 1),
                "total_watch_time_hours": round(total_play_duration / 3600, 1),
            }
        finally:
            conn.close()

    def get_top_performers(
        self, limit: int = 10, sort_by: str = "view_count"
    ) -> list[dict[str, Any]]:
        """
        Get top performers by the specified metric.

        Args:
            limit: Maximum number of performers to return
            sort_by: Metric to sort by (view_count, scene_count, o_count, play_duration)

        Returns:
            List of performer dictionaries with name and metric value
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if sort_by == "view_count":
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [
                    {"name": row[0], "view_count": row[1], "scene_count": row[2]} for row in rows
                ]

            elif sort_by == "scene_count":
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [{"name": row[0], "scene_count": row[1]} for row in rows]

            elif sort_by == "o_count":
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [{"name": row[0], "o_count": row[1], "scene_count": row[2]} for row in rows]

            elif sort_by == "play_duration":
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [
                    {
                        "name": row[0],
                        "play_duration_hours": round((row[1] or 0) / 3600, 2),
                        "scene_count": row[2],
                    }
                    for row in rows
                ]

            else:
                return []
        finally:
            conn.close()

    def get_top_tags(self, limit: int = 10, sort_by: str = "view_count") -> list[dict[str, Any]]:
        """
        Get top tags by the specified metric, excluding configured tags.

        Args:
            limit: Maximum number of tags to return
            sort_by: Metric to sort by (view_count, scene_count, o_count)

        Returns:
            List of tag dictionaries with name and metric value
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Build exclusion clause
            if self.excluded_tags:
                placeholders = ",".join("?" * len(self.excluded_tags))
                exclude_clause = f"WHERE LOWER(t.name) NOT IN ({placeholders})"
                exclude_params: list[Any] = list(self.excluded_tags)
            else:
                exclude_clause = ""
                exclude_params = []

            if sort_by == "view_count":
                query = f"""
                    SELECT
                        t.name,
                        COUNT(svd.view_date) as view_count,
                        COUNT(DISTINCT st.scene_id) as scene_count
                    FROM tags t
                    JOIN scenes_tags st ON t.id = st.tag_id
                    JOIN scenes_view_dates svd ON st.scene_id = svd.scene_id
                    {exclude_clause}
                    GROUP BY t.id, t.name
                    ORDER BY view_count DESC
                    LIMIT ?
                """
                cursor.execute(query, exclude_params + [limit])
                rows = cursor.fetchall()
                return [
                    {"name": row[0], "view_count": row[1], "scene_count": row[2]} for row in rows
                ]

            elif sort_by == "scene_count":
                query = f"""
                    SELECT
                        t.name,
                        COUNT(st.scene_id) as scene_count
                    FROM tags t
                    JOIN scenes_tags st ON t.id = st.tag_id
                    {exclude_clause}
                    GROUP BY t.id, t.name
                    ORDER BY scene_count DESC
                    LIMIT ?
                """
                cursor.execute(query, exclude_params + [limit])
                rows = cursor.fetchall()
                return [{"name": row[0], "scene_count": row[1]} for row in rows]

            elif sort_by == "o_count":
                query = f"""
                    SELECT
                        t.name,
                        COUNT(sod.o_date) as o_count,
                        COUNT(DISTINCT st.scene_id) as scene_count
                    FROM tags t
                    JOIN scenes_tags st ON t.id = st.tag_id
                    JOIN scenes_o_dates sod ON st.scene_id = sod.scene_id
                    {exclude_clause}
                    GROUP BY t.id, t.name
                    ORDER BY o_count DESC
                    LIMIT ?
                """
                cursor.execute(query, exclude_params + [limit])
                rows = cursor.fetchall()
                return [{"name": row[0], "o_count": row[1], "scene_count": row[2]} for row in rows]

            else:
                return []
        finally:
            conn.close()

    def get_top_studios(
        self, limit: int = 10, sort_by: str = "scene_count"
    ) -> list[dict[str, Any]]:
        """
        Get top studios by the specified metric.

        Args:
            limit: Maximum number of studios to return
            sort_by: Metric to sort by (scene_count, view_count)

        Returns:
            List of studio dictionaries with name and metric value
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if sort_by == "view_count":
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [
                    {"name": row[0], "view_count": row[1], "scene_count": row[2]} for row in rows
                ]

            else:  # scene_count (default)
                cursor.execute(
                    """
                    SELECT
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
                rows = cursor.fetchall()
                return [{"name": row[0], "scene_count": row[1]} for row in rows]
        finally:
            conn.close()

    def get_rating_distribution(self) -> dict[str, int]:
        """
        Get distribution of scene ratings from SQLite.

        Returns:
            Dictionary mapping rating ranges to counts
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get rating distribution for watched scenes
            cursor.execute("""
                SELECT
                    CASE
                        WHEN s.rating IS NULL OR s.rating = 0 THEN 'unrated'
                        WHEN s.rating <= 20 THEN '1_star'
                        WHEN s.rating <= 40 THEN '2_star'
                        WHEN s.rating <= 60 THEN '3_star'
                        WHEN s.rating <= 80 THEN '4_star'
                        ELSE '5_star'
                    END as rating_category,
                    COUNT(*) as count
                FROM scenes s
                WHERE EXISTS (
                    SELECT 1 FROM scenes_view_dates svd WHERE svd.scene_id = s.id
                )
                GROUP BY rating_category
            """)

            distribution = {
                "unrated": 0,
                "1_star": 0,
                "2_star": 0,
                "3_star": 0,
                "4_star": 0,
                "5_star": 0,
            }

            for row in cursor.fetchall():
                if row[0] in distribution:
                    distribution[row[0]] = row[1]

            return distribution
        finally:
            conn.close()

    def get_average_scene_duration(self) -> float:
        """
        Get average scene duration in minutes from SQLite.

        Returns:
            Average duration in minutes
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(vf.duration) / 60.0
                FROM scenes_files sf
                JOIN video_files vf ON sf.file_id = vf.file_id
                WHERE sf."primary" = 1
            """)
            result = cursor.fetchone()[0]
            return round(result, 1) if result else 0.0
        finally:
            conn.close()

    def aggregate_for_summary(self) -> dict[str, Any]:
        """
        Aggregate all statistics into a single dictionary for LLM prompt.

        Returns:
            Comprehensive statistics dictionary
        """
        basic = self.get_basic_counts()
        viewing = self.get_viewing_stats()

        return {
            # Basic counts
            "total_scenes": basic.get("scene_count", 0),
            "total_duration_hours": basic.get("total_duration_hours", 0),
            "total_size_gb": basic.get("total_size_gb", 0),
            "performer_count": basic.get("performer_count", 0),
            "tag_count": basic.get("tag_count", 0),
            "studio_count": basic.get("studio_count", 0),
            # Viewing stats
            "watched_count": viewing.get("watched_count", 0),
            "watched_percent": viewing.get("watched_percent", 0),
            "unwatched_count": viewing.get("unwatched_count", 0),
            "total_plays": viewing.get("total_plays", 0),
            "watch_time_hours": viewing.get("total_watch_time_hours", 0),
            "avg_plays_per_watched": viewing.get("avg_plays_per_watched", 0),
            # O-counter
            "total_o_count": basic.get("total_o_count", 0),
            # Top lists (using view_count as default - matches Stash UI Play Count sort)
            "top_performers": self.get_top_performers(10, sort_by="view_count"),
            "top_tags": self.get_top_tags(10, sort_by="view_count"),
            "top_studios": self.get_top_studios(5, sort_by="scene_count"),
            # Rating distribution
            "rating_distribution": self.get_rating_distribution(),
            # Averages
            "avg_scene_duration_minutes": self.get_average_scene_duration(),
        }
