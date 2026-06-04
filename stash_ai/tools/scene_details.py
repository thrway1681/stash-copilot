"""Shared helper: batch-fetch scene details from the Stash SQLite DB.

Extracted from the plugin entry point (#4, commit 4) so the dispatch-seam search
tasks (``find_similar``, ``search_by_text``, ``find_similar_by_frame``,
``find_similar_performers``) share one implementation instead of each calling a
``StashPlugin`` method. Reads the Stash DB directly (per ADR-0002: SQLite, not
GraphQL) via the existing readonly-connection helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .database import get_readonly_connection, get_stash_db_path


def get_scene_details_batch(
    scene_ids: list[int],
    log_callback: Callable[[str, str], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Fetch details for multiple scenes from SQLite in batched queries.

    Returns a dict mapping ``scene_id`` to a details dict (title, date,
    rating100, play_count, o_counter, organized, studio, performers, tags, files,
    interactive). Returns ``{}`` if there are no ids, the DB is missing, or on any
    error (logged).
    """
    log = log_callback or (lambda msg, level: None)
    if not scene_ids:
        return {}

    db_path = get_stash_db_path()
    if not db_path.exists():
        log(f"Database not found at {db_path}", "warning")
        return {}

    try:
        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        # Build placeholders for IN clause
        placeholders = ",".join("?" * len(scene_ids))
        scene_ids_tuple = tuple(scene_ids)

        # Fetch scene base data
        log(f"Fetching details for {len(scene_ids)} scene IDs: {scene_ids[:5]}...", "debug")
        cursor.execute(
            f"""
            SELECT
                s.id,
                s.title,
                s.date,
                s.rating,
                s.organized,
                st.id as studio_id,
                st.name as studio_name
            FROM scenes s
            LEFT JOIN studios st ON s.studio_id = st.id
            WHERE s.id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        scenes: dict[int, dict[str, Any]] = {}
        rows = cursor.fetchall()
        log(f"Found {len(rows)} scenes in database", "debug")

        for row in rows:
            scene_id = row["id"]
            scenes[scene_id] = {
                "id": scene_id,
                "title": row["title"],
                "date": row["date"],
                "rating100": row["rating"],
                "play_count": 0,
                "o_counter": 0,
                "organized": bool(row["organized"]) if row["organized"] is not None else False,
                "studio": {"id": row["studio_id"], "name": row["studio_name"]}
                if row["studio_id"]
                else None,
                "performers": [],
                "tags": [],
                "files": [],
                "interactive": False,
            }

        # Fetch file info (duration, size, resolution)
        cursor.execute(
            f"""
            SELECT
                sf.scene_id,
                f.basename as path,
                f.size,
                vf.duration,
                vf.height,
                vf.width,
                vf.interactive
            FROM scenes_files sf
            JOIN files f ON sf.file_id = f.id
            JOIN video_files vf ON f.id = vf.file_id
            WHERE sf.scene_id IN ({placeholders}) AND sf."primary" = 1
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            scene_id = row["scene_id"]
            if scene_id in scenes:
                scenes[scene_id]["files"].append(
                    {
                        "path": row["path"],
                        "size": row["size"],
                        "duration": row["duration"],
                        "height": row["height"],
                        "width": row["width"],
                    }
                )
                scenes[scene_id]["interactive"] = bool(row["interactive"])

        # Fetch performers
        cursor.execute(
            f"""
            SELECT ps.scene_id, p.id, p.name
            FROM performers_scenes ps
            JOIN performers p ON ps.performer_id = p.id
            WHERE ps.scene_id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            scene_id = row["scene_id"]
            if scene_id in scenes:
                scenes[scene_id]["performers"].append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                    }
                )

        # Fetch tags
        cursor.execute(
            f"""
            SELECT st.scene_id, t.id, t.name
            FROM scenes_tags st
            JOIN tags t ON st.tag_id = t.id
            WHERE st.scene_id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            scene_id = row["scene_id"]
            if scene_id in scenes:
                scenes[scene_id]["tags"].append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                    }
                )

        # Fetch play counts from scenes_view_dates
        cursor.execute(
            f"""
            SELECT scene_id, COUNT(*) as play_count
            FROM scenes_view_dates
            WHERE scene_id IN ({placeholders})
            GROUP BY scene_id
            """,
            scene_ids_tuple,
        )
        play_counts = {row["scene_id"]: row["play_count"] for row in cursor.fetchall()}

        # Fetch o counts from scenes_o_dates
        cursor.execute(
            f"""
            SELECT scene_id, COUNT(*) as o_count
            FROM scenes_o_dates
            WHERE scene_id IN ({placeholders})
            GROUP BY scene_id
            """,
            scene_ids_tuple,
        )
        o_counts = {row["scene_id"]: row["o_count"] for row in cursor.fetchall()}

        # Update scene details with actual counts
        for scene_id in scene_ids:
            if scene_id in scenes:
                scenes[scene_id]["play_count"] = play_counts.get(scene_id, 0)
                scenes[scene_id]["o_counter"] = o_counts.get(scene_id, 0)

        conn.close()
        log(f"Fetched details for {len(scenes)} scenes from SQLite", "debug")
        return scenes

    except Exception as e:
        import traceback

        log(f"Error fetching scene details: {e}", "error")
        log(f"Traceback: {traceback.format_exc()}", "error")
        return {}
