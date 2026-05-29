"""Prompt templates for library statistics summarization."""

from typing import Any

from .loader import get_prompt

# Fallback prompt (used if YAML file not found)
STATS_SUMMARY_PROMPT = """You are analyzing a media library's statistics to provide an engaging, personalized summary for the user.
    Your goal is to highlight interesting patterns and provide helpful insights.

## Library Overview
- Total scenes: {total_scenes}
- Total duration: {total_duration_hours} hours ({total_size_gb} GB)
- Performers in library: {performer_count}
- Tags used: {tag_count}
- Studios: {studio_count}
- Average scene duration: {avg_scene_duration_minutes} minutes

## Viewing Statistics
- Scenes watched: {watched_count} ({watched_percent}% of library)
- Scenes not yet watched: {unwatched_count}
- Total play count: {total_plays} (avg {avg_plays_per_watched} plays per watched scene)
- Estimated watch time: {watch_time_hours} hours
- Total O-count: {total_o_count}

## Top Performers (by views in watched content)
{top_performers_formatted}

## Most Common Tags (in watched content)
{top_tags_formatted}

## Top Studios
{top_studios_formatted}

## Rating Distribution (of watched scenes)
{rating_distribution_formatted}

---

Please provide a friendly, conversational summary (2-3 paragraphs) that:
1. Gives an overview of the library size and how much has been explored
2. Highlights interesting patterns in viewing habits and preferences
3. Notes any standout favorites based on the data
4. Optionally suggests what the user might enjoy exploring based on patterns

Keep the tone casual, helpful, and non-judgmental. Focus on interesting insights rather than just restating numbers. Be concise but engaging."""


def format_list_items(
    items: list[dict[str, Any]],
    name_key: str = "name",
    count_key: str = "view_count",
    label: str = "views",
) -> str:
    """
    Format a list of items for the prompt.

    Args:
        items: List of dictionaries with name and count
        name_key: Key for the name field
        count_key: Key for the count field
        label: Label for the count (e.g., "views", "scenes")

    Returns:
        Formatted string with numbered list
    """
    if not items:
        return "No data available"

    lines = []
    for i, item in enumerate(items[:10], 1):
        name = item.get(name_key, "Unknown")
        count = item.get(count_key, 0)
        lines.append(f"{i}. {name} ({count} {label})")

    return "\n".join(lines)


def format_rating_distribution(distribution: dict[str, int]) -> str:
    """
    Format rating distribution for the prompt.

    Args:
        distribution: Dictionary with rating categories and counts

    Returns:
        Formatted string
    """
    if not distribution:
        return "No rating data available"

    total = sum(distribution.values())
    if total == 0:
        return "No rated scenes"

    lines = []
    labels = {
        "5_star": "5 stars (81-100)",
        "4_star": "4 stars (61-80)",
        "3_star": "3 stars (41-60)",
        "2_star": "2 stars (21-40)",
        "1_star": "1 star (1-20)",
        "unrated": "Unrated",
    }

    for key in ["5_star", "4_star", "3_star", "2_star", "1_star", "unrated"]:
        count = distribution.get(key, 0)
        percent = (count / total * 100) if total > 0 else 0
        lines.append(f"- {labels[key]}: {count} scenes ({percent:.1f}%)")

    return "\n".join(lines)


def format_stats_prompt(stats: dict[str, Any]) -> str:
    """
    Format the complete statistics prompt.

    Args:
        stats: Aggregated statistics dictionary from LibraryStatsAggregator

    Returns:
        Formatted prompt string ready for LLM
    """
    # Format top performers
    top_performers = stats.get("top_performers", [])
    top_performers_formatted = format_list_items(top_performers, "name", "view_count", "views")

    # Format top tags
    top_tags = stats.get("top_tags", [])
    top_tags_formatted = format_list_items(top_tags, "name", "view_count", "views")

    # Format top studios
    top_studios = stats.get("top_studios", [])
    top_studios_formatted = format_list_items(top_studios, "name", "scene_count", "scenes")

    # Format rating distribution
    rating_dist = stats.get("rating_distribution", {})
    rating_distribution_formatted = format_rating_distribution(rating_dist)

    # Load prompt from YAML (hot-reloaded) with fallback
    try:
        prompt_template = get_prompt("stats", "summary", "summary")
    except (FileNotFoundError, KeyError):
        prompt_template = STATS_SUMMARY_PROMPT

    return prompt_template.format(
        total_scenes=stats.get("total_scenes", 0),
        total_duration_hours=stats.get("total_duration_hours", 0),
        total_size_gb=stats.get("total_size_gb", 0),
        performer_count=stats.get("performer_count", 0),
        tag_count=stats.get("tag_count", 0),
        studio_count=stats.get("studio_count", 0),
        avg_scene_duration_minutes=stats.get("avg_scene_duration_minutes", 0),
        watched_count=stats.get("watched_count", 0),
        watched_percent=stats.get("watched_percent", 0),
        unwatched_count=stats.get("unwatched_count", 0),
        total_plays=stats.get("total_plays", 0),
        avg_plays_per_watched=stats.get("avg_plays_per_watched", 0),
        watch_time_hours=stats.get("watch_time_hours", 0),
        total_o_count=stats.get("total_o_count", 0),
        top_performers_formatted=top_performers_formatted,
        top_tags_formatted=top_tags_formatted,
        top_studios_formatted=top_studios_formatted,
        rating_distribution_formatted=rating_distribution_formatted,
    )
