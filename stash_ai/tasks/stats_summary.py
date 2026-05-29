"""Library statistics summary task."""

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..config import LLMConfig
from ..data.aggregators import LibraryStatsAggregator
from ..llm import get_provider
from ..prompts.statistics import format_stats_prompt

if TYPE_CHECKING:
    from ..stash_client import StashClient


class StatsSummaryTask:
    """
    Task for generating AI-powered library statistics summaries.

    This task aggregates library data, formats it into a prompt,
    and uses an LLM to generate a natural language summary.
    """

    def __init__(
        self,
        stash: "StashClient",
        llm_config: LLMConfig,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        excluded_tags: list[str] | None = None,
    ):
        """
        Initialize the stats summary task.

        Args:
            stash: StashClient instance for API calls
            llm_config: LLM configuration
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
            excluded_tags: Optional list of tag names to exclude from analysis
        """
        self.stash = stash
        self.llm_config = llm_config
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)
        self.excluded_tags = excluded_tags or []

        # Initialize aggregator with stash interface and excluded tags
        self.aggregator = LibraryStatsAggregator(stash, excluded_tags=self.excluded_tags)

    def run(self) -> str:
        """
        Run the statistics summary task.

        Returns:
            Generated summary text

        Raises:
            ConnectionError: If LLM is not accessible
            RuntimeError: If summary generation fails
        """
        self.log("Starting library statistics summary generation", "info")
        self.progress(0, 3)

        # Step 1: Aggregate statistics
        self.log("Aggregating library statistics...", "info")
        try:
            stats = self.aggregator.aggregate_for_summary()
        except Exception as e:
            raise RuntimeError(f"Failed to aggregate statistics: {e}") from e
        self.progress(1, 3)

        # Log aggregated stats summary
        self.log("--- Aggregated Statistics ---", "info")
        self.log(
            f"Library: {stats.get('total_scenes', 0)} scenes, "
            f"{stats.get('total_duration_hours', 0)}h total, "
            f"{stats.get('total_size_gb', 0)} GB",
            "info",
        )
        self.log(
            f"Viewing: {stats.get('watched_count', 0)} watched "
            f"({stats.get('watched_percent', 0)}%), "
            f"{stats.get('total_plays', 0)} total plays, "
            f"{stats.get('watch_time_hours', 0)}h watch time",
            "info",
        )
        self.log(
            f"Content: {stats.get('performer_count', 0)} performers, "
            f"{stats.get('tag_count', 0)} tags, "
            f"{stats.get('studio_count', 0)} studios",
            "info",
        )

        # Log top performers/tags at info level
        top_performers = stats.get("top_performers", [])[:5]
        if top_performers:
            performer_list = ", ".join(
                f"{p['name']} ({p.get('view_count', p.get('play_count', 0))})"
                for p in top_performers
            )
            self.log(f"Top performers: {performer_list}", "info")

        top_tags = stats.get("top_tags", [])[:5]
        if top_tags:
            tag_list = ", ".join(
                f"{t['name']} ({t.get('view_count', t.get('play_count', 0))})" for t in top_tags
            )
            self.log(f"Top tags: {tag_list}", "info")

        self.log("-----------------------------", "info")

        # Step 2: Initialize LLM and generate summary
        self.log(f"Connecting to LLM ({self.llm_config.provider})...", "info")
        try:
            llm = get_provider(self.llm_config)
        except ValueError as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}") from e

        # Check LLM health
        if not llm.health_check():
            raise ConnectionError(
                f"Cannot connect to {self.llm_config.provider} at "
                f"{self.llm_config.base_url}. Make sure it's running."
            )
        self.progress(2, 3)

        # Step 3: Generate summary with streaming
        self.log("Generating AI summary (streaming)...", "info")
        prompt = format_stats_prompt(stats)

        # Log the prompt being sent to the AI
        self.log("=" * 60, "debug")
        self.log("PROMPT BEING SENT TO AI:", "debug")
        self.log("=" * 60, "debug")
        for line in prompt.split("\n"):
            self.log(line, "debug")
        self.log("=" * 60, "debug")

        try:
            summary = ""
            last_save_len = 0

            # Write initial empty streaming state
            self._save_summary("", stats, streaming=True)
            self.log("Starting LLM streaming...", "info")

            # Use streaming to write partial results
            token_count = 0
            for token in llm.stream(
                prompt,
                temperature=0.7,
                max_tokens=1024,
            ):
                summary += token
                token_count += 1

                # Update file every ~5 characters for smooth streaming
                if len(summary) - last_save_len >= 5:
                    self._save_summary(summary, stats, streaming=True)
                    self.log(f"Streaming: {len(summary)} chars, {token_count} tokens", "debug")
                    last_save_len = len(summary)

            # Final streaming update before marking complete
            if len(summary) > last_save_len:
                self._save_summary(summary, stats, streaming=True)

        except Exception as e:
            raise RuntimeError(f"Failed to generate summary: {e}") from e

        self.progress(3, 3)
        self.log("Summary generation complete!", "info")

        # Save final summary (not streaming)
        self._save_summary(summary, stats, streaming=False)

        return summary

    def _save_summary(self, summary: str, stats: dict[str, Any], streaming: bool = False) -> None:
        """
        Save the summary to a file for UI access.

        Args:
            summary: Generated summary text
            stats: Statistics data
            streaming: Whether the summary is still being generated
        """
        # Get plugin directory and save to assets folder (accessible via /plugin/stash-copilot/assets/)
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        summary_file = os.path.join(assets_dir, "last_summary.json")

        data = {
            "summary": summary,
            "generated_at": datetime.now().isoformat(),
            "status": "streaming" if streaming else "complete",
            "stats": {
                "total_scenes": stats.get("total_scenes", 0),
                "watched_count": stats.get("watched_count", 0),
                "watched_percent": stats.get("watched_percent", 0),
                "total_plays": stats.get("total_plays", 0),
                "watch_time_hours": stats.get("watch_time_hours", 0),
            },
        }

        try:
            with open(summary_file, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            if streaming:
                self.log(f"Streaming update: {len(summary)} chars written", "debug")
            else:
                self.log(f"Summary saved to {summary_file}", "info")
        except Exception as e:
            self.log(f"Warning: Could not save summary to file: {e}", "warning")
