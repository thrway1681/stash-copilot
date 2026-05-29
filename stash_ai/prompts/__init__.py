"""Prompt templates for LLM interactions."""

from .loader import get_prompt, get_prompts_dir, load_prompt_file
from .statistics import STATS_SUMMARY_PROMPT, format_stats_prompt

__all__ = [
    # Loader functions
    "get_prompt",
    "load_prompt_file",
    "get_prompts_dir",
    # Legacy exports (for backwards compatibility)
    "STATS_SUMMARY_PROMPT",
    "format_stats_prompt",
]
