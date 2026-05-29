"""AI Ask task - query the library with natural language."""

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from ..agent import Agent
from ..config import LLMConfig
from ..prompts.loader import get_prompt

if TYPE_CHECKING:
    from ..stash_client import StashClient


# Fallback system prompt (used if YAML file not found)
SYSTEM_PROMPT = """You are a helpful assistant for a media library application called Stash.
You have access to tools that can query the database to answer questions about the user's library.

When the user asks about performers, tags, scenes, or viewing statistics, use the available tools
to look up the information rather than guessing.

Be helpful, concise, and use the tool results to provide accurate information.
If a tool returns an error, explain what happened and suggest alternatives.

Available information you can look up:
- Tags associated with a performer's scenes
- (More tools will be added in the future)

Always base your answers on actual data from the tools when available."""


def _get_system_prompt() -> str:
    """Load system prompt from YAML with fallback to hardcoded."""
    try:
        return get_prompt("ask", "system", "system")
    except (FileNotFoundError, KeyError):
        return SYSTEM_PROMPT


class AskTask:
    """
    Task for answering natural language questions about the library.

    This task uses an agent with tools to query the database and
    provide accurate answers based on actual library data.
    """

    def __init__(
        self,
        stash: "StashClient",
        llm_config: LLMConfig,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """
        Initialize the ask task.

        Args:
            stash: StashClient instance for API calls
            llm_config: LLM configuration
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
        """
        self.stash = stash
        self.llm_config = llm_config
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

    def run(self, question: str) -> str:
        """
        Run the ask task with a user question.

        Args:
            question: The user's question about their library

        Returns:
            The agent's answer

        Raises:
            ConnectionError: If LLM is not accessible
            RuntimeError: If the task fails
        """
        self.log(f"Processing question: {question}", "info")
        self.progress(0, 2)

        # Create the agent
        try:
            agent = Agent(
                stash=self.stash,
                llm_config=self.llm_config,
                log_callback=self.log,
                max_iterations=5,
            )
        except ValueError as e:
            raise RuntimeError(f"Failed to initialize agent: {e}") from e

        self.log(f"Agent initialized with tools: {agent.get_available_tools()}", "info")
        self.progress(1, 2)

        # Run the agent with hot-reloaded system prompt
        try:
            answer = agent.run(
                query=question,
                system_prompt=_get_system_prompt(),
                temperature=0.7,
            )
        except ConnectionError as e:
            raise ConnectionError(str(e)) from e
        except Exception as e:
            raise RuntimeError(f"Agent failed: {e}") from e

        self.progress(2, 2)
        self.log("Question answered successfully", "info")

        # Save the result for UI access
        self._save_result(question, answer)

        return answer

    def _save_result(self, question: str, answer: str) -> None:
        """
        Save the Q&A result for UI access.

        Args:
            question: The user's question
            answer: The agent's answer
        """
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        result_file = os.path.join(assets_dir, "last_ask.json")

        data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(result_file, "w") as f:
                json.dump(data, f, indent=2)
            self.log(f"Result saved to {result_file}", "debug")
        except Exception as e:
            self.log(f"Warning: Could not save result: {e}", "warning")
