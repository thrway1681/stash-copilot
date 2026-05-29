"""Agent orchestrator for LLM with tool use."""

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .config import LLMConfig
from .llm import get_provider
from .llm.base import BaseLLMProvider, Message
from .tools import get_all_tools
from .tools.base import BaseTool

if TYPE_CHECKING:
    from .stash_client import StashClient


class Agent:
    """
    Agent that orchestrates LLM interactions with tool use.

    The agent manages the conversation loop, executing tools when
    the LLM requests them and continuing until a final response.
    """

    def __init__(
        self,
        stash: "StashClient",
        llm_config: LLMConfig,
        log_callback: Callable[[str, str], None] | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize the agent.

        Args:
            stash: StashClient for database access
            llm_config: LLM configuration
            log_callback: Optional callback for logging (message, level)
            max_iterations: Maximum tool use iterations to prevent loops
        """
        self.stash = stash
        self.llm_config = llm_config
        self.log = log_callback or (lambda msg, level: None)
        self.max_iterations = max_iterations

        # Initialize LLM provider
        self.llm: BaseLLMProvider = get_provider(llm_config)

        # Initialize tools
        self.tools: list[BaseTool] = get_all_tools(stash)
        self.tool_map: dict[str, BaseTool] = {t.name: t for t in self.tools}

        # Get tool schemas for LLM
        self.tool_schemas = [t.to_schema() for t in self.tools]

    def run(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Run the agent with a user query.

        The agent will:
        1. Send the query to the LLM with available tools
        2. Execute any tool calls the LLM makes
        3. Continue the conversation until the LLM provides a final response

        Args:
            query: The user's query or request
            system_prompt: Optional system prompt for context
            temperature: LLM temperature setting

        Returns:
            The agent's final response
        """
        # Build initial messages
        messages: list[Message] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": query})

        self.log(f"Agent starting with query: {query[:100]}...", "info")
        self.log(f"Available tools: {list(self.tool_map.keys())}", "debug")

        # Check if LLM supports tools
        if not self.llm.supports_tools:
            self.log(
                f"Model {self.llm_config.model} doesn't support tools, using direct completion",
                "warning",
            )
            return self._run_without_tools(messages, temperature)

        # Run the agent loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.log(f"Agent iteration {iteration}/{self.max_iterations}", "debug")

            # Get LLM response with tools
            result = self.llm.chat(
                messages=messages,
                tools=self.tool_schemas if self.tools else None,
                temperature=temperature,
            )

            # If no tool calls, we have the final response
            if not result["tool_calls"]:
                final_response = result["content"] or ""
                self.log("Agent completed with final response", "info")
                return final_response

            # Process tool calls
            self.log(f"LLM requested {len(result['tool_calls'])} tool call(s)", "info")

            # Add assistant message with tool calls
            assistant_msg: Message = {
                "role": "assistant",
                "content": result["content"],
                "tool_calls": result["tool_calls"],
            }
            messages.append(assistant_msg)

            # Execute each tool and add results
            for tool_call in result["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = tool_call["id"]

                self.log(f"Executing tool: {tool_name}({tool_args})", "info")

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)

                # Add tool result message
                tool_msg: Message = {
                    "role": "tool",
                    "content": json.dumps(tool_result, indent=2),
                    "tool_call_id": tool_id,
                }
                messages.append(tool_msg)

                if tool_result.get("success"):
                    self.log(f"Tool {tool_name} succeeded", "debug")
                else:
                    self.log(f"Tool {tool_name} failed: {tool_result.get('error')}", "warning")

        # Max iterations reached
        self.log(f"Agent reached max iterations ({self.max_iterations})", "warning")
        return "I've reached the maximum number of steps. Here's what I found so far based on the tool results in our conversation."

    def _run_without_tools(
        self,
        messages: list[Message],
        temperature: float,
    ) -> str:
        """
        Run without tool support (fallback for models that don't support tools).

        Args:
            messages: Conversation messages
            temperature: LLM temperature

        Returns:
            LLM response
        """
        result = self.llm.chat(
            messages=messages,
            tools=None,
            temperature=temperature,
        )
        return result["content"] or ""

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result dictionary
        """
        tool = self.tool_map.get(tool_name)

        if not tool:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            return dict(tool.execute(**arguments))
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Tool execution error: {e!s}",
            }

    def get_available_tools(self) -> list[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return list(self.tool_map.keys())
