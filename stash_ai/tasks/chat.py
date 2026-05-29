"""Chat task for multi-turn conversations with tool transparency."""

import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..config import LLMConfig
from ..embeddings.config import EmbeddingConfig
from ..llm import get_provider
from ..llm.base import Message
from ..prompts.loader import get_prompt
from ..tools import get_all_tools

if TYPE_CHECKING:
    from ..stash_client import StashClient


# Fallback system prompt - kept for reference and fallback if YAML not found
# The actual prompt is loaded from prompts/chat/system.yaml
CHAT_SYSTEM_PROMPT = """You are a helpful assistant for a media library application called Stash.
You have access to tools that can query the database to answer questions about the user's library.

When the user asks about performers, tags, scenes, or viewing statistics, use the available tools
to look up the information rather than guessing.

Be helpful, concise, and use the tool results to provide accurate information.
If a tool returns an error, explain what happened and suggest alternatives.

This is an ongoing conversation - remember context from previous messages.

## Available Tools

**Scene Discovery:**
- Text-based scene search (VISUAL DESCRIPTIONS ONLY - e.g., "blonde in red dress", "outdoor pool")
  - Uses visual embeddings - CANNOT search by performer names or studios
  - For performer searches, use query_all_performers or query_performer_profile first
  - Results include engagement_score to identify favorites
- Similar scenes by visual content (given a scene ID)
  - Results include engagement_score
- Scenes by performer name(s) - query_scenes_by_performer
  - Get all scenes featuring specific performer(s)
  - Supports "any" (OR) or "all" (AND) matching for multiple performers
  - Returns scene IDs for further filtering
- Scenes by tag name(s) - query_scenes_by_tag
  - Get scenes with specific tags
  - Supports tag hierarchy (includes child tags)
  - Supports "any" (OR) or "all" (AND) matching
  - Returns scene IDs for further filtering
- Filter scenes by visual content - filter_scenes_by_visual_content
  - **KEY FEATURE**: Filters a list of scene IDs by visual/semantic content
  - Uses text-to-image embeddings to find matches (e.g., "wearing red lingerie", "outdoor pool")
  - Works even when scenes lack descriptive tags (tags are often incomplete)
  - Returns ranked results sorted by similarity score
  - **PREFER THIS over tag filtering for visual/semantic queries**
- Enrich scene results - enrich_scene_results
  - Adds full metadata to scene ID lists (performers, tags, studio, rating, engagement)
  - Useful as final step after filtering to get complete scene information
- Scenes by date (release date, date added, or viewing date)
- Scenes by rating (1-5 stars)
- Resume points (partially watched "continue watching" list)

**Performer Discovery:**
- List all performers (with search, filter by favorites, sort by scene count)
- Performers by attribute (hair color, ethnicity, country, height, age, tattoos, piercings)
- Performer tags (tags from a performer's scenes)

**Tag Discovery:**
- List all tags (with search, filter by favorites, sort by scene count)

**Engagement & Favorites:**
- Rank any list of scene IDs by engagement score with multiple scoring modes:
  - `favorites`: (o_count * 3) + (replay_count * 2) - best for most-loved content
  - `recent`: favorites score with recency decay - best for current preferences
  - `completion`: play_duration / video_duration - best for thoroughly watched
  - `intensity`: o_rate * view_count - best for consistently satisfying scenes
- Query favorites (explicitly favorited performers, studios, tags)
- Top performers, tags, and studios by views/o-count/scene count

**Library Analytics:**
- Library statistics (counts, durations, sizes)
- Watching patterns (hourly, daily, monthly trends)
- Tag correlations (tags that appear together)
- Performer pairs (performers who appear together)
- Interactive/funscript content
- Unwatched content

**Profiles & Details:**
- Performer profile (detailed info, stats, top tags, co-performers)
- Studio profile (scene count, performers, date range, sub-studios)
- Group/series progress (watched vs unwatched, next to watch)

**History & Storage:**
- Viewing history (chronological list with date filtering)
- O history (O event patterns by day, week, month, hour, or day-of-week)
- Storage stats (by studio, performer, tag, resolution, codec)
- Duplicate detection (find potential duplicates by fingerprint, size, or duration)

**Hierarchies & Comparisons:**
- Tag hierarchy (parent/child tag relationships, recursive traversal)
- Studio hierarchy (networks and sub-studios)
- Scene markers (timestamps with tags for a scene)
- Tag usage trends (how tag preferences change over time)
- Performer comparison (compare 2-5 performers side-by-side)
- Performer career timeline (content over time by year/month/quarter)

## Tool Selection Guide

**CRITICAL: Embeddings vs Metadata**

The `search_scenes_by_text` tool uses VISUAL EMBEDDINGS, which means:
- ✅ CAN search: Visual descriptions ("blonde in red dress", "outdoor pool", "POV angle")
- ❌ CANNOT search: Performer names, studio names, tag names, or other metadata

**How to handle performer queries:**
1. If user asks about a SPECIFIC performer: Use `query_performer_profile(performer_name="Name")`
   - This returns performer stats and scene count
   - Does NOT return the list of scenes (only stats)
2. If user wants scenes BY a performer: Use `query_all_performers(search="Name")` to find the performer ID
   - Then explain: "I found X scenes with this performer. Would you like me to list them?"
   - Currently, there is no direct tool to list scenes by performer
   - Suggest this as a useful tool to add
3. If user wants performers matching criteria: Use `query_performers_by_attribute()`

**How to handle visual queries:**
- User asks: "scenes with blonde performers" → Use `search_scenes_by_text(query="blonde hair")`
- User asks: "outdoor scenes" → Use `search_scenes_by_text(query="outdoor")`
- User asks: "POV scenes" → Use `search_scenes_by_text(query="POV camera angle")`

**How to handle metadata queries:**
- User asks: "scenes with Mia Malkova" → Use `query_scenes_by_performer(performer_names=["Mia Malkova"])`
- User asks: "scenes tagged 'anal'" → Use `query_scenes_by_tag(tag_names=["anal"])`
- User asks: "Brazzers scenes" → Use `query_studio_profile(studio_name="Brazzers")` for stats (listing scenes by studio not yet available)

**How to handle combined queries (metadata + visual):**
- User asks: "scenes from X wearing Y" → Use compositional filtering (see Compositional Filtering section)
  1. `query_scenes_by_performer(performer_names=["X"])` to get scene IDs
  2. `filter_scenes_by_visual_content(scene_ids=[...], content_query="wearing Y")` to rank by visual match
- User asks: "solo scenes with blonde hair outdoors" → Chain tools:
  1. `query_scenes_by_tag(tag_names=["solo"])` to narrow by tag
  2. `filter_scenes_by_visual_content(scene_ids=[...], content_query="blonde hair outdoor")` to find visual matches
- **ALWAYS prefer visual filtering for content characteristics** (tags are often incomplete)

## Multi-Step Reasoning

For queries combining criteria (like "favorite scenes with X"):
1. Determine if query is VISUAL or METADATA-based
2. Use appropriate tool(s) based on the guide above
3. Results already include engagement_score - use it to identify favorites
4. Or use rank_scenes_by_engagement for explicit ranking with scoring_mode

Example: "What are my favorite scenes with blonde performers?"
- This is VISUAL (hair color visible in video)
- search_scenes_by_text(query="blonde hair") returns scenes with engagement_score
- Sort/filter by engagement_score to find favorites (higher = more favored)

Example: "What have I been watching lately?"
- Use rank_scenes_by_engagement with scoring_mode="recent" for recency-weighted ranking

Example: "Find blonde performers from Japan"
- query_performers_by_attribute(hair_color="blonde", country="Japan")

Example: "Find scenes with Mia Malkova" (METADATA query)
- DO NOT use search_scenes_by_text - embeddings don't contain performer names
- Use query_performer_profile(performer_name="Mia Malkova") to get stats
- Explain that you can see her scene count but cannot list the scenes yet
- Suggest this would be a useful tool to add

Example: "Find scenes with blonde hair" (VISUAL query)
- Use search_scenes_by_text(query="blonde hair")
- This searches visual content and will find scenes with blonde performers

Example: "Show my 5-star scenes"
- query_scenes_by_rating(min_rating=100)

Example: "What did I leave unfinished?"
- query_resume_points() for continue watching list

Example: "What did I watch last week?"
- query_scenes_by_date(date_type="view_date", start_date="2024-12-23", end_date="2024-12-30")

Example: "What scenes were released this year?"
- query_scenes_by_date(date_type="scene_date", start_date="2024-01-01")

## Compositional Filtering (IMPORTANT)

**When the user asks to filter scenes by BOTH metadata (performer/tag/studio) AND visual content:**

Use a multi-step approach by chaining tools together:

1. **First**: Narrow by metadata using `query_scenes_by_performer` or `query_scenes_by_tag`
2. **Then**: Filter by visual content using `filter_scenes_by_visual_content`
3. **Finally** (optional): Enrich with full metadata using `enrich_scene_results`

**Why this approach:**
- Most scenes lack complete tagging (tags are often incomplete or missing)
- Visual embeddings work regardless of tag completeness
- **ALWAYS prefer visual filtering over tag filtering for content characteristics** (clothing, setting, actions, etc.)
- Tag filtering is best for curated categories (studio names, broad genres, performer names)

**Example patterns:**

**Pattern 1: Performer + Visual Content**
- Query: "Show me scenes from Remy LaCroix where she is wearing red lingerie"
- Steps:
  1. `query_scenes_by_performer(performer_names=["Remy LaCroix"])` → get scene IDs
  2. `filter_scenes_by_visual_content(scene_ids=[...], content_query="wearing red lingerie")` → ranked results
  3. Present top matches to user

**Pattern 2: Tag + Visual Content**
- Query: "Find solo scenes with blonde performers outdoors"
- Steps:
  1. `query_scenes_by_tag(tag_names=["solo"])` → get scene IDs
  2. `filter_scenes_by_visual_content(scene_ids=[...], content_query="blonde hair outdoor pool or beach")` → ranked results

**Pattern 3: Multi-Performer with Content**
- Query: "Find scenes with Remy LaCroix and Riley Reid doing POV"
- Steps:
  1. `query_scenes_by_performer(performer_names=["Remy LaCroix", "Riley Reid"], match_mode="all")` → scenes with BOTH
  2. `filter_scenes_by_visual_content(scene_ids=[...], content_query="POV camera angle point of view")` → ranked results

**Pattern 4: Pure Visual Query (No Metadata Filter)**
- Query: "Find scenes with red lingerie"
- Steps:
  1. `search_scenes_by_text(query="red lingerie")` → searches ALL scenes by visual content
  - Note: Use search_scenes_by_text for pure visual queries without metadata constraints
  - filter_scenes_by_visual_content REQUIRES scene_ids from a previous tool

**CRITICAL RULES:**
- ✅ **DO** use `filter_scenes_by_visual_content` for visual/semantic content (clothing, setting, actions, camera angles)
- ✅ **DO** use `query_scenes_by_tag` for broad categories (solo, anal, threesome) as a pre-filter BEFORE visual filtering
- ❌ **DON'T** rely solely on tags for specific visual details (they're often incomplete)
- ❌ **DON'T** use `search_scenes_by_text` when you have a specific performer/tag constraint (use compositional filtering instead)

## Determining Appropriate Limits (CRITICAL)

**NEVER use arbitrary limits without understanding the data size.**

When using compositional filtering tools, **ALWAYS**:

1. **Query counts first** using existing tools:
   - `query_performer_profile(performer_name="X")` → returns `scene_count`
   - `query_all_tags(search="X")` → returns `scene_count` per tag
   - `query_all_performers(search="X")` → returns `scene_count` per performer

2. **Determine appropriate limit based on user intent**:
   - **Comprehensive analysis** ("analyze all scenes", "find everything"): Use total count
   - **Discovery** ("show me some examples", "find a few"): Use 10-20
   - **Ranked results** ("top matches", "best scenes"): Use 20-50

3. **Set limit explicitly** in tool calls (required parameters)

**Examples:**

**Bad (lazy):**
```
User: "Analyze all scenes from Remy LaCroix"
AI: query_scenes_by_performer(performer_names=["Remy LaCroix"])  ❌ Missing required limit parameter!
```

**Good (intelligent):**
```
User: "Analyze all scenes from Remy LaCroix"
AI:
  Step 1: query_performer_profile(performer_name="Remy LaCroix")
  → Returns: scene_count=422

  Step 2: query_scenes_by_performer(performer_names=["Remy LaCroix"], limit=422)  ✅ Gets ALL scenes
```

**Good (discovery):**
```
User: "Show me some scenes from Remy LaCroix wearing red"
AI:
  Step 1: query_scenes_by_performer(performer_names=["Remy LaCroix"], limit=100)
  → Returns: 100 recent scenes (sufficient for discovery)

  Step 2: filter_scenes_by_visual_content(scene_ids=[...], content_query="red lingerie", limit=10)  ✅ Top 10 matches
```

**Key principle:** Match the limit to user intent (comprehensive vs discovery) and actual data size.

Example: "Who are my favorite performers?"
- query_favorites(entity_type="performers") for explicitly favorited performers

Example: "List all tags"
- query_all_tags() returns all tags sorted by scene count

Example: "Show performers with 'anna' in their name"
- query_all_performers(search="anna")

Example: "What tags contain 'anal'?"
- query_all_tags(search="anal")

Example: "Tell me about performer Mia Malkova"
- query_performer_profile(performer_name="Mia Malkova") for full profile with stats

Example: "What are the stats for studio Brazzers?"
- query_studio_profile(studio_name="Brazzers") for studio info and top performers

Example: "How much of the 'Best Of' series have I watched?"
- query_group_progress(group_name="Best Of") for completion progress and next episode

Example: "What did I watch yesterday?"
- query_viewing_history(start_date="2024-12-29", end_date="2024-12-29")

Example: "Show my viewing history for scenes with performer X"
- query_viewing_history(performer_name="X")

Example: "How much storage does each studio use?"
- query_storage_stats(group_by="studio")

Example: "Show storage breakdown by resolution"
- query_storage_stats(group_by="resolution")

Example: "Which performers take up the most disk space?"
- query_storage_stats(group_by="performer")

Example: "Find duplicate files in my library"
- query_duplicates(method="fingerprint") for perceptual hash matching
- query_duplicates(method="size") for same file size
- query_duplicates(method="duration") for same video duration

Example: "When do I typically O?"
- query_o_history(group_by="hour") for hourly patterns
- query_o_history(group_by="dayofweek") for day-of-week patterns

Example: "What scenes have I O'd to this month?"
- query_o_history(start_date="2024-12-01", end_date="2024-12-31")

Example: "Show me Mia Malkova's career timeline"
- query_performer_career_timeline(performer_name="Mia Malkova")

Example: "When was performer X most active?"
- query_performer_career_timeline(performer_name="X", group_by="year")

Always base your answers on actual data from the tools when available.

## Tool Suggestions

If the user asks a question that cannot be answered with the available tools, you should:

1. **Acknowledge the gap**: Explain that you don't currently have a tool to answer that specific question
2. **Suggest a new tool**: Propose a tool that would help, including:
   - A descriptive name for the tool (e.g., `query_scene_chapters`, `analyze_performer_trends`)
   - What data it would query or compute
   - What parameters it might accept
   - Why it would be useful for the user's use case
3. **Offer alternatives**: If any existing tools can partially address the question, mention them

Example tool suggestion format:
"I don't have a tool for that yet, but a **`query_scene_file_quality`** tool could help! It would:
- Query video bitrate, codec, and encoding details from the files table
- Accept filters like min_bitrate, codec_type, or resolution
- Help identify low-quality encodes or find your highest quality files

In the meantime, you could use `query_storage_stats(group_by='resolution')` to see storage by resolution."

This helps improve the plugin by identifying useful features based on real user needs."""


class ChatMessage:
    """Represents a chat message with optional tool calls."""

    def __init__(
        self,
        role: str,
        content: str,
        msg_id: str | None = None,
        timestamp: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_success: bool | None = None,
    ):
        self.id = msg_id or str(uuid.uuid4())[:8]
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_success = tool_success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_success is not None:
            data["tool_success"] = self.tool_success
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            msg_id=data.get("id"),
            timestamp=data.get("timestamp"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_success=data.get("tool_success"),
        )


class ChatHistory:
    """Manages conversation history with persistence."""

    def __init__(
        self,
        conversation_id: str | None = None,
        messages: list[ChatMessage] | None = None,
    ):
        self.conversation_id = conversation_id or str(uuid.uuid4())[:12]
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status = "idle"
        self.messages: list[ChatMessage] = messages or []

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatHistory":
        """Create from dictionary."""
        history = cls(
            conversation_id=data.get("conversation_id"),
            messages=[ChatMessage.from_dict(m) for m in data.get("messages", [])],
        )
        history.created_at = data.get("created_at", history.created_at)
        history.updated_at = data.get("updated_at", history.updated_at)
        history.status = data.get("status", "idle")
        return history

    def to_llm_messages(self) -> list[Message]:
        """Convert to LLM message format for context."""
        llm_messages: list[Message] = []

        for msg in self.messages:
            if msg.role == "user":
                llm_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                assistant_msg: Message = {
                    "role": "assistant",
                    "content": msg.content,
                }
                if msg.tool_calls:
                    # Convert tool calls to LLM format
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                        for tc in msg.tool_calls
                    ]
                llm_messages.append(assistant_msg)
            elif msg.role == "tool_result":
                llm_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id or "",
                    }
                )

        return llm_messages


class ChatTask:
    """
    Task for multi-turn chat conversations with the AI agent.

    Supports conversation persistence, streaming responses, and
    transparent tool execution display.
    """

    def __init__(
        self,
        stash: "StashClient",
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig | None = None,
        excluded_tags: list[str] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """
        Initialize the chat task.

        Args:
            stash: StashClient instance for API calls
            llm_config: LLM configuration
            embedding_config: Optional config for text-based scene search
            excluded_tags: Optional list of tag names to exclude from tool results
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
        """
        self.stash = stash
        self.llm_config = llm_config
        self.embedding_config = embedding_config
        self.excluded_tags = excluded_tags or []
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        # Initialize LLM provider
        self.llm = get_provider(llm_config)

        # Initialize tools (includes SearchByTextTool if embedding_config provided)
        self.tools = get_all_tools(stash, embedding_config, excluded_tags)
        self.tool_map = {t.name: t for t in self.tools}
        self.tool_schemas = [t.to_schema() for t in self.tools]

        # Setup assets directory
        self.assets_dir = self._get_assets_dir()
        self.history_file = os.path.join(self.assets_dir, "chat_history.json")

    def _get_assets_dir(self) -> str:
        """Get the plugin assets directory."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        return assets_dir

    def run(self, message: str, conversation_id: str | None = None) -> str:
        """
        Process a chat message.

        Args:
            message: The user's message
            conversation_id: Optional existing conversation ID to continue

        Returns:
            The assistant's response
        """
        self.log(f"Chat message received: {message[:100]}...", "info")
        self.progress(0, 3)

        # Load or create conversation history
        history = self._load_history(conversation_id)
        self.log(f"Conversation ID: {history.conversation_id}", "debug")

        # Add user message
        user_msg = ChatMessage(role="user", content=message)
        history.add_message(user_msg)
        history.status = "streaming"
        self._save_history(history)
        self.progress(1, 3)

        # Load system prompt from YAML (hot-reloaded)
        try:
            system_prompt = get_prompt("chat", "system", "system")
        except (FileNotFoundError, KeyError):
            system_prompt = CHAT_SYSTEM_PROMPT

        # Build messages for LLM (system + history)
        llm_messages: list[Message] = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(history.to_llm_messages())

        self.log(f"Sending {len(llm_messages)} messages to LLM", "debug")
        self.log(f"Available tools: {list(self.tool_map.keys())}", "debug")

        try:
            # Run the agent loop with tool callbacks
            response = self._run_agent_loop(history, llm_messages)

            # Add final assistant message
            assistant_msg = ChatMessage(role="assistant", content=response)
            history.add_message(assistant_msg)
            history.status = "complete"

        except Exception as e:
            self.log(f"Chat error: {e}", "error")
            history.status = "error"
            response = f"I encountered an error: {e!s}"

        self.progress(3, 3)
        history.updated_at = datetime.now().isoformat()
        self._save_history(history)

        self.log("Chat response complete", "info")
        return response

    def _run_agent_loop(
        self,
        history: ChatHistory,
        messages: list[Message],
        max_iterations: int = 5,
    ) -> str:
        """
        Run the agent loop with tool execution and history updates.

        Args:
            history: Chat history to update with tool calls
            messages: Current LLM messages
            max_iterations: Maximum tool iterations

        Returns:
            Final response from the LLM
        """
        # Check if LLM supports tools
        if not self.llm.supports_tools:
            self.log(f"Model {self.llm_config.model} doesn't support tools", "warning")
            result = self.llm.chat(messages=messages, tools=None, temperature=0.7)
            return result["content"] or ""

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            self.log(f"Agent iteration {iteration}/{max_iterations}", "debug")

            # Get LLM response with tools
            result = self.llm.chat(
                messages=messages,
                tools=self.tool_schemas if self.tools else None,
                temperature=0.7,
            )

            # If no tool calls, we have the final response
            if not result["tool_calls"]:
                return result["content"] or ""

            # Process tool calls
            self.log(f"LLM requested {len(result['tool_calls'])} tool call(s)", "info")

            # Create assistant message with tool calls for history
            tool_calls_info = []
            for tc in result["tool_calls"]:
                tool_calls_info.append(
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "status": "pending",
                    }
                )

            # Add assistant message with pending tool calls
            assistant_msg = ChatMessage(
                role="assistant",
                content=result["content"] or "",
                tool_calls=tool_calls_info,
            )
            history.add_message(assistant_msg)
            history.status = "tool_executing"
            self._save_history(history)

            # Add to LLM messages
            llm_assistant_msg: Message = {
                "role": "assistant",
                "content": result["content"],
                "tool_calls": result["tool_calls"],
            }
            messages.append(llm_assistant_msg)

            # Execute each tool and add results
            for i, tool_call in enumerate(result["tool_calls"]):
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = tool_call["id"]

                # Update tool status to executing
                tool_calls_info[i]["status"] = "executing"
                tool_calls_info[i]["started_at"] = datetime.now().isoformat()
                self._save_history(history)

                self.log(f"Executing tool: {tool_name}({tool_args})", "info")

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)

                # Update tool status
                tool_calls_info[i]["status"] = (
                    "completed" if tool_result.get("success") else "failed"
                )
                tool_calls_info[i]["completed_at"] = datetime.now().isoformat()
                tool_calls_info[i]["result"] = {
                    "success": tool_result.get("success", False),
                    "data": self._truncate_data(tool_result.get("data")),
                    "error": tool_result.get("error"),
                }

                # Add tool result message to history
                tool_result_msg = ChatMessage(
                    role="tool_result",
                    content=json.dumps(tool_result.get("data", {}), indent=2)[:1000],
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_success=tool_result.get("success", False),
                )
                history.add_message(tool_result_msg)
                history.status = "streaming"
                self._save_history(history)

                # Add tool result to LLM messages
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
        self.log(f"Agent reached max iterations ({max_iterations})", "warning")
        return "I've reached the maximum number of steps. Here's what I found based on the tool results."

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

    def _truncate_data(self, data: Any, max_items: int = 5) -> Any:
        """Truncate large data structures for display.

        Preserves lists of dicts (scene results with metadata needed by UI
        for card rendering) while truncating lists of primitives (scene_ids,
        tag lists, etc.) to keep stored history compact.
        """
        if data is None:
            return None
        if isinstance(data, list) and len(data) > max_items:
            # Preserve lists of dicts — these are scene results the UI needs
            # for card metadata (similarity scores, timestamps, frame paths).
            # Only truncate lists of primitives (ints, strings).
            if not data or not isinstance(data[0], dict):
                return data[:max_items] + [f"... and {len(data) - max_items} more"]
        if isinstance(data, dict):
            return {k: self._truncate_data(v, max_items) for k, v in data.items()}
        return data

    def _load_history(self, conversation_id: str | None = None) -> ChatHistory:
        """
        Load conversation history from file or create new.

        Args:
            conversation_id: Optional conversation ID to load

        Returns:
            ChatHistory instance
        """
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file) as f:
                    data = json.load(f)

                # If a specific conversation is requested and matches, use it
                if conversation_id and data.get("conversation_id") == conversation_id:
                    self.log(f"Loaded existing conversation: {conversation_id}", "debug")
                    return ChatHistory.from_dict(data)

                # If no specific ID requested, continue existing conversation
                if not conversation_id:
                    self.log(f"Continuing conversation: {data.get('conversation_id')}", "debug")
                    return ChatHistory.from_dict(data)

        except (OSError, json.JSONDecodeError) as e:
            self.log(f"Could not load history: {e}", "warning")

        # Create new conversation
        self.log("Starting new conversation", "debug")
        return ChatHistory()

    def _save_history(self, history: ChatHistory) -> None:
        """
        Save conversation history to file.

        Args:
            history: ChatHistory to save
        """
        try:
            with open(self.history_file, "w") as f:
                json.dump(history.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            self.log(f"History saved ({len(history.messages)} messages)", "debug")
        except Exception as e:
            self.log(f"Warning: Could not save history: {e}", "warning")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
                self.log("Conversation history cleared", "info")
        except Exception as e:
            self.log(f"Warning: Could not clear history: {e}", "warning")
