#!/usr/bin/env python3
"""
StashApp Python Plugin Boilerplate

This plugin demonstrates how to create a Python-based plugin for StashApp.
It can run as a task or be triggered by hooks.

Extended with Stash AI features for LLM-powered library insights.
"""

import json
import os
import sys
from collections.abc import Callable
from typing import Any, cast

from stashapi import log as stash_log
from stashapi.stashapp import StashInterface

# Add plugin directory to path for imports
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

# Import cleanup module early to register signal handlers for graceful shutdown
# This ensures GPU resources are freed when Stash cancels a task (SIGTERM)
try:
    from stash_ai.embeddings import base as _embeddings_base  # noqa: F401
except ImportError:
    pass  # Module may not be available in all contexts

from stash_ai.stash_client import StashApiClient, StashClient  # noqa: E402
from stash_ai.tasks.dispatch import TaskContext, dispatch  # noqa: E402
from stash_ai.tasks.result_store import ResultStore  # noqa: E402


class StashPlugin:
    """Base class for StashApp Python plugins."""

    def __init__(
        self,
        stash_client: StashClient | None = None,
        input_override: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the plugin with Stash connection details.

        Args:
            stash_client: Pre-built Stash client to inject. When provided (local
                dev / tests), stdin reading and the live StashInterface connection
                are skipped — the plugin runs entirely against the injected client.
            input_override: Task/hook input to use instead of reading stdin. Only
                consulted when ``stash_client`` is injected.
        """
        self.stash_url = "http://localhost:9999"
        self.input: dict[str, Any] | None = None
        self.stash: StashClient | None = None

        # Injection path: skip stdin + real connection entirely.
        if stash_client is not None:
            self.stash = stash_client
            self.input = input_override
            return

        # Read input from stdin (provided by Stash)
        self.input = self._read_input()

        # Set connection details from input
        if self.input:
            server_config = self.input.get("server_connection", {})
            self.log(f"server_connection: {server_config}", "debug")
            scheme = server_config.get("Scheme", "http")
            host = server_config.get("Host", "localhost")
            port = server_config.get("Port", 9999)
            stash_dir = server_config.get("Dir")
            if stash_dir:
                os.environ["STASH_CONFIG_DIR"] = stash_dir

            # 0.0.0.0 means "all interfaces" - use 127.0.0.1 for local connections
            if host == "0.0.0.0":
                host = "127.0.0.1"

            self.stash_url = f"{scheme}://{host}:{port}"

            # Initialize StashInterface - pass entire server_connection fragment
            # StashInterface handles authentication automatically. Wrap it in the
            # StashApiClient seam so tasks depend on the StashClient Protocol, not
            # the untyped third-party interface (enables fakes for local dev/tests).
            self.stash = StashApiClient(StashInterface(server_config))

    @property
    def stash_client(self) -> StashClient:
        """The connected Stash client, guaranteed non-None.

        Tasks require a live connection; this raises rather than letting a None
        propagate into a task constructor where the failure would be opaque.
        """
        if self.stash is None:
            raise RuntimeError("Stash connection not initialized")
        return self.stash

    def _read_input(self) -> dict[str, Any] | None:
        """Read and parse JSON input from stdin."""
        try:
            input_str = sys.stdin.read()
            if input_str:
                self.log("Input received", "debug")
                return cast("dict[str, Any]", json.loads(input_str))
        except json.JSONDecodeError as e:
            self.error(f"Failed to parse input JSON: {e}")
            sys.exit(1)
        return None

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message to Stash using proper protocol format.

        Args:
            message: The message to log
            level: Log level (trace, debug, info, warning, error)
        """
        level_map = {
            "trace": stash_log.trace,
            "debug": stash_log.debug,
            "info": stash_log.info,
            "warning": stash_log.warning,
            "error": stash_log.error,
        }
        log_fn = level_map.get(level, stash_log.info)
        log_fn(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "error")

    def progress(self, current: int, total: int) -> None:
        """
        Report progress to Stash (0.0-1.0 range).

        Args:
            current: Current progress value
            total: Total progress value
        """
        value = current / total if total > 0 else 0
        stash_log.progress(value)

    def call_gql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Execute a GraphQL query against Stash API.

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            Response data or None if request failed
        """
        if not self.stash:
            self.error("StashInterface not initialized")
            return None

        try:
            return self.stash.call_GQL(query, variables)
        except Exception as e:
            self.error(f"GraphQL request failed: {e}")
            return None

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        """
        Fetch a scene by ID.

        Args:
            scene_id: The scene ID

        Returns:
            Scene data or None
        """
        query = """
            query FindScene($id: ID!) {
                findScene(id: $id) {
                    id
                    title
                    date
                    rating100
                    play_count
                    o_counter
                    organized
                    files {
                        path
                        size
                        duration
                    }
                    performers {
                        id
                        name
                    }
                    tags {
                        id
                        name
                    }
                }
            }
        """

        data = self.call_gql(query, {"id": scene_id})
        return data.get("findScene") if data else None

    def update_scene(self, scene_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a scene with new data.

        Args:
            scene_id: The scene ID
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        mutation = """
            mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) {
                    id
                }
            }
        """

        input_data = {"id": scene_id, **updates}

        data = self.call_gql(mutation, {"input": input_data})
        return data is not None and "sceneUpdate" in data

    def find_scenes(self, filter_params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Find scenes with optional filters.

        Args:
            filter_params: Optional filter parameters

        Returns:
            List of scenes
        """
        query = """
            query FindScenes($filter: FindFilterType) {
                findScenes(filter: $filter) {
                    count
                    scenes {
                        id
                        title
                        path
                        rating100
                        play_count
                    }
                }
            }
        """

        data = self.call_gql(query, {"filter": filter_params or {}})
        if data and "findScenes" in data:
            return cast("list[dict[str, Any]]", data["findScenes"]["scenes"])
        return []

    def get_plugin_settings(self, plugin_id: str) -> dict[str, Any]:
        """
        Fetch plugin settings from Stash via StashInterface.

        Args:
            plugin_id: The plugin ID (from the yml file name without extension)

        Returns:
            Dictionary of plugin settings
        """
        if not self.stash:
            return {}

        try:
            return self.stash.find_plugin_config(plugin_id)
        except Exception as e:
            self.log(f"Error fetching plugin settings: {e}", "error")
            return {}


class MyPlugin(StashPlugin):
    """Custom plugin implementation."""

    def process_scene(self, scene_id: str) -> None:
        """
        Process a single scene.

        Args:
            scene_id: The scene ID to process
        """
        self.log(f"Processing scene: {scene_id}")

        scene = self.get_scene(scene_id)
        if not scene:
            self.error(f"Scene not found: {scene_id}")
            return

        self.log(f"Scene title: {scene.get('title', 'Untitled')}")

        # Example: Update scene rating based on play count
        play_count = scene.get("play_count", 0)
        if play_count > 10:
            new_rating = min(100, 50 + (play_count * 5))
            updates = {"rating100": new_rating}

            if self.update_scene(scene_id, updates):
                self.log(f"Updated rating to {new_rating}")
            else:
                self.error("Failed to update scene")

    def process_all_scenes(self) -> None:
        """Process all scenes in the library."""
        self.log("Processing all scenes...")

        scenes = self.find_scenes()
        total = len(scenes)

        self.log(f"Found {total} scenes")

        for i, scene in enumerate(scenes, 1):
            self.progress(i, total)
            self.process_scene(scene["id"])

        self.log("Processing complete")

    def run_task(self, task_name: str, args: dict[str, Any]) -> None:
        """
        Run a specific task.

        Args:
            task_name: Name of the task to run
            args: Task arguments
        """
        self.log(f"Running task: {task_name}")

        handler = self._task_handlers().get(task_name)
        if handler is None:
            self.error(f"Unknown task: {task_name}")
            return
        handler(args)

    def _task_handlers(self) -> dict[str, Callable[[dict[str, Any]], None]]:
        """Map task names to their handlers.

        Single source of truth for dispatch, shared by the Stash entry point and
        the ``scripts/run_task.py`` local harness. Handlers are normalised to a
        ``(args) -> None`` shape; the few tasks needing a different call shape are
        adapted with thin wrappers.
        """
        handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "process_all": lambda args: self.process_all_scenes(),
            "process_scene": self._handle_process_scene,
            "stats_summary": self.run_stats_summary,
            "ask": self.run_ask,
            "chat": self.run_chat,
            "clear_chat": self.run_clear_chat,
            "scene_vision": self.run_scene_vision,
            "embed_scenes": self.run_embed_scenes,
            "find_similar": self.run_find_similar,
            "find_similar_by_frame": self.run_find_similar_by_frame,
            "frame_analysis": self.run_frame_analysis,
            "check_frame_analysis": self.check_frame_analysis,
            "run_frame_analysis": self.start_frame_analysis,
            "recommendations": self.run_recommendations,
            "search_by_text": self.run_search_by_text,
            "get_embedding_models": self.run_get_embedding_models,
            "embed_o_moments": self.run_embed_o_moments,
            "embed_cached_frames": self.run_embed_cached_frames,
            "build_frame_index": self.run_build_frame_index,
            "cleanup_orphaned": self.run_cleanup_orphaned,
            "embed_performers": self.run_embed_performers,
            "describe_performers": self.run_describe_performers,
            "find_similar_performers": self.run_find_similar_performers,
            "build_taste_map": self.run_build_taste_map,
            "detect_tag_gaps": self.run_detect_tag_gaps,
            "get_scene_tag_gaps": self.run_get_scene_tag_gaps,
            "preview_tag_impact": self.run_preview_tag_impact,
            "get_tag_suggestions": self.run_get_tag_suggestions,
            "apply_suggested_tag": self.run_apply_suggested_tag,
            "dismiss_suggested_tag": self.run_dismiss_suggested_tag,
            "clear_dismissed_tags": self.run_clear_dismissed_tags,
            "find_duplicate_tags": self.run_find_duplicate_tags,
            "merge_tags": self.run_merge_tags,
            "dismiss_tag_merge": self.run_dismiss_tag_merge,
            "prepare_labeling_session": self.run_prepare_labeling_session,
            "sync_labeling_annotations": self.run_sync_labeling_annotations,
            "export_labeling_dataset": self.run_export_labeling_dataset,
            "get_labeling_sessions": self.run_get_labeling_sessions,
            "eroscripts_validate_auth": self.run_eroscripts_validate_auth,
            "eroscripts_search": self.run_eroscripts_search,
            "eroscripts_download": self.run_eroscripts_download,
            "eroscripts_status": self.run_eroscripts_status,
        }
        return handlers

    def available_tasks(self) -> list[str]:
        """Sorted task names this plugin can dispatch (registry as source of truth)."""
        return sorted(self._task_handlers())

    def _handle_process_scene(self, args: dict[str, Any]) -> None:
        """Dispatch wrapper for ``process_scene``; requires a ``scene_id`` arg."""
        scene_id = args.get("scene_id")
        if scene_id:
            self.process_scene(scene_id)
        else:
            self.error("scene_id argument required")

    def _dispatch(
        self,
        args: dict[str, Any],
        build_task: Callable[[TaskContext], Any],
        *,
        on_result: Callable[[Any, Any], None] | None = None,
    ) -> None:
        """Run ``build_task``'s task through the generic dispatch seam.

        Resolves the plugin settings once, wires the standard
        log/progress/stash dependencies into a :class:`TaskContext`, and
        delegates construction, execution, and uniform error handling to
        :func:`dispatch`. This is the single place per-task handlers get their
        context, replacing the hand-rolled settings fetch + four-clause
        ``try/except`` each used to repeat.
        """

        def build_context() -> TaskContext:
            return TaskContext(
                stash=self.stash_client,
                log=self.log,
                progress=self.progress,
                plugin_settings=self.get_plugin_settings("stash-copilot"),
                args=args,
                request_id=str(args.get("request_id", "")),
            )

        dispatch(
            log=self.log,
            build_context=build_context,
            build_task=build_task,
            on_result=on_result,
        )

    def _result_store(self) -> ResultStore:
        """Return the result store rooted at the plugin's ``assets`` directory.

        The single place handlers persist their frontend-polled result file
        (``assets/{result_key}_{request_id}.json``), replacing the hand-rolled
        ``os.makedirs`` + ``json.dump`` each used to repeat.
        """
        return ResultStore(os.path.join(PLUGIN_DIR, "assets"))

    def run_eroscripts_validate_auth(self, args: dict[str, Any]) -> None:
        """Validate (or clear/re-check) the EroScripts session cookie."""
        try:
            from stash_ai.tasks import eroscripts_auth as task_module

            task_module.run(args, self.log)
        except Exception as e:
            self.error(f"eroscripts_validate_auth failed: {e}")

    def run_eroscripts_search(self, args: dict[str, Any]) -> None:
        """Search discuss.eroscripts.com for funscripts matching a Stash scene."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_search as task_module

            task_module.run(self.stash, args, self.log)
        except Exception as e:
            self.error(f"eroscripts_search failed: {e}")

    def run_eroscripts_download(self, args: dict[str, Any]) -> None:
        """List attachments for an eroscripts topic, or download one and persist."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_download as task_module

            task_module.run(self.stash, args, self.log)
        except Exception as e:
            self.error(f"eroscripts_download failed: {e}")

    def run_eroscripts_status(self, args: dict[str, Any]) -> None:
        """Report whether a scene has a matched funscript + sidecar."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_status as task_module

            task_module.run(self.stash, args, self.log)
        except Exception as e:
            self.error(f"eroscripts_status failed: {e}")

    def run_stats_summary(self, args: dict[str, Any]) -> None:
        """
        Run the AI-powered library statistics summary task.

        Pilot for the dispatch seam (#4): the task now builds itself from the
        :class:`TaskContext` via ``StatsSummaryTask.from_context`` (commit 3),
        and :func:`dispatch` owns execution + uniform error handling; this
        handler only points at the construction hook and declares how to surface
        the task's output.

        Args:
            args: Task arguments containing LLM settings
        """

        def build_task(ctx: TaskContext) -> Any:
            # Import inside the dispatch boundary so an import failure is logged
            # uniformly; construction itself is the task's self-describing hook.
            from stash_ai.tasks.stats_summary import StatsSummaryTask

            return StatsSummaryTask.from_context(ctx)

        def on_result(_task: Any, summary: Any) -> None:
            # Output the summary
            self.log("=" * 50, "info")
            self.log("LIBRARY STATISTICS SUMMARY", "info")
            self.log("=" * 50, "info")
            for line in str(summary).split("\n"):
                self.log(line, "info")
            self.log("=" * 50, "info")

        self._dispatch(args, build_task, on_result=on_result)

    def run_recommendations(self, args: dict[str, Any]) -> None:
        """Run personalized recommendations through the dispatch seam (#4, commit 4).

        Result-producing: ``RecommendationsTask.from_context`` resolves the ~16
        run parameters + model_key from the ``TaskContext``, ``run()`` writes its
        own ``recommendations_{request_id}.json`` (declares
        ``result_key="recommendations"``), ``on_result`` logs the summary, and
        ``dispatch`` owns uniform error handling.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.recommendations import RecommendationsTask

            return RecommendationsTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            self.log("=" * 50, "info")
            self.log("RECOMMENDATIONS", "info")
            self.log("=" * 50, "info")
            self.log(f"Mode: {result['mode']}", "info")
            self.log(f"Scoring: {result['scoring_method']}", "info")
            self.log(
                f"Profile: {result['profile'].get('scene_count', 0)} scenes "
                f"(engagement: {result['profile'].get('total_engagement_score', 0):.1f})",
                "info",
            )
            self.log(f"Results: {len(result['results'])} recommendations", "info")

            for i, rec in enumerate(result["results"][:5], 1):
                scene = rec.get("scene", {})
                title = scene.get("title") or f"Scene {rec['scene_id']}"
                self.log(
                    f"  {i}. {title} (sim={rec['similarity_score']:.3f})",
                    "info",
                )

            if len(result["results"]) > 5:
                self.log(f"  ... and {len(result['results']) - 5} more", "info")

            self.log("=" * 50, "info")

        self._dispatch(args, build_task, on_result=on_result)

    def run_build_taste_map(self, args: dict[str, Any]) -> None:
        """Run the Build Taste Map task through the dispatch seam (#4, commit 4).

        ``TasteMapTask`` is self-describing: it resolves its model_key and run
        parameters from the :class:`TaskContext` in ``from_context``, writes its
        own result file (declaring ``result_key``), and :func:`dispatch` owns
        execution + uniform error handling. This handler only points at the
        construction hook and logs the outcome.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.taste_map import TasteMapTask

            return TasteMapTask.from_context(ctx)

        def on_result(_task: Any, response: Any) -> None:
            if not response:
                return
            if response["status"] == "complete":
                self.log(
                    f"Taste map complete: {response['optimal_k']} clusters, "
                    f"{len(response['scenes'])} scenes",
                    "info",
                )
            else:
                self.log(
                    f"Taste map failed: {response.get('error', 'Unknown error')}",
                    "error",
                )

        self._dispatch(args, build_task, on_result=on_result)

    def run_detect_tag_gaps(self, args: dict[str, Any]) -> None:
        """Run tag gap detection through the dispatch seam (#4, commit 4).

        Result-producing: ``TagGapDetectionTask.from_context`` resolves the
        model_key + request_id/force, ``run()`` writes its own ``tag_gaps_*``
        files (declares ``result_key="tag_gaps"``), ``on_result`` logs the
        complete/failed summary, and ``dispatch`` owns uniform error handling.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            return TagGapDetectionTask.from_context(ctx)

        def on_result(_task: Any, report: Any) -> None:
            if report["status"] == "complete":
                self.log(
                    f"Tag gap detection complete: {report['avg_coverage']:.0%} avg coverage, "
                    f"{report['flagged_scenes']} scenes flagged",
                    "info",
                )
            else:
                self.log(f"Tag gap detection failed: {report.get('error', 'Unknown')}", "error")

        self._dispatch(args, build_task, on_result=on_result)

    def run_get_scene_tag_gaps(self, args: dict[str, Any]) -> None:
        """Get tag-gap detail for a scene (sidebar query), via the dispatch seam (#4).

        Result-producing: ``SceneTagGapsTask`` wraps
        ``TagGapDetectionTask.get_scene_gaps_detail``; ``on_result`` persists the
        result through the seam's ResultStore as ``tag_gaps_scene_{request_id}.
        json`` (request_id defaults to ``scene_{scene_id}``, matching the old
        handler); ``dispatch`` owns uniform error handling. The missing-scene_id
        guard stays here (before the seam).
        """
        scene_id = args.get("scene_id")
        if not scene_id:
            self.error("scene_id argument required")
            return

        # Preserve the old filename: request_id defaults to "scene_{scene_id}",
        # so ResultStore writes tag_gaps_scene_{request_id}.json.
        request_id = args.get("request_id", f"scene_{scene_id}")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_gap_detection import SceneTagGapsTask

            return SceneTagGapsTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            self._result_store().save(task.result_key, request_id, result)

        self._dispatch(args, build_task, on_result=on_result)

    def run_preview_tag_impact(self, args: dict[str, Any]) -> None:
        """Preview a hypothetical tag's coverage impact, via the dispatch seam (#4).

        Result-producing: ``PreviewTagImpactTask`` wraps
        ``TagGapDetectionTask.preview_tag_impact``; ``on_result`` persists the
        result through the seam's ResultStore as ``tag_preview_{request_id}.json``
        (request_id defaults to ``preview_{scene_id}_{tag_name}``, matching the old
        handler); ``dispatch`` owns uniform error handling. The missing-arg guard
        stays here (before the seam).
        """
        scene_id = args.get("scene_id")
        tag_name = args.get("tag_name")
        if not scene_id or not tag_name:
            self.error("scene_id and tag_name arguments required")
            return

        request_id = args.get("request_id", f"preview_{scene_id}_{tag_name}")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_gap_detection import PreviewTagImpactTask

            return PreviewTagImpactTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            self._result_store().save(task.result_key, request_id, result)

        self._dispatch(args, build_task, on_result=on_result)

    def run_get_tag_suggestions(self, args: dict[str, Any]) -> None:
        """Get embedding-based tag suggestions for a scene, via the dispatch seam (#4).

        Result-producing: ``TagSuggestionsTask.from_context`` resolves the
        model_key + scene_id; ``on_result`` persists the result through the seam's
        ResultStore (``tag_suggestions_{request_id}.json``, keyed by the task's
        ``result_key``) and logs the summary; ``dispatch`` owns uniform error
        handling. The missing-scene_id guard stays here (before the seam).
        """
        scene_id = args.get("scene_id")
        if not scene_id:
            self.log("Missing scene_id", "error")
            return

        request_id = args.get("request_id", "")
        self.log(
            f"Computing tag suggestions for scene {int(scene_id)}, request_id={request_id}", "info"
        )
        self.log(f"Args received: {list(args.keys())}", "debug")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

            return TagSuggestionsTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            self._result_store().save(task.result_key, request_id, result)
            if result["status"] == "complete":
                self.log(f"Found {len(result['suggestions'])} tag suggestions", "info")
            else:
                self.log(f"Tag suggestions: {result['error']}", "warning")

        self._dispatch(args, build_task, on_result=on_result)

    def run_apply_suggested_tag(self, args: dict[str, Any]) -> None:
        """Apply a suggested tag to a scene, through the dispatch seam (#4, commit 4).

        Log-only side effect: ``ApplySuggestedTagTask`` reads the scene's tags,
        adds the suggested one if missing (idempotent), and writes back via
        ``sceneUpdate``; the missing-arg guard and idempotency check live in its
        ``run()``, and ``dispatch`` owns uniform error handling.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_suggestion_actions import ApplySuggestedTagTask

            return ApplySuggestedTagTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_dismiss_suggested_tag(self, args: dict[str, Any]) -> None:
        """Dismiss a tag suggestion for a scene, through the dispatch seam (#4, commit 4).

        Log-only side effect: ``DismissSuggestedTagTask`` writes the dismissal to
        EmbeddingStorage and logs the outcome (no result_key); the missing-arg
        guard lives in its ``run()``, and ``dispatch`` owns uniform error handling.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_suggestion_actions import DismissSuggestedTagTask

            return DismissSuggestedTagTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_clear_dismissed_tags(self, args: dict[str, Any]) -> None:
        """Clear all dismissed tags for a scene, through the dispatch seam (#4, commit 4).

        Log-only side effect: ``ClearDismissedTagsTask`` clears the scene's
        dismissed-tag rows in EmbeddingStorage and logs the count; the missing-arg
        guard lives in its ``run()``, and ``dispatch`` owns uniform error handling.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_suggestion_actions import ClearDismissedTagsTask

            return ClearDismissedTagsTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_find_duplicate_tags(self, args: dict[str, Any]) -> None:
        """Find duplicate tags through the dispatch seam (#4, commit 4).

        Result-producing: ``FindDuplicateTagsTask.from_context`` resolves the
        model_key + storage; ``on_result`` persists the result through the seam's
        ResultStore (keyed by the task's ``result_key``) and logs the summary;
        ``dispatch`` owns uniform error handling.
        """
        request_id = args.get("request_id", "")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_dedup import FindDuplicateTagsTask

            return FindDuplicateTagsTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            # Persist for frontend polling via the dispatch seam's result store.
            self._result_store().save(task.result_key, request_id, result)
            if result["status"] == "complete":
                self.log(f"Found {len(result['candidates'])} duplicate tag candidates", "info")
            else:
                self.log(f"Tag dedup: {result.get('error', 'unknown error')}", "warning")

        self._dispatch(args, build_task, on_result=on_result)

    def run_merge_tags(self, args: dict[str, Any]) -> None:
        """Merge one tag into another, through the dispatch seam (#4, commit 4).

        Result-producing: ``MergeTagsTask.from_context`` caches the keep/remove
        tag ids; ``on_result`` persists the result through the seam's ResultStore
        (``tag_merge_{request_id}.json``, keyed by the task's ``result_key``) and
        logs the summary; ``dispatch`` owns uniform error handling. The missing-id
        guard stays here (before the seam) so it's a clean no-op.
        """
        keep_tag_id = int(args.get("keep_tag_id", 0))
        remove_tag_id = int(args.get("remove_tag_id", 0))
        request_id = args.get("request_id", "")

        if not keep_tag_id or not remove_tag_id:
            self.log("Missing keep_tag_id or remove_tag_id", "error")
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_dedup import MergeTagsTask

            return MergeTagsTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            self._result_store().save(task.result_key, request_id, result)
            if result["status"] == "complete":
                self.log(f"Merged tags: {result['scenes_updated']} scenes updated", "info")
            else:
                self.log(f"Tag merge error: {result.get('error')}", "warning")

        self._dispatch(args, build_task, on_result=on_result)

    def run_dismiss_tag_merge(self, args: dict[str, Any]) -> None:
        """Dismiss a tag-merge candidate through the dispatch seam (#4, commit 4).

        Side-effecting: ``DismissTagMergeTask`` records the dismissal in
        EmbeddingStorage; ``on_result`` persists the ``{"status": "complete"}``
        confirmation through the seam's ResultStore (``tag_dismiss_{request_id}.
        json``); ``dispatch`` owns uniform error handling. The missing-name guard
        stays here (before the seam) so it's a clean no-op.
        """
        tag_a_name = args.get("tag_a_name", "")
        tag_b_name = args.get("tag_b_name", "")
        request_id = args.get("request_id", "")

        if not tag_a_name or not tag_b_name:
            self.log("Missing tag_a_name or tag_b_name", "error")
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.tag_dedup import DismissTagMergeTask

            return DismissTagMergeTask.from_context(ctx)

        def on_result(task: Any, result: Any) -> None:
            self._result_store().save(task.result_key, request_id, result)

        self._dispatch(args, build_task, on_result=on_result)

    def run_prepare_labeling_session(self, args: dict[str, Any]) -> None:
        """Prepare a labeling session with uncertainty-sampled frames."""
        request_id = args.get("request_id", "")
        batch_size = int(args.get("batch_size", 200))

        self.log(
            f"Preparing labeling session (batch_size={batch_size}), request_id={request_id}", "info"
        )

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.embeddings.tag_vocabulary import TagVocabulary
            from stash_ai.tasks.labeling import LabelingTask
            from stash_ai.tasks.labeling_types import LabelingConfig

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Determine model key
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if image_provider and image_model:
                embedding_config = EmbeddingConfig(
                    provider=image_provider, model=image_model, device=image_device
                )
                model_key = embedding_config.model_key
            else:
                model_key = "siglip"

            storage = EmbeddingStorage(model_key=model_key)

            # Sync tag vocabulary before preparing session
            self.log("Syncing tag vocabulary...", "info")
            tag_vocab = TagVocabulary(storage=storage, model_key=model_key, log_callback=self.log)
            stash_tags = [t["name"] for t in self.stash_client.find_tags(f={})]
            tag_vocab.ensure_embeddings(stash_tags=stash_tags)

            # Build labeling config
            config = LabelingConfig(
                batch_size=batch_size,
                uncertainty_low=float(plugin_settings.get("label_uncertainty_low", 0.25)),
                uncertainty_high=float(plugin_settings.get("label_uncertainty_high", 0.35)),
                max_suggested_tags=int(plugin_settings.get("label_suggested_tags", 10)),
                caption_template=plugin_settings.get(
                    "label_caption_template", "a scene featuring {tags}"
                ),
            )

            task = LabelingTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
                model_key=model_key,
            )

            result = task.prepare_session(config)

            # Write result JSON
            import json
            from pathlib import Path

            assets_dir = Path(__file__).parent / "assets"
            assets_dir.mkdir(exist_ok=True)
            result_file = assets_dir / f"labeling_session_{request_id}.json"
            result_file.write_text(json.dumps(result, indent=2))

            self.log(f"Labeling session written to {result_file}", "info")

        except Exception as e:
            self.log(f"Error preparing labeling session: {e}", "error")
            import json
            from pathlib import Path

            assets_dir = Path(__file__).parent / "assets"
            result_file = assets_dir / f"labeling_session_{request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "session_id": "",
                        "batch": [],
                        "vocabulary": [],
                        "error": str(e),
                    }
                )
            )

    def run_sync_labeling_annotations(self, args: dict[str, Any]) -> None:
        """Sync annotations from the labeling UI."""
        request_id = args.get("request_id", "")
        payload_json = args.get("payload", "{}")

        try:
            import json
            from pathlib import Path

            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tasks.labeling import LabelingTask

            payload = json.loads(payload_json)
            storage = EmbeddingStorage()

            task = LabelingTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
            )

            task.sync_annotations(payload)

            assets_dir = Path(__file__).parent / "assets"
            result_file = assets_dir / f"labeling_sync_{request_id}.json"
            result_file.write_text(json.dumps({"status": "complete"}))

        except Exception as e:
            self.log(f"Error syncing annotations: {e}", "error")

    def run_export_labeling_dataset(self, args: dict[str, Any]) -> None:
        """Export labeled data as WebDataset."""
        request_id = args.get("request_id", "")
        include_negatives = args.get("include_negatives", "true").lower() == "true"

        self.log(f"Exporting labeling dataset, request_id={request_id}", "info")

        try:
            import json
            from pathlib import Path

            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tasks.labeling import LabelingTask
            from stash_ai.tasks.labeling_types import LabelingConfig

            plugin_settings = self.get_plugin_settings("stash-copilot")
            storage = EmbeddingStorage()
            config = LabelingConfig.from_plugin_settings(plugin_settings)

            task = LabelingTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
            )

            result = task.export_dataset(config, include_negatives=include_negatives)

            assets_dir = Path(__file__).parent / "assets"
            result_file = assets_dir / f"labeling_export_{request_id}.json"
            result_file.write_text(json.dumps(result, indent=2))

            self.log(f"Export result written to {result_file}", "info")

        except Exception as e:
            self.log(f"Error exporting dataset: {e}", "error")
            import json
            from pathlib import Path

            assets_dir = Path(__file__).parent / "assets"
            result_file = assets_dir / f"labeling_export_{request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "export_path": "",
                        "total_images": 0,
                        "total_tags": 0,
                        "error": str(e),
                    }
                )
            )

    def run_get_labeling_sessions(self, args: dict[str, Any]) -> None:
        """List labeling sessions."""
        request_id = args.get("request_id", "")

        try:
            import json
            from pathlib import Path

            from stash_ai.embeddings.storage import EmbeddingStorage

            storage = EmbeddingStorage()
            sessions = storage.list_labeling_sessions()

            assets_dir = Path(__file__).parent / "assets"
            result_file = assets_dir / f"labeling_sessions_{request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "sessions": sessions,
                    },
                    indent=2,
                )
            )

        except Exception as e:
            self.log(f"Error listing sessions: {e}", "error")

    def run_ask(self, args: dict[str, Any]) -> None:
        """Run the AI Ask task through the dispatch seam (#4, commit 4).

        Result-producing: ``AskTask`` writes its own fixed-name
        ``assets/last_ask.json`` (declares ``result_key="last_ask"``);
        ``from_context`` resolves the text-LLM config + question, ``on_result``
        logs the answer, and ``dispatch`` owns the uniform 4-clause error handling
        this handler used to repeat. The empty-question guard stays here (before
        the seam) so an empty question is a clean no-op, not a task error.
        """
        question = args.get("question", "")
        if not question:
            self.error("No question provided")
            return

        self.log(f"AI Ask: {question}", "info")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.ask import AskTask

            return AskTask.from_context(ctx)

        def on_result(_task: Any, answer: Any) -> None:
            self.log("=" * 50, "info")
            self.log("AI ANSWER", "info")
            self.log("=" * 50, "info")
            for line in str(answer).split("\n"):
                self.log(line, "info")
            self.log("=" * 50, "info")

        self._dispatch(args, build_task, on_result=on_result)

    def run_chat(self, args: dict[str, Any]) -> None:
        """Run the Chat task through the dispatch seam (#4, commit 4).

        Multi-turn conversation with tool transparency. Log-only: ``ChatTask``
        persists ``assets/chat_history.json`` itself; ``from_context`` resolves
        the text-LLM config (with the load-bearing max_tokens=8192), embedding
        config, and excluded tags; ``on_result`` logs the response; ``dispatch``
        owns the uniform 4-clause error handling. The empty-message guard stays
        here (before the seam) so an empty message is a clean no-op.
        """
        message = args.get("message", "")
        if not message:
            self.error("No message provided")
            return

        self.log(f"Chat message: {message[:100]}...", "info")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.chat import ChatTask

            return ChatTask.from_context(ctx)

        def on_result(_task: Any, response: Any) -> None:
            self.log("Chat response generated", "info")
            for line in str(response).split("\n"):
                self.log(line, "info")

        self._dispatch(args, build_task, on_result=on_result)

    def run_clear_chat(self, args: dict[str, Any]) -> None:
        """Clear the chat conversation history through the dispatch seam (#4, commit 4).

        Log-only: ``ClearChatTask`` deletes ``assets/chat_history.json`` and
        declares no result_key; ``dispatch`` owns uniform error handling,
        replacing the hand-rolled try/except. The ``clear_chat`` mode ignores its
        args, but the handler now takes ``args`` to match the dispatch shape.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.clear_chat import ClearChatTask

            return ClearChatTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_scene_vision(self, args: dict[str, Any]) -> None:
        """
        Run scene vision analysis using a multimodal LLM.

        Args:
            args: Task arguments containing scene_id, optional message, conversation_id, clear_cache
        """
        try:
            import json as json_module
            import os

            from stash_ai.config import get_text_llm_settings, get_vision_llm_settings
            from stash_ai.tasks.scene_vision import SceneVisionTask

            scene_id = args.get("scene_id", "")
            message = args.get("message", "")
            conversation_id = args.get("conversation_id", "")
            # clear_history: clears conversation history only (for re-analysis with same frames)
            # clear_frames: clears extracted frames cache (forces re-extraction)
            clear_history = args.get("clear_history", "").lower() == "true"
            clear_frames = args.get("clear_frames", "").lower() == "true"
            # Legacy support: clear_cache clears both
            if args.get("clear_cache", "").lower() == "true":
                clear_history = True
                clear_frames = True

            # When clearing history (re-analyze), also clear frames for truly fresh analysis
            # This ensures different frames are selected, leading to varied descriptions
            if clear_history:
                clear_frames = True

            if not scene_id:
                self.error("No scene_id provided")
                return

            self.log(f"Scene Vision Analysis: scene {scene_id}", "info")

            # Clear conversation history if requested (for re-analysis)
            if clear_history:
                self.log("Clearing conversation history for fresh analysis", "info")
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                history_file = os.path.join(
                    plugin_dir, "assets", "scene_vision", f"vision_history_{scene_id}.json"
                )
                self.log(f"Looking for history file: {history_file}", "info")
                if os.path.exists(history_file):
                    os.remove(history_file)
                    self.log("Deleted history file successfully", "info")
                else:
                    self.log("History file does not exist at expected path", "info")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get vision LLM settings (falls back to text LLM if not configured)
            vision_llm = get_vision_llm_settings(plugin_settings, args)
            self.log(f"Using vision model: {vision_llm.provider}/{vision_llm.model}", "info")

            # Get text LLM settings for tag suggestions
            text_llm = get_text_llm_settings(plugin_settings, args)
            if text_llm.provider != vision_llm.provider or text_llm.model != vision_llm.model:
                self.log(f"Using text model for tags: {text_llm.provider}/{text_llm.model}", "info")

            # Get hosted provider max frames setting (default 10)
            hosted_max_frames = int(plugin_settings.get("vision_hosted_max_frames") or "10")

            # Get user confirmation flag (for hosted provider warning bypass)
            user_confirmed = args.get("user_confirmed", "").lower() == "true"
            # Get limited frames flag (use uniformly sampled subset for hosted providers)
            use_limited_frames = args.get("use_limited_frames", "").lower() == "true"

            # Multi-stage vision analysis options
            quick_mode = args.get("quick_mode", "").lower() == "true"
            skip_verification = args.get("skip_verification", "").lower() == "true"
            frame_count_str = args.get("frame_count", "")
            frame_count = (
                int(frame_count_str) if frame_count_str and frame_count_str.isdigit() else None
            )

            # Parse custom prompts (can be JSON string or dict)
            custom_prompts_raw = args.get("custom_prompts", "")
            custom_prompts = None
            if custom_prompts_raw:
                if isinstance(custom_prompts_raw, str):
                    try:
                        import json

                        custom_prompts = json.loads(custom_prompts_raw)
                    except json.JSONDecodeError:
                        self.log(
                            f"Failed to parse custom_prompts as JSON: {custom_prompts_raw[:100]}",
                            "warning",
                        )
                elif isinstance(custom_prompts_raw, dict):
                    custom_prompts = custom_prompts_raw

            # Create LLM configs
            llm_config = vision_llm.to_config()
            tag_llm_config = text_llm.to_config()

            # Parse excluded tags (comma-separated string to list)
            # When parent tags are excluded, their children are also excluded
            excluded_tags_str = plugin_settings.get("excluded_tags", "")
            excluded_tags = (
                [tag.strip() for tag in excluded_tags_str.split(",") if tag.strip()]
                if excluded_tags_str
                else []
            )

            if excluded_tags:
                self.log(f"Excluding tags (and children): {excluded_tags}", "info")

            # Get frame extraction settings
            # Default: 10s interval (0.1 fps), no max (0 = unlimited, extract based on duration)
            frame_interval = float(plugin_settings.get("vision_frame_interval") or "10")
            fps_rate = 1.0 / frame_interval  # Convert interval to fps
            min_frames = int(plugin_settings.get("vision_min_frames") or "1")
            max_frames = int(plugin_settings.get("vision_max_frames") or "0")

            self.log(
                f"Frame extraction: interval={frame_interval}s (fps={fps_rate}), min={min_frames}, max={max_frames} (0=unlimited)",
                "debug",
            )

            # Get custom prompts from args (for prompt iteration via UI)
            custom_system_prompt = args.get("custom_system_prompt", "")
            custom_description_prompt = args.get("custom_description_prompt", "")

            if custom_system_prompt:
                self.log("Using custom system prompt from UI", "debug")
            if custom_description_prompt:
                self.log("Using custom description prompt from UI", "debug")

            # Get image embedding settings for context augmentation
            from stash_ai.embeddings.config import EmbeddingConfig

            image_embedding_config = None
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")

            if image_provider and image_model:
                image_embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device=plugin_settings.get("image_embedding_device") or "auto",
                )
                self.log(f"Vision augmentation enabled: {image_provider}/{image_model}", "info")

            # Create and run the task
            task = SceneVisionTask(
                stash=self.stash_client,
                llm_config=llm_config,
                tag_llm_config=tag_llm_config,
                image_embedding_config=image_embedding_config,
                log_callback=self.log,
                progress_callback=self.progress,
                excluded_tags=excluded_tags,
                fps_rate=fps_rate,
                min_frames=min_frames,
                max_frames=max_frames,
                custom_system_prompt=custom_system_prompt,
                custom_description_prompt=custom_description_prompt,
                hosted_max_frames=hosted_max_frames,
            )

            result = task.run(
                scene_id=scene_id,
                message=message if message else None,
                conversation_id=conversation_id if conversation_id else None,
                clear_frames=clear_frames,
                user_confirmed=user_confirmed,
                use_limited_frames=use_limited_frames,
                quick_mode=quick_mode,
                skip_verification=skip_verification,
                frame_count=frame_count,
                custom_prompts=custom_prompts,
            )

            # Output the result as JSON for frontend consumption
            self.log("=" * 50, "info")
            self.log("SCENE VISION ANALYSIS", "info")
            self.log("=" * 50, "info")

            if result.get("success"):
                self.log(f"Conversation ID: {result.get('conversation_id')}", "info")

                if result.get("description"):
                    self.log("Description:", "info")
                    for line in result["description"].split("\n"):
                        self.log(line, "info")

                if result.get("suggested_tags"):
                    self.log(f"Suggested Tags: {', '.join(result['suggested_tags'])}", "info")

                if result.get("response") and message:
                    self.log("Response:", "info")
                    for line in result["response"].split("\n"):
                        self.log(line, "info")

                # Output JSON result for frontend
                self.log("JSON_RESULT:" + json_module.dumps(result), "debug")
            elif result.get("requires_confirmation"):
                # Hosted provider confirmation needed
                self.log(f"Confirmation required: {result.get('confirmation_reason')}", "info")
                self.log("JSON_RESULT:" + json_module.dumps(result), "debug")
            else:
                self.error(f"Vision analysis failed: {result.get('error')}")

            self.log("=" * 50, "info")

        except ImportError as e:
            self.error(f"Failed to import Stash AI modules: {e}")
        except ConnectionError as e:
            self.error(f"Connection error: {e}")
        except RuntimeError as e:
            self.error(f"Task failed: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_embed_scenes(self, args: dict[str, Any]) -> None:
        """Run scene embedding generation through the dispatch seam (#4, commit 4).

        Log-only: ``EmbedScenesTask.from_context`` resolves the full embedding
        config (the CLIP-vs-VLM-text path) and the scene_id/force selectors from
        the ``TaskContext``; ``run()`` embeds one scene or all; ``on_result`` logs
        the result banner; ``dispatch`` owns the uniform 4-clause error handling
        this handler used to repeat.
        """
        self.log("=== EMBED SCENES TASK STARTED ===", "info")
        self.log(f"Args received: {args}", "debug")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.embed_scenes import EmbedScenesTask

            return EmbedScenesTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            import json as json_module

            self.log("=" * 50, "info")
            self.log("EMBEDDING RESULT", "info")
            self.log("=" * 50, "info")
            self.log(json_module.dumps(result, indent=2), "info")
            self.log("=" * 50, "info")

        self._dispatch(args, build_task, on_result=on_result)

    def run_find_similar(self, args: dict[str, Any]) -> None:
        """Find scenes similar to a scene, via the dispatch seam (#4, commit 4).

        Result-producing: ``FindSimilarTask.from_context`` resolves the model_key
        + all the search/filter args; ``run()`` returns the result dict (or an
        error dict, owning its own runtime error handling); ``on_result`` persists
        it through the seam's ResultStore as ``similar_results_{scene_id}.json``
        (keyed by scene_id, NOT request_id); ``dispatch`` owns uniform error
        handling. The missing-scene_id guard writes an error result file here
        (before the seam), matching the old handler.
        """
        scene_id = args.get("scene_id")
        if not scene_id:
            self.error("scene_id is required")
            self._result_store().save(
                "similar_results",
                scene_id or "unknown",
                {"status": "error", "error": "scene_id is required"},
            )
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.find_similar import FindSimilarTask

            return FindSimilarTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            self._result_store().save("similar_results", str(scene_id), result)

        self._dispatch(args, build_task, on_result=on_result)

    def _get_scene_details_batch(self, scene_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch scene details for multiple scenes from SQLite (shared helper).

        Thin wrapper over ``stash_ai.tools.scene_details.get_scene_details_batch``
        (extracted under #4 commit 4 so the dispatch-seam search tasks share one
        implementation). Retained for the not-yet-migrated search handlers; removed
        in commit 5 once they all use the shared helper directly.
        """
        from stash_ai.tools.scene_details import get_scene_details_batch

        return get_scene_details_batch(scene_ids, self.log)

    def run_frame_analysis(self, args: dict[str, Any]) -> None:
        """Run intra-scene frame analysis through the dispatch seam (#4, commit 4).

        Analyzes frame-to-frame similarity (PCA/t-SNE/UMAP) and selects
        representative frames. ``FrameAnalysisTask.from_context`` resolves the
        embedding / frame-extraction / analysis config; ``run()`` writes its own
        per-scene analysis files; ``on_result`` logs the summary; ``dispatch`` owns
        uniform error handling. The missing-scene_id guard + the initial logs stay
        here (before the seam).
        """
        scene_id = args.get("scene_id")
        if not scene_id:
            self.error("scene_id is required")
            return

        selection_method_arg = args.get("selection_method", "not provided")
        self.log(f"Starting frame analysis for scene {scene_id}...", "info")
        self.log(f"Selection method from args: {selection_method_arg}", "debug")

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.frame_analysis import FrameAnalysisTask

            return FrameAnalysisTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            if result:
                self.log("=" * 50, "info")
                self.log("FRAME ANALYSIS COMPLETE", "info")
                self.log("=" * 50, "info")
                self.log(f"Scene: {result['scene_id']}", "info")
                self.log(f"Frames analyzed: {result['frame_count']}", "info")
                self.log(f"Embedding model: {result['embedding_model']}", "info")
                self.log(f"Embedding dimensions: {result['embedding_dimensions']}", "info")
                self.log("", "info")
                self.log("Representative frames:", "info")
                self.log(f"  Indices: {result['representative']['selected_indices']}", "info")
                self.log(f"  Timestamps: {result['representative']['selected_timestamps']}", "info")
                self.log(f"  Method: {result['representative']['selection_method']}", "info")
                self.log(
                    f"  Diversity score: {result['representative']['diversity_score']:.4f}", "info"
                )
                self.log("=" * 50, "info")
            else:
                self.error("Frame analysis failed")

        self._dispatch(args, build_task, on_result=on_result)

    def check_frame_analysis(self, args: dict[str, Any]) -> None:
        """Poll for frame-analysis results/status for a scene, via the dispatch seam (#4).

        SPECIAL: the UI polls this task's STDOUT for a JSON status, so the result
        is ``print``-ed (not logged). ``CheckFrameAnalysisTask.run()`` reads the
        per-scene status/summary files and owns its own error handling (returns
        ``{"status": "error", ...}``) so stdout always gets valid JSON;
        ``on_result`` prints it. The missing-scene_id guard prints to stdout here
        (before the seam).
        """
        import json as json_module

        scene_id = args.get("scene_id")
        if not scene_id:
            print(json_module.dumps({"status": "error", "error": "scene_id is required"}))
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.frame_analysis import CheckFrameAnalysisTask

            return CheckFrameAnalysisTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            print(json_module.dumps(result))

        self._dispatch(args, build_task, on_result=on_result)

    def start_frame_analysis(self, args: dict[str, Any]) -> None:
        """Start frame analysis for a scene (UI run action), via the dispatch seam (#4).

        SPECIAL: prints a JSON status to STDOUT (the UI IPC channel).
        ``StartFrameAnalysisTask.run()`` writes the ``running`` status file, runs
        the analysis, and returns a ``started`` / ``error`` status (owning its own
        error handling — including writing the error status file — so stdout
        always gets valid JSON); ``on_result`` prints it. The missing-scene_id
        guard prints to stdout here (before the seam).
        """
        import json as json_module

        scene_id = args.get("scene_id")
        if not scene_id:
            print(json_module.dumps({"status": "error", "error": "scene_id is required"}))
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.frame_analysis import StartFrameAnalysisTask

            return StartFrameAnalysisTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            print(json_module.dumps(result))

        self._dispatch(args, build_task, on_result=on_result)

    def _write_similar_result(self, scene_id: str, data: dict[str, Any]) -> None:
        """Write similar scenes result to JSON file for frontend polling."""
        import json as json_module
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")

        # Ensure assets directory exists
        os.makedirs(assets_dir, exist_ok=True)

        result_file = os.path.join(assets_dir, f"similar_results_{scene_id}.json")

        try:
            with open(result_file, "w") as f:
                json_module.dump(data, f)
            self.log(f"Wrote similar results to: {result_file}", "debug")
        except Exception as e:
            self.error(f"Failed to write similar results file: {e}")

    def run_search_by_text(self, args: dict[str, Any]) -> None:
        """Semantic scene search by text, via the dispatch seam (#4, commit 4).

        Result-producing: ``SearchByTextTask.from_context`` resolves the query +
        model/search args; ``run()`` returns the result dict (or an error dict,
        owning its runtime errors); ``on_result`` persists it through the seam's
        ResultStore as ``search_results_{request_id|latest}.json``; ``dispatch``
        owns uniform error handling. The empty-query guard writes its error result
        here (before the seam).
        """
        # Mirror _write_search_result's "request_id or 'latest'" filename rule.
        request_id = args.get("request_id", "") or "latest"

        if not args.get("query", "").strip():
            self._result_store().save(
                "search_results", request_id, {"status": "error", "error": "Query is required"}
            )
            return

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.search_by_text import SearchByTextTask

            return SearchByTextTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            self._result_store().save("search_results", request_id, result)

        self._dispatch(args, build_task, on_result=on_result)

    def run_find_similar_by_frame(self, args: dict[str, Any]) -> None:
        """Find similar scenes by extracting and embedding the current video frame.

        Extracts a single frame at the given timestamp, embeds it with the
        configured image embedding provider, and searches the FAISS frame
        index for visually similar frames across the library.

        Args:
            args: Task arguments containing:
                - scene_id: Scene ID currently playing (required)
                - timestamp: Playback position in seconds (required)
                - limit: Maximum results (default 20)
                - request_id: Unique request ID for frontend polling (required)
        """
        try:
            import numpy as np

            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.frame_search import FrameSearchIndex
            from stash_ai.embeddings.provider import get_embedding_provider
            from stash_ai.tasks.frame_extractor import FrameExtractionConfig, FrameExtractor
            from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

            scene_id = args.get("scene_id", "").strip()
            timestamp_str = args.get("timestamp", "0")
            request_id = args.get("request_id", "")

            if not scene_id:
                self._write_frame_search_result(
                    request_id,
                    {"status": "error", "error": "Scene ID is required", "request_id": request_id},
                )
                return

            try:
                timestamp = float(timestamp_str)
                limit = int(args.get("limit", 20))
            except ValueError as e:
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": f"Invalid parameter: {e}",
                        "request_id": request_id,
                    },
                )
                return
            self.log(
                f"Frame search: scene={scene_id}, timestamp={timestamp:.1f}s, limit={limit}", "info"
            )

            # Step 1: Resolve video file path from Stash SQLite
            db_path = get_stash_db_path()
            conn = get_readonly_connection(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT fo.path || '/' || f.basename as video_path
                    FROM scenes s
                    JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                    JOIN files f ON sf.file_id = f.id
                    JOIN folders fo ON f.parent_folder_id = fo.id
                    JOIN video_files vf ON f.id = vf.file_id
                    WHERE s.id = ?
                    """,
                    (int(scene_id),),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": f"Could not find video file for scene {scene_id}",
                        "request_id": request_id,
                    },
                )
                return

            video_path = row["video_path"]

            # Step 2: Extract frame at timestamp (ephemeral - no disk caching)
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(plugin_dir, "assets")
            extractor = FrameExtractor(
                config=FrameExtractionConfig(),
                cache_dir=os.path.join(assets_dir, "embedded_frames"),
                log_callback=self.log,
            )

            frame_bytes = extractor.extract_frame_at_timestamp(video_path, timestamp)
            if frame_bytes is None:
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": f"Failed to extract frame at {timestamp:.1f}s",
                        "request_id": request_id,
                    },
                )
                return

            self.log(f"Extracted frame: {len(frame_bytes)} bytes", "debug")

            # Step 3: Embed the frame
            plugin_settings = self.get_plugin_settings("stash-copilot")
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": "No image embedding provider configured. Set up in Plugin Settings.",
                        "request_id": request_id,
                    },
                )
                return

            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )
            model_key = embedding_config.model_key

            embedder = get_embedding_provider(embedding_config)
            if not hasattr(embedder, "embed_image"):
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": f"Provider '{image_provider}' does not support image embedding.",
                        "request_id": request_id,
                    },
                )
                return
            result = embedder.embed_image(frame_bytes)
            query_embedding = np.array(result["embedding"], dtype=np.float32)

            self.log(f"Embedded frame: {result['dimensions']} dims", "debug")

            # Step 4: Load frame search index
            frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

            if not frame_index.exists:
                self._write_frame_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": f"Frame search index not found for model '{model_key}'. Run 'Build Frame Search Index' task first.",
                        "request_id": request_id,
                    },
                )
                return

            # Step 5: Search for similar frames
            # Over-fetch frames since many top matches may belong to the same scene.
            # After aggregate_to_scenes(), we need enough unique scenes to fill `limit`.
            frame_matches = frame_index.search(query_embedding, top_k=2000)

            # Step 6: Filter out query scene's own frames
            frame_matches = [m for m in frame_matches if m.scene_id != int(scene_id)]

            # Step 7: Aggregate to best match per scene
            scene_matches = frame_index.aggregate_to_scenes(frame_matches)

            # Step 8: Truncate to limit
            scene_matches = scene_matches[:limit]

            # Step 9: Fetch scene details
            scene_details = self._get_scene_details_batch([m.scene_id for m in scene_matches])

            # Step 10: Build result data
            result_data = []
            for m in scene_matches:
                scene = scene_details.get(m.scene_id, {})
                frame_path = (
                    f"embedded_frames/scene_{m.scene_id}/frame_{m.best_frame_index:04d}.jpg"
                )
                result_data.append(
                    {
                        "scene_id": m.scene_id,
                        "similarity": m.similarity,
                        "matched_timestamp": m.best_timestamp,
                        "matched_frame_index": m.best_frame_index,
                        "frame_path": frame_path,
                        "scene": scene,
                    }
                )

            # Step 11: Write results
            self._write_frame_search_result(
                request_id,
                {
                    "status": "complete",
                    "query_scene_id": int(scene_id),
                    "query_timestamp": timestamp,
                    "model_key": model_key,
                    "results": result_data,
                    "limit": limit,
                    "request_id": request_id,
                },
            )

            self.log(f"Frame search complete: {len(result_data)} scenes found", "info")

        except ImportError as e:
            self.error(f"Failed to import modules: {e}")
            self._write_frame_search_result(
                args.get("request_id", ""),
                {
                    "status": "error",
                    "error": f"Failed to import modules: {e}",
                    "request_id": args.get("request_id", ""),
                },
            )
        except Exception as e:
            self.error(f"Frame search error: {e}")
            self._write_frame_search_result(
                args.get("request_id", ""),
                {"status": "error", "error": str(e), "request_id": args.get("request_id", "")},
            )

    def _write_frame_search_result(self, request_id: str, data: dict[str, Any]) -> None:
        """Write frame search results to JSON file for frontend polling."""
        import json as json_module
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        filename = f"frame_search_{request_id or 'latest'}.json"
        result_file = os.path.join(assets_dir, filename)

        try:
            with open(result_file, "w") as f:
                json_module.dump(data, f)
            self.log(f"Wrote frame search results to: {result_file}", "debug")
        except Exception as e:
            self.error(f"Failed to write frame search results: {e}")

    def run_get_embedding_models(self, args: dict[str, Any]) -> None:
        """List available embedding models + stats, through the dispatch seam (#4, commit 4).

        Result-producing: ``GetEmbeddingModelsTask`` writes its own
        ``assets/embedding_models_{request_id|latest}.json`` on both success and
        error (declares ``result_key="embedding_models"``), so it owns its file
        I/O and error handling; ``dispatch`` just runs it.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.embedding_models import GetEmbeddingModelsTask

            return GetEmbeddingModelsTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_embed_o_moments(self, args: dict[str, Any]) -> None:
        """Run O-moment embedding through the dispatch seam (#4, commit 4).

        Log-only: ``EmbedOMomentsTask`` resolves its embedding + O-moment config
        and the single-scene/all-scenes selectors from the ``TaskContext`` in
        ``from_context``, and its ``run()`` embeds and logs the summary; missing
        image-embedding config surfaces via ``dispatch``'s RuntimeError branch.
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.embed_o_moments import EmbedOMomentsTask

            return EmbedOMomentsTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_embed_cached_frames(self, args: dict[str, Any]) -> None:
        """Run the cached-frames embedding backfill through the dispatch seam (#4, commit 4).

        Backfills frame embeddings for scenes whose frames were cached before
        individual frame storage existed (enables smart frame selection for VLM
        analysis). Log-only: ``EmbedCachedFramesTask`` resolves its config + the
        force/scene_id selectors from the ``TaskContext`` in ``from_context``;
        ``on_result`` logs the summary; ``dispatch`` owns uniform error handling
        (missing image-embedding config surfaces via its RuntimeError branch).
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.embed_cached_frames import EmbedCachedFramesTask

            return EmbedCachedFramesTask.from_context(ctx)

        def on_result(_task: Any, result: Any) -> None:
            if not result:
                return
            self.log("Embed Cached Frames complete:", "info")
            self.log(f"  Total: {result.get('total', 0)}", "info")
            self.log(f"  Processed: {result.get('processed', 0)}", "info")
            self.log(f"  Skipped: {result.get('skipped', 0)}", "info")
            self.log(f"  Errors: {result.get('errors', 0)}", "info")
            if result.get("error_details"):
                for err in result["error_details"][:5]:
                    self.log(f"  - {err}", "warning")

        self._dispatch(args, build_task, on_result=on_result)

    def _cleanup_deleted_scene(self, scene_id: int) -> None:
        """
        Clean up all embeddings and data for a deleted scene.

        Called by Scene.Destroy.Post hook when a scene is deleted from Stash.

        Args:
            scene_id: ID of the deleted scene
        """
        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            self.log(f"Cleaning up embeddings for deleted scene {scene_id}...", "info")

            # Create storage instance (model_key doesn't matter for delete_all_scene_data)
            storage = EmbeddingStorage()
            result = storage.delete_all_scene_data(scene_id)

            total_deleted = sum(result.values())
            if total_deleted > 0:
                self.log(
                    f"Cleaned up {total_deleted} items for deleted scene {scene_id}: "
                    f"{result['embeddings']} embeddings, {result['o_moments']} o-moments, "
                    f"{result['frames']} frames, {result['segments']} segments",
                    "info",
                )
            else:
                self.log(f"No embeddings found for deleted scene {scene_id}", "debug")

        except ImportError as e:
            self.log(f"Failed to import storage module: {e}", "warning")
        except Exception as e:
            self.log(f"Error cleaning up deleted scene {scene_id}: {e}", "warning")

    def run_build_frame_index(self, args: dict[str, Any]) -> None:
        """
        Build FAISS index for frame-level semantic search.

        Args:
            args: Task arguments containing:
                - model_key: Optional model key (defaults to configured model)
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.frame_search import FrameSearchIndex
            from stash_ai.embeddings.storage import EmbeddingStorage

            # Get model key from args or settings
            requested_model_key = args.get("model_key", "").strip()

            plugin_settings = self.get_plugin_settings("stash-copilot")

            if requested_model_key:
                model_key = requested_model_key
            else:
                image_provider = plugin_settings.get("image_embedding_provider")
                image_model = plugin_settings.get("image_embedding_model")

                if not image_provider or not image_model:
                    self.error(
                        "Image embedding provider not configured. Set up in Plugin Settings first."
                    )
                    return

                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device="cpu",  # Not used for indexing
                )
                model_key = embedding_config.model_key

            self.log(f"Building frame search index for model: {model_key}", "info")

            # Initialize storage and index
            storage = EmbeddingStorage(model_key=model_key)
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(plugin_dir, "assets")

            frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

            # Build with progress reporting
            def progress_callback(current: int, total: int) -> None:
                self.progress(current, total)
                if current % 50000 == 0 or current == total:
                    self.log(f"Indexed {current:,} / {total:,} frames", "info")

            info = frame_index.build(
                storage=storage,
                progress_callback=progress_callback,
            )

            self.log(
                f"Frame search index built successfully:\n"
                f"  Model: {info.model_key}\n"
                f"  Frames: {info.frame_count:,}\n"
                f"  Scenes: {info.scene_count:,}\n"
                f"  Dimensions: {info.dimensions}",
                "info",
            )

        except ValueError as e:
            self.error(str(e))
        except Exception as e:
            self.error(f"Failed to build frame search index: {e}")
            import traceback

            self.log(traceback.format_exc(), "debug")

    def run_cleanup_orphaned(self, args: dict[str, Any]) -> None:
        """Clean up embeddings for deleted scenes, through the dispatch seam (#4, commit 4).

        Log-only maintenance task: ``CleanupOrphanedTask`` reads valid scene IDs
        from the Stash DB, finds orphaned embeddings, and (unless ``dry_run``)
        deletes them with progress logging; ``dispatch`` owns uniform error
        handling (a missing DB surfaces via its RuntimeError branch).
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.cleanup_orphaned import CleanupOrphanedTask

            return CleanupOrphanedTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_embed_performers(self, args: dict[str, Any]) -> None:
        """Generate performer embeddings through the dispatch seam (#4, commit 4).

        Log-only: ``EmbedPerformersTask.from_context`` resolves the embedding
        config, engagement weights, and task config (min/max scenes from args),
        and caches the performer_id/force selectors; ``run()`` embeds one performer
        or all and logs the summary; ``dispatch`` owns uniform error handling
        (missing image-embedding config surfaces via its RuntimeError branch).
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.embed_performers import EmbedPerformersTask

            return EmbedPerformersTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_describe_performers(self, args: dict[str, Any]) -> None:
        """Generate AI performer descriptions through the dispatch seam (#4, commit 4).

        Log-only: ``DescribePerformerTask.from_context`` resolves the VLM config,
        storage model_key, and task config, and caches the performer_id/force
        selectors; ``run()`` describes one performer or all (writing descriptions
        to the embeddings DB) and logs the summary; ``dispatch`` owns uniform
        error handling (missing image-embedding config surfaces via its
        RuntimeError branch).
        """

        def build_task(ctx: TaskContext) -> Any:
            from stash_ai.tasks.describe_performer import DescribePerformerTask

            return DescribePerformerTask.from_context(ctx)

        self._dispatch(args, build_task)

    def run_find_similar_performers(self, args: dict[str, Any]) -> None:
        """
        Find performers visually similar to a given performer.

        Args:
            args: Task arguments containing:
                - performer_id: Required source performer ID
                - limit: Maximum results (default 10)
                - min_similarity: Minimum similarity threshold (default 0.5)
                - request_id: Optional request ID for frontend tracking
        """

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.embed_performers import EmbedPerformersTask

            performer_id = args.get("performer_id")
            if not performer_id:
                self._write_similar_performers_result(
                    "", {"status": "error", "error": "performer_id is required"}
                )
                return

            limit = int(args.get("limit") or "10")
            min_similarity = float(args.get("min_similarity") or "0.0")
            request_id = args.get("request_id") or str(performer_id)

            self.log(f"Finding performers similar to {performer_id}...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get image embedding config
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self._write_similar_performers_result(
                    request_id,
                    {"status": "error", "error": "Image embedding provider not configured"},
                )
                return

            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )

            # Create task (reusing EmbedPerformersTask for find_similar_performers)
            task = EmbedPerformersTask(
                stash=self.stash_client,
                embedding_config=embedding_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            result = task.find_similar_performers(
                performer_id=int(performer_id),
                limit=limit,
                min_similarity=min_similarity,
            )

            if result.get("success"):
                self.log(
                    f"Found {len(result.get('similar_performers', []))} similar performers", "info"
                )
                self._write_similar_performers_result(
                    request_id,
                    {
                        "status": "complete",
                        "source_performer": result.get("source_performer"),
                        "results": result.get("similar_performers", []),
                        "total_found": result.get("total_found", 0),
                    },
                )
            else:
                self._write_similar_performers_result(
                    request_id,
                    {
                        "status": "error",
                        "error": result.get("error"),
                    },
                )

        except ImportError as e:
            self._write_similar_performers_result(
                args.get("request_id", ""),
                {"status": "error", "error": f"Failed to import modules: {e}"},
            )
        except Exception as e:
            self._write_similar_performers_result(
                args.get("request_id", ""), {"status": "error", "error": str(e)}
            )

    def _write_similar_performers_result(self, request_id: str, data: dict[str, Any]) -> None:
        """Write similar performers result to JSON file for frontend polling."""
        import json as json_module
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        result_file = os.path.join(assets_dir, f"similar_performers_{request_id}.json")

        try:
            with open(result_file, "w") as f:
                json_module.dump(data, f)
            self.log(f"Wrote similar performers results to: {result_file}", "debug")
        except Exception as e:
            self.error(f"Failed to write similar performers results file: {e}")

    def run_hook(self, hook_context: dict[str, Any]) -> None:
        """
        Run as a hook (triggered by Stash events).

        Args:
            hook_context: Hook context data
        """
        hook_type = hook_context.get("hookContext", {}).get("type")
        self.log(f"Hook triggered: {hook_type}")

        if hook_type == "Scene.Create.Post":
            scene_id = hook_context.get("hookContext", {}).get("id")
            if scene_id:
                self.process_scene(scene_id)
        elif hook_type == "Scene.Destroy.Post":
            scene_id = hook_context.get("hookContext", {}).get("id")
            if scene_id:
                self._cleanup_deleted_scene(int(scene_id))

    def run(self) -> None:
        """Main entry point for the plugin."""
        if not self.input:
            self.error("No input provided")
            sys.exit(1)

        # Check if running as a task or hook
        if "args" in self.input:
            # Running as a task
            mode = self.input["args"].get("mode", "")
            self.run_task(mode, self.input["args"])
        elif "hookContext" in self.input:
            # Running as a hook
            self.run_hook(self.input)
        else:
            self.error("Unknown execution mode")
            sys.exit(1)


def main() -> None:
    """Main function."""
    plugin = MyPlugin()
    plugin.run()


if __name__ == "__main__":
    main()
