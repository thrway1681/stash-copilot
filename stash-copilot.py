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
        self._read_input()

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

    def _read_input(self) -> None:
        """Read and parse JSON input from stdin."""
        try:
            input_str = sys.stdin.read()
            if input_str:
                self.input = json.loads(input_str)
                self.log("Input received", "debug")
        except json.JSONDecodeError as e:
            self.error(f"Failed to parse input JSON: {e}")
            sys.exit(1)

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
            return data["findScenes"]["scenes"]
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
            "clear_chat": lambda args: self.run_clear_chat(),
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
            "preference_recs": self.run_preference_recs,
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
        # Preference tasks share one handler that also needs the task name.
        for name in (
            "preference_start",
            "preference_compare",
            "preference_swipe",
            "preference_end",
            "preference_stats",
            "preference_reset",
        ):
            handlers[name] = self._make_preference_handler(name)
        return handlers

    def _make_preference_handler(
        self, task_name: str
    ) -> Callable[[dict[str, Any]], None]:
        """Bind ``task_name`` into a preference handler with the standard shape."""

        def handler(args: dict[str, Any]) -> None:
            self.run_preference_trainer(task_name, args)

        return handler

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

    def run_eroscripts_validate_auth(self, args: dict[str, Any]) -> None:
        """Validate (or clear/re-check) the EroScripts session cookie."""
        try:
            from stash_ai.tasks import eroscripts_auth as task_module
            task_module.run(args, self.log)
        except Exception as e:  # noqa: BLE001
            self.error(f"eroscripts_validate_auth failed: {e}")

    def run_eroscripts_search(self, args: dict[str, Any]) -> None:
        """Search discuss.eroscripts.com for funscripts matching a Stash scene."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_search as task_module
            task_module.run(self.stash, args, self.log)
        except Exception as e:  # noqa: BLE001
            self.error(f"eroscripts_search failed: {e}")

    def run_eroscripts_download(self, args: dict[str, Any]) -> None:
        """List attachments for an eroscripts topic, or download one and persist."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_download as task_module
            task_module.run(self.stash, args, self.log)
        except Exception as e:  # noqa: BLE001
            self.error(f"eroscripts_download failed: {e}")

    def run_eroscripts_status(self, args: dict[str, Any]) -> None:
        """Report whether a scene has a matched funscript + sidecar."""
        if self.stash is None:
            self.error("Stash connection unavailable")
            return
        try:
            from stash_ai.tasks import eroscripts_status as task_module
            task_module.run(self.stash, args, self.log)
        except Exception as e:  # noqa: BLE001
            self.error(f"eroscripts_status failed: {e}")

    def run_stats_summary(self, args: dict[str, Any]) -> None:
        """
        Run the AI-powered library statistics summary task.

        Args:
            args: Task arguments containing LLM settings
        """
        try:
            from stash_ai.config import get_text_llm_settings
            from stash_ai.tasks.stats_summary import StatsSummaryTask

            self.log("Initializing Stash AI statistics summary...", "info")

            # Fetch plugin settings from Stash via GraphQL
            # The plugin ID is the yml filename without extension: "stash-copilot"
            plugin_settings = self.get_plugin_settings("stash-copilot")
            self.log(f"Plugin settings from Stash: {plugin_settings}", "debug")

            # Get text LLM settings
            text_llm = get_text_llm_settings(plugin_settings, args)
            self.log(f"Using LLM provider: {text_llm.provider}", "info")
            self.log(f"Using model: {text_llm.model}", "info")

            llm_config = text_llm.to_config()

            # Parse excluded tags (comma-separated string to list)
            excluded_tags_str = plugin_settings.get("excluded_tags", "")
            excluded_tags = (
                [tag.strip() for tag in excluded_tags_str.split(",") if tag.strip()]
                if excluded_tags_str
                else []
            )

            if excluded_tags:
                self.log(f"Excluding tags: {excluded_tags}", "info")

            # Create and run the task with the already-connected StashInterface
            task = StatsSummaryTask(
                stash=self.stash_client,
                llm_config=llm_config,
                log_callback=self.log,
                progress_callback=self.progress,
                excluded_tags=excluded_tags,
            )

            summary = task.run()

            # Output the summary
            self.log("=" * 50, "info")
            self.log("LIBRARY STATISTICS SUMMARY", "info")
            self.log("=" * 50, "info")
            for line in summary.split("\n"):
                self.log(line, "info")
            self.log("=" * 50, "info")

        except ImportError as e:
            self.error(f"Failed to import Stash AI modules: {e}")
            self.error("Make sure the stash_ai package is properly installed.")
        except ConnectionError as e:
            self.error(f"Connection error: {e}")
        except RuntimeError as e:
            self.error(f"Task failed: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_recommendations(self, args: dict[str, Any]) -> None:
        """
        Run the personalized recommendations task.

        Args:
            args: Task arguments containing recommendation settings
        """
        try:
            from stash_ai.tasks.recommendations import RecommendationsTask

            self.log("Initializing recommendation generation...", "info")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Parse arguments with fallback to plugin settings
            mode = args.get("rec_mode", "discover_new")
            scoring_method = args.get("scoring_method", "base_weighted")
            limit = int(args.get("limit", 120))  # 10 pages x 12 per page
            per_page = int(args.get("per_page", 12))
            request_id = args.get("request_id", "")

            # Engagement weights from settings or defaults
            top_scenes = int(plugin_settings.get("rec_top_scenes") or args.get("top_scenes", 20))
            o_weight = float(plugin_settings.get("rec_o_weight") or args.get("o_weight", 3.0))
            view_weight = float(
                plugin_settings.get("rec_view_weight") or args.get("view_weight", 1.5)
            )
            duration_weight = float(
                plugin_settings.get("rec_duration_weight") or args.get("duration_weight", 1.0)
            )
            rating_weight = float(
                plugin_settings.get("rec_rating_weight") or args.get("rating_weight", 1.5)
            )
            half_life = float(
                plugin_settings.get("rec_time_decay_days") or args.get("half_life_days", 30.0)
            )
            min_similarity = float(args.get("min_similarity", 0.1))

            # Seed scene for scene-specific recommendations
            seed_scene_id_str = args.get("seed_scene_id", "")
            seed_scene_id = int(seed_scene_id_str) if seed_scene_id_str else None
            seed_weight = float(args.get("seed_weight", 0.3))
            engagement_weight = float(args.get("engagement_weight", 0.6))

            # Session-based recommendations (scene IDs from current session)
            session_scene_ids_str = args.get("session_scene_ids", "")
            session_scene_ids = (
                [int(x.strip()) for x in session_scene_ids_str.split(",") if x.strip()]
                if session_scene_ids_str
                else None
            )

            self.log(
                f"Running recommendations: mode={mode}, scoring={scoring_method}",
                "info",
            )
            if session_scene_ids:
                self.log(f"Session mode: {len(session_scene_ids)} scenes", "info")
            if seed_scene_id:
                self.log(f"Seed scene: {seed_scene_id} (weight: {seed_weight})", "info")
            self.log(
                f"Weights: o_count={o_weight}, views={view_weight}, "
                f"duration={duration_weight}, rating={rating_weight}",
                "debug",
            )

            # Get model_key from image embedding settings (defaults to siglip)
            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"  # Default
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key
                self.log(f"Using embedding model: {model_key}", "debug")

            task = RecommendationsTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            result = task.run(
                mode=mode,
                scoring_method=scoring_method,
                limit=limit,
                per_page=per_page,
                top_scenes_for_profile=top_scenes,
                o_weight=o_weight,
                view_weight=view_weight,
                duration_weight=duration_weight,
                rating_weight=rating_weight,
                half_life_days=half_life,
                min_similarity=min_similarity,
                request_id=request_id,
                seed_scene_id=seed_scene_id,
                seed_weight=seed_weight,
                engagement_weight=engagement_weight,
                session_scene_ids=session_scene_ids,
            )

            # Output summary
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

        except ImportError as e:
            self.error(f"Failed to import recommendation modules: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_build_taste_map(self, args: dict[str, Any]) -> None:
        """Run the Build Taste Map task."""
        try:
            from stash_ai.tasks.taste_map import TasteMapTask

            self.log("Initializing taste map generation...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get model_key from image embedding settings (same as recommendations)
            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"  # Default
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TasteMapTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            num_clusters_str = args.get("num_clusters", "")
            num_clusters = int(num_clusters_str) if num_clusters_str else None

            response = task.run(
                request_id=args.get("request_id", ""),
                scoring_method=args.get("scoring_method", "base_weighted"),
                num_clusters=num_clusters,
            )

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

        except Exception as e:
            self.error(f"Build Taste Map failed: {e}")

    def run_detect_tag_gaps(self, args: dict[str, Any]) -> None:
        """Run the tag gap detection task."""
        try:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            self.log("Initializing tag gap detection...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TagGapDetectionTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            force = args.get("force", "false").lower() == "true"
            report = task.run(
                request_id=args.get("request_id", ""),
                force=force,
            )

            if report["status"] == "complete":
                self.log(
                    f"Tag gap detection complete: {report['avg_coverage']:.0%} avg coverage, "
                    f"{report['flagged_scenes']} scenes flagged",
                    "info",
                )
            else:
                self.log(f"Tag gap detection failed: {report.get('error', 'Unknown')}", "error")

        except Exception as e:
            self.error(f"Detect Tag Gaps failed: {e}")

    def run_get_scene_tag_gaps(self, args: dict[str, Any]) -> None:
        """Get tag gap detail for a specific scene (sidebar query)."""
        try:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            scene_id = args.get("scene_id")
            if not scene_id:
                self.error("scene_id argument required")
                return

            plugin_settings = self.get_plugin_settings("stash-copilot")

            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TagGapDetectionTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            result = task.get_scene_gaps_detail(int(scene_id))

            # Write basic result immediately so frontend doesn't timeout
            request_id = args.get("request_id", f"scene_{scene_id}")
            assets_dir = os.path.join(PLUGIN_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            filepath = os.path.join(assets_dir, f"tag_gaps_scene_{request_id}.json")

            with open(filepath, "w") as f:
                json.dump(result, f)

            # Similar scenes computation is slow (O(N) queries across all scenes)
            # Skip for now - requires optimization (pre-computed avg embeddings)
            # TODO: Optimize find_similar_uncovered with pre-computed scene vectors

        except Exception as e:
            self.error(f"Get scene tag gaps failed: {e}")

    def run_preview_tag_impact(self, args: dict[str, Any]) -> None:
        """Preview the coverage impact of a hypothetical tag on a scene."""
        try:
            from stash_ai.tasks.tag_gap_detection import TagGapDetectionTask

            scene_id = args.get("scene_id")
            tag_name = args.get("tag_name")
            if not scene_id or not tag_name:
                self.error("scene_id and tag_name arguments required")
                return

            plugin_settings = self.get_plugin_settings("stash-copilot")

            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            task = TagGapDetectionTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            result = task.preview_tag_impact(int(scene_id), tag_name)

            request_id = args.get("request_id", f"preview_{scene_id}_{tag_name}")
            assets_dir = os.path.join(PLUGIN_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)

            filepath = os.path.join(assets_dir, f"tag_preview_{request_id}.json")
            with open(filepath, "w") as f:
                json.dump(result, f)

        except Exception as e:
            self.error(f"Preview tag impact failed: {e}")

    def run_get_tag_suggestions(self, args: dict[str, Any]) -> None:
        """Get embedding-based tag suggestions for a scene."""
        scene_id = args.get("scene_id")
        if not scene_id:
            self.log("Missing scene_id", "error")
            return

        scene_id = int(scene_id)
        request_id = args.get("request_id", "")

        self.log(f"Computing tag suggestions for scene {scene_id}, request_id={request_id}", "info")
        self.log(f"Args received: {list(args.keys())}", "debug")

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tasks.tag_suggestions import TagSuggestionsTask

            # Get model_key from plugin settings (same as other embedding tasks)
            plugin_settings = self.get_plugin_settings("stash-copilot")
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if image_provider and image_model:
                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device=image_device,
                )
                model_key = embedding_config.model_key
            else:
                # Fallback to the most common model key in use
                model_key = "openclip:ViT-H-14"

            self.log(f"Using embedding model: {model_key}", "debug")

            storage = EmbeddingStorage(model_key=model_key)
            task = TagSuggestionsTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
                model_key=model_key,
            )

            result = task.run(scene_id=scene_id)

            # Save result for frontend polling
            if request_id:
                assets_dir = os.path.join(PLUGIN_DIR, "assets")
                os.makedirs(assets_dir, exist_ok=True)
                result_path = os.path.join(assets_dir, f"tag_suggestions_{request_id}.json")
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

            if result["status"] == "complete":
                self.log(f"Found {len(result['suggestions'])} tag suggestions", "info")
            else:
                self.log(f"Tag suggestions: {result['error']}", "warning")

        except Exception as e:
            self.error(f"Get tag suggestions failed: {e}")

    def run_apply_suggested_tag(self, args: dict[str, Any]) -> None:
        """Apply a suggested tag to a scene."""
        scene_id = int(args.get("scene_id", 0))
        tag_id = int(args.get("tag_id", 0))

        if not scene_id or not tag_id:
            self.log("Missing scene_id or tag_id", "error")
            return

        try:
            # Get current tags
            result = self.stash_client.call_GQL(
                """
                query FindScene($id: ID!) {
                    findScene(id: $id) { tags { id } }
                }
                """,
                {"id": str(scene_id)},
            )

            current_ids = [int(t["id"]) for t in result["findScene"]["tags"]]
            if tag_id in current_ids:
                self.log("Tag already on scene", "info")
                return

            new_ids = current_ids + [tag_id]

            # Update scene
            self.stash_client.call_GQL(
                """
                mutation SceneUpdate($input: SceneUpdateInput!) {
                    sceneUpdate(input: $input) { id }
                }
                """,
                {"input": {"id": str(scene_id), "tag_ids": [str(i) for i in new_ids]}},
            )

            self.log(f"Applied tag {tag_id} to scene {scene_id}", "info")

        except Exception as e:
            self.log(f"Failed to apply tag: {e}", "error")

    def run_dismiss_suggested_tag(self, args: dict[str, Any]) -> None:
        """Dismiss a tag suggestion for a scene."""
        scene_id = int(args.get("scene_id", 0))
        tag_id = int(args.get("tag_id", 0))

        if not scene_id or not tag_id:
            self.log("Missing scene_id or tag_id", "error")
            return

        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            storage = EmbeddingStorage(model_key="siglip")
            storage.save_dismissed_tag(scene_id, tag_id)
            self.log(f"Dismissed tag {tag_id} for scene {scene_id}", "info")

        except Exception as e:
            self.log(f"Failed to dismiss tag: {e}", "error")

    def run_clear_dismissed_tags(self, args: dict[str, Any]) -> None:
        """Clear all dismissed tags for a scene."""
        scene_id = int(args.get("scene_id", 0))

        if not scene_id:
            self.log("Missing scene_id", "error")
            return

        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            storage = EmbeddingStorage(model_key="siglip")
            count = storage.clear_dismissed_tags(scene_id)
            self.log(f"Cleared {count} dismissed tags for scene {scene_id}", "info")

        except Exception as e:
            self.log(f"Failed to clear dismissed tags: {e}", "error")

    def run_find_duplicate_tags(self, args: dict[str, Any]) -> None:
        """Find duplicate tags using embedding similarity."""
        request_id = args.get("request_id", "")

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tasks.tag_dedup import FindDuplicateTagsTask

            plugin_settings = self.get_plugin_settings("stash-copilot")
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")

            if image_provider and image_model:
                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                )
                model_key = embedding_config.model_key
            else:
                model_key = "openclip:ViT-H-14"

            storage = EmbeddingStorage(model_key=model_key)
            task = FindDuplicateTagsTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
                model_key=model_key,
            )

            result = task.run()

            # Save result for frontend polling
            if request_id:
                assets_dir = os.path.join(PLUGIN_DIR, "assets")
                os.makedirs(assets_dir, exist_ok=True)
                result_path = os.path.join(assets_dir, f"tag_dedup_{request_id}.json")
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

            if result["status"] == "complete":
                self.log(f"Found {len(result['candidates'])} duplicate tag candidates", "info")
            else:
                self.log(f"Tag dedup: {result.get('error', 'unknown error')}", "warning")

        except Exception as e:
            self.error(f"Find duplicate tags failed: {e}")

    def run_merge_tags(self, args: dict[str, Any]) -> None:
        """Merge one tag into another."""
        keep_tag_id = int(args.get("keep_tag_id", 0))
        remove_tag_id = int(args.get("remove_tag_id", 0))
        request_id = args.get("request_id", "")

        if not keep_tag_id or not remove_tag_id:
            self.log("Missing keep_tag_id or remove_tag_id", "error")
            return

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tasks.tag_dedup import MergeTagsTask

            plugin_settings = self.get_plugin_settings("stash-copilot")
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")

            if image_provider and image_model:
                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                )
                model_key = embedding_config.model_key
            else:
                model_key = "openclip:ViT-H-14"

            storage = EmbeddingStorage(model_key=model_key)
            task = MergeTagsTask(
                stash=self.stash_client,
                storage=storage,
                log_callback=self.log,
                model_key=model_key,
            )

            result = task.run(keep_tag_id=keep_tag_id, remove_tag_id=remove_tag_id)

            # Save result for frontend
            if request_id:
                assets_dir = os.path.join(PLUGIN_DIR, "assets")
                os.makedirs(assets_dir, exist_ok=True)
                result_path = os.path.join(assets_dir, f"tag_merge_{request_id}.json")
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

            if result["status"] == "complete":
                self.log(f"Merged tags: {result['scenes_updated']} scenes updated", "info")
            else:
                self.log(f"Tag merge error: {result.get('error')}", "warning")

        except Exception as e:
            self.error(f"Merge tags failed: {e}")

    def run_dismiss_tag_merge(self, args: dict[str, Any]) -> None:
        """Dismiss a tag merge candidate (not duplicates)."""
        tag_a_name = args.get("tag_a_name", "")
        tag_b_name = args.get("tag_b_name", "")
        request_id = args.get("request_id", "")

        if not tag_a_name or not tag_b_name:
            self.log("Missing tag_a_name or tag_b_name", "error")
            return

        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            storage = EmbeddingStorage()
            storage.save_dismissed_tag_merge(tag_a_name, tag_b_name)
            self.log(f"Dismissed merge: {tag_a_name} / {tag_b_name}", "info")

            # Save confirmation for frontend
            if request_id:
                assets_dir = os.path.join(PLUGIN_DIR, "assets")
                os.makedirs(assets_dir, exist_ok=True)
                result_path = os.path.join(assets_dir, f"tag_dismiss_{request_id}.json")
                with open(result_path, "w") as f:
                    json.dump({"status": "complete"}, f)

        except Exception as e:
            self.log(f"Failed to dismiss tag merge: {e}", "error")

    def run_prepare_labeling_session(self, args: dict[str, Any]) -> None:
        """Prepare a labeling session with uncertainty-sampled frames."""
        request_id = args.get("request_id", "")
        batch_size = int(args.get("batch_size", 200))

        self.log(f"Preparing labeling session (batch_size={batch_size}), request_id={request_id}", "info")

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
            tag_vocab = TagVocabulary(
                storage=storage, model_key=model_key, log_callback=self.log
            )
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
            result_file.write_text(json.dumps({
                "status": "error",
                "session_id": "",
                "batch": [],
                "vocabulary": [],
                "error": str(e),
            }))

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
            result_file.write_text(json.dumps({
                "status": "error",
                "export_path": "",
                "total_images": 0,
                "total_tags": 0,
                "error": str(e),
            }))

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
            result_file.write_text(json.dumps({
                "status": "complete",
                "sessions": sessions,
            }, indent=2))

        except Exception as e:
            self.log(f"Error listing sessions: {e}", "error")

    def run_preference_trainer(self, task_name: str, args: dict[str, Any]) -> None:
        """Run preference training tasks (start, compare, swipe, end)."""
        try:
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.preferences.session import PreferenceSessionManager
            from stash_ai.preferences.types import (
                PreferenceSessionConfig,
                SwipeDirection,
            )

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get model_key from image embedding settings
            from stash_ai.embeddings.config import EmbeddingConfig

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            storage = EmbeddingStorage(model_key=model_key)

            manager = PreferenceSessionManager(
                storage=storage,
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            request_id = args.get("request_id", "")
            response = None

            if task_name == "preference_start":
                # Parse exploration_rate from args (0.0 to 1.0)
                exploration_str = args.get("exploration_rate", "0.2")
                try:
                    exploration_rate = max(0.0, min(1.0, float(exploration_str)))
                except ValueError:
                    exploration_rate = 0.2

                # Parse pure_random flag (bypasses cluster-based bootstrapping)
                pure_random_str = args.get("pure_random", "false").lower()
                pure_random = pure_random_str in ("true", "1", "yes")

                session_config = PreferenceSessionConfig(
                    mode=args.get("session_mode", "swipe"),
                    batch_size=int(args.get("batch_size", "20")),
                    model_key=model_key,
                    exploration_rate=exploration_rate,
                    pure_random=pure_random,
                )
                response = manager.start_session(session_config)
                self.log(
                    f"Preference session started: {response.session_id}, "
                    f"{len(response.pairs)} pairs, phase={response.phase}",
                    "info",
                )

            elif task_name == "preference_compare":
                session_id = args.get("session_id", "")
                scene_a_id = int(args.get("scene_a_id", "0"))
                scene_b_id = int(args.get("scene_b_id", "0"))
                winner_id = int(args.get("winner_id", "0"))
                rt_str = args.get("response_time_ms", "")
                response_time = int(rt_str) if rt_str else None

                response = manager.record_comparison(
                    session_id=session_id,
                    scene_a_id=scene_a_id,
                    scene_b_id=scene_b_id,
                    winner_id=winner_id,
                    response_time_ms=response_time,
                )
                self.log(
                    f"Comparison recorded: {winner_id} preferred, "
                    f"confidence={response.convergence.confidence_pct if response.convergence else '?'}%",
                    "info",
                )

            elif task_name == "preference_swipe":
                session_id = args.get("session_id", "")
                scene_id = int(args.get("scene_id", "0"))
                direction = SwipeDirection(args.get("direction", "skip"))
                rt_str = args.get("response_time_ms", "")
                response_time = int(rt_str) if rt_str else None

                response = manager.record_swipe(
                    session_id=session_id,
                    scene_id=scene_id,
                    direction=direction,
                    response_time_ms=response_time,
                )
                self.log(
                    f"Swipe recorded: scene {scene_id} = {direction.value}",
                    "info",
                )

            elif task_name == "preference_end":
                session_id = args.get("session_id", "")
                response = manager.end_session(session_id)
                self.log(
                    f"Session ended: {response.n_comparisons} comparisons, "
                    f"confidence={response.convergence.confidence_pct if response.convergence else '?'}%",
                    "info",
                )

            elif task_name == "preference_stats":
                response = manager.get_model_stats()
                self.log(
                    f"Preference stats: {response.n_comparisons} comparisons, "
                    f"phase={response.phase}",
                    "info",
                )

            elif task_name == "preference_reset":
                response = manager.reset_model()
                self.log("Preference model reset complete", "info")

            # Save results under request_id for frontend polling
            if response and request_id:
                self._save_preference_result(response, request_id)

        except ImportError as e:
            self.error(f"Failed to import preference modules: {e}")
        except Exception as e:
            self.error(f"Preference trainer failed: {e}")

    def _save_preference_result(self, response: Any, request_id: str) -> None:
        """Save preference trainer response as JSON for frontend polling."""
        import json as json_module
        from dataclasses import fields

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(assets_dir, exist_ok=True)

        filepath = os.path.join(assets_dir, f"preference_trainer_{request_id}.json")

        def _serialize(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, list):
                return [_serialize(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            # Handle numpy types
            try:
                import numpy as np

                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
            except ImportError:
                pass
            return obj

        try:
            data = _serialize(response)
            with open(filepath, "w", encoding="utf-8") as f:
                json_module.dump(data, f, indent=2, default=str)
            self.log(f"Results saved for frontend: preference_trainer_{request_id}.json", "debug")
        except (OSError, TypeError) as e:
            self.log(f"Failed to save preference result for frontend: {e}", "warning")

    def run_preference_recs(self, args: dict[str, Any]) -> None:
        """Run preference-based recommendations from the trained preference model."""
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.preference_recs import PreferenceRecsTask

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get model_key from image embedding settings
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"
            if image_provider and image_model:
                config = EmbeddingConfig(provider=image_provider, model=image_model)
                model_key = config.model_key

            limit = int(args.get("limit", "24"))
            mode = args.get("rec_mode", "discover")
            request_id = args.get("request_id", "")

            task = PreferenceRecsTask(
                stash=self.stash_client,
                log_callback=self.log,
                progress_callback=self.progress,
                model_key=model_key,
            )

            task.run(limit=limit, mode=mode, request_id=request_id)

        except ImportError as e:
            self.error(f"Failed to import preference recs modules: {e}")
        except Exception as e:
            self.error(f"Preference recs failed: {e}")

    def run_ask(self, args: dict[str, Any]) -> None:
        """
        Run the AI Ask task - answer questions using tools.

        Args:
            args: Task arguments containing the question and LLM settings
        """
        try:
            from stash_ai.config import get_text_llm_settings
            from stash_ai.tasks.ask import AskTask

            question = args.get("question", "")
            if not question:
                self.error("No question provided")
                return

            self.log(f"AI Ask: {question}", "info")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get text LLM settings
            text_llm = get_text_llm_settings(plugin_settings, args)
            self.log(f"Using LLM: {text_llm.provider}/{text_llm.model}", "info")

            llm_config = text_llm.to_config()

            # Create and run the task
            task = AskTask(
                stash=self.stash_client,
                llm_config=llm_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            answer = task.run(question)

            # Output the answer
            self.log("=" * 50, "info")
            self.log("AI ANSWER", "info")
            self.log("=" * 50, "info")
            for line in answer.split("\n"):
                self.log(line, "info")
            self.log("=" * 50, "info")

        except ImportError as e:
            self.error(f"Failed to import Stash AI modules: {e}")
        except ConnectionError as e:
            self.error(f"Connection error: {e}")
        except RuntimeError as e:
            self.error(f"Task failed: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_chat(self, args: dict[str, Any]) -> None:
        """
        Run the Chat task - multi-turn conversation with tool transparency.

        Args:
            args: Task arguments containing message and optional conversation_id
        """
        try:
            from stash_ai.config import get_text_llm_settings
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.chat import ChatTask

            message = args.get("message", "")
            conversation_id = args.get("conversation_id")

            if not message:
                self.error("No message provided")
                return

            self.log(f"Chat message: {message[:100]}...", "info")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get text LLM settings
            text_llm = get_text_llm_settings(plugin_settings, args)
            self.log(f"Using LLM: {text_llm.provider}/{text_llm.model}", "info")

            # Chat needs higher max_tokens than default (1024) because tool calls
            # may include large argument payloads (e.g., hundreds of scene IDs).
            # With 1024 tokens, the LLM response gets truncated mid-tool-call,
            # causing arguments to arrive as empty strings.
            llm_config = text_llm.to_config(max_tokens=8192)

            # Get image embedding config for text-based scene search
            embedding_config = None
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if image_provider and image_model:
                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device=image_device,
                )
                self.log(f"Text search enabled with: {image_provider}/{image_model}", "debug")

            # Parse excluded tags (comma-separated string to list)
            excluded_tags_str = plugin_settings.get("excluded_tags", "")
            excluded_tags = (
                [tag.strip() for tag in excluded_tags_str.split(",") if tag.strip()]
                if excluded_tags_str
                else []
            )

            if excluded_tags:
                self.log(f"Excluding tags from AI tools: {excluded_tags}", "debug")

            # Create and run the task
            task = ChatTask(
                stash=self.stash_client,
                llm_config=llm_config,
                embedding_config=embedding_config,
                excluded_tags=excluded_tags,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            response = task.run(message, conversation_id)

            # Output the response
            self.log("Chat response generated", "info")
            for line in response.split("\n"):
                self.log(line, "info")

        except ImportError as e:
            self.error(f"Failed to import Stash AI modules: {e}")
        except ConnectionError as e:
            self.error(f"Connection error: {e}")
        except RuntimeError as e:
            self.error(f"Task failed: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_clear_chat(self) -> None:
        """
        Clear the chat conversation history.
        """
        try:
            import os

            # Get the chat history file path
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            history_file = os.path.join(plugin_dir, "assets", "chat_history.json")

            if os.path.exists(history_file):
                os.remove(history_file)
                self.log("Chat history cleared", "info")
            else:
                self.log("No chat history to clear", "info")

        except Exception as e:
            self.error(f"Failed to clear chat history: {e}")

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
        """
        Run the scene embedding generation task.

        Args:
            args: Task arguments containing optional scene_id and force flag
        """
        self.log("=== EMBED SCENES TASK STARTED ===", "info")
        self.log(f"Args received: {args}", "debug")

        try:
            import json as json_module

            from stash_ai.config import get_text_llm_settings, get_vision_llm_settings
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.embed_scenes import EmbedConfig, EmbedScenesTask

            self.log("Initializing scene embedding generation...", "info")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")
            self.log(f"Plugin settings: {plugin_settings}", "debug")

            # Image embedding config (CLIP/OpenCLIP/SigLIP) - check first to determine if VLM is needed
            image_embedding_config = None
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            self.log(
                f"Image embedding config: provider={image_provider}, model={image_model}, device={image_device}",
                "info",
            )
            use_clip = bool(image_provider and image_model)

            if use_clip:
                image_embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device=image_device,
                )
                self.log(
                    f"Using CLIP-style embeddings: {image_provider}/{image_model} on {image_device}",
                    "info",
                )
                self.log(
                    "VLM will NOT be used for visual embeddings (CLIP embeds images directly)",
                    "info",
                )
            else:
                self.log(
                    "No image embedder configured - will use VLM text descriptions for visual embeddings",
                    "info",
                )

            # Get text LLM settings (for base_url used by Ollama embedding model)
            text_llm = get_text_llm_settings(plugin_settings, args)

            # Text embedding model config (for metadata: performers, tags, studio)
            # Only needed if NOT using CLIP (CLIP can embed text too)
            embedding_model = (
                args.get("embedding_model")
                or plugin_settings.get("embedding_model")
                or "nomic-embed-text"
            )

            if use_clip:
                # When using CLIP, we use CLIP for both image and text embeddings
                # No Ollama text embedding needed
                self.log(
                    f"Using {image_provider}/{image_model} for both visual AND metadata embeddings",
                    "info",
                )
                embedding_config = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device=image_device,
                )
            else:
                self.log(f"Using Ollama text embedding model: {embedding_model}", "info")
                embedding_config = EmbeddingConfig(
                    provider="ollama",
                    model=embedding_model,
                    base_url=text_llm.base_url,
                )

            # Get vision LLM settings - only needed as fallback if CLIP not configured
            vision_llm = get_vision_llm_settings(plugin_settings, args)
            vlm_config = vision_llm.to_config()
            if not use_clip:
                self.log(
                    f"VLM for visual descriptions: {vision_llm.provider}/{vision_llm.model}", "info"
                )

            # Embedding task config
            visual_weight = float(
                args.get("visual_weight") or plugin_settings.get("embed_visual_weight") or "0.7"
            )

            # Get frame extraction settings (shared with vision task)
            frame_interval = float(plugin_settings.get("vision_frame_interval") or "10")
            fps_rate = 1.0 / frame_interval  # Convert interval to fps
            min_frames = int(plugin_settings.get("vision_min_frames") or "1")
            max_frames = int(plugin_settings.get("vision_max_frames") or "0")

            # Parallel processing settings
            num_workers = int(plugin_settings.get("embed_num_workers") or "2")

            embed_config = EmbedConfig(
                visual_weight=visual_weight,
                use_cached_descriptions=True,
                fps_rate=fps_rate,
                min_frames=min_frames,
                max_frames=max_frames,
                num_workers=num_workers,
            )

            self.log(f"Visual embedding weight: {visual_weight}", "info")
            self.log(
                f"Frame extraction: interval={frame_interval}s (fps={fps_rate}), min={min_frames}, max={max_frames}",
                "info",
            )
            self.log(f"Scene workers: {num_workers}", "info")

            # Create task
            task = EmbedScenesTask(
                stash=self.stash_client,
                vlm_config=vlm_config,
                embedding_config=embedding_config,
                image_embedding_config=image_embedding_config,
                embed_config=embed_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            # Check for single scene or batch mode
            scene_id = args.get("scene_id")
            force = args.get("force", "").lower() == "true"

            if scene_id:
                self.log(f"Embedding single scene: {scene_id}", "info")
                result = task.embed_scene(int(scene_id), force=force, success_tag="Embedded")
            else:
                self.log("Embedding all scenes...", "info")
                result = task.embed_all(force=force, success_tag="Embedded")

            # Output result
            self.log("=" * 50, "info")
            self.log("EMBEDDING RESULT", "info")
            self.log("=" * 50, "info")
            self.log(json_module.dumps(result, indent=2), "info")
            self.log("=" * 50, "info")

        except ImportError as e:
            self.error(f"Failed to import embedding modules: {e}")
        except ConnectionError as e:
            self.error(f"Connection error: {e}")
        except RuntimeError as e:
            self.error(f"Task failed: {e}")
        except Exception as e:
            self.error(f"Unexpected error: {e}")

    def run_find_similar(self, args: dict[str, Any]) -> None:
        """
        Find scenes similar to a given scene using embeddings.

        Args:
            args: Task arguments containing scene_id and optional limit
        """
        try:
            import json as json_module

            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

            scene_id = args.get("scene_id")
            if not scene_id:
                self.error("scene_id is required")
                self._write_similar_result(
                    scene_id or "unknown", {"status": "error", "error": "scene_id is required"}
                )
                return

            limit = int(args.get("limit", 10))  # Results per page
            offset = int(args.get("offset", 0))  # Pagination offset
            min_similarity = float(args.get("min_similarity", 0.0))
            exclude_common_performers = (
                args.get("exclude_common_performers", "false").lower() == "true"
            )
            request_id = args.get("request_id", "")  # Unique request ID for frontend validation

            # Visual weight for dynamic embedding blend (0.0-1.0)
            # If not provided, uses stored composite embedding
            visual_weight_str = args.get("visual_weight", "")
            visual_weight: float | None = None
            if visual_weight_str:
                try:
                    visual_weight = float(visual_weight_str)
                    # Clamp to valid range
                    visual_weight = max(0.0, min(1.0, visual_weight))
                except ValueError:
                    self.log(
                        f"Invalid visual_weight: {visual_weight_str}, using default", "warning"
                    )

            # Parse exclusion filter names (comma-separated)
            exclude_performer_names_str = args.get("exclude_performer_names", "")
            exclude_tag_names_str = args.get("exclude_tag_names", "")
            exclude_performer_names = [
                n.strip() for n in exclude_performer_names_str.split(",") if n.strip()
            ]
            exclude_tag_names = [n.strip() for n in exclude_tag_names_str.split(",") if n.strip()]

            filter_desc = ""
            if exclude_common_performers:
                filter_desc += " (excluding common performers)"
            if exclude_performer_names:
                filter_desc += f" (excluding performers: {', '.join(exclude_performer_names)})"
            if exclude_tag_names:
                filter_desc += f" (excluding tags: {', '.join(exclude_tag_names)})"

            self.log(
                f"Finding scenes similar to: {scene_id} (offset={offset}, limit={limit}){filter_desc}",
                "info",
            )

            # Get model_key from plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")
            from stash_ai.embeddings.config import EmbeddingConfig as EmbedCfg

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            model_key = "siglip"  # Default
            if image_provider and image_model:
                cfg = EmbedCfg(provider=image_provider, model=image_model)
                model_key = cfg.model_key

            self.log(f"Using embedding model: {model_key}", "info")
            storage = EmbeddingStorage(model_key=model_key)

            # Get query embedding
            query_record = storage.get_embedding(int(scene_id))
            if not query_record:
                error_msg = f"Scene {scene_id} has no embedding. Run embed_scenes task first."
                self.error(error_msg)
                self._write_similar_result(scene_id, {"status": "error", "error": error_msg})
                return

            # Get source scene's performer IDs if filtering by performers
            source_performer_ids: set[int] = set()
            exclude_performer_ids: set[int] = set()
            exclude_tag_ids: set[int] = set()
            needs_db_filtering = (
                exclude_common_performers or exclude_performer_names or exclude_tag_names
            )

            if needs_db_filtering:
                db_path = get_stash_db_path()
                if db_path.exists():
                    conn = get_readonly_connection(db_path)
                    cursor = conn.cursor()

                    # Get source scene's performers for "different performers" tab
                    if exclude_common_performers:
                        cursor.execute(
                            "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                            (int(scene_id),),
                        )
                        source_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                        self.log(
                            f"Source scene has {len(source_performer_ids)} performers", "debug"
                        )

                    # Look up performer IDs for exclude_performer_names
                    if exclude_performer_names:
                        placeholders = ",".join("?" * len(exclude_performer_names))
                        cursor.execute(
                            f"SELECT id FROM performers WHERE name IN ({placeholders}) COLLATE NOCASE",
                            exclude_performer_names,
                        )
                        exclude_performer_ids = {row["id"] for row in cursor.fetchall()}
                        self.log(
                            f"Found {len(exclude_performer_ids)} performer IDs to exclude", "debug"
                        )

                    # Look up tag IDs for exclude_tag_names
                    if exclude_tag_names:
                        placeholders = ",".join("?" * len(exclude_tag_names))
                        cursor.execute(
                            f"SELECT id FROM tags WHERE name IN ({placeholders}) COLLATE NOCASE",
                            exclude_tag_names,
                        )
                        exclude_tag_ids = {row["id"] for row in cursor.fetchall()}
                        self.log(f"Found {len(exclude_tag_ids)} tag IDs to exclude", "debug")

                    conn.close()

            # Find similar - fetch more if filtering to ensure we find enough results
            fetch_limit = 500 if needs_db_filtering else limit
            fetch_offset = 0 if needs_db_filtering else offset

            # Build find_similar arguments
            find_similar_kwargs = {
                "query_embedding": query_record["composite_embedding"],
                "limit": fetch_limit + offset if needs_db_filtering else limit,
                "offset": fetch_offset,
                "exclude_scene_ids": [int(scene_id)],
                "min_similarity": min_similarity,
            }

            # Add dynamic weight parameters if visual_weight is provided
            if visual_weight is not None:
                query_visual = query_record.get("visual_embedding")
                query_metadata = query_record.get("metadata_embedding")
                if query_visual and query_metadata:
                    find_similar_kwargs["visual_weight"] = visual_weight
                    find_similar_kwargs["query_visual_embedding"] = query_visual
                    find_similar_kwargs["query_metadata_embedding"] = query_metadata
                    self.log(f"Using dynamic visual weight: {visual_weight:.2f}", "info")
                else:
                    self.log(
                        f"Query scene missing separate embeddings (visual={query_visual is not None}, "
                        f"metadata={query_metadata is not None}). Re-run 'Embed Scenes' to enable dynamic weights.",
                        "warning",
                    )

            results = storage.find_similar(**find_similar_kwargs)

            # Apply database-level filtering (performers and tags)
            if needs_db_filtering:
                db_path = get_stash_db_path()
                if db_path.exists():
                    conn = get_readonly_connection(db_path)
                    cursor = conn.cursor()

                    filtered_results = []
                    for r in results:
                        should_exclude = False

                        # Check exclude_common_performers (different performers tab)
                        if exclude_common_performers and source_performer_ids:
                            cursor.execute(
                                "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                                (r.scene_id,),
                            )
                            scene_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                            if source_performer_ids.intersection(scene_performer_ids):
                                should_exclude = True

                        # Check exclude_performer_ids (manual exclusion filter)
                        if not should_exclude and exclude_performer_ids:
                            cursor.execute(
                                "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                                (r.scene_id,),
                            )
                            scene_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                            if exclude_performer_ids.intersection(scene_performer_ids):
                                should_exclude = True

                        # Check exclude_tag_ids (manual tag exclusion filter)
                        if not should_exclude and exclude_tag_ids:
                            cursor.execute(
                                "SELECT tag_id FROM scenes_tags WHERE scene_id = ?", (r.scene_id,)
                            )
                            scene_tag_ids = {row["tag_id"] for row in cursor.fetchall()}
                            if exclude_tag_ids.intersection(scene_tag_ids):
                                should_exclude = True

                        if not should_exclude:
                            filtered_results.append(r)

                    conn.close()
                    self.log(
                        f"Filtered {len(results)} results to {len(filtered_results)} after applying exclusions",
                        "debug",
                    )
                    results = filtered_results

                # Apply offset and limit to filtered results
                results = results[offset : offset + limit]

            # Output results
            self.log("=" * 50, "info")
            self.log("SIMILAR SCENES", "info")
            self.log("=" * 50, "info")

            if not results:
                self.log("No similar scenes found", "info")
            else:
                for r in results:
                    self.log(f"Scene {r.scene_id}: similarity={r.similarity:.4f}", "info")
                    if r.visual_description:
                        preview = (
                            r.visual_description[:100] + "..."
                            if len(r.visual_description) > 100
                            else r.visual_description
                        )
                        self.log(f"  Description: {preview}", "info")

            self.log("=" * 50, "info")

            # Fetch full scene details from SQLite for all results
            scene_details = self._get_scene_details_batch([r.scene_id for r in results])

            # Build result data with embedded scene details
            result_data = []
            for r in results:
                scene = scene_details.get(r.scene_id, {})
                result_data.append(
                    {"scene_id": r.scene_id, "similarity": r.similarity, "scene": scene}
                )

            # has_more is true if we got a full page of results
            has_more = len(results) == limit

            # Write results to JSON file for frontend polling
            # Include filter mode so frontend knows which tab these results belong to
            filter_mode = "different-performers" if exclude_common_performers else "all"
            result_json = {
                "status": "complete",
                "query_scene_id": int(scene_id),
                "model_key": model_key,
                "results": result_data,
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "filter_mode": filter_mode,
                "request_id": request_id,
            }
            # Include visual_weight in response if it was used
            if visual_weight is not None:
                result_json["visual_weight"] = visual_weight

            self._write_similar_result(scene_id, result_json)

            # Also output as JSON for programmatic access
            self.log("JSON_RESULT:" + json_module.dumps(result_data), "debug")

        except ImportError as e:
            self.error(f"Failed to import embedding modules: {e}")
            self._write_similar_result(
                args.get("scene_id", "unknown"),
                {"status": "error", "error": f"Failed to import embedding modules: {e}"},
            )
        except Exception as e:
            self.error(f"Unexpected error: {e}")
            self._write_similar_result(
                args.get("scene_id", "unknown"),
                {"status": "error", "error": f"Unexpected error: {e}"},
            )

    def _get_scene_details_batch(self, scene_ids: list[int]) -> dict[int, dict[str, Any]]:
        """
        Fetch scene details for multiple scenes from SQLite in a single query.

        Args:
            scene_ids: List of scene IDs to fetch

        Returns:
            Dict mapping scene_id to scene details dict
        """
        if not scene_ids:
            return {}

        from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

        db_path = get_stash_db_path()
        if not db_path.exists():
            self.log(f"Database not found at {db_path}", "warning")
            return {}

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            # Build placeholders for IN clause
            placeholders = ",".join("?" * len(scene_ids))
            scene_ids_tuple = tuple(scene_ids)

            # Fetch scene base data
            self.log(
                f"Fetching details for {len(scene_ids)} scene IDs: {scene_ids[:5]}...", "debug"
            )
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
            self.log(f"Found {len(rows)} scenes in database", "debug")

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
            self.log(f"Fetched details for {len(scenes)} scenes from SQLite", "debug")
            return scenes

        except Exception as e:
            import traceback

            self.log(f"Error fetching scene details: {e}", "error")
            self.log(f"Traceback: {traceback.format_exc()}", "error")
            return {}

    def run_frame_analysis(self, args: dict[str, Any]) -> None:
        """
        Run frame embedding analysis for a scene.

        Analyzes frame-to-frame similarity within a scene using dimensionality
        reduction (PCA, t-SNE, UMAP) and selects representative frames.

        Args:
            args: Task arguments containing scene_id and optional parameters
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.frame_analysis import (
                FrameAnalysisConfig,
                FrameAnalysisTask,
            )
            from stash_ai.tasks.frame_extractor import FrameExtractionConfig

            scene_id = args.get("scene_id")
            if not scene_id:
                self.error("scene_id is required")
                return

            selection_method_arg = args.get("selection_method", "not provided")
            self.log(f"Starting frame analysis for scene {scene_id}...", "info")
            self.log(f"Selection method from args: {selection_method_arg}", "debug")

            # Get plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Image embedding config (required for frame analysis)
            image_provider = plugin_settings.get("image_embedding_provider") or "openclip"
            image_model = plugin_settings.get("image_embedding_model") or "ViT-B-32"
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            image_embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )
            self.log(
                f"Using image embedder: {image_provider}/{image_model} on {image_device}", "info"
            )

            # Frame extraction config
            frame_interval = float(
                args.get("frame_interval") or plugin_settings.get("vision_frame_interval") or "10"
            )
            min_frames = int(
                args.get("min_frames") or plugin_settings.get("vision_min_frames") or "1"
            )

            frame_config = FrameExtractionConfig(
                interval_seconds=frame_interval,
                min_frames=min_frames,
                max_frames=0,  # No limit for analysis
                frame_width=640,
            )

            # Analysis config
            n_representative = int(
                args.get("n_representative")
                or plugin_settings.get("frame_analysis_n_frames")
                or "8"
            )
            selection_method_str = (
                args.get("selection_method")
                or plugin_settings.get("frame_analysis_method")
                or "kmeans"
            )
            # Validate selection method
            valid_methods = ("kmeans", "maximin", "coverage")
            if selection_method_str not in valid_methods:
                self.log(
                    f"Invalid selection method '{selection_method_str}', using 'kmeans'", "warning"
                )
                selection_method_str = "kmeans"

            self.log(f"Using selection method: {selection_method_str}", "info")

            # Dynamic frame count settings
            dynamic_frame_count_str = (
                plugin_settings.get("frame_analysis_dynamic") or "true"
            ).lower()
            dynamic_frame_count = dynamic_frame_count_str in ("true", "1", "yes")

            frames_per_minute = float(
                plugin_settings.get("frame_analysis_frames_per_minute") or "1.0"
            )
            dynamic_min_frames = int(plugin_settings.get("frame_analysis_min_frames") or "4")
            dynamic_max_frames = int(plugin_settings.get("frame_analysis_max_frames") or "50")

            # Compare methods setting
            compare_methods_str = (plugin_settings.get("frame_analysis_compare") or "true").lower()
            compare_methods = compare_methods_str in ("true", "1", "yes")

            # Type narrowing for Literal type
            from typing import Literal, cast

            selection_method = cast(
                "Literal['kmeans', 'maximin', 'coverage']", selection_method_str
            )

            analysis_config = FrameAnalysisConfig(
                n_representative=n_representative,
                selection_method=selection_method,
                reduction_methods=["pca", "tsne", "umap"],
                dynamic_frame_count=dynamic_frame_count,
                frames_per_minute=frames_per_minute,
                min_frames=dynamic_min_frames,
                max_frames=dynamic_max_frames,
                compare_methods=compare_methods,
            )

            if dynamic_frame_count:
                self.log(
                    f"Analysis config: dynamic frames ({frames_per_minute}/min, "
                    f"min={dynamic_min_frames}, max={dynamic_max_frames}), "
                    f"method={selection_method}",
                    "info",
                )
            else:
                self.log(
                    f"Analysis config: {n_representative} frames, method={selection_method}", "info"
                )

            # Create and run task
            task = FrameAnalysisTask(
                stash=self.stash_client,
                image_embedding_config=image_embedding_config,
                analysis_config=analysis_config,
                frame_config=frame_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            result = task.run(int(scene_id))

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

        except ImportError as e:
            self.error(f"Failed to import frame analysis modules: {e}")
        except Exception as e:
            import traceback

            self.error(f"Frame analysis error: {e}")
            self.log(f"Traceback: {traceback.format_exc()}", "debug")

    def check_frame_analysis(self, args: dict[str, Any]) -> None:
        """
        Check if frame analysis results exist for a scene.

        Used by the UI to poll for cached results or task completion status.

        Args:
            args: Task arguments containing scene_id
        """
        import json as json_module
        import os

        scene_id = args.get("scene_id")
        if not scene_id:
            print(json_module.dumps({"status": "error", "error": "scene_id is required"}))
            return

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(plugin_dir, "assets", f"frame_analysis_{scene_id}")

        # Check for status file first
        status_file = os.path.join(output_dir, "analysis_status.json")
        summary_file = os.path.join(output_dir, "analysis_summary.json")

        try:
            # If summary exists, return complete status with results
            if os.path.exists(summary_file):
                with open(summary_file) as f:
                    results = json_module.load(f)
                print(
                    json_module.dumps(
                        {
                            "status": "complete",
                            "results": results,
                        }
                    )
                )
                return

            # Check status file for running/error state
            if os.path.exists(status_file):
                with open(status_file) as f:
                    status_data = json_module.load(f)
                print(json_module.dumps(status_data))
                return

            # No results or status file
            print(json_module.dumps({"status": "not_found"}))

        except Exception as e:
            print(
                json_module.dumps(
                    {
                        "status": "error",
                        "error": str(e),
                    }
                )
            )

    def start_frame_analysis(self, args: dict[str, Any]) -> None:
        """
        Start frame analysis for a scene (called from UI).

        This runs the analysis synchronously but writes status files
        so the UI can poll for progress.

        Args:
            args: Task arguments containing scene_id
        """
        import json as json_module
        import os

        scene_id = args.get("scene_id")
        if not scene_id:
            print(json_module.dumps({"status": "error", "error": "scene_id is required"}))
            return

        # Create output directory and write initial status
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(plugin_dir, "assets", f"frame_analysis_{scene_id}")
        os.makedirs(output_dir, exist_ok=True)

        status_file = os.path.join(output_dir, "analysis_status.json")
        summary_file = os.path.join(output_dir, "analysis_summary.json")

        try:
            # Clear old results so polling doesn't find stale data
            if os.path.exists(summary_file):
                os.remove(summary_file)

            # Write running status
            with open(status_file, "w") as f:
                json_module.dump(
                    {
                        "status": "running",
                        "scene_id": scene_id,
                    },
                    f,
                )

            # Run the actual analysis
            self.run_frame_analysis(args)

            # Return started status (UI will poll for completion)
            print(
                json_module.dumps(
                    {
                        "status": "started",
                        "scene_id": scene_id,
                    }
                )
            )

        except Exception as e:
            # Write error status
            with open(status_file, "w") as f:
                json_module.dump(
                    {
                        "status": "error",
                        "scene_id": scene_id,
                        "error": str(e),
                    },
                    f,
                )
            print(
                json_module.dumps(
                    {
                        "status": "error",
                        "error": str(e),
                    }
                )
            )

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
        """
        Search scenes by natural language text query (semantic search).

        Args:
            args: Task arguments containing:
                - query: Text query string (required)
                - limit: Maximum results (default 24)
                - offset: Pagination offset (default 0)
                - request_id: Unique request ID for frontend validation
                - model_key: Optional model key to search (e.g., "openclip:ViT-H-14")
                             If not provided, uses currently configured model from settings
        """

        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.provider import get_embedding_provider
            from stash_ai.embeddings.storage import EmbeddingStorage

            query = args.get("query", "").strip()
            if not query:
                self._write_search_result("", {"status": "error", "error": "Query is required"})
                return

            limit = int(args.get("limit", 24))
            offset = int(args.get("offset", 0))
            request_id = args.get("request_id", "")
            requested_model_key = args.get("model_key", "").strip()
            frame_search = args.get("frame_search", "").lower() == "true"

            self.log(
                f"Searching scenes for: '{query}' (limit={limit}, offset={offset}, frame_search={frame_search})",
                "info",
            )

            # Get device setting from plugin settings
            plugin_settings = self.get_plugin_settings("stash-copilot")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            # If model_key is provided, use it; otherwise fall back to plugin settings
            if requested_model_key:
                # Create config from model_key
                embedding_config = EmbeddingConfig.from_model_key(
                    requested_model_key, device=image_device
                )
                model_key = requested_model_key
                self.log(f"Using requested model: {model_key}", "info")
            else:
                # Fall back to plugin settings
                image_provider = plugin_settings.get("image_embedding_provider")
                image_model = plugin_settings.get("image_embedding_model")

                if not image_provider or not image_model:
                    self._write_search_result(
                        request_id,
                        {
                            "status": "error",
                            "error": "Image embedding provider not configured. Set up in Plugin Settings.",
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
            storage = EmbeddingStorage(model_key=model_key)

            # Check for embeddings
            stats = storage.get_stats()
            if stats["total_embeddings"] == 0:
                self._write_search_result(
                    request_id,
                    {
                        "status": "error",
                        "error": "No scene embeddings found. Run 'Embed All Scenes' task first.",
                    },
                )
                return

            # Frame-level search using FAISS index
            if frame_search:
                import numpy as np

                from stash_ai.embeddings.frame_search import FrameSearchIndex

                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                assets_dir = os.path.join(plugin_dir, "assets")

                frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

                if not frame_index.exists:
                    self._write_search_result(
                        request_id,
                        {
                            "status": "error",
                            "error": f"Frame search index not built for model '{model_key}'. Run 'Build Frame Search Index' task first.",
                        },
                    )
                    return

                # Embed the query text
                try:
                    result = embedder.embed_text(query)
                    query_embedding = np.array(result["embedding"], dtype=np.float32)
                except Exception as e:
                    self._write_search_result(
                        request_id, {"status": "error", "error": f"Failed to embed query: {e!s}"}
                    )
                    return

                # Search frames
                frame_matches = frame_index.search(query_embedding, top_k=2000)

                # Aggregate to scenes
                scene_matches = frame_index.aggregate_to_scenes(frame_matches)

                # Apply pagination
                paginated = scene_matches[offset : offset + limit]

                # Fetch scene details
                scene_details = self._get_scene_details_batch([m.scene_id for m in paginated])

                # Build result data with frame info
                result_data = []
                for m in paginated:
                    scene = scene_details.get(m.scene_id, {})
                    # Format frame path
                    frame_path = (
                        f"embedded_frames/scene_{m.scene_id}/frame_{m.best_frame_index:04d}.jpg"
                    )
                    result_data.append(
                        {
                            "scene_id": m.scene_id,
                            "similarity": m.similarity,
                            "best_frame_index": m.best_frame_index,
                            "best_timestamp": m.best_timestamp,
                            "frame_path": frame_path,
                            "scene": scene,
                        }
                    )

                has_more = len(scene_matches) > (offset + limit)

                self._write_search_result(
                    request_id,
                    {
                        "status": "complete",
                        "query": query,
                        "model_key": model_key,
                        "frame_search": True,
                        "results": result_data,
                        "offset": offset,
                        "limit": limit,
                        "has_more": has_more,
                        "request_id": request_id,
                        "total_scenes": len(scene_matches),
                    },
                )

                self.log(f"Frame search complete: {len(result_data)} scenes for '{query}'", "info")
                return

            # Embed the query text
            try:
                result = embedder.embed_text(query)
                query_embedding = result["embedding"]
            except Exception as e:
                self._write_search_result(
                    request_id, {"status": "error", "error": f"Failed to embed query: {e!s}"}
                )
                return

            # Find similar scenes
            # Note: Text-to-image search typically has lower similarity scores (0.01-0.10)
            # so we use no minimum threshold and rely on relative ranking
            results = storage.find_similar(
                query_embedding=query_embedding,
                limit=limit,
                offset=offset,
                min_similarity=0.0,
            )

            # Fetch full scene details from SQLite
            scene_details = self._get_scene_details_batch([r.scene_id for r in results])

            # Build result data with embedded scene details
            result_data = []
            for r in results:
                scene = scene_details.get(r.scene_id, {})
                result_data.append(
                    {"scene_id": r.scene_id, "similarity": r.similarity, "scene": scene}
                )

            # has_more is true if we got a full page of results
            has_more = len(results) == limit

            # Write results to JSON file for frontend polling
            self._write_search_result(
                request_id,
                {
                    "status": "complete",
                    "query": query,
                    "model_key": model_key,
                    "results": result_data,
                    "offset": offset,
                    "limit": limit,
                    "has_more": has_more,
                    "request_id": request_id,
                    "total_embeddings": stats["total_embeddings"],
                },
            )

            self.log(f"Search complete: {len(result_data)} results for '{query}'", "info")

        except ImportError as e:
            self.error(f"Failed to import embedding modules: {e}")
            self._write_search_result(
                args.get("request_id", ""),
                {"status": "error", "error": f"Failed to import embedding modules: {e}"},
            )
        except Exception as e:
            self.error(f"Search error: {e}")
            self._write_search_result(
                args.get("request_id", ""), {"status": "error", "error": str(e)}
            )

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
                    request_id, {"status": "error", "error": "Scene ID is required", "request_id": request_id}
                )
                return

            try:
                timestamp = float(timestamp_str)
                limit = int(args.get("limit", 20))
            except ValueError as e:
                self._write_frame_search_result(
                    request_id, {"status": "error", "error": f"Invalid parameter: {e}", "request_id": request_id}
                )
                return
            self.log(f"Frame search: scene={scene_id}, timestamp={timestamp:.1f}s, limit={limit}", "info")

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
                    {"status": "error", "error": f"Could not find video file for scene {scene_id}", "request_id": request_id},
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
                    {"status": "error", "error": f"Failed to extract frame at {timestamp:.1f}s", "request_id": request_id},
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
                    {"status": "error", "error": "No image embedding provider configured. Set up in Plugin Settings.", "request_id": request_id},
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
                    {"status": "error", "error": f"Provider '{image_provider}' does not support image embedding.", "request_id": request_id},
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
                {"status": "error", "error": f"Failed to import modules: {e}", "request_id": args.get("request_id", "")},
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

    def _write_search_result(self, request_id: str, data: dict[str, Any]) -> None:
        """Write text search results to JSON file for frontend polling."""
        import json as json_module
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")

        # Ensure assets directory exists
        os.makedirs(assets_dir, exist_ok=True)

        # Use request_id for filename to support concurrent searches
        filename = f"search_results_{request_id or 'latest'}.json"
        result_file = os.path.join(assets_dir, filename)

        try:
            with open(result_file, "w") as f:
                json_module.dump(data, f)
            self.log(f"Wrote search results to: {result_file}", "debug")
        except Exception as e:
            self.error(f"Failed to write search results file: {e}")

    def run_get_embedding_models(self, args: dict[str, Any]) -> None:
        """
        Get available embedding models and their statistics.

        Returns a JSON file with all model keys that have stored embeddings,
        along with their embedding counts and dimensions.

        Args:
            args: Task arguments containing:
                - request_id: Unique request ID for frontend validation
        """

        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            request_id = args.get("request_id", "")

            self.log("Fetching available embedding models...", "info")

            # Create storage instance (model_key doesn't matter for getting all models)
            storage = EmbeddingStorage(model_key="siglip")

            # Get available model keys
            model_keys = storage.get_available_model_keys()

            # Get stats for each model
            models_data = []
            for model_key in model_keys:
                model_storage = EmbeddingStorage(model_key=model_key)
                stats = model_storage.get_stats()
                models_data.append(
                    {
                        "model_key": model_key,
                        "count": stats["total_embeddings"],
                        "dimensions": list(stats["dimensions_distribution"].keys())[0]
                        if stats["dimensions_distribution"]
                        else None,
                        "oldest": stats["oldest_embedding"],
                        "newest": stats["newest_embedding"],
                    }
                )

            # Get current plugin settings for reference
            plugin_settings = self.get_plugin_settings("stash-copilot")
            current_provider = plugin_settings.get("image_embedding_provider", "")
            current_model = plugin_settings.get("image_embedding_model", "")

            # Build current model key for comparison
            current_model_key = None
            if current_provider and current_model:
                if current_provider == "siglip":
                    current_model_key = "siglip"
                else:
                    current_model_key = f"{current_provider}:{current_model}"

            # Write results
            result_data = {
                "status": "complete",
                "models": models_data,
                "current_model_key": current_model_key,
                "request_id": request_id,
            }

            self._write_embedding_models_result(request_id, result_data)
            self.log(f"Found {len(models_data)} embedding models", "info")

        except ImportError as e:
            self.error(f"Failed to import embedding modules: {e}")
            self._write_embedding_models_result(
                args.get("request_id", ""),
                {"status": "error", "error": f"Failed to import embedding modules: {e}"},
            )
        except Exception as e:
            self.error(f"Error getting embedding models: {e}")
            self._write_embedding_models_result(
                args.get("request_id", ""), {"status": "error", "error": str(e)}
            )

    def _write_embedding_models_result(self, request_id: str, data: dict[str, Any]) -> None:
        """Write embedding models result to JSON file for frontend polling."""
        import json as json_module
        import os

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")

        os.makedirs(assets_dir, exist_ok=True)

        filename = f"embedding_models_{request_id or 'latest'}.json"
        result_file = os.path.join(assets_dir, filename)

        try:
            with open(result_file, "w") as f:
                json_module.dump(data, f)
            self.log(f"Wrote embedding models to: {result_file}", "debug")
        except Exception as e:
            self.error(f"Failed to write embedding models file: {e}")

    def run_embed_o_moments(self, args: dict[str, Any]) -> None:
        """
        Run the O-moment embedding generation task.

        Creates embeddings from frames around O markers for
        "Peak Moments" recommendations.

        Args:
            args: Task arguments containing optional scene_id and force flag
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.embed_o_moments import EmbedOMomentsConfig, EmbedOMomentsTask

            self.log("Initializing O-moment embedding generation...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get image embedding config
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self.error(
                    "Image embedding provider and model are required for O-moment embedding. "
                    "Please configure image_embedding_provider and image_embedding_model in plugin settings."
                )
                return

            # Build embedding config
            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )

            self.log(f"Using {image_provider}/{image_model} for O-moment embeddings", "info")

            # Build O-moment config from settings
            window_seconds = float(
                args.get("window_seconds") or plugin_settings.get("o_moment_window") or "120"
            )
            frames_per_window = int(
                args.get("frames_per_window") or plugin_settings.get("o_moment_frames") or "12"
            )
            o_tag_name = args.get("o_tag") or plugin_settings.get("o_tag_name") or "O"

            embed_config = EmbedOMomentsConfig(
                window_seconds=window_seconds,
                frames_per_window=frames_per_window,
                o_tag_name=o_tag_name,
            )

            self.log(
                f"O-moment config: window={window_seconds}s, frames={frames_per_window}, tag='{o_tag_name}'",
                "debug",
            )

            # Create task
            task = EmbedOMomentsTask(
                stash=self.stash_client,
                embedding_config=embedding_config,
                embed_config=embed_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            # Check for specific scene or all scenes
            scene_id = args.get("scene_id")
            force = str(args.get("force", "")).lower() == "true"

            if scene_id:
                self.log(f"Embedding O-moments for scene {scene_id}...", "info")
                result = task.embed_scene_o_moments(int(scene_id), force=force)

                self.log(f"Result: {result}", "info")
                if result.get("success"):
                    self.log(
                        f"Embedded {result.get('embedded', 0)} O-moments, "
                        f"skipped {result.get('skipped', 0)} (already embedded)",
                        "info",
                    )
            else:
                self.log("Embedding O-moments for all scenes with O markers...", "info")

                # Check for scene_ids filter
                scene_ids_str = args.get("scene_ids", "")
                scene_ids = None
                if scene_ids_str:
                    scene_ids = [int(s.strip()) for s in scene_ids_str.split(",") if s.strip()]

                result = task.embed_all_o_moments(force=force, scene_ids=scene_ids)

                self.log("O-moment embedding complete:", "info")
                self.log(f"  Total scenes: {result.get('total_scenes', 0)}", "info")
                self.log(f"  Total markers: {result.get('total_markers', 0)}", "info")
                self.log(f"  Embedded: {result.get('embedded', 0)}", "info")
                self.log(f"  Skipped: {result.get('skipped', 0)}", "info")
                self.log(f"  Errors: {result.get('errors', 0)}", "info")

                if result.get("error_details"):
                    for err in result["error_details"][:5]:
                        self.log(f"  - {err}", "warning")

        except ImportError as e:
            self.error(f"Failed to import O-moment embedding modules: {e}")
        except Exception as e:
            self.error(f"O-moment embedding failed: {e}")

    def run_embed_cached_frames(self, args: dict[str, Any]) -> None:
        """
        Run the cached frames embedding task.

        Backfills frame embeddings for scenes that have frames cached
        but were embedded before individual frame storage was implemented.
        This enables smart frame selection for VLM analysis.

        Args:
            args: Task arguments containing optional scene_id and force flag
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.embed_cached_frames import EmbedCachedFramesTask

            self.log("Initializing cached frame embedding backfill...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get image embedding config
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self.error(
                    "Image embedding provider and model are required for frame embedding. "
                    "Please configure image_embedding_provider and image_embedding_model in plugin settings."
                )
                return

            # Build embedding config
            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )

            # Parallel processing settings (reuse embed_num_workers from main embedding task)
            num_workers = int(plugin_settings.get("embed_num_workers") or "2")

            self.log(f"Using {image_provider}/{image_model} for frame embeddings", "info")
            self.log(f"Scene workers: {num_workers}", "info")

            # Create task
            task = EmbedCachedFramesTask(
                stash=self.stash_client,
                embedding_config=embedding_config,
                log_callback=self.log,
                progress_callback=self.progress,
                num_workers=num_workers,
            )

            # Check for specific scene or all scenes
            scene_id_str = args.get("scene_id")
            scene_id = int(scene_id_str) if scene_id_str else None
            force = str(args.get("force", "")).lower() == "true"

            result = task.run(force=force, scene_id=scene_id)

            self.log("Embed Cached Frames complete:", "info")
            self.log(f"  Total: {result.get('total', 0)}", "info")
            self.log(f"  Processed: {result.get('processed', 0)}", "info")
            self.log(f"  Skipped: {result.get('skipped', 0)}", "info")
            self.log(f"  Errors: {result.get('errors', 0)}", "info")

            if result.get("error_details"):
                for err in result["error_details"][:5]:
                    self.log(f"  - {err}", "warning")

        except ImportError as e:
            self.error(f"Failed to import embed cached frames modules: {e}")
        except Exception as e:
            self.error(f"Embed cached frames failed: {e}")

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
        """
        Clean up embeddings for scenes that no longer exist in Stash.

        This maintenance task finds and removes orphaned embeddings for scenes
        that were deleted while the plugin was disabled or before the
        Scene.Destroy.Post hook was implemented.

        Args:
            args: Task arguments containing:
                - dry_run: "true" to only report what would be deleted
        """
        try:
            from stash_ai.embeddings.storage import EmbeddingStorage
            from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

            dry_run = str(args.get("dry_run", "true")).lower() == "true"

            self.log(f"Starting orphaned embeddings cleanup (dry_run={dry_run})...", "info")

            # Get all valid scene IDs from Stash database
            db_path = get_stash_db_path()
            if not db_path.exists():
                self.error("Stash database not found")
                return

            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM scenes")
            valid_scene_ids = [row["id"] for row in cursor.fetchall()]
            conn.close()

            self.log(f"Found {len(valid_scene_ids)} scenes in Stash database", "info")

            # Create storage instance (model_key doesn't matter for orphan detection)
            storage = EmbeddingStorage()
            orphaned_ids = storage.get_orphaned_scene_ids(valid_scene_ids)

            if not orphaned_ids:
                self.log("No orphaned embeddings found", "info")
                return

            self.log(f"Found {len(orphaned_ids)} orphaned scene IDs", "info")

            if dry_run:
                self.log(
                    f"DRY RUN: Would delete embeddings for {len(orphaned_ids)} "
                    f"orphaned scenes: {orphaned_ids[:10]}{'...' if len(orphaned_ids) > 10 else ''}",
                    "info",
                )
                return

            # Delete embeddings for each orphaned scene
            total_deleted = 0
            for i, scene_id in enumerate(orphaned_ids):
                result = storage.delete_all_scene_data(scene_id)
                deleted = sum(result.values())
                total_deleted += deleted

                if (i + 1) % 10 == 0 or i == len(orphaned_ids) - 1:
                    self.log(f"Progress: {i + 1}/{len(orphaned_ids)} scenes processed", "info")
                    self.progress(i + 1, len(orphaned_ids))

            self.log(
                f"Cleanup complete: deleted {total_deleted} items "
                f"from {len(orphaned_ids)} orphaned scenes",
                "info",
            )

        except ImportError as e:
            self.error(f"Failed to import cleanup modules: {e}")
        except Exception as e:
            self.error(f"Orphaned embeddings cleanup failed: {e}")

    def run_embed_performers(self, args: dict[str, Any]) -> None:
        """
        Generate embeddings for performers from aggregated scene embeddings.

        Args:
            args: Task arguments containing:
                - performer_id: Optional specific performer ID
                - force: "true" to regenerate existing embeddings
        """
        try:
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.recommendations.types import EngagementWeights
            from stash_ai.tasks.embed_performers import (
                EmbedPerformersTask,
                EmbedPerformersTaskConfig,
            )

            self.log("Initializing performer embedding generation...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get image embedding config
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self.error(
                    "Image embedding provider and model are required for performer embedding. "
                    "Please configure image_embedding_provider and image_embedding_model in plugin settings."
                )
                return

            # Build embedding config
            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )

            self.log(f"Using {image_provider}/{image_model} for performer embeddings", "info")

            # Get engagement weights from settings
            weights = cast(
                "EngagementWeights",
                {
                    "o_count": float(plugin_settings.get("rec_o_weight") or "20.0"),
                    "view_count": float(plugin_settings.get("rec_view_weight") or "2.0"),
                    "play_duration": float(plugin_settings.get("rec_duration_weight") or "1.0"),
                    "rating": float(plugin_settings.get("rec_rating_weight") or "1.5"),
                },
            )

            # Task config
            min_scenes = int(args.get("min_scenes") or "2")
            max_scenes = int(args.get("max_scenes") or "50")

            task_config = EmbedPerformersTaskConfig(
                min_scenes=min_scenes,
                max_scenes=max_scenes,
                use_engagement_weighting=True,
                include_unwatched=True,
            )

            # Create task
            task = EmbedPerformersTask(
                stash=self.stash_client,
                embedding_config=embedding_config,
                task_config=task_config,
                weights=weights,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            # Check for specific performer or all performers
            performer_id = args.get("performer_id")
            force = str(args.get("force", "")).lower() == "true"

            if performer_id:
                self.log(f"Embedding performer {performer_id}...", "info")
                result = task.embed_performer(int(performer_id), force=force)

                if result.get("success"):
                    if result.get("skipped"):
                        self.log(f"Skipped: {result.get('message')}", "info")
                    else:
                        self.log(
                            f"Embedded {result.get('performer_name')}: "
                            f"{result.get('contributing_scenes')} scenes, "
                            f"score {result.get('total_engagement_score', 0):.2f}",
                            "info",
                        )
                else:
                    self.error(f"Failed: {result.get('error')}")
            else:
                self.log("Embedding all performers...", "info")
                result = task.embed_all_performers(force=force)

                self.log("=" * 50, "info")
                self.log("PERFORMER EMBEDDING COMPLETE", "info")
                self.log("=" * 50, "info")
                self.log(f"Total performers: {result.get('total_performers', 0)}", "info")
                self.log(f"Embedded: {result.get('embedded', 0)}", "info")
                self.log(f"Skipped (already embedded): {result.get('skipped', 0)}", "info")
                self.log(f"Insufficient scenes: {result.get('insufficient_scenes', 0)}", "info")
                self.log(f"Errors: {result.get('errors', 0)}", "info")

                if result.get("error_details"):
                    for err in result["error_details"][:5]:
                        self.log(f"  - {err}", "warning")

        except ImportError as e:
            self.error(f"Failed to import performer embedding modules: {e}")
        except Exception as e:
            import traceback

            self.error(f"Performer embedding failed: {e}")
            self.log(f"Traceback: {traceback.format_exc()}", "debug")

    def run_describe_performers(self, args: dict[str, Any]) -> None:
        """
        Generate AI-powered descriptions for performers using VLM.

        Args:
            args: Task arguments containing:
                - performer_id: Optional specific performer ID
                - force: "true" to regenerate existing descriptions
        """
        try:
            from stash_ai.config import get_vision_llm_settings
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.tasks.describe_performer import (
                DescribePerformerTask,
                DescribePerformerTaskConfig,
            )

            self.log("Initializing performer description generation...", "info")

            plugin_settings = self.get_plugin_settings("stash-copilot")

            # Get vision LLM settings
            vision_llm = get_vision_llm_settings(plugin_settings, args)
            self.log(f"Using VLM: {vision_llm.provider}/{vision_llm.model}", "info")

            llm_config = vision_llm.to_config()

            # Get image embedding model key for storage
            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if not image_provider or not image_model:
                self.error(
                    "Image embedding provider and model are required. "
                    "Please configure image_embedding_provider and image_embedding_model in plugin settings."
                )
                return

            embedding_config = EmbeddingConfig(
                provider=image_provider,
                model=image_model,
                device=image_device,
            )
            model_key = embedding_config.model_key

            # Task config
            frames_per_scene = int(args.get("frames_per_scene") or "4")
            max_scenes = int(args.get("max_scenes") or "8")

            task_config = DescribePerformerTaskConfig(
                frames_per_scene=frames_per_scene,
                max_scenes=max_scenes,
            )

            # Create task
            task = DescribePerformerTask(
                stash=self.stash_client,
                llm_config=llm_config,
                model_key=model_key,
                task_config=task_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            # Check for specific performer or all performers
            performer_id = args.get("performer_id")
            force = str(args.get("force", "")).lower() == "true"

            if performer_id:
                self.log(f"Describing performer {performer_id}...", "info")
                result = task.describe_performer(int(performer_id), force=force)

                if result.get("success"):
                    if result.get("skipped"):
                        self.log("Skipped (already has description)", "info")
                    else:
                        self.log(
                            f"Generated description for {result.get('performer_name')} "
                            f"({result.get('frames_analyzed')} frames from {result.get('scenes_analyzed')} scenes)",
                            "info",
                        )
                else:
                    self.error(f"Failed: {result.get('error')}")
            else:
                self.log("Describing all performers with embeddings...", "info")
                result = task.describe_all_performers(force=force)

                self.log("=" * 50, "info")
                self.log("PERFORMER DESCRIPTION COMPLETE", "info")
                self.log("=" * 50, "info")
                self.log(f"Total performers: {result.get('total_performers', 0)}", "info")
                self.log(f"Described: {result.get('described', 0)}", "info")
                self.log(f"Skipped: {result.get('skipped', 0)}", "info")
                self.log(f"Errors: {result.get('errors', 0)}", "info")

                if result.get("error_details"):
                    for err in result["error_details"][:5]:
                        self.log(f"  - {err}", "warning")

        except ImportError as e:
            self.error(f"Failed to import performer description modules: {e}")
        except Exception as e:
            import traceback

            self.error(f"Performer description failed: {e}")
            self.log(f"Traceback: {traceback.format_exc()}", "debug")

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
