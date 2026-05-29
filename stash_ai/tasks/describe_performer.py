"""Task for generating AI-powered performer descriptions using VLM."""

import base64
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import LLMConfig
from ..embeddings.storage import EmbeddingStorage
from ..llm import get_provider
from ..llm.base import BaseLLMProvider, Message
from ..recommendations.performer_profile import PerformerProfileBuilder
from .frame_extractor import FrameExtractionConfig, FrameExtractor

if TYPE_CHECKING:
    from ..stash_client import StashClient


# System prompt for performer description generation
PERFORMER_DESCRIPTION_SYSTEM_PROMPT = """You are an expert at describing adult performers based on visual analysis.
Your job is to create a detailed description of the performer's appearance across multiple scenes.

Always use common terminology: "tits" for breasts, "ass" for buttocks, "pussy" for vagina.
Be factual and describe only what you can visually observe.
Focus on consistent physical attributes that identify this performer across scenes."""

# Main description prompt template
PERFORMER_DESCRIPTION_PROMPT = """I'm showing you {frame_count} frames from {scene_count} different scenes featuring the same performer: {performer_name}.

Based on these frames, provide a detailed visual description of this performer.

**Include these details:**
1. **Physical Build:** Body type (petite, athletic, curvy, etc.), approximate height appearance
2. **Hair:** Color, length, style (if it varies between scenes, note that)
3. **Face:** Distinctive facial features, makeup style preferences
4. **Body Features:** Tits size/shape, ass shape, any distinctive features
5. **Skin:** Skin tone, tan lines, any visible tattoos or piercings
6. **Style:** Common clothing/lingerie styles, jewelry preferences

**Important:**
- Only describe what you can actually see in the frames
- If attributes vary between scenes (e.g., hair color changes), note both
- Focus on features that would help identify this performer

Write your description as a cohesive 2-3 paragraph profile."""


@dataclass
class DescribePerformerTaskConfig:
    """Configuration for performer description generation."""

    # Number of frames per scene to analyze
    frames_per_scene: int = 4

    # Maximum scenes to sample from
    max_scenes: int = 8

    # Frame width for extraction
    frame_width: int = 640

    # Include tag analysis
    include_tags: bool = True


class DescribePerformerTask:
    """
    Task for generating AI-powered performer descriptions.

    Uses VLM to analyze frames from multiple scenes featuring a performer
    to generate a comprehensive visual description.

    This enables:
    - Searchable performer descriptions
    - Visual profile summaries
    - Enhanced performer page metadata
    """

    def __init__(
        self,
        stash: "StashClient",
        llm_config: LLMConfig,
        model_key: str,
        task_config: DescribePerformerTaskConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize the performer description task.

        Args:
            stash: StashClient instance
            llm_config: Config for VLM provider
            model_key: Model key for embedding storage
            task_config: Optional task configuration
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
        """
        self.stash = stash
        self.llm_config = llm_config
        self.config = task_config or DescribePerformerTaskConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        # Initialize storage
        self.storage = EmbeddingStorage(model_key=model_key)

        # Initialize profile builder
        self.profile_builder = PerformerProfileBuilder(
            storage=self.storage,
            log_callback=self.log,
        )

        # Setup frame extractor
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cache_dir = os.path.join(plugin_dir, "assets", "performer_frames")
        self.frame_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                frame_width=self.config.frame_width,
            ),
            cache_dir=cache_dir,
            log_callback=self.log,
        )

        # Initialize LLM provider lazily
        self._llm_provider: BaseLLMProvider | None = None

    @property
    def llm(self) -> BaseLLMProvider:
        """Lazy-load LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_provider(self.llm_config)
        return self._llm_provider

    def _get_scene_file_path(self, scene_id: int) -> str | None:
        """Get the file path for a scene."""
        from ..tools.database import get_readonly_connection, get_stash_db_path

        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT f.path
            FROM scenes s
            JOIN scenes_files sf ON s.id = sf.scene_id
            JOIN files f ON sf.file_id = f.id
            WHERE s.id = ?
            ORDER BY sf."primary" DESC
            LIMIT 1
        """,
            (scene_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return row["path"] if row else None

    def _get_scene_duration(self, scene_id: int) -> float | None:
        """Get the duration of a scene in seconds."""
        from ..tools.database import get_readonly_connection, get_stash_db_path

        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT vf.duration
            FROM scenes s
            JOIN scenes_files sf ON s.id = sf.scene_id
            JOIN files f ON sf.file_id = f.id
            JOIN video_files vf ON f.id = vf.file_id
            WHERE s.id = ?
            ORDER BY sf."primary" DESC
            LIMIT 1
        """,
            (scene_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return float(row["duration"]) if row else None

    def _extract_frames_for_scene(
        self,
        scene_id: int,
        n_frames: int,
    ) -> list[str]:
        """
        Extract evenly-spaced frames from a scene.

        Args:
            scene_id: Scene ID
            n_frames: Number of frames to extract

        Returns:
            List of base64-encoded frame images
        """
        file_path = self._get_scene_file_path(scene_id)
        if not file_path or not os.path.exists(file_path):
            self.log(f"Scene {scene_id} file not found: {file_path}", "warning")
            return []

        duration = self._get_scene_duration(scene_id)
        if not duration:
            self.log(f"Scene {scene_id} has no duration", "warning")
            return []

        # Calculate evenly-spaced timestamps
        interval = duration / (n_frames + 1)
        timestamps = [interval * (i + 1) for i in range(n_frames)]

        frames_base64 = []
        for ts in timestamps:
            try:
                frame_bytes = self.frame_extractor.extract_frame_at_timestamp(file_path, ts)
                if frame_bytes:
                    frames_base64.append(base64.b64encode(frame_bytes).decode())
            except (OSError, ValueError) as e:
                self.log(f"Error extracting frame at {ts}s: {e}", "warning")

        return frames_base64

    def describe_performer(
        self,
        performer_id: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Generate AI description for a performer.

        Args:
            performer_id: Stash performer ID
            force: If True, regenerate even if description exists

        Returns:
            Dict with description and metadata
        """
        # Get performer info
        performer = self.profile_builder.get_performer_by_id(performer_id)
        if not performer:
            return {
                "success": False,
                "performer_id": performer_id,
                "error": "Performer not found",
            }

        performer_name = performer["name"]

        # Check if already has description
        record = self.storage.get_performer_embedding(performer_id)
        if record and record.get("visual_description") and not force:
            self.log(f"Performer {performer_name} already has description", "debug")
            return {
                "success": True,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "skipped": True,
                "description": record["visual_description"],
            }

        # Get performer's scenes
        scenes = self.profile_builder.get_performer_scenes(performer_id)
        if not scenes:
            return {
                "success": False,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "error": "No scenes found for performer",
            }

        # Sort by engagement (o_count > play_count > rating)
        scenes.sort(
            key=lambda s: (s["o_count"], s["play_count"], s["rating"] or 0),
            reverse=True,
        )

        # Take top scenes up to max_scenes
        selected_scenes = scenes[: self.config.max_scenes]

        self.log(
            f"Analyzing {len(selected_scenes)} scenes for {performer_name}",
            "info",
        )

        # Extract frames from each scene
        all_frames: list[str] = []
        for scene in selected_scenes:
            scene_id = scene["scene_id"]
            frames = self._extract_frames_for_scene(scene_id, self.config.frames_per_scene)
            all_frames.extend(frames)

            if len(all_frames) >= self.config.frames_per_scene * self.config.max_scenes:
                break

        if not all_frames:
            return {
                "success": False,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "error": "Could not extract frames from any scenes",
            }

        self.log(
            f"Extracted {len(all_frames)} frames from {len(selected_scenes)} scenes",
            "debug",
        )

        # Build prompt
        prompt = PERFORMER_DESCRIPTION_PROMPT.format(
            frame_count=len(all_frames),
            scene_count=len(selected_scenes),
            performer_name=performer_name,
        )

        # Prepare images for LLM
        images = [base64.b64decode(f) for f in all_frames]

        # Call VLM
        try:
            messages: list[Message] = [
                {"role": "system", "content": PERFORMER_DESCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            self.log(f"Calling VLM with {len(images)} images...", "debug")
            result = self.llm.chat(messages, images=images)

            description = result["content"] if result else None

            if not description:
                return {
                    "success": False,
                    "performer_id": performer_id,
                    "performer_name": performer_name,
                    "error": "VLM returned empty response",
                }

            # Get top tags if enabled
            top_tags = None
            if self.config.include_tags:
                # Aggregate tags from scenes
                from collections import Counter

                tag_counter: Counter[str] = Counter()
                for scene in scenes:
                    tag_counter.update(scene.get("tags", []))
                top_tags = [tag for tag, _ in tag_counter.most_common(10)]

            # Update performer embedding with description
            if record:
                self.storage.update_performer_description(
                    performer_id=performer_id,
                    visual_description=description,
                    top_tags=json.dumps(top_tags) if top_tags else None,
                )
            else:
                self.log(
                    f"Performer {performer_name} has no embedding - description not stored",
                    "warning",
                )

            self.log(f"Generated description for {performer_name}", "info")

            return {
                "success": True,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "description": description,
                "top_tags": top_tags,
                "frames_analyzed": len(all_frames),
                "scenes_analyzed": len(selected_scenes),
            }

        except (ValueError, RuntimeError, OSError) as e:
            self.log(f"Error generating description for {performer_name}: {e}", "error")
            return {
                "success": False,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "error": str(e),
            }

    def describe_all_performers(
        self,
        force: bool = False,
        performer_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Generate descriptions for all performers with embeddings.

        Args:
            force: If True, regenerate all descriptions
            performer_ids: Optional list of specific performer IDs

        Returns:
            Summary of description generation
        """
        # Get performers to process
        if performer_ids:
            performers_to_process = performer_ids
        else:
            # Only process performers that have embeddings
            performers_to_process = self.storage.get_embedded_performer_ids()

        if not performers_to_process:
            self.log("No performers with embeddings found", "warning")
            return {
                "total_performers": 0,
                "described": 0,
                "skipped": 0,
                "errors": 0,
            }

        self.log(f"Processing {len(performers_to_process)} performers", "info")

        described = 0
        skipped = 0
        errors = 0
        error_details: list[str] = []

        for i, performer_id in enumerate(performers_to_process):
            self.progress(i, len(performers_to_process))

            result = self.describe_performer(performer_id, force=force)

            if result.get("success"):
                if result.get("skipped"):
                    skipped += 1
                else:
                    described += 1
            else:
                errors += 1
                if len(error_details) < 10:
                    error_details.append(f"Performer {performer_id}: {result.get('error')}")

        self.progress(len(performers_to_process), len(performers_to_process))

        return {
            "total_performers": len(performers_to_process),
            "described": described,
            "skipped": skipped,
            "errors": errors,
            "error_details": error_details,
        }
