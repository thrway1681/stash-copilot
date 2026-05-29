"""Image labeling task — uncertainty sampling and session management."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..stash_client import StashClient

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.tasks.labeling_types import (
    FrameSuggestion,
    LabelingConfig,
    LabelingFrameItem,
    LabelingSessionResult,
)


class LabelingTask:
    """Prepare labeling sessions with uncertainty-sampled frames."""

    def __init__(
        self,
        stash: StashClient,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def _compute_uncertainty(
        self,
        frame_similarities: NDArray[np.float32],
        low: float,
        high: float,
    ) -> int:
        """Count how many tags fall in the confusion zone for a single frame."""
        return int(np.sum((frame_similarities >= low) & (frame_similarities <= high)))

    def _rank_by_uncertainty(
        self,
        similarities: NDArray[np.float32],
        frame_keys: list[tuple[int, int]],
        low: float,
        high: float,
        limit: int,
    ) -> list[tuple[int, int]]:
        """Rank frames by uncertainty score and return top `limit`."""
        scores = []
        for i in range(similarities.shape[0]):
            score = self._compute_uncertainty(similarities[i], low, high)
            scores.append((score, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [frame_keys[idx] for _, idx in scores[:limit]]

    def _get_suggested_tags(
        self,
        frame_sims: NDArray[np.float32],
        tag_info: list[dict[str, str]],
        max_tags: int,
    ) -> list[FrameSuggestion]:
        """Get top suggested tags for a single frame, ordered by similarity."""
        indexed = [(float(frame_sims[i]), i) for i in range(len(tag_info))]
        indexed.sort(key=lambda x: -x[0])
        suggestions: list[FrameSuggestion] = []
        for sim, idx in indexed[:max_tags]:
            suggestions.append(
                FrameSuggestion(
                    tag_text=tag_info[idx]["text"],
                    tag_source=tag_info[idx]["source"],
                    similarity=round(sim, 4),
                )
            )
        return suggestions

    def _get_scene_tags(self, scene_id: int) -> list[str]:
        """Get existing tags for a scene from Stash."""
        try:
            scene = self.stash.find_scene(scene_id)
            if scene and "tags" in scene:
                return [t["name"] for t in scene["tags"]]
        except Exception:
            pass
        return []

    def _get_scene_title(self, scene_id: int) -> str:
        """Get scene title from Stash."""
        try:
            scene = self.stash.find_scene(scene_id)
            if scene:
                title = scene.get("title", "")
                if not title and scene.get("files"):
                    path = scene["files"][0].get("path", "")
                    title = Path(path).stem
                return title or f"Scene {scene_id}"
        except Exception:
            pass
        return f"Scene {scene_id}"

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}:{secs:02d}"

    def prepare_session(self, config: LabelingConfig) -> LabelingSessionResult:
        """Prepare a labeling session with uncertainty-sampled frames."""
        self.log("Preparing labeling session...", "info")

        # 1. Count embeddings and load a capped random sample to stay within
        #    memory budget (~150 MB for 50K × 768 × float32 instead of ~12 GB
        #    for 4M+).  Uncertainty ranking works well on a random subset.
        self.log("Loading frame embeddings...", "info")
        conn = self.storage._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM frame_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        total_count = cursor.fetchone()[0]
        if total_count == 0:
            conn.close()
            return LabelingSessionResult(
                status="no_embeddings",
                session_id="",
                batch=[],
                vocabulary=[],
                error="No frame embeddings found. Run 'Embed All Scenes' first.",
            )

        self.log(f"Total frame embeddings: {total_count}", "info")

        # 2. Exclude already-labeled frames
        labeled_keys = self.storage.get_labeled_frame_keys()
        self.log(f"Excluding {len(labeled_keys)} already-labeled frames", "info")

        # Cap candidates to avoid loading millions of embeddings into RAM.
        # Two-phase sampling: first pick rowids (fast, no BLOB reads), then
        # fetch full rows only for the chosen subset.
        max_candidates = config.max_candidates
        use_sampling = total_count > max_candidates
        if use_sampling:
            self.log(
                f"Sampling {max_candidates} of {total_count} candidates "
                f"(memory-safe mode)",
                "info",
            )
            cursor.execute(
                """SELECT rowid FROM frame_embeddings WHERE model_key = ?
                ORDER BY RANDOM() LIMIT ?""",
                (self.model_key, max_candidates),
            )
            sampled_rowids = [r[0] for r in cursor.fetchall()]
            self.log(f"Sampled {len(sampled_rowids)} rowids, fetching embeddings...", "info")
            # Batch fetch in chunks to avoid SQLite variable limit
            chunk_size = 500
            rows_iter: list[Any] = []
            for i in range(0, len(sampled_rowids), chunk_size):
                chunk = sampled_rowids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor.execute(
                    f"""SELECT scene_id, frame_index, timestamp, embedding
                    FROM frame_embeddings WHERE rowid IN ({placeholders})""",
                    chunk,
                )
                rows_iter.extend(cursor.fetchall())
        else:
            cursor.execute(
                """SELECT scene_id, frame_index, timestamp, embedding
                FROM frame_embeddings WHERE model_key = ?
                ORDER BY scene_id, frame_index""",
                (self.model_key,),
            )
            rows_iter = cursor.fetchall()

        frame_embeddings_list: list[list[float]] = []
        frame_keys: list[tuple[int, int]] = []
        frame_timestamps: dict[tuple[int, int], float] = {}

        for row in rows_iter:
            key = (row["scene_id"], row["frame_index"])
            if key in labeled_keys:
                continue
            frame_embeddings_list.append(
                self.storage._unpack_embedding(row["embedding"])
            )
            frame_keys.append(key)
            frame_timestamps[key] = row["timestamp"]

        conn.close()

        if not frame_keys:
            return LabelingSessionResult(
                status="complete",
                session_id="",
                batch=[],
                vocabulary=[],
                error="All frames have been labeled!",
            )

        frame_embeddings = np.array(frame_embeddings_list, dtype=np.float32)
        del frame_embeddings_list  # Free the Python list immediately
        self.log(f"{len(frame_keys)} unlabeled candidate frames loaded", "info")

        # 3. Load tag embeddings
        tag_data = self.storage.get_all_tag_embeddings(self.model_key)
        if not tag_data:
            return LabelingSessionResult(
                status="error",
                session_id="",
                batch=[],
                vocabulary=[],
                error="No tag embeddings found. Ensure tag vocabulary is built.",
            )

        tag_info = [{"text": t["text"], "source": t["source"]} for t in tag_data]
        tag_embeddings = np.array(
            [t["embedding"] for t in tag_data], dtype=np.float32
        )

        # 4. Compute similarity matrix (normalize in-place to halve peak memory)
        self.log("Computing frame-tag similarities...", "info")
        frame_norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        frame_embeddings /= frame_norms + 1e-8  # In-place normalize
        tag_norms = np.linalg.norm(tag_embeddings, axis=1, keepdims=True)
        tag_embeddings /= tag_norms + 1e-8  # In-place normalize
        similarities = np.dot(frame_embeddings, tag_embeddings.T)
        del frame_embeddings, tag_embeddings  # Free before building batch

        # 5. Rank by uncertainty
        self.log("Ranking frames by uncertainty...", "info")
        selected_keys = self._rank_by_uncertainty(
            similarities,
            frame_keys,
            low=config.uncertainty_low,
            high=config.uncertainty_high,
            limit=config.batch_size,
        )

        # 6. Build batch items
        self.log(f"Building batch of {len(selected_keys)} frames...", "info")
        scene_tags_cache: dict[int, list[str]] = {}
        scene_title_cache: dict[int, str] = {}
        key_to_idx = {k: i for i, k in enumerate(frame_keys)}
        plugin_dir = Path(__file__).parent.parent.parent
        batch: list[LabelingFrameItem] = []

        for scene_id, frame_index in selected_keys:
            idx = key_to_idx[(scene_id, frame_index)]
            suggested = self._get_suggested_tags(
                similarities[idx], tag_info, config.max_suggested_tags
            )
            if scene_id not in scene_tags_cache:
                scene_tags_cache[scene_id] = self._get_scene_tags(scene_id)
                scene_title_cache[scene_id] = self._get_scene_title(scene_id)

            frame_path = str(
                plugin_dir / "assets" / "embedded_frames"
                / f"scene_{scene_id}" / f"frame_{frame_index:04d}.jpg"
            )
            timestamp = frame_timestamps.get((scene_id, frame_index), 0.0)

            batch.append(
                LabelingFrameItem(
                    scene_id=scene_id,
                    frame_index=frame_index,
                    frame_path=frame_path,
                    timestamp=self._format_timestamp(timestamp),
                    uncertainty_score=float(
                        self._compute_uncertainty(
                            similarities[idx],
                            config.uncertainty_low,
                            config.uncertainty_high,
                        )
                    ),
                    suggested_tags=suggested,
                    scene_tags=scene_tags_cache[scene_id],
                    scene_title=scene_title_cache[scene_id],
                )
            )

        # 7. Create session in DB
        session_id = self.storage.create_labeling_session(
            sampling_method="uncertainty",
            batch_size=config.batch_size,
            total_frames=len(batch),
            config_json=json.dumps({
                "uncertainty_low": config.uncertainty_low,
                "uncertainty_high": config.uncertainty_high,
                "max_suggested_tags": config.max_suggested_tags,
                "model_key": self.model_key,
            }),
        )

        # 8. Build vocabulary list for autocomplete
        vocabulary = sorted(set(t["text"] for t in tag_info))

        self.log(
            f"Session {session_id} ready: {len(batch)} frames, "
            f"{len(vocabulary)} vocabulary items",
            "info",
        )

        return LabelingSessionResult(
            status="complete",
            session_id=session_id,
            batch=batch,
            vocabulary=vocabulary,
            error=None,
        )

    def sync_annotations(self, payload: dict[str, Any]) -> None:
        """Sync annotations from frontend to storage."""
        session_id = payload["session_id"]
        annotations = payload.get("annotations", [])
        progress = payload.get("progress", [])

        if annotations:
            self.storage.save_annotations(session_id, annotations)

        for p in progress:
            self.storage.update_labeling_progress(
                session_id,
                scene_id=p["scene_id"],
                frame_index=p["frame_index"],
                status=p["status"],
            )

        # Update session counts
        labeled = sum(1 for p in progress if p["status"] == "labeled")
        skipped = sum(1 for p in progress if p["status"] == "skipped")

        session = self.storage.get_labeling_session(session_id)
        if session:
            self.storage.update_labeling_session(
                session_id,
                labeled_count=session["labeled_count"] + labeled,
                skipped_count=session["skipped_count"] + skipped,
            )

        self.log(
            f"Synced {len(annotations)} annotations, {labeled} labeled, {skipped} skipped",
            "info",
        )

    def _generate_caption(self, tags: list[str], template: str) -> str:
        """Generate a caption from confirmed tags."""
        if len(tags) == 0:
            return ""
        elif len(tags) == 1:
            tag_str = tags[0]
        elif len(tags) == 2:
            tag_str = f"{tags[0]} and {tags[1]}"
        else:
            tag_str = ", ".join(tags[:-1]) + f", and {tags[-1]}"
        return template.replace("{tags}", tag_str)

    def _generate_negative_caption(self, tags: list[str]) -> str:
        """Generate a negative caption from rejected tags."""
        if not tags:
            return ""
        return "not featuring " + ", not featuring ".join(tags)

    def export_dataset(
        self,
        config: LabelingConfig,
        output_dir: Path | None = None,
        include_negatives: bool = True,
    ) -> dict[str, Any]:
        """Export labeled data as WebDataset tar."""
        import io
        import tarfile
        from collections import defaultdict
        from datetime import datetime, timezone

        self.log("Exporting dataset...", "info")

        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "assets" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Group confirmed annotations by frame
        all_annotations = self.storage.get_all_confirmed_annotations()
        if not all_annotations:
            return {
                "status": "error",
                "export_path": "",
                "total_images": 0,
                "total_tags": 0,
                "error": "No confirmed annotations to export.",
            }

        frame_tags: dict[tuple[int, int], list[str]] = defaultdict(list)
        for ann in all_annotations:
            key = (ann["scene_id"], ann["frame_index"])
            frame_tags[key].append(ann["tag_text"])

        # Collect rejected tags for negatives
        rejected_tags: dict[tuple[int, int], list[str]] = defaultdict(list)
        if include_negatives:
            conn = self.storage._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT scene_id, frame_index, tag_text FROM frame_annotations WHERE label = 'rejected'"
            )
            for row in cursor.fetchall():
                key = (row["scene_id"], row["frame_index"])
                rejected_tags[key].append(row["tag_text"])
            conn.close()

        # 2. Build tar
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tar_path = output_dir / f"dataset_{timestamp}.tar"
        plugin_dir = Path(__file__).parent.parent.parent

        tag_counts: dict[str, int] = defaultdict(int)
        total_images = 0

        with tarfile.open(tar_path, "w") as tar:
            for (scene_id, frame_index), tags in frame_tags.items():
                frame_path = (
                    plugin_dir / "assets" / "embedded_frames"
                    / f"scene_{scene_id}" / f"frame_{frame_index:04d}.jpg"
                )
                if not frame_path.exists():
                    self.log(f"Frame not found: {frame_path}", "warning")
                    continue

                base_name = f"scene{scene_id}_frame{frame_index:04d}"
                tar.add(str(frame_path), arcname=f"{base_name}.jpg")

                caption = self._generate_caption(tags, config.caption_template)
                caption_bytes = caption.encode("utf-8")
                caption_info = tarfile.TarInfo(name=f"{base_name}.txt")
                caption_info.size = len(caption_bytes)
                tar.addfile(caption_info, io.BytesIO(caption_bytes))

                if include_negatives and (scene_id, frame_index) in rejected_tags:
                    neg_caption = self._generate_negative_caption(
                        rejected_tags[(scene_id, frame_index)]
                    )
                    if neg_caption:
                        neg_bytes = neg_caption.encode("utf-8")
                        neg_info = tarfile.TarInfo(name=f"{base_name}_neg.txt")
                        neg_info.size = len(neg_bytes)
                        tar.addfile(neg_info, io.BytesIO(neg_bytes))

                total_images += 1
                for tag in tags:
                    tag_counts[tag] += 1

            # Add metadata
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_images": total_images,
                "total_tags": len(tag_counts),
                "caption_template": config.caption_template,
                "include_negatives": include_negatives,
                "tag_stats": dict(tag_counts),
            }
            meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
            meta_info = tarfile.TarInfo(name="metadata.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

        self.log(f"Exported {total_images} images to {tar_path}", "info")

        return {
            "status": "complete",
            "export_path": str(tar_path),
            "total_images": total_images,
            "total_tags": len(tag_counts),
            "error": None,
        }
