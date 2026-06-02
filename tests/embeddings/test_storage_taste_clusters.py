"""Tests for EmbeddingStorage.get_taste_clusters (typed read objects)."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.recommendations.types import TasteCluster


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary embedding-store database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db: str) -> EmbeddingStorage:
    """Create storage instance with a temp database."""
    return EmbeddingStorage(db_path=temp_db, model_key="test_model")


def _make_cluster(cluster_id: int) -> TasteCluster:
    """Build a TasteCluster with distinct, round-trippable field values."""
    return TasteCluster(
        cluster_id=cluster_id,
        centroid=np.array([float(cluster_id), 0.1, 0.2], dtype=np.float32),
        scene_ids=[cluster_id * 10, cluster_id * 10 + 1],
        engagement_total=float(cluster_id) * 100.0,
        engagement_share=0.25 * cluster_id,
        auto_label=f"label-{cluster_id}",
        user_label=None,
        weight_override=None,
        excluded=False,
        tag_matches=[{"text": f"tag-{cluster_id}", "similarity": 0.9, "source": "stash_tag"}],
    )


class TestGetTasteClustersTyped:
    def test_no_clusters_returns_empty_list(self, storage: EmbeddingStorage) -> None:
        assert storage.get_taste_clusters("test_model") == []

    def test_returns_typed_cluster_objects(self, storage: EmbeddingStorage) -> None:
        storage.save_taste_clusters([_make_cluster(1)], "test_model")

        clusters = storage.get_taste_clusters("test_model")

        assert len(clusters) == 1
        assert isinstance(clusters[0], TasteCluster)

    def test_round_trips_field_values(self, storage: EmbeddingStorage) -> None:
        storage.save_taste_clusters([_make_cluster(2)], "test_model")

        cluster = storage.get_taste_clusters("test_model")[0]

        assert cluster.cluster_id == 2
        assert cluster.scene_ids == [20, 21]
        assert cluster.engagement_total == pytest.approx(200.0)
        assert cluster.engagement_share == pytest.approx(0.5)
        assert cluster.auto_label == "label-2"
        assert cluster.user_label is None
        assert cluster.weight_override is None
        assert cluster.excluded is False
        assert cluster.tag_matches == [
            {"text": "tag-2", "similarity": 0.9, "source": "stash_tag"}
        ]

    def test_centroid_is_float32_array(self, storage: EmbeddingStorage) -> None:
        storage.save_taste_clusters([_make_cluster(3)], "test_model")

        cluster = storage.get_taste_clusters("test_model")[0]

        assert isinstance(cluster.centroid, np.ndarray)
        assert cluster.centroid.dtype == np.float32
        np.testing.assert_allclose(cluster.centroid, [3.0, 0.1, 0.2], rtol=1e-6)

    def test_ordered_by_cluster_id(self, storage: EmbeddingStorage) -> None:
        storage.save_taste_clusters(
            [_make_cluster(3), _make_cluster(1), _make_cluster(2)], "test_model"
        )

        clusters = storage.get_taste_clusters("test_model")

        assert [c.cluster_id for c in clusters] == [1, 2, 3]

    def test_respects_model_key(self, temp_db: str) -> None:
        model_a = EmbeddingStorage(db_path=temp_db, model_key="model_a")
        model_b = EmbeddingStorage(db_path=temp_db, model_key="model_b")
        model_a.save_taste_clusters([_make_cluster(1)], "model_a")
        model_b.save_taste_clusters([_make_cluster(2)], "model_b")

        clusters_a = model_a.get_taste_clusters("model_a")

        assert [c.cluster_id for c in clusters_a] == [1]
