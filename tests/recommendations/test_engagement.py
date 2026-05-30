"""Tests for the canonical engagement scoring formula and DB interface (ADR-0004)."""

from datetime import datetime, timedelta, timezone

import pytest

from stash_ai.recommendations.engagement import EngagementCalculator
from stash_ai.recommendations.types import EngagementScoringMethod, SceneEngagementData


def _scene(
    scene_id: int = 1,
    view_count: int = 1,
    o_count: int = 0,
    play_duration: float = 0.0,
    rating: int | None = None,
    last_played: str | None = None,
    first_played: str | None = None,
) -> SceneEngagementData:
    return {
        "scene_id": scene_id,
        "view_count": view_count,
        "o_count": o_count,
        "play_duration": play_duration,
        "last_played": last_played,
        "first_played": first_played,
        "rating": rating,
    }


class TestCalculateBaseScore:
    """Pure unit tests for: o_count*20 + replays*2 + stars*1.5"""

    def test_o_count_weight(self) -> None:
        """o_count contributes 20 per count."""
        calc = EngagementCalculator()
        score, components = calc.calculate_base_score(_scene(o_count=3))
        assert components["o_count"] == pytest.approx(60.0)
        assert score == pytest.approx(60.0)

    def test_replay_count_weight(self) -> None:
        """Each replay (view beyond first) contributes 2."""
        calc = EngagementCalculator()
        # 5 views = 4 replays → 4 * 2 = 8
        score, components = calc.calculate_base_score(_scene(view_count=5))
        assert components["view_count"] == pytest.approx(8.0)
        assert score == pytest.approx(8.0)

    def test_replays_is_max_views_minus_one_zero(self) -> None:
        """Single view produces zero replays (no negative)."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(view_count=1))
        assert components["view_count"] == pytest.approx(0.0)

    def test_replays_zero_views(self) -> None:
        """Zero views also produces zero replay contribution."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(view_count=0))
        assert components["view_count"] == pytest.approx(0.0)

    def test_rating100_conversion(self) -> None:
        """rating100=100 → stars=5.0 → 5.0 * 1.5 = 7.5."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(rating=100))
        assert components["rating"] == pytest.approx(7.5)

    def test_rating100_partial(self) -> None:
        """rating100=60 → stars=3.0 → 3.0 * 1.5 = 4.5."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(rating=60))
        assert components["rating"] == pytest.approx(4.5)

    def test_unrated_none_no_penalty(self) -> None:
        """rating=None gives zero rating contribution (no penalty)."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(rating=None))
        assert components["rating"] == pytest.approx(0.0)

    def test_unrated_zero_no_penalty(self) -> None:
        """rating=0 gives zero rating contribution (no penalty)."""
        calc = EngagementCalculator()
        _, components = calc.calculate_base_score(_scene(rating=0))
        assert components["rating"] == pytest.approx(0.0)

    def test_play_duration_excluded_from_formula(self) -> None:
        """play_duration has zero effect on the score (ADR-0004 excludes it)."""
        calc = EngagementCalculator()
        base = _scene(view_count=3, o_count=2, rating=80)
        with_duration = _scene(view_count=3, o_count=2, rating=80, play_duration=360000.0)
        score_base, components_base = calc.calculate_base_score(base)
        score_with, _ = calc.calculate_base_score(with_duration)
        assert score_base == pytest.approx(score_with)
        assert "play_duration" not in components_base

    def test_full_canonical_formula(self) -> None:
        """o_count*20 + replays*2 + stars*1.5 = 80 + 14 + 7.5 = 101.5."""
        calc = EngagementCalculator()
        # o=4, views=8 (replays=7), rating100=100
        data = _scene(scene_id=7, view_count=8, o_count=4, play_duration=3600.0, rating=100)
        score, components = calc.calculate_base_score(data)
        assert score == pytest.approx(101.5)
        assert components["o_count"] == pytest.approx(80.0)
        assert components["view_count"] == pytest.approx(14.0)
        assert components["rating"] == pytest.approx(7.5)

    def test_zero_engagement(self) -> None:
        """Unrated, unviewed, no O's → score of 0."""
        calc = EngagementCalculator()
        score, components = calc.calculate_base_score(_scene())
        assert score == pytest.approx(0.0)
        assert all(v == pytest.approx(0.0) for v in components.values())


class TestTimeDecayMultiplier:
    """Tests for the exponential time-decay multiplier."""

    def test_no_last_played_returns_min_weight(self) -> None:
        """None last_played returns the configured min_weight floor."""
        calc = EngagementCalculator()
        mult = calc.calculate_time_decay_multiplier(None)
        assert mult == pytest.approx(calc.time_decay["min_weight"])

    def test_very_recent_near_one(self) -> None:
        """Play from yesterday gives a multiplier close to 1.0."""
        calc = EngagementCalculator()
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mult = calc.calculate_time_decay_multiplier(yesterday)
        assert mult > 0.97

    def test_one_half_life_gives_half(self) -> None:
        """Exactly 30 days ago (one half-life) → multiplier ≈ 0.5."""
        calc = EngagementCalculator()
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        mult = calc.calculate_time_decay_multiplier(thirty_days_ago)
        assert mult == pytest.approx(0.5, abs=0.02)

    def test_very_old_clamps_to_min_weight(self) -> None:
        """365-day-old play clamps to the min_weight floor."""
        calc = EngagementCalculator()
        old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        mult = calc.calculate_time_decay_multiplier(old)
        assert mult == pytest.approx(calc.time_decay["min_weight"])

    def test_invalid_date_returns_min_weight(self) -> None:
        """Unparseable date string falls back to min_weight."""
        calc = EngagementCalculator()
        mult = calc.calculate_time_decay_multiplier("not-a-date")
        assert mult == pytest.approx(calc.time_decay["min_weight"])


class TestCalculateScore:
    """Tests for the full calculate_score() dispatcher."""

    def test_base_weighted_uses_raw_score(self) -> None:
        """BASE_WEIGHTED method returns raw score in raw_score field."""
        calc = EngagementCalculator()
        data = _scene(o_count=2, view_count=3, rating=100)
        result = calc.calculate_score(data, EngagementScoringMethod.BASE_WEIGHTED)
        # 2*20 + 2*2 + 5*1.5 = 40 + 4 + 7.5 = 51.5
        assert result.raw_score == pytest.approx(51.5)
        assert result.time_decayed_score == pytest.approx(51.5)

    def test_time_decayed_applies_multiplier(self) -> None:
        """TIME_DECAYED method applies a decay multiplier < 1 for old scenes."""
        calc = EngagementCalculator()
        old_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        data = _scene(o_count=2, view_count=3, rating=100, last_played=old_date)
        result = calc.calculate_score(data, EngagementScoringMethod.TIME_DECAYED)
        assert result.time_decayed_score < result.raw_score

    def test_score_result_contains_component_breakdown(self) -> None:
        """EngagementScore.components contains o_count, view_count, rating keys."""
        calc = EngagementCalculator()
        data = _scene(o_count=1, view_count=2, rating=80)
        result = calc.calculate_score(data)
        assert "o_count" in result.components
        assert "view_count" in result.components
        assert "rating" in result.components
        assert "play_duration" not in result.components


class TestGetEngagement:
    """DB-level tests for get_engagement(scene_ids=None)."""

    def test_all_engaged_scenes(self, patched_engagement_db: None) -> None:
        """get_engagement(None) returns all scenes with views or O's."""
        calc = EngagementCalculator()
        data = calc.get_engagement()
        # Fixture has multiple scenes with view history (see schema.py)
        assert len(data) > 0
        for scene_id, d in data.items():
            assert d["scene_id"] == scene_id
            assert "view_count" in d
            assert "o_count" in d
            assert "rating" in d

    def test_filter_by_scene_ids(self, patched_engagement_db: None) -> None:
        """get_engagement([1, 7]) returns only the requested scene IDs."""
        calc = EngagementCalculator()
        data = calc.get_engagement(scene_ids=[1, 7])
        assert set(data.keys()) <= {1, 7}

    def test_known_o_counts(self, patched_engagement_db: None) -> None:
        """Scene 7 has 4 O's and 8 views per fixture data."""
        calc = EngagementCalculator()
        data = calc.get_engagement(scene_ids=[7])
        assert 7 in data
        assert data[7]["o_count"] == 4
        assert data[7]["view_count"] == 8

    def test_get_all_scene_engagement_alias(self, patched_engagement_db: None) -> None:
        """get_all_scene_engagement() is a thin alias for get_engagement(None)."""
        calc = EngagementCalculator()
        via_alias = calc.get_all_scene_engagement()
        via_new = calc.get_engagement()
        assert set(via_alias.keys()) == set(via_new.keys())

    def test_empty_scene_ids_list(self, patched_engagement_db: None) -> None:
        """get_engagement([]) with an empty list returns an empty dict."""
        calc = EngagementCalculator()
        data = calc.get_engagement(scene_ids=[])
        assert data == {}

    def test_nonexistent_scene_ids_excluded(self, patched_engagement_db: None) -> None:
        """Scene IDs not in the DB are silently absent from results."""
        calc = EngagementCalculator()
        data = calc.get_engagement(scene_ids=[99999])
        assert len(data) == 0


class TestRank:
    """DB-level tests for rank()."""

    def test_rank_all_returns_sorted_scores(self, patched_engagement_db: None) -> None:
        """rank() with no scene_ids returns all engaged scenes sorted by score."""
        calc = EngagementCalculator()
        scores = calc.rank()
        assert len(scores) > 0
        for i in range(len(scores) - 1):
            assert scores[i].raw_score >= scores[i + 1].raw_score

    def test_rank_with_scene_ids(self, patched_engagement_db: None) -> None:
        """rank(scene_ids=[1, 7]) ranks only those scenes."""
        calc = EngagementCalculator()
        scores = calc.rank(scene_ids=[1, 7])
        scene_ids_returned = {s.scene_id for s in scores}
        assert scene_ids_returned <= {1, 7}

    def test_rank_limit(self, patched_engagement_db: None) -> None:
        """limit parameter caps the number of returned scores."""
        calc = EngagementCalculator()
        scores = calc.rank(limit=3)
        assert len(scores) <= 3

    def test_rank_time_decayed_ordering(self, patched_engagement_db: None) -> None:
        """TIME_DECAYED method returns scores ordered by time_decayed_score."""
        calc = EngagementCalculator()
        scores = calc.rank(method=EngagementScoringMethod.TIME_DECAYED)
        assert len(scores) > 0
        for i in range(len(scores) - 1):
            assert scores[i].time_decayed_score >= scores[i + 1].time_decayed_score

    def test_rank_canonical_scores(self, patched_engagement_db: None) -> None:
        """Scene 10 (o=4, views=10, rating=100) scores 105.5 by canonical formula."""
        calc = EngagementCalculator()
        scores = calc.rank(scene_ids=[10])
        assert len(scores) == 1
        # o_count=4, replays=9, rating100=100→stars=5: 4*20 + 9*2 + 5*1.5 = 105.5
        assert scores[0].raw_score == pytest.approx(105.5)

    def test_get_top_engaged_scenes_delegates_to_rank(
        self, patched_engagement_db: None
    ) -> None:
        """get_top_engaged_scenes() returns same result as rank(limit=N)."""
        calc = EngagementCalculator()
        via_legacy = calc.get_top_engaged_scenes(limit=5)
        via_rank = calc.rank(limit=5)
        assert [s.scene_id for s in via_legacy] == [s.scene_id for s in via_rank]
