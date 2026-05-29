"""Unit tests for stash_ai.eroscripts.query_builder."""

from __future__ import annotations

from stash_ai.eroscripts.query_builder import (
    CATEGORY_FILTER,
    QueryInputs,
    build_modal_queries,
    default_initial_query,
    filename_to_query_terms,
    strip_noise,
    with_category,
)


class TestStripNoise:
    def test_empty_input_returns_empty(self) -> None:
        assert strip_noise("") == ""
        assert strip_noise(None) == ""  # type: ignore[arg-type]

    def test_lowercases_and_collapses_separators(self) -> None:
        assert strip_noise("My.Awesome_Title-Here") == "my awesome title here"

    def test_strips_resolution_tags(self) -> None:
        assert strip_noise("Scene 1080p") == "scene"
        assert strip_noise("Scene 4K UHD") == "scene"
        assert strip_noise("Scene 720p HD") == "scene"

    def test_strips_codec_and_container_tags(self) -> None:
        assert strip_noise("Title x264 AAC mp4") == "title"
        assert strip_noise("Title HEVC 10bit") == "title"

    def test_strips_web_dl_split(self) -> None:
        # WEB-DL splits to ["web", "dl"] — both must be in the noise list.
        assert "dl" not in strip_noise("Title.WEB-DL.x264-RARBG")
        assert "rarbg" not in strip_noise("Title.WEB-DL.x264-RARBG")
        assert strip_noise("Title.WEB-DL.x264-RARBG") == "title"

    def test_drops_non_alphanumerics(self) -> None:
        assert strip_noise("title!@# (2024)") == "title 2024"

    def test_collapses_whitespace_runs(self) -> None:
        assert strip_noise("a   b  c") == "a b c"

    def test_preserves_meaningful_words(self) -> None:
        # Performer name + studio shouldn't be touched.
        assert strip_noise("Lily Rader Blacked") == "lily rader blacked"


class TestFilenameToQueryTerms:
    def test_strips_extension(self) -> None:
        assert filename_to_query_terms("video.mp4") == "video"
        assert filename_to_query_terms("video.MKV") == "video"

    def test_handles_path_prefix(self) -> None:
        assert filename_to_query_terms("/path/to/video.mp4") == "video"

    def test_strips_release_noise(self) -> None:
        out = filename_to_query_terms("Lily.Rader.BLACKED.XXX.1080p.WEB-DL.x264-RARBG.mp4")
        assert out == "lily rader blacked"

    def test_empty_input(self) -> None:
        assert filename_to_query_terms("") == ""


class TestWithCategory:
    def test_appends_category_filter(self) -> None:
        assert with_category("query") == f"query {CATEGORY_FILTER}"

    def test_empty_query_returns_just_filter(self) -> None:
        assert with_category("") == CATEGORY_FILTER
        assert with_category("   ") == CATEGORY_FILTER


class TestBuildModalQueries:
    def test_emits_only_distinct_variants(self) -> None:
        # Title and filename normalize to the same string → dedupe to 1 query.
        qs = build_modal_queries(QueryInputs(
            title="Same Stuff",
            filename="Same.Stuff.1080p.mp4",
        ))
        assert len(qs) == 1
        assert qs[0] == f"same stuff {CATEGORY_FILTER}"

    def test_emits_three_variants_when_inputs_diverge(self) -> None:
        qs = build_modal_queries(QueryInputs(
            title="Threesome with Two BBC",
            filename="LilRdr_Scene_2024.04.15.mp4",
            studio="Blacked",
            performers=["Lily Rader"],
        ))
        assert len(qs) == 3

    def test_enriched_query_dedupes_repeated_tokens(self) -> None:
        # "blacked" appears in both title and studio → enriched query
        # should contain it once.
        qs = build_modal_queries(QueryInputs(
            title="Lily Blacked Threesome",
            filename="lily.blacked.mp4",
            studio="Blacked",
            performers=["Lily"],
        ))
        # The enriched variant is the third query (or the one with extra tokens).
        # Find any query with all three tokens; it should not contain "blacked"
        # twice.
        for q in qs:
            occurrences = q.lower().split().count("blacked")
            assert occurrences <= 1, f"duplicated 'blacked' in {q!r}"

    def test_all_queries_have_category_filter(self) -> None:
        qs = build_modal_queries(QueryInputs(
            title="X", filename="y.mp4", studio="Z", performers=["W"],
        ))
        for q in qs:
            assert q.endswith(CATEGORY_FILTER)

    def test_filename_only_returns_one_query(self) -> None:
        qs = build_modal_queries(QueryInputs(
            title=None, filename="ScenePack_001.mkv",
        ))
        assert qs == [f"scenepack 001 {CATEGORY_FILTER}"]

    def test_no_inputs_returns_empty(self) -> None:
        qs = build_modal_queries(QueryInputs(title=None, filename=None))
        assert qs == []


class TestDefaultInitialQuery:
    def test_prefers_title_when_available(self) -> None:
        out = default_initial_query(QueryInputs(
            title="My Scene", filename="my.scene.1080p.mp4",
        ))
        assert out == "my scene"

    def test_falls_back_to_filename_when_title_empty(self) -> None:
        out = default_initial_query(QueryInputs(
            title="", filename="My.Scene.1080p.mp4",
        ))
        assert out == "my scene"

    def test_no_category_filter_in_initial_query(self) -> None:
        # The initial query is what gets prefilled into the search input —
        # the category filter is appended separately at search time.
        out = default_initial_query(QueryInputs(
            title="Some Scene", filename=None,
        ))
        assert CATEGORY_FILTER not in out
