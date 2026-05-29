"""Unit tests for stash_ai.eroscripts.client — HTML parsing + response shape."""

from __future__ import annotations

from unittest.mock import MagicMock

from stash_ai.eroscripts.client import (
    _AttachmentLinkParser,
    _extract_external_links,
    _extract_funscript_attachments,
    _is_login_redirect,
    _parse_search_results,
)


# ---------- HTML attachment parsing -----------------------------------------

class TestAttachmentParser:
    def test_extracts_friendly_filename_not_hash(self) -> None:
        # Real shape: href is the opaque short-URL hash, inner text is the
        # uploader-provided filename. We must surface the inner text.
        html = (
            '<a class="attachment" '
            'href="/uploads/short-url/xZqem6J6lIfXZ89NX7IwrTLNtY8.funscript">'
            'Friendly Name.funscript</a>'
        )
        atts = _extract_funscript_attachments(html)
        assert len(atts) == 1
        assert atts[0].filename == "Friendly Name.funscript"
        assert atts[0].url == "/uploads/short-url/xZqem6J6lIfXZ89NX7IwrTLNtY8.funscript"

    def test_extracts_multiple_attachments(self) -> None:
        # Common multi-axis pattern: main + roll + pitch.
        html = """
            <p>Here are the scripts:</p>
            <a class="attachment" href="/uploads/short-url/aaa.funscript">scene.funscript</a>
            <a class="attachment" href="/uploads/short-url/bbb.funscript">scene.roll.funscript</a>
            <a class="attachment" href="/uploads/short-url/ccc.funscript">scene.pitch.funscript</a>
        """
        atts = _extract_funscript_attachments(html)
        names = [a.filename for a in atts]
        assert names == ["scene.funscript", "scene.roll.funscript", "scene.pitch.funscript"]

    def test_dedupes_repeated_hrefs(self) -> None:
        # Same href twice (e.g. linked from both an icon and the filename
        # text) should produce one attachment, not two.
        html = (
            '<a class="attachment" href="/uploads/short-url/abc.funscript">My.funscript</a>'
            '<a class="attachment" href="/uploads/short-url/abc.funscript">My.funscript</a>'
        )
        atts = _extract_funscript_attachments(html)
        assert len(atts) == 1

    def test_falls_back_to_url_filename_when_inner_empty(self) -> None:
        # Sometimes the link wraps only an icon → no inner text. Fall back
        # to the URL hash so the attachment is still selectable.
        html = (
            '<a href="/uploads/short-url/abc123.funscript">'
            '<svg></svg>'
            '</a>'
        )
        atts = _extract_funscript_attachments(html)
        assert len(atts) == 1
        assert atts[0].filename == "abc123.funscript"

    def test_ignores_non_funscript_extensions(self) -> None:
        # Image/text files shouldn't show up as funscript attachments.
        html = (
            '<a class="attachment" href="/uploads/short-url/img.jpg">preview</a>'
            '<a class="attachment" href="/uploads/short-url/notes.txt">readme</a>'
            '<a class="attachment" href="/uploads/short-url/yes.funscript">good.funscript</a>'
        )
        atts = _extract_funscript_attachments(html)
        assert [a.filename for a in atts] == ["good.funscript"]

    def test_accepts_zip_attachments(self) -> None:
        # Some packs ship as .zip rather than bare .funscript.
        html = (
            '<a class="attachment" href="/uploads/short-url/pack.zip">scripts.zip</a>'
        )
        atts = _extract_funscript_attachments(html)
        assert len(atts) == 1
        assert atts[0].filename == "scripts.zip"

    def test_empty_html_returns_empty(self) -> None:
        assert _extract_funscript_attachments("") == []
        assert _extract_funscript_attachments(None) == []  # type: ignore[arg-type]

    def test_handles_malformed_html_gracefully(self) -> None:
        # Truncated/garbage HTML must not raise.
        html = '<a href="/uploads/short-url/x.funscript">fine.funscript<a unclosed'
        # Should not raise; whatever it returns is acceptable.
        atts = _extract_funscript_attachments(html)
        assert isinstance(atts, list)

    def test_parser_class_directly(self) -> None:
        # Verify the underlying class works the same as the helper.
        parser = _AttachmentLinkParser()
        parser.feed('<a href="/uploads/short-url/test.funscript">My.funscript</a>')
        atts = parser.attachments()
        assert len(atts) == 1
        assert atts[0].filename == "My.funscript"


# ---------- External-host link extraction -----------------------------------

class TestExternalLinks:
    def test_extracts_paywall_hosts(self) -> None:
        html = (
            '<a href="https://patreon.com/scripter">Patreon</a>'
            '<a href="https://funscripts.gumroad.com/l/abc">Gumroad</a>'
            '<a href="https://mega.nz/file/xyz">Mega</a>'
        )
        links = _extract_external_links(html)
        assert any("patreon.com" in u for u in links)
        assert any("gumroad.com" in u for u in links)
        assert any("mega.nz" in u for u in links)

    def test_dedupes_repeated_links(self) -> None:
        html = (
            '<a href="https://mega.nz/file/x">link</a>'
            '<a href="https://mega.nz/file/x">link</a>'
        )
        assert _extract_external_links(html) == ["https://mega.nz/file/x"]

    def test_empty_html(self) -> None:
        assert _extract_external_links("") == []

    def test_ignores_unrelated_domains(self) -> None:
        html = '<a href="https://google.com/search">google</a>'
        assert _extract_external_links(html) == []


# ---------- Search response merging -----------------------------------------

class TestParseSearchResults:
    def test_merges_topics_with_matching_posts(self) -> None:
        data = {
            "topics": [
                {
                    "id": 100, "title": "Test Topic", "slug": "test-topic",
                    "tags": [{"name": "vr"}, {"name": "pov"}],
                    "thumbnails": [{"url": "https://example.com/thumb.jpg",
                                    "width": 200, "height": 100}],
                    "category_id": 14,
                    "created_at": "2024-01-01T00:00:00Z",
                    "excerpt": "topic excerpt",
                },
            ],
            "posts": [
                {"id": 999, "topic_id": 100, "username": "alice",
                 "avatar_template": "/u/alice/{size}.png",
                 "like_count": 12, "blurb": "post blurb"},
            ],
        }
        out = _parse_search_results(data)
        assert len(out) == 1
        r = out[0]
        assert r.topic_id == 100
        assert r.title == "Test Topic"
        assert r.creator_username == "alice"
        assert r.like_count == 12
        # Post blurb takes precedence over topic excerpt — that's where
        # Discourse puts the meaningful first-post snippet.
        assert r.excerpt == "post blurb"
        assert r.tags == ["vr", "pov"]
        assert r.url == "https://discuss.eroscripts.com/t/test-topic/100"

    def test_topic_without_matching_post_still_emitted(self) -> None:
        # Edge case: search returned a topic but the corresponding post
        # wasn't in the response (rare but possible). We should still
        # produce a result, just with empty creator/likes.
        data = {
            "topics": [{"id": 1, "title": "Lonely", "slug": "lonely",
                        "tags": [], "thumbnails": []}],
            "posts": [],
        }
        out = _parse_search_results(data)
        assert len(out) == 1
        assert out[0].creator_username is None
        assert out[0].like_count == 0
        assert out[0].excerpt == ""

    def test_handles_missing_topic_id(self) -> None:
        # Defensive: a topic record without an integer id should be skipped.
        data = {"topics": [{"title": "no id"}], "posts": []}
        assert _parse_search_results(data) == []

    def test_url_falls_back_to_id_only_when_slug_missing(self) -> None:
        data = {
            "topics": [{"id": 5, "title": "x", "slug": "", "tags": [], "thumbnails": []}],
            "posts": [],
        }
        out = _parse_search_results(data)
        assert out[0].url == "https://discuss.eroscripts.com/t/5"


# ---------- Login-redirect detection ----------------------------------------

class TestIsLoginRedirect:
    def _resp(self, status: int, location: str = "") -> MagicMock:
        m = MagicMock()
        m.status_code = status
        m.headers = {"Location": location}
        return m

    def test_302_to_login_is_redirect(self) -> None:
        assert _is_login_redirect(self._resp(302, "/login")) is True

    def test_302_to_other_path_is_not_login_redirect(self) -> None:
        assert _is_login_redirect(self._resp(302, "/topic/foo")) is False

    def test_200_is_not_redirect(self) -> None:
        assert _is_login_redirect(self._resp(200, "")) is False

    def test_307_to_login_is_redirect(self) -> None:
        assert _is_login_redirect(self._resp(307, "/login?return=/")) is True
