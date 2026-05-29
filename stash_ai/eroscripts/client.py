"""Discourse REST client for discuss.eroscripts.com.

Implements the three operations the plugin needs:
- ``validate_session``: confirms the cookie is valid and returns the username
- ``search``: hits ``/search.json`` and returns merged topic+post data
- ``get_topic``: fetches the topic JSON used to extract attachment URLs
- ``download_attachment``: GETs an upload URL with cookie attached

The client is intentionally small and stateless beyond the cookie. All HTTP
errors are converted into typed results so callers don't have to handle
``requests`` exceptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

import requests


BASE_URL = "https://discuss.eroscripts.com"
USER_AGENT = "Stash-Copilot/0.1 (eroscripts integration)"
DEFAULT_TIMEOUT = 15  # seconds


@dataclass
class SessionInfo:
    """Result of ``validate_session``."""

    valid: bool
    username: Optional[str] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class SearchResult:
    """One ranked entry returned by ``search``.

    Combines topic-level fields (title, tags, thumbnails) with post-level
    fields (creator username, like_count, blurb) — the search response splits
    these across ``topics[]`` and ``posts[]`` arrays joined by topic id.
    """

    topic_id: int
    title: str
    slug: str
    url: str
    excerpt: str
    creator_username: Optional[str]
    creator_avatar_template: Optional[str]
    like_count: int
    created_at: Optional[str]
    tags: list[str]
    thumbnails: list[dict]  # raw Discourse thumbnail objects (multi-resolution)
    category_id: Optional[int]


@dataclass
class SearchResponse:
    """Result of ``search``."""

    ok: bool
    results: list[SearchResult]
    error: Optional[str] = None
    status_code: Optional[int] = None
    rate_limited: bool = False


@dataclass
class TopicAttachment:
    """A downloadable file linked from a topic's first post."""

    filename: str                   # e.g. "MyScript_v3.funscript"
    url: str                        # short-url form, e.g. "/uploads/short-url/xyz.funscript"
    size_bytes: Optional[int] = None  # populated via HEAD probe; None if unknown


@dataclass
class TopicResponse:
    """Result of ``get_topic``."""

    ok: bool
    topic_id: int
    title: Optional[str] = None
    cooked_html: Optional[str] = None
    funscript_attachments: Optional[list[TopicAttachment]] = None
    external_links: Optional[list[str]] = None  # mega.nz, gumroad, patreon, etc.
    error: Optional[str] = None
    status_code: Optional[int] = None


_FUNSCRIPT_HREF_PATTERN = re.compile(
    r'^(?:/uploads/short-url/[^?]+\.(?:funscript|zip))$|^upload://[^?]+\.(?:funscript|zip)$',
    re.IGNORECASE,
)
_EXTERNAL_HOST_HREF = re.compile(
    r'href="(https?://(?:[^"]*\.)?(?:mega\.nz|gumroad\.com|patreon\.com|drive\.google\.com|mediafire\.com|dropbox\.com)/[^"]+)"',
    re.IGNORECASE,
)


class EroScriptsClient:
    """Stateless-ish wrapper over ``requests`` for the Discourse endpoints."""

    def __init__(self, session_cookie: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._cookie = session_cookie
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        self._session.cookies.set("_t", session_cookie, domain="discuss.eroscripts.com")

    # ------------------------------------------------------------------ auth
    def validate_session(self) -> SessionInfo:
        try:
            resp = self._session.get(
                urljoin(BASE_URL, "/session/current.json"),
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            return SessionInfo(valid=False, error=f"Network error: {e}")

        if resp.status_code != 200:
            return SessionInfo(
                valid=False,
                status_code=resp.status_code,
                error="Session is invalid or expired (received "
                      f"HTTP {resp.status_code}). Re-copy your `_t` cookie.",
            )
        try:
            data = resp.json()
        except ValueError:
            return SessionInfo(
                valid=False,
                status_code=resp.status_code,
                error="EroScripts returned a non-JSON response.",
            )
        user = data.get("current_user") or {}
        username = user.get("username")
        if not username:
            return SessionInfo(
                valid=False,
                status_code=resp.status_code,
                error="Cookie is anonymous — make sure you are logged into "
                      "eroscripts.com before copying.",
            )
        return SessionInfo(valid=True, username=username, status_code=resp.status_code)

    # ---------------------------------------------------------------- search
    def search(self, query: str) -> SearchResponse:
        try:
            resp = self._session.get(
                urljoin(BASE_URL, "/search.json"),
                params={"q": query},
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            return SearchResponse(ok=False, results=[], error=f"Network error: {e}")

        if resp.status_code == 429:
            return SearchResponse(
                ok=False, results=[], status_code=429, rate_limited=True,
                error="EroScripts is rate-limiting requests. Wait 30 seconds.",
            )
        if resp.status_code in (401, 403) or _is_login_redirect(resp):
            return SearchResponse(
                ok=False, results=[], status_code=resp.status_code,
                error="Your eroscripts session has expired. Re-paste cookie.",
            )
        if resp.status_code != 200:
            return SearchResponse(
                ok=False, results=[], status_code=resp.status_code,
                error=f"EroScripts returned HTTP {resp.status_code}.",
            )
        try:
            data = resp.json()
        except ValueError:
            return SearchResponse(
                ok=False, results=[], status_code=resp.status_code,
                error="EroScripts returned a malformed search response.",
            )

        return SearchResponse(ok=True, results=_parse_search_results(data))

    # ----------------------------------------------------------------- topic
    def get_topic(self, topic_id: int) -> TopicResponse:
        try:
            resp = self._session.get(
                urljoin(BASE_URL, f"/t/{topic_id}.json"),
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            return TopicResponse(ok=False, topic_id=topic_id, error=f"Network error: {e}")

        if resp.status_code in (401, 403) or _is_login_redirect(resp):
            return TopicResponse(
                ok=False, topic_id=topic_id, status_code=resp.status_code,
                error="Your eroscripts session has expired. Re-paste cookie.",
            )
        if resp.status_code == 404:
            return TopicResponse(
                ok=False, topic_id=topic_id, status_code=404,
                error="Topic not found (it may have been deleted).",
            )
        if resp.status_code != 200:
            return TopicResponse(
                ok=False, topic_id=topic_id, status_code=resp.status_code,
                error=f"EroScripts returned HTTP {resp.status_code}.",
            )
        try:
            data = resp.json()
        except ValueError:
            return TopicResponse(
                ok=False, topic_id=topic_id, status_code=resp.status_code,
                error="EroScripts returned a malformed topic response.",
            )

        posts = (data.get("post_stream") or {}).get("posts") or []
        cooked = posts[0].get("cooked", "") if posts else ""
        attachments = _extract_funscript_attachments(cooked)
        external = _extract_external_links(cooked)
        # Best-effort: probe each attachment with a HEAD request to surface
        # file size in the modal picker. We do this sequentially with a
        # short per-request timeout — for the typical 1-3 attachments this
        # adds < 1s and keeps the picker UX informative. Sizes are optional;
        # any HEAD failure leaves `size_bytes` as None and the JS just hides
        # the size suffix for that one button.
        for a in attachments:
            a.size_bytes = self._probe_size(a.url)
        return TopicResponse(
            ok=True,
            topic_id=topic_id,
            title=data.get("title"),
            cooked_html=cooked,
            funscript_attachments=attachments,
            external_links=external,
        )

    # ---------------------------------------------------------- size probe
    def _probe_size(self, attachment_url: str) -> Optional[int]:
        """Return the attachment size via a HEAD request, or None on error.

        Discourse redirects ``/uploads/short-url/...`` to the underlying CDN
        URL on a 30x; we follow the redirect and use the Content-Length
        from the final response. Auth cookie travels via the existing
        session so we don't need to re-pass it.
        """
        if not attachment_url or attachment_url.startswith("upload://"):
            return None
        url = attachment_url
        if url.startswith("/"):
            url = urljoin(BASE_URL, url)
        try:
            resp = self._session.head(url, timeout=5, allow_redirects=True)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        cl = resp.headers.get("Content-Length")
        if not cl:
            return None
        try:
            return int(cl)
        except ValueError:
            return None

    # ------------------------------------------------------------- download
    def download_attachment(self, attachment_url: str) -> tuple[bool, bytes, Optional[str]]:
        """Download an attachment URL. Follows Discourse's CDN redirect.

        Returns ``(ok, content, error_message)``.
        """
        url = attachment_url
        if url.startswith("upload://"):
            # Discourse short upload form — would normally hit /uploads/lookup-urls
            # but for now we don't surface upload:// links from the parser.
            return False, b"", "Unsupported upload:// link form."
        if url.startswith("/"):
            url = urljoin(BASE_URL, url)
        try:
            resp = self._session.get(
                url,
                timeout=self._timeout,
                allow_redirects=True,
                stream=False,
            )
        except requests.RequestException as e:
            return False, b"", f"Network error: {e}"
        if resp.status_code in (401, 403) or _is_login_redirect(resp):
            return False, b"", "Session expired during download. Re-paste cookie."
        if resp.status_code == 404:
            return False, b"", "Attachment is no longer available (HTTP 404)."
        if resp.status_code != 200:
            return False, b"", f"Download failed: HTTP {resp.status_code}."
        return True, resp.content, None


# ============================================================ parse helpers


def _is_login_redirect(resp: requests.Response) -> bool:
    """Discourse returns 302 → /login when a session is invalid."""
    if resp.status_code not in (301, 302, 303, 307, 308):
        return False
    location = resp.headers.get("Location", "")
    return "/login" in location


def _parse_search_results(data: dict) -> list[SearchResult]:
    """Merge ``topics[]`` and ``posts[]`` from a /search.json payload.

    Discourse search returns both arrays; we join by ``post.topic_id`` so
    each topic gets its first-post creator/likes/excerpt without an extra
    request.
    """
    topics = data.get("topics") or []
    posts_by_topic: dict[int, dict] = {}
    for p in data.get("posts") or []:
        tid = p.get("topic_id")
        if isinstance(tid, int) and tid not in posts_by_topic:
            posts_by_topic[tid] = p

    results: list[SearchResult] = []
    for t in topics:
        tid = t.get("id")
        if not isinstance(tid, int):
            continue
        post = posts_by_topic.get(tid, {})
        slug = t.get("slug", "")
        results.append(SearchResult(
            topic_id=tid,
            title=t.get("title", "(untitled)"),
            slug=slug,
            url=f"{BASE_URL}/t/{slug}/{tid}" if slug else f"{BASE_URL}/t/{tid}",
            excerpt=post.get("blurb") or t.get("excerpt") or "",
            creator_username=post.get("username"),
            creator_avatar_template=post.get("avatar_template"),
            like_count=int(post.get("like_count") or 0),
            created_at=t.get("created_at"),
            tags=[tag.get("name") for tag in (t.get("tags") or []) if tag.get("name")],
            thumbnails=t.get("thumbnails") or [],
            category_id=t.get("category_id"),
        ))
    return results


class _AttachmentLinkParser(HTMLParser):
    """Walk the cooked HTML and capture ``<a href=...>filename</a>`` pairs.

    Discourse renders attachments as ``<a class="attachment" href="/uploads/
    short-url/<hash>.funscript">My Friendly Name.funscript</a>``. The href
    carries an opaque short-URL hash; the link's inner text is the user-
    facing filename a human will recognize. We grab both.
    """

    def __init__(self) -> None:
        super().__init__()
        self._links: list[tuple[str, list[str]]] = []  # (href, text-fragments)
        self._depth = 0  # nesting depth inside a matching <a>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            href = next((v for k, v in attrs if k == "href" and v), None) or ""
            if _FUNSCRIPT_HREF_PATTERN.match(href):
                self._links.append((href, []))
                self._depth = 1
                return
        if self._depth:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:  # noqa: ARG002 — required signature
        if self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth and self._links:
            self._links[-1][1].append(data)

    def attachments(self) -> list[TopicAttachment]:
        out: list[TopicAttachment] = []
        seen: set[str] = set()
        for href, fragments in self._links:
            if href in seen:
                continue
            seen.add(href)
            display = "".join(fragments).strip()
            if not display:
                # Fallback: derive from the URL hash (degraded UX, but better
                # than dropping the attachment entirely).
                display = href.rsplit("/", 1)[-1].split("?", 1)[0]
            out.append(TopicAttachment(filename=display, url=href))
        return out


def _extract_funscript_attachments(cooked_html: str) -> list[TopicAttachment]:
    """Pull ``.funscript``/``.zip`` short-URL hrefs out of the post HTML.

    Uses an HTML parser rather than a flat regex so we can recover the
    user-friendly link text (e.g. ``"MyScript_v3.funscript"``) instead of
    the opaque Discourse upload hash that lives in the URL.
    """
    parser = _AttachmentLinkParser()
    try:
        parser.feed(cooked_html or "")
    except Exception:  # noqa: BLE001 — malformed HTML shouldn't break search
        return []
    return parser.attachments()


def _extract_external_links(cooked_html: str) -> list[str]:
    """Pull external paywall/host links so we can surface them to the user."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _EXTERNAL_HOST_HREF.finditer(cooked_html or ""):
        href = m.group(1)
        if href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out
