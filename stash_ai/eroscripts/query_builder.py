"""Query construction for eroscripts search.

Builds the three search-query variants used by the modal picker
(filename / title / title+studio+performer) and a single fallback variant
for any caller that wants one query. All variants get the
``#free-scripts`` Discourse category filter appended so paid/Patreon-gated
threads are excluded — the smoketest confirmed unfiltered queries return
mixed Free + Paid results, while ``#free-scripts`` narrows to direct
attachments only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

CATEGORY_FILTER = "#free-scripts"


_NOISE_TOKENS = {
    # Resolution / quality tags
    "1080p",
    "1080",
    "720p",
    "720",
    "2160p",
    "2160",
    "4k",
    "uhd",
    "8k",
    "480p",
    "540p",
    "hd",
    "fhd",
    "qhd",
    # Codec / container
    "x264",
    "x265",
    "h264",
    "h265",
    "hevc",
    "av1",
    "xvid",
    "webdl",
    "web",
    "webrip",
    "bluray",
    "brrip",
    "dvdrip",
    "dl",
    "mp4",
    "mkv",
    "avi",
    "mov",
    "wmv",
    "m4v",
    # Source / generic
    "xxx",
    "porn",
    "rip",
    "internal",
    "ddp",
    "ddp51",
    "aac",
    "10bit",
    "8bit",
    "hdr",
    "sdr",
    "dolby",
    # Release groups commonly seen in scene names
    "rarbg",
    "yify",
    "yts",
    "etrg",
}

_TOKEN_SPLIT = re.compile(r"[._\-\s]+")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9 ]+")


def strip_noise(text: str) -> str:
    """Remove release-group noise tokens and collapse separators.

    Lowercase, replace ``._-`` with spaces, drop tokens in the noise set,
    drop non-alphanumerics, collapse whitespace.
    """
    if not text:
        return ""
    base = _TOKEN_SPLIT.sub(" ", text.lower())
    base = _NON_ALNUM.sub(" ", base)
    tokens = [t for t in base.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


def filename_to_query_terms(filename: str) -> str:
    """Strip extension and noise from a filename for use as a search query."""
    if not filename:
        return ""
    base = filename.rsplit("/", 1)[-1]
    name, _ = os.path.splitext(base)
    return strip_noise(name)


@dataclass
class QueryInputs:
    """Stash scene fields used to construct queries."""

    title: str | None
    filename: str | None
    studio: str | None = None
    performers: list[str] | None = None


def with_category(query: str) -> str:
    """Append the ``#free-scripts`` filter to a base query."""
    base = query.strip()
    if not base:
        return CATEGORY_FILTER
    return f"{base} {CATEGORY_FILTER}"


def build_modal_queries(inputs: QueryInputs) -> list[str]:
    """Build the parallel query set for the modal multi-query merge.

    Returns up to three deduplicated, non-empty queries (filename, title,
    enriched). Each already has the category filter appended.
    """
    out: list[str] = []
    seen: set[str] = set()

    def push(text: str) -> None:
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        out.append(with_category(cleaned))

    filename_q = filename_to_query_terms(inputs.filename or "")
    title_q = strip_noise(inputs.title or "")

    push(filename_q)
    push(title_q)

    enriched_tokens: list[str] = []
    enriched_seen: set[str] = set()

    def _add_tokens(text: str) -> None:
        for tok in text.split():
            if tok and tok not in enriched_seen:
                enriched_seen.add(tok)
                enriched_tokens.append(tok)

    if title_q:
        _add_tokens(title_q)
    if inputs.studio:
        _add_tokens(strip_noise(inputs.studio))
    if inputs.performers:
        first = next((p for p in inputs.performers if p), None)
        if first:
            _add_tokens(strip_noise(first))
    push(" ".join(enriched_tokens))

    return out


def default_initial_query(inputs: QueryInputs) -> str:
    """Best single query for prefilling the modal search box.

    Prefers stripped title; falls back to stripped filename. The category
    filter is *not* appended here — callers append it via ``with_category``
    when issuing the search, so the user sees a clean string in the input.
    """
    title = strip_noise(inputs.title or "")
    if title:
        return title
    return filename_to_query_terms(inputs.filename or "")
