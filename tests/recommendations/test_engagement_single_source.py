"""Single-source guard for the Engagement Score (issue #1, commit 9 / ADR-0004).

The Engagement Score formula and the engagement query must live in *exactly one*
module -- ``EngagementCalculator`` in ``stash_ai/recommendations/engagement.py``.
Historically the formula drifted across four+ call sites (different weights, an
inconsistent rating scale, an ``o*3`` docstring disagreeing with ``o*20`` code) and
the ``scenes_view_dates``/``scenes_o_dates`` scoring query was copy-pasted into
several tools. These tests fail if any of that reach-through comes back.

The guards intentionally target the *scoring* computation and any SQL that embeds an
engagement score -- not raw access to ``scenes_view_dates`` / ``scenes_o_dates``,
which are shared infrastructure that legitimately back unrelated statistics (e.g.
per-performer view/o sums) and frame the canonical query itself.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root: tests/recommendations/<this file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "stash_ai"

# The single module that is allowed to own the engagement formula + query.
OWNER = PACKAGE_ROOT / "recommendations" / "engagement.py"


def _source_files() -> list[Path]:
    """All first-party Python sources (package + top-level entry points)."""
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    for entry in ("stash-copilot.py", "standalone_embed.py"):
        path = REPO_ROOT / entry
        if path.exists():
            files.append(path)
    return files


def test_scoring_functions_defined_in_exactly_one_module() -> None:
    """The engagement formula + query are defined only in engagement.py.

    Each is a ``def`` that must appear in exactly one file, and that file must be
    ``recommendations/engagement.py``. Catches a second copy of the score
    computation or the engagement-fetch query being introduced elsewhere.
    """
    canonical_defs = (
        "get_engagement",  # the engagement query
        "calculate_base_score",  # the weighted formula
        "calculate_time_decay_multiplier",  # the recency-decay term
        "calculate_score",  # base/time-decayed dispatch
    )

    for name in canonical_defs:
        pattern = re.compile(rf"^\s*def {re.escape(name)}\b", re.MULTILINE)
        defining_files = [
            path for path in _source_files() if pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert defining_files == [OWNER], (
            f"`def {name}` (part of the canonical Engagement Score per ADR-0004) "
            f"must be defined only in {OWNER.relative_to(REPO_ROOT)}, "
            f"but was found in: {[str(p.relative_to(REPO_ROOT)) for p in defining_files]}. "
            "Route callers through EngagementCalculator instead of reimplementing it."
        )


def test_no_engagement_score_embedded_in_sql() -> None:
    """No module may compute an engagement score inside a SQL string.

    Two fingerprints of the old SQL-side scoring (removed from the date/rating
    query tools): the ``as engagement_score`` projection alias and the weighted
    ``* 20.0`` o-count multiplier. Both must appear in zero source files -- scoring
    is pure Python in EngagementCalculator (ADR-0004: "SQL-side ORDER BY/scoring in
    the tools is removed").
    """
    sql_score_alias = re.compile(r"as\s+engagement_score\b", re.IGNORECASE)
    sql_weight_multiply = re.compile(r"\*\s*20\.0")  # SQL float weight, e.g. o_count * 20.0

    offenders: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        if sql_score_alias.search(text) or sql_weight_multiply.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "Engagement score is embedded in SQL in: "
        f"{offenders}. The score must be computed via EngagementCalculator.calculate_score "
        "in Python (ADR-0004), not reimplemented in a query."
    )
