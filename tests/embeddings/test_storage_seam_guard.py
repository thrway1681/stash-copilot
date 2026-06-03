"""Seam guard for the local EmbeddingStore (issue #3, commit 8).

``EmbeddingStorage`` owns all access to its own SQLite tables and BLOB encoding.
Callers use the high-level operations it exposes (bulk fetch, typed clusters,
frame count/iter/sample, distinct scene ids, the tag-gap threshold cache), never a
raw connection or the pack/unpack helpers. Commit 7 made those primitives private
by name-mangling them (``__get_connection`` / ``__pack_embedding`` /
``__unpack_embedding``); this test fails if any module *outside the store* reaches
back through that seam, so the reach-throughs the refactor removed can't come back.

The guards target the EmbeddingStore-specific primitives only:

* Name-mangled access (``_EmbeddingStorage__...``) -- the one way to reach a private
  member of ``EmbeddingStorage`` from outside the class -- catches a reach-through to
  the connection or the pack/unpack helpers.
* The blob pack/unpack helper names, which are unique to ``EmbeddingStorage``'s BLOB
  layout.

The raw-connection helper name ``_get_connection`` is intentionally *not* grepped
generically: it is a generic name legitimately owned by genuinely separate stores
that are out of scope here -- the Stash-DB ``LibraryStatsAggregator``
(``data/aggregators.py``) and ``StandaloneEmbeddingStorage``
(``standalone_embed.py``), each with its own connection. A reach-through into
*EmbeddingStorage*'s connection is caught by the mangled-name guard instead.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: tests/embeddings/<this file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "stash_ai"

# The single module allowed to own the store's primitives.
OWNER = PACKAGE_ROOT / "embeddings" / "storage.py"


def _source_files() -> list[Path]:
    """First-party Python sources (package + top-level entry scripts), minus the owner."""
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    for entry in ("stash-copilot.py", "standalone_embed.py"):
        path = REPO_ROOT / entry
        if path.exists():
            files.append(path)
    return [path for path in files if path != OWNER]


def test_no_external_mangled_access_to_storage_primitives() -> None:
    """No module outside the store reaches its private members via name mangling.

    After commit 7, the connection and pack/unpack helpers are name-mangled, so the
    only way to touch them from outside ``EmbeddingStorage`` is the mangled prefix
    ``_EmbeddingStorage__``. It must appear in zero first-party sources -- a hit means
    a caller is reaching past the interface into a private primitive again.
    """
    token = "_EmbeddingStorage__"

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _source_files()
        if token in path.read_text(encoding="utf-8")
    ]

    assert not offenders, (
        f"Name-mangled access to EmbeddingStorage's private members ({token}...) "
        f"found outside the store in: {offenders}. Route callers through the store's "
        "high-level operations instead of reaching into its connection or pack/unpack helpers."
    )


def test_pack_unpack_primitives_only_in_storage() -> None:
    """The blob pack/unpack helpers live only in the store.

    ``pack_embedding`` / ``unpack_embedding`` encode the store's BLOB layout; they are
    unique to ``EmbeddingStorage``. Their appearance in any other first-party source
    means a caller re-implemented or reached the pack/unpack primitive instead of using
    a high-level operation that returns ready-made vectors.
    """
    primitives = ("pack_embedding", "unpack_embedding")

    offenders: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in primitives):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "EmbeddingStorage's blob pack/unpack primitives are referenced outside the store in: "
        f"{offenders}. The store owns its BLOB encoding -- callers must use the high-level "
        "operations (bulk fetch, frame iter/sample, etc.) that return decoded vectors."
    )
