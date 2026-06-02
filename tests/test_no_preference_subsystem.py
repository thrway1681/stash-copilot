"""Guard for the removed swipe/preference subsystem (issue #2, commit 10 / ADR-0001).

The explicit-preference subsystem -- the ``stash_ai/preferences`` package (a Bayesian
Bradley-Terry model + pair selection + session management), its seven ``preference_*``
plugin tasks, the swipe-trainer UI, the three ``preference_*`` database tables, and the
engine's optional preference-model blend -- was removed across commits 1-9 of issue #2.
Taste is now inferred implicitly from Engagement (see ``CONTEXT.md`` / ADR-0001).

These tests fail if any piece of the subsystem creeps back: the package, an import of
it, or a ``preference_*`` plugin task. The *unrelated* ``performer_preference``
recommendation mode is explicitly allowed -- it is not part of this subsystem.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# Repo root: tests/<this file> -> parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "stash_ai"
PLUGIN_MANIFEST = REPO_ROOT / "stash-copilot.yml"

# The one rec-mode that legitimately contains the substring "preference" -- the
# performer-preference recommendation mode is unrelated to the removed subsystem.
ALLOWED_PREFERENCE_TOKENS = frozenset({"performer_preference"})


def _source_files() -> list[Path]:
    """All first-party Python sources (package + top-level entry points)."""
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    for entry in ("stash-copilot.py", "standalone_embed.py"):
        path = REPO_ROOT / entry
        if path.exists():
            files.append(path)
    return files


def test_preferences_package_is_gone() -> None:
    """The ``stash_ai/preferences`` package directory must not exist."""
    package_dir = PACKAGE_ROOT / "preferences"
    assert not package_dir.exists(), (
        f"The preferences subsystem package reappeared at "
        f"{package_dir.relative_to(REPO_ROOT)}. It was removed per ADR-0001 -- "
        "taste is inferred implicitly from Engagement, not explicit swiping."
    )


def test_no_module_imports_the_preferences_package() -> None:
    """No first-party source may import ``stash_ai.preferences`` (any spelling)."""
    # Absolute (`stash_ai.preferences`) and relative (`from .preferences`,
    # `from ..preferences`) imports of the removed package.
    import_pattern = re.compile(r"(?:from|import)\s+(?:stash_ai\.preferences|\.{1,2}preferences)\b")

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _source_files()
        if import_pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        f"The removed preferences package is imported in: {offenders}. "
        "The subsystem was deleted per ADR-0001; do not reintroduce a dependency on it."
    )


def test_no_preference_plugin_task_remains() -> None:
    """No plugin task in the manifest may dispatch a ``preference_*`` mode.

    The seven swipe-trainer tasks were dropped from ``stash-copilot.yml`` (commit 4).
    The unrelated ``performer_preference`` recommendation mode travels as a ``rec_mode``
    argument value, never as a task ``mode``, so it is unaffected by this check.
    """
    manifest = yaml.safe_load(PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", []) or []

    offenders: list[str] = []
    for task in tasks:
        mode = (task.get("defaultArgs") or {}).get("mode", "")
        if (
            isinstance(mode, str)
            and mode.startswith("preference")
            and mode not in ALLOWED_PREFERENCE_TOKENS
        ):
            offenders.append(f"{task.get('name', '<unnamed>')!r} (mode={mode!r})")

    assert not offenders, (
        f"Preference-subsystem plugin tasks reappeared in the manifest: {offenders}. "
        "The seven swipe-trainer tasks were removed per ADR-0001."
    )
