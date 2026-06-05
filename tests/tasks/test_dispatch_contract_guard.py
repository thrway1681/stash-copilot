"""Cross-stack contract guard for the dispatch seam (issue #5, commit 5).

Issue #5 collapsed the frontend's hand-rolled task invocations into a single
``TASKS`` registry + ``dispatchTask`` seam in ``stash-copilot.js``. Each registry
entry binds three things that live in three different files and *must* agree, or a
feature silently breaks at runtime with no compile-time signal:

  1. ``name``      -> the EXACT task name in ``stash-copilot.yml`` that Stash runs.
                      A typo here makes ``runPluginTask`` a no-op (Stash finds no
                      task by that name) -- the UI spins forever, no error.
  2. ``resultKey`` -> the backend ``result_key`` a task class declares, which
                      ``ResultStore`` writes to ``assets/{result_key}_{id}.json``
                      (issue #4). If the JS polls ``foo_{id}.json`` but the backend
                      writes ``bar_{id}.json``, the poll loop just times out.

There is no shared schema between the JS literal, the YAML, and the Python class
attribute -- only this guard. It parses all three and asserts they line up, so a
rename on any one side fails CI instead of shipping a dead button.

Parsing (not importing) is deliberate: the registry is a JS object literal and the
YAML is data, so we read them as text. The regexes are pinned to the registry's
one-entry-per-line shape (see the ``// the registry stays trivially parseable``
note in stash-copilot.js); a sanity check on the parsed count guards against a
silent regex breakage that would make every per-entry assertion vacuously pass.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root: tests/tasks/<this file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "stash_ai"
JS_FILE = REPO_ROOT / "stash-copilot.js"
YML_FILE = REPO_ROOT / "stash-copilot.yml"

# resultKeys the frontend polls as FIXED-name files the backend overwrites in
# place (freshness via a timestamp baseline, not a per-request filename). These
# tasks deliberately declare NO ``result_key`` class attribute -- the task code
# writes ``<key>.json`` directly. The honesty of this allowlist is itself checked
# by ``test_fixed_file_allowlist_is_written_by_backend`` below.
FIXED_FILE_ALLOWLIST = {"chat_history", "last_summary"}

# Valid ``keying`` discriminants (how dispatchTask derives the polled file stem).
VALID_KEYINGS = {"request_id", "scene_id", "fixed", "none"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_js_tasks() -> list[dict[str, object]]:
    """Parse the ``const TASKS = { ... }`` registry from stash-copilot.js.

    Returns one dict per entry: ``{key, name, resultKey (str|None), keying}``.
    Relies on the registry's one-object-literal-per-line shape. Each entry nests a
    single ``defaultArgs: { ... }`` object, so the entry-body capture allows exactly
    one level of nesting (``\\{(?:[^{}]|\\{[^{}]*\\})*\\}``) -- this keeps every field
    (including ones that follow ``defaultArgs``) inside the captured body rather than
    truncating at the first inner brace.
    """
    text = _read(JS_FILE)
    block_match = re.search(r"const TASKS\s*=\s*\{(.*?)\n\s*\};", text, re.DOTALL)
    assert block_match, "Could not locate the `const TASKS = { ... };` block in stash-copilot.js"
    block = block_match.group(1)

    entries: list[dict[str, object]] = []
    for entry in re.finditer(r"(\w+)\s*:\s*\{((?:[^{}]|\{[^{}]*\})*)\}", block):
        key = entry.group(1)
        body = entry.group(2)

        name_m = re.search(r"name\s*:\s*'([^']*)'", body)
        result_m = re.search(r"resultKey\s*:\s*(null|'([^']*)')", body)
        keying_m = re.search(r"keying\s*:\s*'([^']*)'", body)

        assert name_m, f"TASKS entry '{key}' has no parseable name: {body!r}"
        assert result_m, f"TASKS entry '{key}' has no parseable resultKey: {body!r}"
        assert keying_m, f"TASKS entry '{key}' has no parseable keying: {body!r}"

        entries.append(
            {
                "key": key,
                "name": name_m.group(1),
                "resultKey": result_m.group(2),  # None when `resultKey: null`
                "keying": keying_m.group(1),
            }
        )
    return entries


def _parse_yml_task_names() -> set[str]:
    """Collect task ``name`` values from stash-copilot.yml's ``tasks:`` section only.

    Sliced to the ``tasks:`` block (top-level ``tasks:`` -> next top-level ``hooks:``)
    so hook names and the settings keys can't leak in. Parsed by regex to avoid a
    hard PyYAML dependency for the test suite.
    """
    lines = _read(YML_FILE).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "tasks:")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()),
        len(lines),
    )
    names: set[str] = set()
    for ln in lines[start + 1 : end]:
        m = re.match(r"\s*-\s+name:\s*(.+?)\s*$", ln)
        if m:
            names.add(m.group(1).strip().strip("'\""))
    return names


def _backend_result_keys() -> set[str]:
    """Every ``result_key`` declared by a backend task class.

    Matches both annotated and bare forms::

        result_key = "tag_gaps"
        result_key: ClassVar[str] = "tag_dedup"
    """
    pattern = re.compile(r"""result_key\s*(?::\s*[^=\n]+)?=\s*["']([a-z_]+)["']""")
    keys: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        keys.update(pattern.findall(_read(path)))
    return keys


# --- the parsed registry, shared across tests -------------------------------
_JS_TASKS = _parse_js_tasks()


def test_registry_parsed_nonempty() -> None:
    """Guard the guard: a broken regex must fail loudly, not pass vacuously.

    The registry has ~29 entries; if parsing yields far fewer, the per-entry
    assertions below would pass for the wrong reason (nothing to check).
    """
    assert len(_JS_TASKS) >= 25, (
        f"Parsed only {len(_JS_TASKS)} TASKS entries from stash-copilot.js -- the "
        "registry shape likely changed and the parse regex needs updating. Refusing "
        "to let the contract assertions pass vacuously."
    )
    # Every entry's keying must be a known discriminant (drives stem derivation).
    bad = [(t["key"], t["keying"]) for t in _JS_TASKS if t["keying"] not in VALID_KEYINGS]
    assert not bad, f"TASKS entries with an unknown `keying`: {bad}"


def test_js_task_names_match_yml() -> None:
    """Every TASKS ``name`` is an exact task name in stash-copilot.yml.

    dispatchTask runs ``runPluginTask(spec.name, ...)`` under this literal. If it
    doesn't match a yml task name verbatim, Stash silently runs nothing.
    """
    yml_names = _parse_yml_task_names()
    missing = sorted(
        {str(t["name"]) for t in _JS_TASKS} - yml_names,
    )
    assert not missing, (
        f"TASKS `name`s with no matching task in stash-copilot.yml: {missing}. "
        f"dispatchTask would call runPluginTask under a name Stash doesn't know, a "
        f"silent no-op. Known yml task names: {sorted(yml_names)}"
    )


def test_js_result_keys_have_backend_writer() -> None:
    """Every polled ``resultKey`` is a backend-declared ``result_key`` (or allowlisted).

    The frontend polls ``assets/{resultKey}_{id}.json``; the backend writes that
    file only if a task class declares the matching ``result_key``. A mismatch is a
    guaranteed poll timeout. FIXED-file resultKeys (chat_history, last_summary) are
    exempt -- their tasks write a fixed-name file directly (checked separately).
    """
    backend_keys = _backend_result_keys()
    offenders: list[str] = []
    for task in _JS_TASKS:
        result_key = task["resultKey"]
        if result_key is None or result_key in FIXED_FILE_ALLOWLIST:
            continue
        if result_key not in backend_keys:
            offenders.append(f"{task['key']} -> resultKey '{result_key}'")

    assert not offenders, (
        "TASKS resultKeys with no backend `result_key` writer "
        f"(would poll a file that's never written -> timeout): {offenders}. "
        f"Backend-declared result_keys: {sorted(backend_keys)}"
    )


def test_fixed_file_allowlist_is_written_by_backend() -> None:
    """Keep the FIXED-file allowlist honest: each entry's ``<key>.json`` is written.

    The allowlist exempts resultKeys from the ``result_key``-class-attr check
    *because* their tasks persist a fixed-name file directly. If such a task is ever
    migrated to ResultStore (or the filename changes), its allowlist entry goes
    stale -- this test flags that by requiring the literal ``"<key>.json"`` to appear
    in some backend source.
    """
    sources = "\n".join(_read(p) for p in PACKAGE_ROOT.rglob("*.py"))
    for key in sorted(FIXED_FILE_ALLOWLIST):
        assert f"{key}.json" in sources, (
            f"FIXED-file allowlist entry '{key}' claims the backend writes "
            f"'{key}.json' directly, but no backend source references that filename. "
            "Update FIXED_FILE_ALLOWLIST (or the task) so the contract guard stays honest."
        )
