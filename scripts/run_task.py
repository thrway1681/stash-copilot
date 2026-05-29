#!/usr/bin/env python3
"""Run a Stash Copilot task locally, with no Stash server.

By default the task runs against an in-memory ``FakeStashClient`` (optionally
seeded from a JSON fixture), so you can iterate on task logic in seconds. Pass
``--real`` to run against a live Stash instead.

Examples::

    # Run a task against an empty in-memory fake
    uv run python scripts/run_task.py stats_summary

    # Seed the fake from a fixture and pass task args
    uv run python scripts/run_task.py process_scene \\
        --fixture tests/fixtures/sample_library.json --arg scene_id=1

    # Run against a live Stash
    uv run python scripts/run_task.py recommendations \\
        --real --stash-url http://localhost:3000 --api-key "$STASH_API_KEY"

This harness writes the same ``assets/<name>_<request_id>.json`` files the
frontend polls, because it invokes the real task handlers unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_FILE = REPO_ROOT / "stash-copilot.py"


def _load_plugin_module() -> Any:
    """Import the hyphenated ``stash-copilot.py`` entry-point module."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("stash_copilot_entry", PLUGIN_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plugin module from {PLUGIN_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stash Copilot task locally.")
    parser.add_argument("task", help="Task name (the same value Stash sends as 'mode').")
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task argument; repeatable.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="JSON fixture to seed the in-memory FakeStashClient.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run against a live Stash instead of the fake.",
    )
    parser.add_argument("--stash-url", default="http://localhost:9999")
    parser.add_argument("--api-key", default="")
    return parser.parse_args(argv)


def _build_args(pairs: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--arg must be KEY=VALUE, got: {pair!r}")
        key, value = pair.split("=", 1)
        args[key] = value
    return args


def _build_real_client(stash_url: str, api_key: str) -> Any:
    from urllib.parse import urlparse

    from stashapi.stashapp import StashInterface

    from stash_ai.stash_client import StashApiClient

    parsed = urlparse(stash_url)
    conn: dict[str, Any] = {
        "Scheme": parsed.scheme or "http",
        "Host": parsed.hostname or "localhost",
        "Port": parsed.port or (443 if parsed.scheme == "https" else 9999),
    }
    if api_key:
        conn["ApiKey"] = api_key
    return StashApiClient(StashInterface(conn))


def _build_fake_client(fixture: Path | None) -> Any:
    from tests.fakes import FakeStashClient

    if fixture is not None:
        return FakeStashClient.from_fixture(fixture)
    return FakeStashClient()


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    task_args = _build_args(ns.arg)

    if ns.real:
        client = _build_real_client(ns.stash_url, ns.api_key)
        print(f"[run_task] real Stash at {ns.stash_url}")
    else:
        client = _build_fake_client(ns.fixture)
        seed = ns.fixture if ns.fixture else "empty"
        print(f"[run_task] in-memory fake (seed: {seed})")

    module = _load_plugin_module()
    plugin = module.MyPlugin(stash_client=client, input_override={"args": task_args})

    print(f"[run_task] dispatching task {ns.task!r} with args {task_args}")
    plugin.run_task(ns.task, task_args)
    print("[run_task] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
