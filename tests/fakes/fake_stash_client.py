"""In-memory :class:`StashClient` for local development and tests.

``FakeStashClient`` satisfies the ``StashClient`` Protocol with real in-memory
behaviour backed by plain dicts/lists, so tasks can run with no Stash server.

Load it from a JSON fixture::

    client = FakeStashClient.from_fixture("tests/fixtures/sample_library.json")

or construct it directly with ``scenes=...``, ``tags=...``, etc.

``call_GQL`` routes the handful of GraphQL operations the codebase actually uses
to the in-memory data. Unknown operations raise ``NotImplementedError`` naming the
query — fail loud rather than silently returning ``{}`` (which is how truncated
real responses have masked bugs before).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

GqlHandler = Callable[[str, dict[str, Any]], dict[str, Any]]


class FakeStashClient:
    """Structural ``StashClient`` implementation over in-memory data."""

    def __init__(
        self,
        scenes: list[dict[str, Any]] | None = None,
        tags: list[dict[str, Any]] | None = None,
        plugin_config: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        gql_handlers: dict[str, GqlHandler] | None = None,
    ) -> None:
        self._scenes: dict[int, dict[str, Any]] = {
            int(s["id"]): dict(s) for s in (scenes or [])
        }
        self._tags: list[dict[str, Any]] = [dict(t) for t in (tags or [])]
        self._plugin_config: dict[str, Any] = dict(plugin_config or {})
        self._configuration: dict[str, Any] = dict(configuration or {})
        self._gql_handlers: dict[str, GqlHandler] = dict(gql_handlers or {})
        # Recorded calls, useful for assertions in tests.
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        # Job ids handed out by metadata_scan.
        self._next_job = 0

    @classmethod
    def from_fixture(cls, path: str | Path) -> FakeStashClient:
        """Build a client from a JSON fixture.

        Expected shape (all keys optional)::

            {
              "scenes": [{"id": "1", "title": "...", "tags": [...]}],
              "tags": [{"id": "1", "name": "..."}],
              "plugin_config": {...},
              "configuration": {...}
            }
        """
        data = json.loads(Path(path).read_text())
        return cls(
            scenes=data.get("scenes"),
            tags=data.get("tags"),
            plugin_config=data.get("plugin_config"),
            configuration=data.get("configuration"),
        )

    def register_gql_handler(self, operation: str, handler: GqlHandler) -> None:
        """Register a handler for a GraphQL operation matched by substring."""
        self._gql_handlers[operation] = handler

    # ------------------------------------------------------------------
    # StashClient Protocol
    # ------------------------------------------------------------------
    def call_GQL(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        variables = variables or {}
        self.calls.append(("call_GQL", (query,), {"variables": variables}))

        # User-registered handlers take precedence (substring match).
        for op, handler in self._gql_handlers.items():
            if op in query:
                return handler(query, variables)

        return self._default_gql(query, variables)

    def find_scene(
        self, id: int, fragment: str | None = None
    ) -> dict[str, Any] | None:
        self.calls.append(("find_scene", (id,), {"fragment": fragment}))
        return self._scenes.get(int(id))

    def find_scenes(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append(("find_scenes", (), {"f": f, "filter": filter, "q": q}))
        return list(self._scenes.values())

    def find_tags(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append(("find_tags", (), {"f": f, "filter": filter, "q": q}))
        return list(self._tags)

    def update_scene(
        self, update_input: dict[str, Any], create: bool = False
    ) -> dict[str, Any] | None:
        self.calls.append(("update_scene", (update_input,), {"create": create}))
        scene_id = int(update_input["id"])
        scene = self._scenes.setdefault(scene_id, {"id": str(scene_id)})
        scene.update(update_input)
        return {"id": str(scene_id)}

    def metadata_scan(
        self,
        paths: list[str] | None = None,
        flags: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(("metadata_scan", (), {"paths": paths, "flags": flags}))
        self._next_job += 1
        return f"fake-job-{self._next_job}"

    def get_configuration(self, fragment: str | None = None) -> dict[str, Any]:
        self.calls.append(("get_configuration", (), {"fragment": fragment}))
        return dict(self._configuration)

    def find_plugin_config(
        self, plugin_id: str, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append(("find_plugin_config", (plugin_id,), {"defaults": defaults}))
        merged = dict(defaults or {})
        merged.update(self._plugin_config)
        return merged

    # ------------------------------------------------------------------
    # Default GraphQL routing for the operations the codebase uses
    # ------------------------------------------------------------------
    def _default_gql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        # findScene(id: ...) { ... }
        if "findScene" in query:
            scene_id = variables.get("id")
            scene = self._scenes.get(int(scene_id)) if scene_id is not None else None
            return {"findScene": scene}

        # allTags { id name }
        if "allTags" in query:
            return {"allTags": [{"id": t["id"], "name": t["name"]} for t in self._tags]}

        # findScenes(filter: ...) { count scenes { ... } }
        if "findScenes" in query:
            scenes = list(self._scenes.values())
            return {"findScenes": {"count": len(scenes), "scenes": scenes}}

        # sceneUpdate(input: ...) { id }
        if "sceneUpdate" in query:
            scene_input = variables.get("input", {})
            if "id" in scene_input:
                self.update_scene(scene_input)
            return {"sceneUpdate": {"id": scene_input.get("id")}}

        op = self._operation_name(query)
        raise NotImplementedError(
            f"FakeStashClient has no handler for GraphQL operation {op!r}. "
            f"Register one via register_gql_handler(), or extend _default_gql(). "
            f"Query:\n{query.strip()[:500]}"
        )

    @staticmethod
    def _operation_name(query: str) -> str:
        match = re.search(r"\b(query|mutation)\s+(\w+)", query)
        return match.group(2) if match else "<anonymous>"
