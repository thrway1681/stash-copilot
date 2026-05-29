"""Typed seam over the third-party Stash API.

Every outbound call to the Stash server flows through ``StashClient``. Production
wires :class:`StashApiClient`, which delegates to ``stashapi.StashInterface``; tests
substitute an in-memory fake that satisfies the same Protocol.

The Protocol method signatures mirror ``stashapi.StashInterface`` so existing call
sites need only a type-annotation change, not a behavioral one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from stashapi.stashapp import StashInterface


@runtime_checkable
class StashClient(Protocol):
    """Structural interface for the subset of Stash operations this plugin uses."""

    def call_GQL(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def find_scene(self, id: int, fragment: str | None = None) -> dict[str, Any] | None: ...

    def find_scenes(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]: ...

    def find_tags(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]: ...

    def update_scene(
        self, update_input: dict[str, Any], create: bool = False
    ) -> dict[str, Any] | None: ...

    def metadata_scan(
        self,
        paths: list[str] | None = None,
        flags: dict[str, Any] | None = None,
    ) -> str: ...

    def get_configuration(self, fragment: str | None = None) -> dict[str, Any]: ...

    def find_plugin_config(
        self, plugin_id: str, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


class StashApiClient:
    """:class:`StashClient` backed by a live ``stashapi.StashInterface``.

    A thin pass-through. The only logic here is restoring ``StashInterface``'s own
    default argument values when a caller passes ``None``, so behavior is identical
    to calling the wrapped interface directly.
    """

    def __init__(self, stash: StashInterface) -> None:
        self._stash = stash

    @property
    def raw(self) -> StashInterface:
        """The wrapped ``StashInterface`` for code not yet migrated to the seam."""
        return self._stash

    def call_GQL(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._stash.call_GQL(query, variables if variables is not None else {}),
        )

    def find_scene(self, id: int, fragment: str | None = None) -> dict[str, Any] | None:
        return cast("dict[str, Any] | None", self._stash.find_scene(id, fragment=fragment))

    def find_scenes(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]:
        return cast(
            "list[dict[str, Any]]",
            self._stash.find_scenes(
                f=f if f is not None else {},
                filter=filter if filter is not None else {"per_page": -1},
                q=q,
                fragment=fragment,
                get_count=get_count,
            ),
        )

    def find_tags(
        self,
        f: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        q: str = "",
        fragment: str | None = None,
        get_count: bool = False,
    ) -> list[dict[str, Any]]:
        return cast(
            "list[dict[str, Any]]",
            self._stash.find_tags(
                f=f if f is not None else {},
                filter=filter if filter is not None else {"per_page": -1},
                q=q,
                fragment=fragment,
                get_count=get_count,
            ),
        )

    def update_scene(
        self, update_input: dict[str, Any], create: bool = False
    ) -> dict[str, Any] | None:
        return cast(
            "dict[str, Any] | None",
            self._stash.update_scene(update_input, create=create),
        )

    def metadata_scan(
        self,
        paths: list[str] | None = None,
        flags: dict[str, Any] | None = None,
    ) -> str:
        return cast(
            "str",
            self._stash.metadata_scan(
                paths=paths if paths is not None else [],
                flags=flags if flags is not None else {},
            ),
        )

    def get_configuration(self, fragment: str | None = None) -> dict[str, Any]:
        return cast("dict[str, Any]", self._stash.get_configuration(fragment=fragment))

    def find_plugin_config(
        self, plugin_id: str, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._stash.find_plugin_config(
                plugin_id, defaults=defaults if defaults is not None else {}
            ),
        )
