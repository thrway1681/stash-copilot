"""Tests for the StashClient seam: the fake, the Protocol, and plugin injection."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from stash_ai.stash_client import StashClient
from tests.fakes import FakeStashClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sample_library.json"


@pytest.fixture
def fake() -> FakeStashClient:
    return FakeStashClient.from_fixture(FIXTURE)


def _load_plugin_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "stash_copilot_entry_test", REPO_ROOT / "stash-copilot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProtocolConformance:
    def test_fake_satisfies_protocol(self, fake: FakeStashClient) -> None:
        assert isinstance(fake, StashClient)


class TestFakeBehaviour:
    def test_find_scene_returns_seeded_scene(self, fake: FakeStashClient) -> None:
        scene = fake.find_scene(1)
        assert scene is not None
        assert scene["title"] == "Sample Scene One"

    def test_find_scene_missing_returns_none(self, fake: FakeStashClient) -> None:
        assert fake.find_scene(999) is None

    def test_find_scenes_and_tags(self, fake: FakeStashClient) -> None:
        assert len(fake.find_scenes()) == 2
        assert {t["name"] for t in fake.find_tags()} == {"demo", "favorites"}

    def test_update_scene_mutates_in_memory(self, fake: FakeStashClient) -> None:
        fake.update_scene({"id": "2", "rating100": 90})
        scene = fake.find_scene(2)
        assert scene is not None
        assert scene["rating100"] == 90

    def test_metadata_scan_returns_job_id(self, fake: FakeStashClient) -> None:
        assert fake.metadata_scan(paths=["/x"]) == "fake-job-1"
        assert fake.metadata_scan(paths=["/y"]) == "fake-job-2"

    def test_find_plugin_config_merges_defaults(self, fake: FakeStashClient) -> None:
        merged = fake.find_plugin_config("stash-copilot", defaults={"extra": "d"})
        assert merged["extra"] == "d"
        assert merged["text_llm_provider"] == "ollama"

    def test_get_configuration(self, fake: FakeStashClient) -> None:
        assert fake.get_configuration()["general"]["databasePath"]


class TestCallGqlRouting:
    def test_find_scene_query(self, fake: FakeStashClient) -> None:
        result = fake.call_GQL("query FindScene { findScene { id } }", {"id": "1"})
        assert result["findScene"]["title"] == "Sample Scene One"

    def test_all_tags_query(self, fake: FakeStashClient) -> None:
        names = [t["name"] for t in fake.call_GQL("query { allTags { id } }")["allTags"]]
        assert names == ["demo", "favorites"]

    def test_scene_update_mutation(self, fake: FakeStashClient) -> None:
        fake.call_GQL(
            "mutation SceneUpdate { sceneUpdate { id } }",
            {"input": {"id": "1", "rating100": 70}},
        )
        scene = fake.find_scene(1)
        assert scene is not None and scene["rating100"] == 70

    def test_unknown_operation_fails_loud(self, fake: FakeStashClient) -> None:
        with pytest.raises(NotImplementedError, match="Mystery"):
            fake.call_GQL("query Mystery { unknownThing { id } }")

    def test_registered_handler_takes_precedence(self, fake: FakeStashClient) -> None:
        fake.register_gql_handler("customOp", lambda q, v: {"customOp": 42})
        assert fake.call_GQL("query { customOp }")["customOp"] == 42

    def test_calls_are_recorded(self, fake: FakeStashClient) -> None:
        fake.find_scene(1)
        assert any(name == "find_scene" for name, _, _ in fake.calls)


class TestPluginInjection:
    def test_plugin_runs_task_against_fake_without_server(
        self, fake: FakeStashClient
    ) -> None:
        module = _load_plugin_module()
        plugin = module.MyPlugin(
            stash_client=fake, input_override={"args": {"scene_id": "1"}}
        )
        # No stdin, no StashInterface — the injected fake is the client.
        assert plugin.stash is fake
        plugin.run_task("process_scene", {"scene_id": "1"})
        # process_scene bumps rating for high play_count scenes (25 -> capped 100).
        scene = fake.find_scene(1)
        assert scene is not None and scene["rating100"] == 100
