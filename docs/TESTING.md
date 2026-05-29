# Testing & Local Development

How to clone, set up, and exercise Stash Copilot on any machine — with or without a
running Stash server. Everything needed is committed (code, tests, fixtures,
`uv.lock`, the dev Docker compose), so a fresh clone reproduces the same toolchain.

## 1. Prerequisites

- **git**
- **[uv](https://docs.astral.sh/uv/)** — manages Python and dependencies
- **Docker** (optional) — only needed for the "real Stash server" workflow in §6

Python itself is handled by uv (project requires **3.10+**). If the right Python
isn't present, run `uv python install 3.12`.

## 2. Clone & install

```bash
git clone <your-fork-or-remote> stash-copilot
cd stash-copilot
uv sync --extra dev      # installs exact locked versions + dev tools (pytest, mypy, ruff)
```

`uv sync` makes `.venv/` match `uv.lock` precisely, so the dependency set is
identical on every machine. (On a machine where you previously installed extra
packages, `uv sync` will prune them back to the lockfile — that's expected.)

## 3. Run the test suite

```bash
uv run --extra dev pytest -q
```

Expected: **all pass.**

Run a single file or test while iterating:

```bash
uv run --extra dev pytest tests/test_stash_client_seam.py -q
uv run --extra dev pytest tests/test_stash_client_seam.py::TestPluginInjection -q
```

## 4. Type-check & lint

```bash
uv run --extra dev mypy stash-copilot.py stash_ai/         # strict typing, package + entry point
uv run --extra dev ruff check stash-copilot.py stash_ai/   # lint the package + entry point
uv run --extra dev ruff format --check stash-copilot.py stash_ai/
```

These are **clean and CI-blocking** for the shipped package (`stash-copilot.py` +
`stash_ai/`); keep them at zero. The scope deliberately excludes throwaway
`poc_*.py` / `standalone_embed.py` / experimental `tools/` scripts (running
`mypy .` or `ruff check .` over the whole repo reports findings in those).

## 5. Run a task with NO Stash server (fast loop)

The quickest way to exercise task logic. Tasks run against an in-memory
`FakeStashClient` seeded from a JSON fixture — no server, no database.

```bash
# List the flags
uv run python scripts/run_task.py --help

# Run a task against the committed sample library
uv run python scripts/run_task.py process_scene \
    --fixture tests/fixtures/sample_library.json --arg scene_id=1

# Empty in-memory library (no fixture)
uv run python scripts/run_task.py stats_summary
```

- `--arg KEY=VALUE` is repeatable and maps to the task args Stash would normally send.
- Edit `tests/fixtures/sample_library.json` to add scenes/tags/config. It's plain
  JSON and is committed, so your seed data travels with the repo.
- Tasks that call out to an LLM still need that provider configured; pure
  data/embedding tasks run fully offline.

## 6. Run against a real (Dockerized) Stash server

For end-to-end checks against a real Stash API. This runs a throwaway Stash server
in Docker, **isolated from your real `~/.stash`** (all state lives in volumes).
Requires a running Docker daemon.

> The plugin's Python/ML (torch, faiss) runs on the **host**, not inside the Stash
> container — the official `stashapp/stash` image is Alpine/musl and can't install
> the glibc ML wheels. Docker here provides only a reproducible Stash *server*.

### Two host/container boundaries to bridge

Because the ML code runs on the host but Stash runs in the container, two things
the host needs to reach live inside the container by default. The committed
`docker-compose.dev.yml` is set up to bridge both:

1. **The SQLite DB.** This plugin's data tasks read `stash-go.sqlite` *directly
   from the filesystem* (see `get_stash_db_path()`), not over GraphQL. The config
   dir is therefore a **host bind mount** (`./docker/stash-config:/root/.stash`),
   so the host can open the DB. Point the host at it with
   `export STASH_CONFIG_DIR="$PWD/docker/stash-config"`.

2. **Media files.** Embedding reads each scene's video by the absolute path Stash
   recorded in the DB. For that path to resolve on the host too, mount the media at
   the **same absolute path** inside the container and point the Stash library
   there. Set `STASH_MEDIA_DIR` and `STASH_MEDIA_MOUNT` to the same absolute host
   path (both default to `./docker/media` → `/data`, which is fine for non-ML use).

### Walkthrough (headless, no UI clicks)

```bash
export MEDIA_ABS="$PWD/docker/media"
export STASH_MEDIA_DIR="$MEDIA_ABS" STASH_MEDIA_MOUNT="$MEDIA_ABS"
export STASH_CONFIG_DIR="$PWD/docker/stash-config"

docker compose -f docker-compose.dev.yml up -d        # http://localhost:3000

# Complete first-run setup via GraphQL (library = the parity path):
curl -s localhost:3000/graphql -H 'Content-Type: application/json' -d "{\"query\":\"mutation(\$i:SetupInput!){setup(input:\$i)}\",\"variables\":{\"i\":{\"configLocation\":\"/root/.stash/config.yml\",\"stashes\":[{\"path\":\"$MEDIA_ABS\",\"excludeVideo\":false,\"excludeImage\":false}],\"databaseFile\":\"/root/.stash/stash-go.sqlite\",\"generatedLocation\":\"/generated\",\"cacheLocation\":\"/cache\",\"storeBlobsInDatabase\":true,\"blobsLocation\":\"/root/.stash/blobs\"}}}"

# Drop videos into ./docker/media, then scan:
curl -s localhost:3000/graphql -H 'Content-Type: application/json' -d '{"query":"mutation($i:ScanMetadataInput!){metadataScan(input:$i)}","variables":{"i":{"scanGenerateCovers":true,"scanGenerateSprites":true,"scanGeneratePhashes":true}}}'

# For CLIP/OpenCLIP embedding (visual, no LLM/VLM needed), set the plugin config
# Stash reads via find_plugin_config — otherwise embedding falls back to a VLM:
curl -s localhost:3000/graphql -H 'Content-Type: application/json' -d '{"query":"mutation($id:ID!,$in:Map!){configurePlugin(plugin_id:$id,input:$in)}","variables":{"id":"stash-copilot","input":{"image_embedding_provider":"openclip","image_embedding_model":"ViT-B-32","image_embedding_device":"mps"}}}'

uv run python scripts/run_task.py embed_scenes  --real --stash-url http://localhost:3000
uv run python scripts/run_task.py recommendations --real --stash-url http://localhost:3000 --arg mode=discover

docker compose -f docker-compose.dev.yml down      # stop, keep DB+volumes
docker compose -f docker-compose.dev.yml down -v   # stop, wipe for a clean slate
```

Notes:
- `--api-key` is only needed if you enabled auth in the dev server; omit it locally.
- **Recommendations need engagement data.** Discover builds a taste profile from
  watched scenes and needs **≥3 engaged scenes that have embeddings**, plus some
  *unwatched* embedded scenes to surface. Seed engagement with the `sceneAddO`,
  `sceneAddPlay`, and `sceneUpdate(rating100:)` mutations before expecting results.
- `embed_scenes` downloads the OpenCLIP weights on first run (`ViT-B-32` ≈ 150 MB;
  the default `ViT-H-14` ≈ 3.9 GB). On Apple Silicon use `image_embedding_device:"mps"`.

## 7. Picking up on another computer

What travels via **git** (so a clone reproduces it exactly):

- All source, tests, and the `tests/fixtures/*.json` fake data
- `uv.lock` — exact dependency versions
- `docker-compose.dev.yml` — the dev-server definition

What is **machine-local** (does NOT travel via git):

- The dev Stash library (lives in Docker named volumes). To resume on a new
  machine: `docker compose -f docker-compose.dev.yml up -d`, re-add media under
  `./docker/media/`, and scan. Or just rely on the fixture-based fake (§5), which
  needs no server at all and is fully portable.

So the reproducible workflow on any machine is:

```bash
git clone … && cd stash-copilot
uv sync --extra dev
uv run --extra dev pytest -q
uv run python scripts/run_task.py <task> --fixture tests/fixtures/sample_library.json
```
