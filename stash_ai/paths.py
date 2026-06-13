"""Resolution of the plugin's generated-runtime-data directory (ADR-0007).

Under the installed-plugin model (ADR-0006) Stash owns the plugin directory and a
plugin update *replaces* it, so any data the plugin writes inside its own tree is
destroyed on every update. This module is the single seam that resolves a stable,
update-safe data root **outside** the plugin directory plus the typed sub-paths
beneath it. Every consumer of generated runtime state (the embeddings store,
extracted frames, caches, histories, exports, the EroScripts cookie) is meant to
resolve its location here instead of computing a ``__file__``-relative path.

The data root is ``$STASH_CONFIG_DIR/stash-copilot/`` when ``STASH_CONFIG_DIR`` is
set (Stash exports it for plugins), otherwise ``~/.stash/stash-copilot/``. There is
no plugin-directory fallback: the break from the old in-plugin ``assets/`` location
is intentional and clean (ADR-0007). The sub-path leaf names mirror the old
``assets/`` layout so a one-time Phase-B copy maps old -> new without renaming. This
module only ever *reads* the new location; it never auto-copies existing data.

This commit adds the module with **no callers**; the shipped consumers are routed
onto it in subsequent commits of the relocation issue (#14), and
``eroscripts_auth_path`` is consumed by the secrets-rerouting issue (#15).
"""

import os
from pathlib import Path

# The single declaration of the on-disk layout. Leaf names match the legacy
# in-plugin ``assets/`` names so Phase B can copy old -> new without renaming.
_DATA_DIR_NAME = "stash-copilot"
_EMBEDDINGS_DB_NAME = "stash_copilot.sqlite"
_FRAMES_CACHE_DIRNAME = "embedded_frames"
_SCENE_VISION_DIRNAME = "scene_vision"
_O_MOMENT_CACHE_DIRNAME = "o_moment_cache"
_PERFORMER_FRAMES_DIRNAME = "performer_frames"
_EXPORTS_DIRNAME = "exports"
_EROSCRIPTS_AUTH_NAME = "eroscripts_auth.json"


def data_dir() -> Path:
    """Resolve (and create) the plugin's update-safe data root.

    Resolves to ``$STASH_CONFIG_DIR/stash-copilot/`` when ``STASH_CONFIG_DIR`` is
    set, else ``~/.stash/stash-copilot/``. The directory is created on demand so
    callers can write into it without a pre-flight ``mkdir``.

    Returns:
        Path: The existing data-root directory.
    """
    config_dir = os.environ.get("STASH_CONFIG_DIR")
    base = Path(config_dir) if config_dir else Path.home() / ".stash"
    root = base / _DATA_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def embeddings_db_path() -> Path:
    """Path to the main embeddings SQLite store under the data root."""
    return data_dir() / _EMBEDDINGS_DB_NAME


def frames_cache_dir() -> Path:
    """Directory holding extracted 1-FPS scene frames (the embedding frames cache)."""
    return data_dir() / _FRAMES_CACHE_DIRNAME


def scene_vision_dir() -> Path:
    """Directory holding per-scene vision-analysis conversation history."""
    return data_dir() / _SCENE_VISION_DIRNAME


def o_moment_cache_dir() -> Path:
    """Directory holding cached O-marker frames for O-moment embedding."""
    return data_dir() / _O_MOMENT_CACHE_DIRNAME


def performer_frames_dir() -> Path:
    """Directory holding cached performer frames for describe-performer."""
    return data_dir() / _PERFORMER_FRAMES_DIRNAME


def exports_dir() -> Path:
    """Directory for generated exports (datasets, snapshots) the plugin writes."""
    return data_dir() / _EXPORTS_DIRNAME


def eroscripts_auth_path() -> Path:
    """Path to the EroScripts session-cookie file under the data root.

    Declared here for the secrets-rerouting issue (#15), which moves the cookie out
    of the plugin tree onto this update-safe path.
    """
    return data_dir() / _EROSCRIPTS_AUTH_NAME
