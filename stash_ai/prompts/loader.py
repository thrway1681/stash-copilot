"""Load prompts from external YAML files for hot-reload support."""

from pathlib import Path

import yaml


def get_prompts_dir() -> Path:
    """Get the prompts directory path.

    Returns:
        Path to the prompts/ directory at the repo root.
    """
    # prompts/ is at repo root, same level as stash_ai/
    return Path(__file__).parent.parent.parent / "prompts"


def load_prompt_file(category: str, name: str) -> dict[str, str]:
    """Load prompts from a YAML file. Always reads from disk for hot-reload.

    Args:
        category: Subdirectory (e.g., "vision", "chat")
        name: File name without extension (e.g., "system", "description")

    Returns:
        Dict of prompt_name -> prompt_text

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompts_dir = get_prompts_dir()
    file_path = prompts_dir / category / f"{name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        prompts: dict[str, str] = yaml.safe_load(f)

    return prompts


def get_prompt(category: str, file: str, name: str) -> str:
    """Get a specific prompt by name. Always hot-reloads from disk.

    Args:
        category: Subdirectory (e.g., "vision")
        file: File name without extension (e.g., "system")
        name: Prompt name within file (e.g., "professional")

    Returns:
        The prompt text

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        KeyError: If prompt name not found in file
    """
    prompts = load_prompt_file(category, file)
    if name not in prompts:
        available = list(prompts.keys())
        raise KeyError(
            f"Prompt '{name}' not found in {category}/{file}.yaml. Available prompts: {available}"
        )
    return prompts[name]
