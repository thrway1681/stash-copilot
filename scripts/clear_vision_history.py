#!/usr/bin/env python3
"""Clear all existing vision history files for schema migration."""

import glob
import os
from pathlib import Path


def main():
    """Clear all vision history JSON files from the assets directory."""
    # Find assets directory
    plugin_dir = Path(__file__).parent.parent
    assets_dir = plugin_dir / "assets"

    if not assets_dir.exists():
        print("Assets directory not found")
        return

    # Find all vision history files
    pattern = str(assets_dir / "vision_history_*.json")
    history_files = glob.glob(pattern)

    if not history_files:
        print("No vision history files found")
        return

    print(f"Found {len(history_files)} vision history files")

    # Confirm deletion
    response = input("Delete all vision history files? [y/N] ")
    if response.lower() != "y":
        print("Aborted")
        return

    # Delete files
    deleted = 0
    for f in history_files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"Deleted {deleted} files")


if __name__ == "__main__":
    main()
