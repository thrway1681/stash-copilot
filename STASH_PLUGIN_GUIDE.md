# Stash Plugin Development Best Practices Guide

This document provides comprehensive guidance for developing plugins for StashApp, compiled from official documentation, community examples, and the existing stash-copilot implementation.

---

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [YAML Configuration](#yaml-configuration)
3. [Python Backend Plugins](#python-backend-plugins)
4. [JavaScript UI Plugins](#javascript-ui-plugins)
5. [Communication Patterns](#communication-patterns)
6. [Hook System](#hook-system)
7. [GraphQL API Integration](#graphql-api-integration)
8. [Best Practices](#best-practices)
9. [Common Pitfalls](#common-pitfalls)
10. [File Structure Template](#file-structure-template)
11. [Resources](#resources)

> **Compatibility:** This guide targets **Stash v0.25.0 and later** (current: v0.30.1). Some features require specific versions: `args_map` and `PluginApi.Event` (v0.25.0+), `PluginApi.patch.instead` multi-hook (v0.27.0+), GraphQL content-type change (v0.29.0+).

---

## Plugin Architecture Overview

Stash supports a **two-tier plugin system**:

| Type | Purpose | Execution Context | Primary Use Cases |
|------|---------|-------------------|-------------------|
| **Python** | Backend logic | Server-side | API integration, file operations, data processing, automated tasks |
| **JavaScript** | UI customization | Browser | DOM manipulation, interactive elements, visual enhancements |

Both types require YAML configuration files that define metadata, entry points, and integration hooks.

### Key Concepts

1. **Tasks**: User-triggered operations callable from Stash UI (Settings > Tasks)
2. **Hooks**: Event-driven triggers that fire on database operations (Scene.Create.Post, etc.)
3. **UI Injection**: JavaScript/CSS files injected into the Stash web interface

---

## YAML Configuration

The plugin manifest (`plugin-name.yml`) is the entry point for all plugins.

### Complete Configuration Schema

```yaml
# Required fields
name: Plugin Display Name
description: Brief description of what the plugin does
version: 1.0.0

# Optional metadata
url: https://github.com/your/repo

# Execution configuration (for Python/executable plugins)
exec:
  - python              # or ./run-plugin.sh for wrapper scripts
  - plugin-name.py

# Interface type (determines communication protocol)
interface: raw          # Options: raw (JSON), rpc, js

# Error logging level for stderr
errLog: info            # Options: trace, debug, info, warning, error

# Plugin settings (configurable via Stash UI)
settings:
  setting_name:
    displayName: Human Readable Name
    description: What this setting does
    type: STRING        # Options: STRING, NUMBER, BOOLEAN

# Manual tasks (appear in Settings > Tasks)
tasks:
  - name: Task Name
    description: Description shown in UI
    execArgs:                    # Arguments appended to exec command
      - --task
      - task_name
    defaultArgs:                 # Default values if not provided at runtime
      mode: task_mode
      optional_arg: default_value

# Event-driven hooks
hooks:
  - name: Hook Name
    description: When this hook triggers
    triggeredBy:
      - Scene.Create.Post
      - Scene.Update.Post

# UI injection (JavaScript plugins)
ui:
  requires:
    - another-plugin-id   # Plugin dependencies
  javascript:
    - plugin.js           # JS files to inject
  css:
    - plugin.css          # CSS files to inject
  assets:
    /: assets             # Map URL paths to directories
  # Content Security Policy overrides (nested under ui:)
  csp:
    script-src:
      - https://cdn.example.com
    style-src:
      - https://cdn.example.com
    connect-src:
      - https://api.example.com
```

### Interface Types

| Interface | Description | Use Case |
|-----------|-------------|----------|
| `raw` | JSON-encoded stdin/stdout | Most Python plugins |
| `rpc` | RPC protocol | Complex bidirectional communication |
| `js` | JavaScript interface | Pure JS plugins (no external process) |

---

## Python Backend Plugins

### Input/Output Protocol

Stash communicates with Python plugins via stdin/stdout JSON:

**Input (received via stdin):**
```json
{
  "server_connection": {
    "Scheme": "http",
    "Host": "localhost",
    "Port": 9999,
    "SessionCookie": {"Name": "session", "Value": "..."},
    "Dir": "/root/.stash",
    "PluginDir": "/root/.stash/plugins/your-plugin"
  },
  "args": {
    "mode": "task_name",
    "scene_id": "123",
    "custom_arg": "value"
  }
}
```

**Output (write to stdout/stderr):**
```json
{"output": "Task completed successfully"}
```

For errors:
```json
{"error": "Something went wrong"}
```

### Base Plugin Class Template

```python
#!/usr/bin/env python3
"""StashApp Python Plugin Template"""

import sys
import json
import os
from typing import Dict, Any, Optional
from stashapi.stashapp import StashInterface
from stashapi import log as stash_log

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)


class StashPlugin:
    """Base class for StashApp Python plugins."""

    def __init__(self):
        self.stash_url = "http://localhost:9999"
        self.input = None
        self.stash = None
        self._read_input()

        if self.input:
            server_config = self.input.get("server_connection", {})
            scheme = server_config.get("Scheme", "http")
            host = server_config.get("Host", "localhost")
            port = server_config.get("Port", 9999)

            # 0.0.0.0 means "all interfaces" - use 127.0.0.1 for local
            if host == "0.0.0.0":
                host = "127.0.0.1"

            self.stash_url = f"{scheme}://{host}:{port}"
            # StashInterface handles auth automatically
            self.stash = StashInterface(server_config)

    def _read_input(self):
        """Read and parse JSON input from stdin."""
        try:
            input_str = sys.stdin.read()
            if input_str:
                self.input = json.loads(input_str)
        except json.JSONDecodeError as e:
            self.error(f"Failed to parse input JSON: {e}")
            sys.exit(1)

    def log(self, message: str, level: str = "info"):
        """Log a message to Stash."""
        level_map = {
            "trace": stash_log.trace,
            "debug": stash_log.debug,
            "info": stash_log.info,
            "warning": stash_log.warning,
            "error": stash_log.error,
        }
        log_fn = level_map.get(level, stash_log.info)
        log_fn(message)

    def error(self, message: str):
        """Log an error message."""
        self.log(message, "error")

    def progress(self, current: int, total: int):
        """Report progress (0.0-1.0 range)."""
        value = current / total if total > 0 else 0
        stash_log.progress(value)

    def get_plugin_settings(self, plugin_id: str) -> Dict[str, Any]:
        """Fetch plugin settings from Stash."""
        if not self.stash:
            return {}
        try:
            return self.stash.find_plugin_config(plugin_id)
        except Exception as e:
            self.log(f"Error fetching settings: {e}", "error")
            return {}


class MyPlugin(StashPlugin):
    """Your plugin implementation."""

    def run(self):
        """Main entry point."""
        args = self.input.get("args", {})
        mode = args.get("mode", "")

        if mode == "my_task":
            self.run_my_task(args)
        else:
            self.error(f"Unknown mode: {mode}")

    def run_my_task(self, args: Dict[str, Any]):
        """Example task implementation."""
        self.log("Starting my task...")
        # Your logic here
        self.log("Task complete!")


if __name__ == "__main__":
    plugin = MyPlugin()
    plugin.run()
```

### Using stashapi Library

> **Two Python packages are available:**
>
> | Package | Version | Notes |
> |---------|---------|-------|
> | `stashapp-tools` | 0.2.59+ | More established, widely used in community plugins |
> | `stashapi` | 0.1.3+ | Newer package, same author (stg-annon) |
>
> Both work identically. Install with: `pip install stashapp-tools` or `pip install stashapi`

The library provides `StashInterface` for API interactions:

```python
from stashapi.stashapp import StashInterface
from stashapi import log as stash_log

# Initialize with server_connection from input
stash = StashInterface(server_connection)

# Common operations
scene = stash.find_scene(scene_id)
scenes = stash.find_scenes(f={"per_page": 100})
stash.update_scene({"id": scene_id, "title": "New Title"})

# Direct GraphQL
result = stash.call_GQL("""
    query FindScene($id: ID!) {
        findScene(id: $id) {
            id
            title
            performers { name }
        }
    }
""", {"id": scene_id})

# Get plugin configuration
config = stash.find_plugin_config("plugin-id")
```

### Logging Convention

Always use the stashapi log module for output:

```python
from stashapi import log

log.trace("Verbose debugging info")
log.debug("Development debugging")
log.info("Normal operation messages")
log.warning("Non-fatal issues")
log.error("Failures requiring attention")
log.progress(0.5)  # Progress bar (0.0-1.0)
```

The log module outputs JSON to stderr in the format Stash expects:
```json
{"output": "message", "level": "info"}
```

---

## JavaScript UI Plugins

> **Breaking Change (v0.29.0):** GraphQL responses now use content-type `application/graphql-response+json` instead of `application/json`. Update any code that intercepts or filters GraphQL requests by content-type.

### Structure Pattern (IIFE)

JavaScript plugins should use Immediately Invoked Function Expressions to prevent namespace pollution:

```javascript
(function() {
    'use strict';

    // Plugin configuration
    const PLUGIN_ID = 'my-plugin';

    // State management
    const state = {
        initialized: false,
        // ... plugin-specific state
    };

    // Helper functions
    function log(message, level = 'info') {
        console[level](`[${PLUGIN_ID}] ${message}`);
    }

    // GraphQL helper using csLib
    async function callGQL(query, variables = {}) {
        if (window.csLib && window.csLib.callGQL) {
            return await window.csLib.callGQL({ query, variables });
        }
        // Fallback implementation
        const response = await fetch('/graphql', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, variables })
        });
        const result = await response.json();
        return result.data;
    }

    // Main setup function
    function setup() {
        if (state.initialized) return;
        state.initialized = true;
        log('Plugin initialized');
        // Setup logic here
    }

    // Route detection and initialization
    function init() {
        // Use csLib if available
        if (window.csLib && window.csLib.PathElementListener) {
            csLib.PathElementListener('/scenes/', setup);
        } else {
            // Fallback: MutationObserver or polling
            const observer = new MutationObserver(() => {
                if (window.location.pathname.includes('/scenes/')) {
                    setup();
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
    }

    // Wait for DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
```

### csLib Utility Library

The CommunityScriptsUILibrary provides helper functions:

```javascript
// GraphQL calls
const data = await csLib.callGQL({
    query: `query { findScenes { count } }`,
    variables: {}
});

// Wait for DOM element
csLib.waitForElement('.scene-card', (element) => {
    // Element is now available
});

// Route-based initialization
csLib.PathElementListener('/scenes/', () => {
    // Called when navigating to /scenes/ routes
});

// Plugin configuration
const config = await csLib.getConfiguration('my-plugin', { default: true });
await csLib.setConfiguration('my-plugin', { ...config, newSetting: 'value' });
```

### JavaScript Plugin API (v0.25.0+)

Stash exposes additional APIs for UI plugins beyond csLib:

#### Event System
```javascript
// Listen for navigation changes (alternative to MutationObserver)
PluginApi.Event.addEventListener('stash:location', (event) => {
    console.log('Navigated to:', event.detail.data.location.pathname);
});
```

#### Available Libraries
- `Mousetrap` - Keyboard shortcut handling
- `MousetrapPause` - Pause/resume keyboard shortcuts

#### Exposed React Hooks
- `PluginApi.hooks.useToast` - Toast notifications

#### Exposed Components
- Studio, Performer, Tag, Gallery select controls
- Date, Country, Folder input components

#### Function Patching (v0.27.0+)
```javascript
// Multiple plugins can now hook the same function
PluginApi.patch.instead('TargetComponent.method', function(original, ...args) {
    // Your override logic
    return original.apply(this, args);
});
```

### DOM Manipulation Best Practices

```javascript
// Create elements programmatically
function createButton(text, onClick) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-primary my-plugin-btn';
    btn.textContent = text;
    btn.addEventListener('click', onClick);
    return btn;
}

// Find Stash UI elements safely
function findSceneCard(sceneId) {
    return document.querySelector(`[data-scene-id="${sceneId}"]`);
}

// Inject into existing UI
function injectControls(container) {
    const existingControls = container.querySelector('.scene-toolbar');
    if (existingControls && !container.querySelector('.my-plugin-btn')) {
        existingControls.appendChild(createButton('My Action', handleAction));
    }
}

// Cleanup when plugin disabled/page changes
function cleanup() {
    document.querySelectorAll('.my-plugin-btn').forEach(el => el.remove());
}
```

### Embedded JavaScript Plugins (interface: js)

When using `interface: js`, plugins execute entirely within Stash without spawning external processes. These have access to special API objects:

```javascript
// GraphQL operations
const result = await gql.Do(`query { findScenes { count } }`);

// Logging (note: capitalized methods)
log.Info("Informational message");
log.Debug("Debug info");
log.Error("Error message");
log.Progress(0.5); // Progress bar (0.0-1.0)

// Utilities
util.Sleep(1000); // Pause execution for 1000ms
```

> **Important:** The `gql`, `log`, and `util` objects are **only available** for embedded JS plugins (`interface: js`). They are NOT available for UI-injected JavaScript files (which use csLib instead).

---

## Communication Patterns

### Python Plugin to Frontend

For complex data, write to the assets directory and have frontend poll:

```python
# Python: Write results to assets
import json
import os

results = {"status": "complete", "data": [...]}
assets_dir = os.path.join(PLUGIN_DIR, "assets")
os.makedirs(assets_dir, exist_ok=True)

with open(os.path.join(assets_dir, f"results_{request_id}.json"), "w") as f:
    json.dump(results, f)
```

```javascript
// JavaScript: Poll for results
async function pollResults(requestId) {
    const response = await fetch(`/plugin/my-plugin/assets/results_${requestId}.json`);
    if (response.ok) {
        return await response.json();
    }
    return null;
}

async function waitForResults(requestId, interval = 1000, timeout = 60000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        const results = await pollResults(requestId);
        if (results && results.status === 'complete') {
            return results;
        }
        await new Promise(r => setTimeout(r, interval));
    }
    throw new Error('Timeout waiting for results');
}
```

### Running Plugin Tasks from JavaScript

```javascript
// UPDATED (v0.25.0+): args is deprecated, use args_map
async function runPluginTask(pluginId, taskName, args = {}, description = '') {
    const mutation = `
        mutation RunPluginTask(
            $plugin_id: ID!,
            $task_name: String,
            $args_map: Map,
            $description: String
        ) {
            runPluginTask(
                plugin_id: $plugin_id,
                task_name: $task_name,
                args_map: $args_map,
                description: $description
            )
        }
    `;

    const variables = {
        plugin_id: pluginId,
        task_name: taskName || null,  // Optional since v0.25.0
        args_map: args,               // New: replaces deprecated 'args'
        description: description || null  // Custom task queue description
    };

    return await callGQL(mutation, variables);
}

// Usage
await runPluginTask('my-plugin', 'Process Scene', { scene_id: '123' });

// Also available: runPluginOperation - immediate execution (no task queue)
// Returns plugin output directly instead of queuing
```

---

## Hook System

### Available Hooks

All hooks fire **after** their respective operations (Post-trigger pattern):

| Category | Hooks |
|----------|-------|
| **Scene** | `Scene.Create.Post`, `Scene.Update.Post`, `Scene.Destroy.Post` |
| **SceneMarker** | `SceneMarker.Create.Post`, `SceneMarker.Update.Post`, `SceneMarker.Destroy.Post` |
| **Image** | `Image.Create.Post`, `Image.Update.Post`, `Image.Destroy.Post` |
| **Gallery** | `Gallery.Create.Post`, `Gallery.Update.Post`, `Gallery.Destroy.Post` |
| **GalleryChapter** | `GalleryChapter.Create.Post`, `GalleryChapter.Update.Post`, `GalleryChapter.Destroy.Post` |
| **Performer** | `Performer.Create.Post`, `Performer.Update.Post`, `Performer.Destroy.Post` |
| **Studio** | `Studio.Create.Post`, `Studio.Update.Post`, `Studio.Destroy.Post` |
| **Tag** | `Tag.Create.Post`, `Tag.Update.Post`, `Tag.Merge.Post`, `Tag.Destroy.Post` |
| **Group** | `Group.Create.Post`, `Group.Update.Post`, `Group.Destroy.Post` |

> **Note (v0.27.0):** Movies have been **renamed to Groups** with new capabilities (orderable sub-groups, descriptions). The `Movie.*` hooks (`Movie.Create.Post`, `Movie.Update.Post`, `Movie.Destroy.Post`) are deprecated - use `Group.*` hooks instead. Existing `Movie.*` hooks work for backwards compatibility only.

### Hook Input Structure

```json
{
  "server_connection": { ... },
  "args": {
    "hookContext": {
      "id": "123",
      "type": "Scene.Update.Post",
      "input": { /* update input data */ },
      "inputFields": ["title", "rating100"]
    }
  }
}
```

### Hook Implementation Example

```yaml
# In plugin.yml
hooks:
  - name: Auto-tag on scan
    description: Automatically adds tags based on file path
    triggeredBy:
      - Scene.Create.Post
```

```python
# In plugin.py
def run(self):
    args = self.input.get("args", {})
    hook_context = args.get("hookContext")

    if hook_context:
        self.handle_hook(hook_context)
    else:
        mode = args.get("mode", "")
        # Handle tasks

def handle_hook(self, context):
    hook_type = context.get("type")
    entity_id = context.get("id")

    if hook_type == "Scene.Create.Post":
        self.process_new_scene(entity_id)
    elif hook_type == "Scene.Update.Post":
        input_fields = context.get("inputFields", [])
        if "tags" in input_fields:
            self.validate_tags(entity_id)
```

---

## GraphQL API Integration

### Common Queries

```graphql
# Find single scene with full details
query FindScene($id: ID!) {
    findScene(id: $id) {
        id
        title
        date
        rating100
        o_counter
        play_count
        play_duration
        organized

        # Playback tracking
        resume_time
        last_played_at
        play_history

        # Interactive content
        interactive
        interactive_speed
        captions { ... }

        files {
            path
            size
            duration
            video_codec
            audio_codec
            width
            height
        }
        performers {
            id
            name
        }
        studio {
            id
            name
        }
        tags {
            id
            name
        }

        # Groups (replaces deprecated movies)
        groups {
            id
            name
        }
    }
}

# Find scenes with filtering and pagination
query FindScenes($filter: FindFilterType, $scene_filter: SceneFilterType) {
    findScenes(filter: $filter, scene_filter: $scene_filter) {
        count
        scenes {
            id
            title
            rating100
        }
    }
}

# Variables example
{
    "filter": {
        "page": 1,
        "per_page": 25,
        "sort": "play_count",
        "direction": "DESC"
    },
    "scene_filter": {
        "rating100": { "value": 80, "modifier": "GREATER_THAN" }
    }
}
```

### Common Mutations

```graphql
# Update scene
mutation SceneUpdate($input: SceneUpdateInput!) {
    sceneUpdate(input: $input) {
        id
    }
}

# Variables
{
    "input": {
        "id": "123",
        "title": "New Title",
        "rating100": 85,
        "tag_ids": ["1", "2", "3"]
    }
}

# Increment O-counter
mutation SceneAddO($id: ID!) {
    sceneAddO(id: $id)
}

# Record play activity
mutation SceneAddPlay($id: ID!) {
    sceneAddPlay(id: $id)
}
```

---

## Best Practices

### Python Plugins

1. **Dependency Management**
   - Document all dependencies in `requirements.txt`
   - Use `stashapp-tools` for Stash API interactions
   - Consider bundling dependencies or using virtual environments

2. **Error Handling**
   - Use specific exception types, not bare `except:`
   - Log errors with context for debugging
   - Gracefully degrade when optional features unavailable

3. **Performance**
   - Batch GraphQL queries when possible
   - Use pagination for large result sets
   - Report progress for long-running tasks

4. **Typing**
   - Use type hints throughout (`TypedDict`, `Optional`, etc.)
   - Helps both human understanding and AI assistance

5. **Logging**
   - Use appropriate log levels
   - Include context in log messages
   - Avoid logging sensitive data

### JavaScript Plugins

1. **Encapsulation**
   - Always use IIFE pattern
   - Prefix CSS classes with plugin name
   - Avoid polluting global namespace

2. **Performance**
   - Minimize DOM manipulations
   - Use event delegation for dynamic elements
   - Debounce/throttle expensive operations

3. **Cleanup**
   - Remove event listeners when not needed
   - Clean up DOM elements on page changes
   - Clear intervals/timeouts

4. **User Experience**
   - Provide visual feedback for async operations
   - Handle errors gracefully with user-friendly messages
   - Follow Stash's existing UI patterns

5. **Compatibility**
   - Test with current stable Stash release
   - Use csLib when available, fallback otherwise
   - Don't assume specific DOM structure (may change)

---

## Common Pitfalls

### Python

| Pitfall | Solution |
|---------|----------|
| `ModuleNotFoundError: stashapi` | Install with `pip install stashapp-tools` |
| Externally-managed Python (Docker) | Create venv in mounted volume: `python -m venv /root/.stash/plugins/my-plugin/venv`, then update exec to use `./venv/bin/python` |
| Plugin not finding modules | Add plugin directory to `sys.path` at top of script |
| Host 0.0.0.0 connection errors | Replace with 127.0.0.1 for local connections |
| Settings not loading | Use correct plugin ID (yml filename without extension) |
| `python` vs `python3` naming | If system requires `python3`, update exec: `exec: [python3, plugin.py]` |
| Dependencies lost on container restart | Install packages in mounted volume path, not system Python |

### JavaScript

| Pitfall | Solution |
|---------|----------|
| CSP blocking CDN scripts | Bundle libraries locally in assets folder |
| Elements not found | Use `waitForElement()` or MutationObserver |
| State lost on navigation | Use localStorage or poll server for persistent state |
| Multiple initialization | Track initialized state, use guards |
| Memory leaks | Clean up observers, listeners, intervals |
| GraphQL interception broken (v0.29.0+) | Content-type changed from `application/json` to `application/graphql-response+json`. Update any code checking content-type headers. |

### General

| Pitfall | Solution |
|---------|----------|
| Hook infinite loops | Stash prevents via cookie-based context tracking |
| Large payloads via stdin | Write to assets directory, poll from frontend |
| Permissions in Docker | Use mounted volumes, check file ownership |
| Stale asset files | Include timestamps or request IDs in filenames |

---

## File Structure Template

```
my-plugin/
├── my-plugin.yml           # Plugin manifest (required)
├── my-plugin.py            # Python entry point
├── my-plugin.js            # JavaScript UI code
├── my-plugin.css           # Styles
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── run-plugin.sh           # Optional wrapper script
├── assets/                 # Static assets and runtime data
│   ├── icon.png
│   └── results/            # Runtime output files
└── my_plugin/              # Python package (for complex plugins)
    ├── __init__.py
    ├── config.py
    ├── tasks/
    │   └── __init__.py
    └── utils/
        └── __init__.py
```

### Wrapper Script (run-plugin.sh)

Useful for virtual environment activation or pre-processing:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if exists
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Run the plugin
exec python "$@"
```

### Alternative: UV Package Manager

Modern Python projects can use [UV](https://github.com/astral-sh/uv) instead of pip for faster dependency management:

```bash
#!/bin/bash
# run-plugin.sh with UV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
uv run python "$1"
```

Benefits of UV:
- Faster package installs (10-100x vs pip)
- Automatic virtual environment management
- Lockfile support for reproducible builds
- Drop-in replacement for pip commands

---

## Resources

### Official Documentation
- [Stash Docs](https://docs.stashapp.cc) - Official guides and troubleshooting
- [In-App Manual](https://docs.stashapp.cc/in-app-manual) - Plugin documentation

### Community Resources
- [CommunityScripts Repository](https://github.com/stashapp/CommunityScripts) - Community plugins and examples
- [Plugin Repository Template](https://github.com/stashapp/plugins-repo-template) - Template for hosting your own plugins
- [stashapp-tools (PyPI)](https://pypi.org/project/stashapp-tools/) - Python API wrapper

### Support Channels
- [Discourse Forum](https://discourse.stashapp.cc) - Community support and discussions
- [Discord](https://discord.gg/2TsNFKt) - Real-time chat
- [GitHub Discussions](https://github.com/stashapp/stash/discussions) - Feature discussions

### Technical References
- [Plugin Package (Go)](https://pkg.go.dev/github.com/stashapp/stash/pkg/plugin) - Plugin system internals
- [Hook Package (Go)](https://pkg.go.dev/github.com/stashapp/stash/pkg/plugin/hook) - Hook trigger definitions
- [DeepWiki - Plugin Development](https://deepwiki.com/stashapp/CommunityScripts/6.2-plugin-development) - Technical deep dive
