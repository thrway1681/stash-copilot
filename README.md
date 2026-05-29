# Stash Copilot

AI-powered assistant for your [StashApp](https://github.com/stashapp/stash) library. Chat with your library, get personalized recommendations, analyze scenes with vision AI, and discover similar content.

## Features

### Core AI Features
- **AI Chat Interface** - Natural language conversations about your library using tool-based queries
- **Library Insights** - Get AI-generated summaries of your viewing habits and statistics
- **Scene Vision Analysis** - Analyze scene content using multimodal LLMs (Gemma 3, LLaVA, etc.)
- **Smart Tag Suggestions** - Vision AI suggests relevant tags with confidence scores

### Recommendations & Discovery
- **Personalized Recommendations** - AI-powered suggestions based on your viewing history and engagement
- **Similar Scenes** - Find visually and contextually similar scenes using embeddings
- **Taste Clustering** - Automatic grouping of your preferences into taste profiles
- **Peak Moments** - Recommendations based on O-moment embeddings (scenes with similar highlights)

### Scene Page Integration
- **Analyze Tab** - Vision analysis, tag suggestions, and AI chat for any scene
- **Similar Tab** - Find similar scenes with adjustable visual/metadata weighting
- **Recs Tab** - Personalized recommendations seeded from the current scene

## Screenshots

*Coming soon*

## Requirements

- [StashApp](https://github.com/stashapp/stash) v0.24+
- Python 3.10+ with [UV](https://docs.astral.sh/uv/)
- **LLM Provider** (one of):
  - **Ollama** (recommended for local) - `llama3`, `gemma3`, `mistral`
  - **OpenRouter** - Access to many models via API
  - **OpenAI API** - GPT-4o, GPT-4
  - **Anthropic API** - Claude models
- **For vision features**: `gemma-3`, `llava`, `pixtral`, or any vision-capable model
- **For recommendations**: Embeddings are computed locally using OpenCLIP (GPU recommended)

## Installation

1. Clone or download this repository to your Stash plugins directory:
   ```bash
   cd ~/.stash/plugins
   git clone https://github.com/thrway1681/stash-copilot.git
   ```

2. Install Python dependencies:
   ```bash
   cd stash-copilot
   uv sync
   ```

3. Restart Stash or reload plugins from Settings → Plugins

4. Configure the plugin in Settings → Plugins → Stash Copilot:
   - Set your **Ollama Host** (default: `http://localhost:11434`)
   - Set your **Ollama Model** (e.g., `llama3`, `mistral`)
   - Set your **Vision Model** for scene analysis (e.g., `llava`)

## Usage

### AI Chat

1. Click the **Stash Copilot** icon in the navigation bar
2. Go to the **Chat** tab
3. Ask questions about your library:
   - "What are my most watched tags?"
   - "Which performers have the highest rated scenes?"
   - "Show me stats for scenes added this month"

### Scene Vision Analysis

1. Navigate to any scene page
2. Click **"Analyze with AI"** button
3. The AI will analyze the scene's sprite sheet and:
   - Describe the visual content
   - Suggest relevant tags from your library
4. Click suggested tags to apply them to the scene

### Library Summary

1. Go to Settings → Tasks
2. Run **"Generate Library Summary"**
3. View the AI-generated summary of your library statistics

### Recommendations

1. First, run **"Embed All Scenes"** from Settings → Tasks to index your library
2. Navigate to any scene and click the **"Recs"** tab
3. Toggle between **Discover** (unwatched) and **Re-watch** (favorites) modes
4. Adjust the slider to blend between your overall taste and the current scene

### Similar Scenes

1. Navigate to any scene and click the **"Similar"** tab
2. Adjust the **Meta ↔ Visual** slider to weight metadata vs visual similarity
3. Use **Diff. Performers** sub-tab to find similar scenes with different performers

## Available Tools

The AI has access to these database query tools:

| Tool | Description | Example Query |
|------|-------------|---------------|
| Library Stats | Overall library statistics | "How big is my library?" |
| Scene Search | Search scenes with filters | "Find scenes with tag X" |
| Performer Stats | Performer statistics | "Who are my top performers?" |
| Tag Stats | Tag usage statistics | "What are my most used tags?" |
| Tag Correlations | Find co-occurring tags | "What tags appear with 'outdoor'?" |
| Recent Activity | Recently played/added content | "What did I watch this week?" |
| Rating Distribution | Scene rating breakdown | "How are my scenes rated?" |
| Studio Stats | Studio statistics | "Which studios have the most content?" |
| Performer Scene History | Scenes by performer | "Show me scenes with performer X" |
| Scene Details | Detailed scene info | "Tell me about scene ID 123" |
| File Info | File/storage statistics | "How much storage am I using?" |
| Marker Search | Search scene markers | "Find all bookmarked moments" |
| Common Tags | Tags shared by top performers | "What do my favorites have in common?" |

## Configuration

### Plugin Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **LLM Provider** | `ollama`, `openrouter`, `openai`, or `anthropic` | `ollama` |
| Ollama Host | URL of Ollama server | `http://localhost:11434` |
| Ollama Model | Text generation model | (required) |
| API Key | For cloud providers | (empty) |
| Vision Model | Model for scene analysis | `llava` |
| Excluded Tags | Tags to hide from AI | (empty) |

### Recommendation Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Top Scenes | Scenes used for profile building | 20 |
| O-Count Weight | Weight for O-counter in scoring | 20.0 |
| View Weight | Weight per replay (views beyond first) | 2.0 |
| Duration Weight | Weight per hour of play time | 1.0 |
| Rating Weight | Weight per star (only if rated) | 1.5 |
| Time Decay Days | Half-life for recency weighting | 30 |

### Using OpenAI or Anthropic

1. Set **LLM Provider** to `openai` or `anthropic`
2. Enter your **API Key**
3. Set **Ollama Model** to the model name (e.g., `gpt-4`, `claude-3-sonnet`)

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy stash_ai/

# Test plugin standalone (will fail without stdin, but checks imports)
uv run python stash-copilot.py
```

### Project Structure

```
stash-copilot/
├── stash-copilot.yml      # Plugin manifest
├── stash-copilot.py       # Python backend entry point
├── stash-copilot.js       # JavaScript frontend
├── stash-copilot.css      # Styles
├── run-plugin.sh          # UV wrapper script
├── stash_ai/              # AI functionality package
│   ├── config.py          # LLM configuration
│   ├── llm/               # LLM providers (Ollama, OpenRouter, OpenAI, Anthropic)
│   ├── tools/             # Database query tools for AI chat
│   ├── tasks/             # Task implementations (analysis, embedding, etc.)
│   ├── embeddings/        # Image embedding providers and storage
│   ├── recommendations/   # Recommendation engine and taste profiling
│   ├── preferences/       # Bayesian preference learning
│   └── prompts/           # YAML prompt templates
├── assets/                # Runtime data
│   ├── embeddings.db      # Scene embeddings database
│   ├── embedded_frames/   # Cached video frames
│   └── recommendations_*.json
└── tests/                 # Test suite
```

## Troubleshooting

### Plugin not loading
- Check Stash logs for errors
- Ensure `run-plugin.sh` is executable: `chmod +x run-plugin.sh`
- Verify UV is installed: `uv --version`

### Ollama connection failed
- Ensure Ollama is running: `ollama serve`
- Check the host URL in settings matches your Ollama server
- For remote Ollama, ensure the port is accessible

### Vision analysis not working
- Install a vision model: `ollama pull llava`
- Set the Vision Model in plugin settings
- Ensure the scene has generated sprite thumbnails

### No AI response
- Check browser console for errors (F12)
- Verify the model is downloaded: `ollama list`
- Try a smaller model if running out of memory

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License - see LICENSE file

## Acknowledgments

- [StashApp](https://github.com/stashapp/stash) - The media organizer this plugin extends
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [stashapi](https://github.com/stashapp/stashapp-tools) - Python library for Stash
