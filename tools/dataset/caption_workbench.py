#!/usr/bin/env python3
"""
Caption Workbench — local server for interactive caption comparison.

Serves the workbench UI and proxies API calls to Gemini and Claude.

Usage:
    uv run python tools/dataset/caption_workbench.py [port]
    # Default port: 8765, then open http://localhost:8765
"""

from __future__ import annotations

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import requests

from tools.dataset.constants import CAPTION_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HTML_FILE = Path(__file__).resolve().parent / "caption_workbench.html"
FRAMES_DIR = PROJECT_ROOT / "assets" / "embedded_frames"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", ""):
            self._serve_file(HTML_FILE, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/prompt":
            self._send_json({"prompt": CAPTION_PROMPT})
        elif path == "/api/scenes":
            self._list_scenes()
        elif path.startswith("/assets/embedded_frames/"):
            full = PROJECT_ROOT / path.lstrip("/")
            if full.is_file() and full.suffix in (".jpg", ".jpeg", ".png"):
                mime = "image/png" if full.suffix == ".png" else "image/jpeg"
                self._serve_file(full, mime)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/caption":
            self._proxy_caption()
        elif self.path == "/api/models":
            self._list_models()
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _list_scenes(self) -> None:
        scenes: dict[str, list[str]] = {}
        if FRAMES_DIR.is_dir():
            for d in sorted(FRAMES_DIR.iterdir()):
                if d.is_dir() and d.name.startswith("scene_"):
                    frames = sorted(
                        (f.name for f in d.glob("frame_*.jpg")),
                        key=lambda n: int(n.split("_")[1].split(".")[0]),
                    )
                    if frames:
                        scenes[d.name] = frames
        self._send_json(scenes)

    def _list_models(self) -> None:
        try:
            body = self._read_json()
        except Exception as e:
            self._send_json({"error": f"Invalid request: {e}"}, 400)
            return

        provider = body.get("provider", "")
        api_key = body.get("api_key", "")
        if not provider or not api_key:
            self._send_json({"error": "Missing provider or api_key"}, 400)
            return

        try:
            if provider == "gemini":
                models = self._fetch_gemini_models(api_key)
            elif provider == "claude":
                models = self._fetch_claude_models(api_key)
            else:
                self._send_json({"error": f"Unknown provider: {provider}"}, 400)
                return
            self._send_json({"models": models})
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            detail = ""
            if e.response is not None:
                try:
                    detail = json.dumps(e.response.json())
                except Exception:
                    detail = e.response.text[:500]
            self._send_json({"error": f"API {status}: {detail}"}, 502)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _fetch_gemini_models(self, api_key: str) -> list[dict[str, str]]:
        resp = requests.get(
            f"{GEMINI_API_BASE}/models",
            params={"key": api_key, "pageSize": 100},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        models = []
        for m in data.get("models", []):
            # Only include models that support generateContent (vision/text gen)
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            model_id = m.get("name", "").removeprefix("models/")
            display = m.get("displayName", model_id)
            models.append({"id": model_id, "name": display})

        # Sort newest first: reverse-alphabetical puts higher version numbers on top
        models.sort(key=lambda x: x["id"], reverse=True)
        return models

    def _fetch_claude_models(self, api_key: str) -> list[dict[str, str]]:
        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        models = []
        for m in data.get("data", []):
            model_id = m.get("id", "")
            display = m.get("display_name", model_id)
            created = m.get("created_at", "")
            models.append({"id": model_id, "name": display, "created_at": created})

        # Sort newest first by creation date (ISO format sorts lexically)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return models

    def _proxy_caption(self) -> None:
        try:
            body = self._read_json()
        except Exception as e:
            self._send_json({"error": f"Invalid request: {e}"}, 400)
            return

        provider = body.get("provider", "")
        model = body.get("model", "")
        api_key = body.get("api_key", "")
        scene = body.get("scene", "scene_10065")
        frame = body.get("frame", "")
        prompt = body.get("prompt", "")
        temperature = float(body.get("temperature", 0.2))

        if not all([provider, model, api_key, frame, prompt]):
            self._send_json({"error": "Missing required fields"}, 400)
            return

        frame_path = FRAMES_DIR / scene / frame
        if not frame_path.is_file():
            self._send_json({"error": f"Frame not found: {scene}/{frame}"}, 404)
            return

        frame_b64 = base64.b64encode(frame_path.read_bytes()).decode()

        try:
            if provider == "gemini":
                result = self._call_gemini(model, api_key, frame_b64, prompt, temperature)
            elif provider == "claude":
                result = self._call_claude(model, api_key, frame_b64, prompt, temperature)
            else:
                self._send_json({"error": f"Unknown provider: {provider}"}, 400)
                return
            self._send_json(result)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            detail = ""
            if e.response is not None:
                try:
                    detail = json.dumps(e.response.json())
                except Exception:
                    detail = e.response.text[:1000]
            self._send_json({"error": f"API {status}: {detail}"}, 502)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _call_gemini(
        self, model: str, api_key: str, frame_b64: str, prompt: str, temperature: float
    ) -> dict:
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
        }
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "promptFeedback" in data:
            reason = data["promptFeedback"].get("blockReason")
            if reason:
                raise RuntimeError(f"Prompt blocked: {reason}")

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates in Gemini response")

        candidate = candidates[0]
        finish = candidate.get("finishReason", "STOP")
        if finish not in ("STOP", "MAX_TOKENS"):
            raise RuntimeError(f"Generation stopped: {finish}")

        caption = candidate["content"]["parts"][0]["text"]

        # Extract token usage from usageMetadata
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        # Gemini provides per-modality breakdown in promptTokensDetails
        details = usage.get("promptTokensDetails", [])
        image_tokens = 0
        text_tokens = 0
        for d in details:
            modality = d.get("modality", "")
            count = d.get("tokenCount", 0)
            if modality == "IMAGE":
                image_tokens = count
            elif modality == "TEXT":
                text_tokens = count

        # If no detailed breakdown, estimate text tokens from total
        if not details and input_tokens:
            text_tokens = input_tokens
            image_tokens = 0

        return {
            "caption": caption,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "image_tokens": image_tokens,
                "text_tokens": text_tokens,
            },
        }

    def _call_claude(
        self, model: str, api_key: str, frame_b64: str, prompt: str, temperature: float
    ) -> dict:
        payload: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": frame_b64,
                }},
                {"type": "text", "text": prompt},
            ]}],
        }
        if temperature != 1.0:
            payload["temperature"] = temperature

        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("type") == "error":
            raise RuntimeError(data.get("error", {}).get("message", str(data)))

        texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        if not texts:
            raise RuntimeError("No text in Claude response")
        caption = "".join(texts)

        # Extract token usage
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        # Anthropic doesn't provide image/text breakdown directly.
        # Estimate text tokens (~4 chars per token for English) from prompt length.
        estimated_text_tokens = len(prompt) // 4
        image_tokens = max(0, input_tokens - estimated_text_tokens)

        return {
            "caption": caption,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "image_tokens": image_tokens,
                "text_tokens": estimated_text_tokens,
                "text_tokens_estimated": True,
            },
        }

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"  {format % args}\n")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadedServer(("", port), Handler)
    print(f"Caption Workbench: http://localhost:{port}")
    print(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
