"""OpenRouter chat/completions wrapper that returns caption + real token usage.

Mirrors the interface of gemini_api.py but calls OpenRouter's OpenAI-compatible
API instead of Gemini's generateContent endpoint.  Returns the same CaptionResult
so the caption runner can dispatch transparently.

Every API call returns a CaptionResult with:
- caption: the text output
- input_tokens: from usage.prompt_tokens
- output_tokens: from usage.completion_tokens
"""
from __future__ import annotations

import time

import requests

from tools.dataset.api_budget import DailyLimitReached
from tools.dataset.gemini_api import CaptionResult

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def caption_frame(
    frame_b64: str,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_retries: int = 3,
) -> CaptionResult:
    """Caption a single frame via OpenRouter chat/completions.

    Uses the OpenAI-compatible multimodal format (image_url content parts).
    Retries on 429 (rate limit) and 5xx (server error) with exponential backoff.

    Returns CaptionResult with caption text and actual token usage.
    """
    url = f"{OPENROUTER_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/stashapp/stash",
        "X-Title": "Stash Copilot Caption Pipeline",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return parse_response(resp.json())
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                body_text = ""
                if e.response is not None:
                    try:
                        body_text = e.response.text.lower()
                    except Exception:
                        pass
                is_daily = any(kw in body_text for kw in (
                    "per_day", "per day", "daily",
                    "resource_exhausted", "resource exhausted",
                    "quota exceeded", "quota",
                    "rate limit",  # OpenRouter uses "rate limit" phrasing
                ))
                if is_daily:
                    detail = e.response.text[:300] if e.response is not None else str(e)
                    raise DailyLimitReached(f"OpenRouter API quota: {detail}")
                # Momentary rate limit — retry with backoff
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            elif status >= 500:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.ConnectionError as e:
            last_error = e
            wait = 2 ** attempt
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def parse_response(response: dict) -> CaptionResult:
    """Extract caption + token usage from an OpenRouter chat/completions response.

    Raises RuntimeError if the response is empty or has no content.
    """
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("No choices in OpenRouter response")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    finish_reason = choices[0].get("finish_reason", "stop")

    if not content or not content.strip():
        raise RuntimeError(
            f"Empty caption from OpenRouter (finish_reason={finish_reason})"
        )

    # Content moderation — OpenRouter may return a refusal
    if finish_reason == "content_filter":
        # If we still got text, use it (partial is better than nothing)
        if content.strip():
            pass  # fall through to return
        else:
            raise RuntimeError("Response blocked by content filter")

    caption = content.strip()

    # Extract actual token counts from usage
    usage = response.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return CaptionResult(
        caption=caption,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
