"""Gemini generateContent wrapper that returns caption + real token usage.

Every API call returns a CaptionResult with:
- caption: the text output
- input_tokens: from usageMetadata.promptTokenCount (measured, not estimated)
- output_tokens: from usageMetadata.candidatesTokenCount (measured)

These feed directly into ApiBudget.record_usage() for accurate cost tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from tools.dataset.api_budget import DailyLimitReached

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class CaptionResult:
    """Result from a single generateContent call."""
    caption: str
    input_tokens: int
    output_tokens: int


def parse_response(response: dict) -> CaptionResult:
    """Extract caption + token usage from a Gemini generateContent response.

    Raises RuntimeError if the response is blocked or empty.
    """
    # Check for prompt-level blocking
    if "promptFeedback" in response:
        reason = response["promptFeedback"].get("blockReason")
        if reason:
            raise RuntimeError(f"Prompt blocked: {reason}")

    candidates = response.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates in Gemini response")

    candidate = candidates[0]
    finish = candidate.get("finishReason", "STOP")

    parts = candidate.get("content", {}).get("parts", [])
    has_text = parts and "text" in parts[0]

    # If blocked AND no partial text was generated, raise
    if finish not in ("STOP", "MAX_TOKENS", "PROHIBITED_CONTENT"):
        raise RuntimeError(f"Generation blocked: {finish}")

    if not has_text:
        raise RuntimeError(f"No text in response (finishReason={finish})")

    caption = parts[0]["text"].strip()

    # If we got text but it was cut short by PROHIBITED_CONTENT,
    # still use it — partial captions are better than no captions
    if not caption:
        raise RuntimeError(f"Empty caption (finishReason={finish})")

    # Extract actual token counts from usageMetadata
    usage = response.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    return CaptionResult(
        caption=caption,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def caption_frame(
    frame_b64: str,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_retries: int = 3,
) -> CaptionResult:
    """Caption a single frame via Gemini generateContent.

    Retries on 429 (rate limit) and 5xx (server error) with exponential backoff.

    Returns CaptionResult with caption text and actual token usage.
    """
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
        "safetySettings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=120)
            resp.raise_for_status()
            return parse_response(resp.json())
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                # Parse the response body to distinguish daily quota from
                # momentary rate limit.  Gemini returns JSON like:
                #   {"error": {"status": "RESOURCE_EXHAUSTED",
                #    "message": "...per_model_per_day..."}}
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
                ))
                if is_daily:
                    detail = e.response.text[:300] if e.response is not None else str(e)
                    raise DailyLimitReached(f"Gemini API quota: {detail}")
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
