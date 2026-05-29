"""Tests for tools.dataset.gemini_api — Gemini generateContent wrapper."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.gemini_api import (
    caption_frame,
    parse_response,
    CaptionResult,
)
from tools.dataset.api_budget import DailyLimitReached


def _make_response(caption: str, input_tok: int = 1500, output_tok: int = 100) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": caption}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": input_tok,
            "candidatesTokenCount": output_tok,
            "totalTokenCount": input_tok + output_tok,
        },
    }


def test_parse_response_extracts_caption_and_usage() -> None:
    resp = _make_response("A doggy style scene.", input_tok=1500, output_tok=80)
    result = parse_response(resp)
    assert result.caption == "A doggy style scene."
    assert result.input_tokens == 1500
    assert result.output_tokens == 80


def test_parse_response_raises_on_blocked() -> None:
    resp = {
        "candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}],
        "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
    }
    with pytest.raises(RuntimeError, match="blocked"):
        parse_response(resp)


def test_parse_response_salvages_prohibited_content_with_text() -> None:
    """PROHIBITED_CONTENT with partial text should return the caption, not raise."""
    resp = {
        "candidates": [{
            "finishReason": "PROHIBITED_CONTENT",
            "content": {"parts": [{"text": "A partial caption that got cut off"}]},
        }],
        "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 40, "totalTokenCount": 1540},
    }
    result = parse_response(resp)
    assert result.caption == "A partial caption that got cut off"
    assert result.input_tokens == 1500
    assert result.output_tokens == 40


def test_parse_response_raises_on_prohibited_content_no_text() -> None:
    """PROHIBITED_CONTENT with no text should raise."""
    resp = {
        "candidates": [{
            "finishReason": "PROHIBITED_CONTENT",
            "content": {"parts": []},
        }],
        "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 0, "totalTokenCount": 1500},
    }
    with pytest.raises(RuntimeError, match="No text"):
        parse_response(resp)


def test_parse_response_raises_on_empty() -> None:
    resp = {"candidates": [], "usageMetadata": {}}
    with pytest.raises(RuntimeError):
        parse_response(resp)


@patch("tools.dataset.gemini_api.requests.post")
def test_caption_frame_returns_result(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response("Cowgirl POV.", 1500, 60)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = caption_frame("base64data", "prompt", "gemini-3-flash-preview", "key", 1.0)
    assert result.caption == "Cowgirl POV."
    assert result.input_tokens == 1500
    assert result.output_tokens == 60


@patch("tools.dataset.gemini_api.time.sleep")
@patch("tools.dataset.gemini_api.requests.post")
def test_caption_frame_raises_daily_limit_on_429_resource_exhausted(
    mock_post: MagicMock, mock_sleep: MagicMock,
) -> None:
    """A 429 with RESOURCE_EXHAUSTED / per_day should raise DailyLimitReached immediately."""
    import requests as _req

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = (
        '{"error":{"code":429,"message":"Quota exceeded for metric: '
        'generativelanguage.googleapis.com/generate_requests_per_model_per_day",'
        '"status":"RESOURCE_EXHAUSTED"}}'
    )
    http_err = _req.exceptions.HTTPError(response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_err
    mock_post.return_value = mock_resp

    with pytest.raises(DailyLimitReached, match="quota"):
        caption_frame("base64data", "prompt", "gemini-3-flash-preview", "key", 1.0)

    # Should NOT have retried — raise immediately on daily limit
    mock_sleep.assert_not_called()


@patch("tools.dataset.gemini_api.time.sleep")
@patch("tools.dataset.gemini_api.requests.post")
def test_caption_frame_retries_on_429_without_quota_keywords(
    mock_post: MagicMock, mock_sleep: MagicMock,
) -> None:
    """A 429 without quota keywords should retry with backoff."""
    import requests as _req

    # First two calls: 429 without quota keywords, third call: success
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = '{"error":{"code":429,"message":"Rate limit temporarily exceeded"}}'
    http_err = _req.exceptions.HTTPError(response=mock_resp_429)
    mock_resp_429.raise_for_status.side_effect = http_err

    mock_resp_ok = MagicMock()
    mock_resp_ok.json.return_value = _make_response("Success.", 1500, 60)
    mock_resp_ok.raise_for_status = MagicMock()

    mock_post.side_effect = [mock_resp_429, mock_resp_429, mock_resp_ok]

    # Should NOT raise — should retry and eventually succeed
    # Wait: the mock_resp_429 returns raise_for_status side_effect, but
    # mock_resp_ok doesn't, so we need to set up mock_post to return different responses
    result = caption_frame("base64data", "prompt", "gemini-3-flash-preview", "key", 1.0)
    assert result.caption == "Success."
    assert mock_sleep.call_count == 2  # retried twice before success
