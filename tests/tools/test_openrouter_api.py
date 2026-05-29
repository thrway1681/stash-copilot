"""Tests for tools.dataset.openrouter_api — OpenRouter caption wrapper."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.openrouter_api import (
    caption_frame,
    parse_response,
)
from tools.dataset.gemini_api import CaptionResult
from tools.dataset.api_budget import DailyLimitReached


def _make_response(
    caption: str,
    prompt_tokens: int = 1500,
    completion_tokens: int = 100,
) -> dict:
    return {
        "choices": [{
            "message": {"content": caption},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_parse_response_extracts_caption_and_usage() -> None:
    resp = _make_response("A scene with two performers.", prompt_tokens=1500, completion_tokens=80)
    result = parse_response(resp)
    assert result.caption == "A scene with two performers."
    assert result.input_tokens == 1500
    assert result.output_tokens == 80


def test_parse_response_raises_on_empty_choices() -> None:
    resp = {"choices": [], "usage": {}}
    with pytest.raises(RuntimeError, match="No choices"):
        parse_response(resp)


def test_parse_response_raises_on_empty_content() -> None:
    resp = {
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    with pytest.raises(RuntimeError, match="Empty caption"):
        parse_response(resp)


def test_parse_response_content_filter_with_text() -> None:
    """content_filter with partial text should return the caption."""
    resp = {
        "choices": [{
            "message": {"content": "A partial caption"},
            "finish_reason": "content_filter",
        }],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 20},
    }
    result = parse_response(resp)
    assert result.caption == "A partial caption"


@patch("tools.dataset.openrouter_api.requests.post")
def test_caption_frame_returns_result(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response("Cowgirl POV.", 1500, 60)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = caption_frame("base64data", "prompt", "google/gemini-3-flash-preview", "sk-or-key", 1.0)
    assert result.caption == "Cowgirl POV."
    assert result.input_tokens == 1500
    assert result.output_tokens == 60

    # Verify Authorization header uses Bearer format
    call_kwargs = mock_post.call_args
    assert "Bearer sk-or-key" in call_kwargs.kwargs["headers"]["Authorization"]


@patch("tools.dataset.openrouter_api.time.sleep")
@patch("tools.dataset.openrouter_api.requests.post")
def test_caption_frame_raises_daily_limit_on_429_quota(
    mock_post: MagicMock, mock_sleep: MagicMock,
) -> None:
    """A 429 with quota keywords should raise DailyLimitReached immediately."""
    import requests as _req

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = '{"error":{"message":"Rate limit exceeded: quota exceeded for today","type":"rate_limit"}}'
    http_err = _req.exceptions.HTTPError(response=mock_resp)
    mock_resp.raise_for_status.side_effect = http_err
    mock_post.return_value = mock_resp

    with pytest.raises(DailyLimitReached, match="OpenRouter API quota"):
        caption_frame("base64data", "prompt", "google/gemini-3-flash-preview", "sk-or-key", 1.0)

    mock_sleep.assert_not_called()


@patch("tools.dataset.openrouter_api.time.sleep")
@patch("tools.dataset.openrouter_api.requests.post")
def test_caption_frame_retries_on_transient_429(
    mock_post: MagicMock, mock_sleep: MagicMock,
) -> None:
    """A 429 without quota keywords should retry and succeed."""
    import requests as _req

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = '{"error":{"message":"Too many requests, please slow down"}}'
    http_err = _req.exceptions.HTTPError(response=mock_resp_429)
    mock_resp_429.raise_for_status.side_effect = http_err

    mock_resp_ok = MagicMock()
    mock_resp_ok.json.return_value = _make_response("Success.", 1500, 60)
    mock_resp_ok.raise_for_status = MagicMock()

    mock_post.side_effect = [mock_resp_429, mock_resp_429, mock_resp_ok]

    result = caption_frame("base64data", "prompt", "google/gemini-3-flash-preview", "sk-or-key", 1.0)
    assert result.caption == "Success."
    assert mock_sleep.call_count == 2
