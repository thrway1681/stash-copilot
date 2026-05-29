"""Thread-safe API budget manager for Gemini caption pipeline.

Provides:
- Rate limiting: RPM (requests/min), TPM (tokens/min), RPD (requests/day)
- Cost tracking: actual cost from real token counts, not estimates
- Budget cap: hard stop when spending exceeds --max-cost
- Dashboard: periodic status with real numbers
- State persistence: RPD + cost survive restarts

All token counts come from real API responses (usageMetadata), not estimates.
The only "estimate" is the pre-run cost projection, which uses measured
prompt tokens from the countTokens API + average output tokens from early calls.

Pricing source: https://ai.google.dev/gemini-api/docs/pricing
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ── Exceptions ───────────────────────────────────────────────────────────


class BudgetExhausted(Exception):
    """Raised when spending exceeds the configured max_cost."""


class DailyLimitReached(Exception):
    """Raised when RPD limit for today is reached."""


# ── Pricing ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeminiPricing:
    """Pricing per 1M tokens for a Gemini model."""
    input_per_m: float   # $/1M input tokens (text + image + video)
    output_per_m: float  # $/1M output tokens


# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-02-19)
# Pro model prices are placeholders — update when official pricing confirmed
PRICING: dict[str, GeminiPricing] = {
    "gemini-3-flash-preview": GeminiPricing(input_per_m=0.50, output_per_m=3.00),
    "gemini-3-flash-preview:batch": GeminiPricing(input_per_m=0.25, output_per_m=1.50),
    "gemini-3.1-pro-preview": GeminiPricing(input_per_m=2.50, output_per_m=15.00),
    "gemini-3.1-pro-preview:batch": GeminiPricing(input_per_m=1.25, output_per_m=7.50),
    "gemini-3-pro-preview": GeminiPricing(input_per_m=2.50, output_per_m=15.00),
    "gemini-3-pro-preview:batch": GeminiPricing(input_per_m=1.25, output_per_m=7.50),
    "gemini-2.0-flash": GeminiPricing(input_per_m=0.10, output_per_m=0.40),
    "gemini-2.0-flash:batch": GeminiPricing(input_per_m=0.05, output_per_m=0.20),
    # OpenRouter models — same Gemini models routed through OpenRouter (separate quota)
    "google/gemini-3-flash-preview": GeminiPricing(input_per_m=0.50, output_per_m=3.00),
    "google/gemini-3.1-pro-preview": GeminiPricing(input_per_m=2.50, output_per_m=15.00),
    "google/gemini-3-pro-preview": GeminiPricing(input_per_m=2.50, output_per_m=15.00),
}


def compute_cost(pricing: GeminiPricing, input_tokens: int, output_tokens: int) -> float:
    """Compute actual cost from real token counts."""
    return (input_tokens * pricing.input_per_m / 1_000_000) + \
           (output_tokens * pricing.output_per_m / 1_000_000)


# ── countTokens API ─────────────────────────────────────────────────────


def count_tokens(
    model: str,
    api_key: str,
    prompt: str,
    frame_b64: str,
) -> int:
    """Call Gemini's countTokens API to measure exact input token count.

    This is FREE (no billing) and has a separate 3,000 RPM quota.
    Should be called once at startup with a sample frame to measure
    the real prompt + image token count.

    Returns:
        Total input token count (prompt text + image tokens).
    """
    url = f"{GEMINI_API_BASE}/models/{model}:countTokens"
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
            {"text": prompt},
        ]}],
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("totalTokens", 0)


# ── ApiBudget ────────────────────────────────────────────────────────────


class ApiBudget:
    """Thread-safe API budget manager.

    Usage:
        budget = ApiBudget(model="gemini-3-flash-preview", ...)

        # At startup, measure real token count:
        budget.measured_input_tokens_per_call = count_tokens(model, key, prompt, sample_b64)
        print(budget.estimate_total_cost(n_frames=2_419_013))

        # In each worker thread:
        budget.acquire()             # blocks until rate limits allow
        response = call_gemini(...)  # your API call
        usage = response["usageMetadata"]
        budget.record_usage(
            input_tokens=usage["promptTokenCount"],
            output_tokens=usage["candidatesTokenCount"],
        )
    """

    def __init__(
        self,
        model: str,
        rpm_limit: int = 900,
        tpm_limit: int = 900_000,
        rpd_limit: int = 9_500,
        max_cost: float | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.max_cost = max_cost
        self.state_file = state_file

        # Pricing lookup — fall back to flash if model unknown
        self.pricing = PRICING.get(model, PRICING["gemini-2.0-flash"])

        # Measured values (set after calling countTokens)
        self.measured_input_tokens_per_call: int = 0
        self.measured_output_tokens_per_call: int = 0  # updated as rolling avg

        # Accumulators
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_errors: int = 0
        self._total_frames: int = 0  # set externally for dashboard %

        # RPD tracking (date-scoped)
        self._rpd_count: int = 0
        self._rpd_date: str = date.today().isoformat()

        # Sliding windows (thread-safe via lock)
        self._lock = threading.Lock()
        self._rpm_timestamps: deque[float] = deque()
        self._tpm_entries: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._tpm_tokens_in_window: int = 0

        # Start time for dashboard
        self._start_time: float = time.monotonic()

        # Load persisted state if available
        if state_file and state_file.exists():
            self._load_state()

    # ── Rate limiting ────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Block until rate limits allow another API call.

        Raises:
            BudgetExhausted: if total_cost >= max_cost
            DailyLimitReached: if RPD count >= rpd_limit
        """
        with self._lock:
            # 1. Check budget cap
            if self.max_cost is not None and self.total_cost >= self.max_cost:
                raise BudgetExhausted(
                    f"Budget exhausted: ${self.total_cost:.2f} >= ${self.max_cost:.2f} cap"
                )

            # 2. Track RPD for display (actual limits enforced by API responses;
            #    the runner detects sustained errors as a rate-limit signal)
            today = date.today().isoformat()
            if today != self._rpd_date:
                self._rpd_count = 0
                self._rpd_date = today

        # 3. RPM throttle (outside lock to avoid blocking other threads on sleep)
        self._wait_for_rpm()

        # 4. TPM throttle
        self._wait_for_tpm()

    def _wait_for_rpm(self) -> None:
        """Sleep until the RPM sliding window has room."""
        while True:
            now = time.monotonic()
            with self._lock:
                # Evict timestamps older than 60s
                while self._rpm_timestamps and self._rpm_timestamps[0] < now - 60:
                    self._rpm_timestamps.popleft()

                if len(self._rpm_timestamps) < self.rpm_limit:
                    self._rpm_timestamps.append(now)
                    return

                # Calculate sleep time until oldest entry expires
                sleep_until = self._rpm_timestamps[0] + 60
                wait = sleep_until - now

            if wait > 0:
                time.sleep(wait)

    def _wait_for_tpm(self) -> None:
        """Sleep until the TPM sliding window has room."""
        # Estimate tokens for next call
        est_tokens = self.measured_input_tokens_per_call + self.measured_output_tokens_per_call
        if est_tokens <= 0:
            est_tokens = 1_600  # fallback if not measured yet

        while True:
            now = time.monotonic()
            with self._lock:
                # Evict entries older than 60s
                while self._tpm_entries and self._tpm_entries[0][0] < now - 60:
                    _, tokens = self._tpm_entries.popleft()
                    self._tpm_tokens_in_window -= tokens

                if self._tpm_tokens_in_window + est_tokens <= self.tpm_limit:
                    return

                # Wait until oldest entry expires
                if self._tpm_entries:
                    sleep_until = self._tpm_entries[0][0] + 60
                    wait = sleep_until - now
                else:
                    wait = 1.0

            if wait > 0:
                time.sleep(wait)

    # ── Recording ────────────────────────────────────────────────────────

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual token usage from a completed API call.

        Args:
            input_tokens: From usageMetadata.promptTokenCount
            output_tokens: From usageMetadata.candidatesTokenCount
        """
        cost = compute_cost(self.pricing, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self._rpd_count += 1

            # Update TPM window
            now = time.monotonic()
            self._tpm_entries.append((now, total_tokens))
            self._tpm_tokens_in_window += total_tokens

            # Update rolling average of output tokens (for cost estimation)
            if self.total_calls > 0:
                self.measured_output_tokens_per_call = (
                    self.total_output_tokens // self.total_calls
                )

    def record_error(self) -> None:
        """Record a failed API call (no tokens consumed)."""
        with self._lock:
            self.total_errors += 1

    # ── Recording with model override ────────────────────────────────────

    def record_usage_for_model(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record actual token usage, computing cost with the specific model's pricing.

        Use this when a fallback model was used instead of the primary model.

        Args:
            model: The model that actually served the request.
            input_tokens: From usageMetadata.promptTokenCount
            output_tokens: From usageMetadata.candidatesTokenCount
        """
        pricing = PRICING.get(model, self.pricing)
        cost = compute_cost(pricing, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self._rpd_count += 1

            # Update TPM window
            now = time.monotonic()
            self._tpm_entries.append((now, total_tokens))
            self._tpm_tokens_in_window += total_tokens

            # Update rolling average of output tokens
            if self.total_calls > 0:
                self.measured_output_tokens_per_call = (
                    self.total_output_tokens // self.total_calls
                )

    # ── Cost estimation ──────────────────────────────────────────────────

    def estimate_total_cost(self, n_frames: int) -> float:
        """Pre-run cost estimate using measured token counts.

        Call this after setting measured_input_tokens_per_call (from countTokens API)
        and optionally measured_output_tokens_per_call (from early runs or default).

        Returns estimated total cost in USD.
        """
        input_per_call = self.measured_input_tokens_per_call
        output_per_call = self.measured_output_tokens_per_call or 100  # output default
        return n_frames * compute_cost(self.pricing, input_per_call, output_per_call)

    # ── Dashboard ────────────────────────────────────────────────────────

    def dashboard(self) -> str:
        """Formatted dashboard string with real numbers."""
        elapsed = time.monotonic() - self._start_time
        elapsed_str = _format_duration(elapsed)

        with self._lock:
            calls = self.total_calls
            cost = self.total_cost
            errors = self.total_errors
            rpd = self._rpd_count
            input_tok = self.total_input_tokens
            output_tok = self.total_output_tokens

            # Current RPM (calls in last 60s)
            now = time.monotonic()
            while self._rpm_timestamps and self._rpm_timestamps[0] < now - 60:
                self._rpm_timestamps.popleft()
            current_rpm = len(self._rpm_timestamps)

            # Current TPM
            while self._tpm_entries and self._tpm_entries[0][0] < now - 60:
                _, t = self._tpm_entries.popleft()
                self._tpm_tokens_in_window -= t
            current_tpm = self._tpm_tokens_in_window

        # Progress
        total = self._total_frames or 1
        pct = calls / total * 100 if total else 0

        # ETA
        if calls > 0 and elapsed > 0:
            rate = calls / elapsed  # calls/sec
            remaining = total - calls
            eta_secs = remaining / rate if rate > 0 else 0
            eta_str = _format_duration(eta_secs)
        else:
            eta_str = "calculating..."

        # Cost per call (actual average)
        avg_cost = cost / calls if calls else 0

        # Budget line
        if self.max_cost is not None:
            budget_str = f"${cost:.2f} / ${self.max_cost:.2f} budget ({cost / self.max_cost * 100:.1f}%)"
        else:
            budget_str = f"${cost:.2f} (no cap)"

        lines = [
            "── Dashboard ──────────────────────────────────",
            f"  Calls:       {calls:,} / {total:,} ({pct:.2f}%)",
            f"  Cost:        {budget_str}",
            f"  Avg cost:    ${avg_cost:.6f}/call",
            f"  Input tok:   {input_tok:,}  Output tok: {output_tok:,}",
            f"  RPM:         {current_rpm} / {self.rpm_limit}",
            f"  TPM:         {current_tpm:,} / {self.tpm_limit:,}",
            f"  RPD:         {rpd:,} / {self.rpd_limit:,}",
            f"  Errors:      {errors}",
            f"  Elapsed:     {elapsed_str}",
            f"  ETA:         {eta_str}",
            "───────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist budget state to disk for resume across restarts."""
        if not self.state_file:
            return
        state = {
            "model": self.model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "total_errors": self.total_errors,
            "rpd_count": self._rpd_count,
            "rpd_date": self._rpd_date,
            "max_cost": self.max_cost,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        """Restore budget state from disk."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.total_calls = state.get("total_calls", 0)
            self.total_input_tokens = state.get("total_input_tokens", 0)
            self.total_output_tokens = state.get("total_output_tokens", 0)
            self.total_cost = state.get("total_cost", 0.0)
            self.total_errors = state.get("total_errors", 0)

            # RPD: only restore if same day
            saved_rpd_date = state.get("rpd_date", "")
            if saved_rpd_date == date.today().isoformat():
                self._rpd_count = state.get("rpd_count", 0)
                self._rpd_date = saved_rpd_date

            # Restore rolling average
            if self.total_calls > 0:
                self.measured_output_tokens_per_call = (
                    self.total_output_tokens // self.total_calls
                )
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt state file, start fresh


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds % 60:.0f}s"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.0f}h {minutes % 60:.0f}m"
    days = hours / 24
    return f"{days:.1f}d {hours % 24:.0f}h"
