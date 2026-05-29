"""Tests for tools.dataset.api_budget — rate limiting, cost tracking, dashboard."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.api_budget import (
    ApiBudget,
    BudgetExhausted,
    DailyLimitReached,
    GeminiPricing,
    PRICING,
    compute_cost,
)


# ── Pricing ──────────────────────────────────────────────────────────────


def test_pricing_table_has_known_models() -> None:
    assert "gemini-3-flash-preview" in PRICING
    assert "gemini-2.0-flash" in PRICING
    assert "gemini-3.1-pro-preview" in PRICING
    assert "gemini-3-pro-preview" in PRICING


def test_compute_cost_uses_actual_tokens() -> None:
    pricing = PRICING["gemini-3-flash-preview"]
    # 1,500 input tokens × $0.50/1M = $0.00075
    # 100 output tokens × $3.00/1M = $0.0003
    cost = compute_cost(pricing, input_tokens=1_500, output_tokens=100)
    assert abs(cost - 0.00105) < 0.00001


def test_compute_cost_batch_pricing() -> None:
    pricing = PRICING["gemini-3-flash-preview:batch"]
    cost = compute_cost(pricing, input_tokens=1_500, output_tokens=100)
    assert abs(cost - 0.000525) < 0.00001


# ── Budget cap ───────────────────────────────────────────────────────────


def test_budget_exhausted_raises() -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=0.001,
    )
    # Simulate recording usage that exceeds budget
    budget.record_usage(input_tokens=1_500, output_tokens=100)  # $0.00105
    with pytest.raises(BudgetExhausted):
        budget.acquire()


def test_no_budget_cap_allows_unlimited() -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=None,
    )
    budget.record_usage(input_tokens=1_500, output_tokens=100)
    budget.acquire()  # should not raise


# ── RPD tracking ─────────────────────────────────────────────────────────


def test_rpd_over_limit_still_acquires() -> None:
    """RPD is tracked for display only; acquire() no longer raises."""
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=2,
        max_cost=None,
    )
    budget.record_usage(input_tokens=100, output_tokens=10)
    budget.record_usage(input_tokens=100, output_tokens=10)
    budget.acquire()  # should not raise — API enforces its own limits


# ── RPM throttling ───────────────────────────────────────────────────────


def test_rpm_throttle_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RPM is at limit, acquire() should sleep until window clears."""
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=2,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=None,
    )
    # Fill RPM window
    now = time.monotonic()
    budget._rpm_timestamps.append(now)
    budget._rpm_timestamps.append(now)

    sleep_calls: list[float] = []
    # After sleep is called, advance clock past the 60s window so entries expire
    clock = [now + 0.1]

    def fake_sleep(s: float) -> None:
        sleep_calls.append(s)
        clock[0] = now + 61  # advance past window expiry

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    budget.acquire()
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] > 0


# ── TPM throttling ───────────────────────────────────────────────────────


def test_tpm_throttle_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=2000,  # must be > est_tokens so the request can eventually fit
        rpd_limit=10_000,
        max_cost=None,
    )
    # Set measured tokens so est_tokens = 50+10 = 60 (fits in 2000 limit)
    budget.measured_input_tokens_per_call = 50
    budget.measured_output_tokens_per_call = 10

    now = time.monotonic()
    # Fill TPM window to the limit
    budget._tpm_entries.append((now, 2000))
    budget._tpm_tokens_in_window = 2000

    sleep_calls: list[float] = []
    clock = [now + 0.1]

    def fake_sleep(s: float) -> None:
        sleep_calls.append(s)
        clock[0] = now + 61  # advance past window expiry

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    budget.acquire()
    assert len(sleep_calls) >= 1


# ── State persistence ────────────────────────────────────────────────────


def test_state_persistence(tmp_path: Path) -> None:
    state_file = tmp_path / "budget_state.json"
    budget1 = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=100.0,
        state_file=state_file,
    )
    budget1.record_usage(input_tokens=1_500, output_tokens=100)
    budget1.record_usage(input_tokens=1_500, output_tokens=100)
    budget1.save_state()

    # New instance should restore state
    budget2 = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=100.0,
        state_file=state_file,
    )
    assert budget2.total_calls == 2
    assert budget2.total_cost > 0
    assert budget2.total_input_tokens == 3_000
    assert budget2.total_output_tokens == 200


# ── Thread safety ────────────────────────────────────────────────────────


def test_concurrent_record_usage() -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=10_000,
        tpm_limit=100_000_000,
        rpd_limit=100_000,
        max_cost=None,
    )
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(100):
                budget.record_usage(input_tokens=100, output_tokens=10)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert budget.total_calls == 1000


# ── Dashboard ────────────────────────────────────────────────────────────


def test_dashboard_format() -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=900,
        tpm_limit=900_000,
        rpd_limit=9_500,
        max_cost=50.0,
    )
    budget.record_usage(input_tokens=1_500, output_tokens=100)
    budget._total_frames = 2_419_013

    text = budget.dashboard()
    assert "Calls:" in text
    assert "Cost:" in text
    assert "$50.00" in text  # budget cap
    assert "RPD:" in text


# ── Cost estimation ──────────────────────────────────────────────────────


def test_estimate_total_cost_from_measured() -> None:
    budget = ApiBudget(
        model="gemini-3-flash-preview",
        rpm_limit=900,
        tpm_limit=900_000,
        rpd_limit=9_500,
        max_cost=None,
    )
    # Simulate measuring prompt tokens via countTokens API
    budget.measured_input_tokens_per_call = 1_500
    budget.measured_output_tokens_per_call = 100

    estimated = budget.estimate_total_cost(n_frames=2_419_013)
    # 2,419,013 × $0.00105 = ~$2,540
    assert 2_400 < estimated < 2_700
