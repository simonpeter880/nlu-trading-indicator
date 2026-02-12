"""
Tests for retry utilities with exponential backoff.
"""

import asyncio
import time

import pytest

from indicator.utils.retry import (
    ExponentialBackoff,
    RetryContext,
    RetryError,
    retry_async,
    retry_sync,
)


class TestExponentialBackoff:
    """Tests for ExponentialBackoff class."""

    def test_calculate_no_jitter(self):
        """Test exponential backoff calculation without jitter."""
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=60.0, jitter=False)

        assert backoff.calculate(0) == 1.0
        assert backoff.calculate(1) == 2.0
        assert backoff.calculate(2) == 4.0
        assert backoff.calculate(3) == 8.0
        assert backoff.calculate(4) == 16.0
        assert backoff.calculate(5) == 32.0
        assert backoff.calculate(6) == 60.0  # Capped at max_delay

    def test_calculate_with_jitter(self):
        """Test exponential backoff calculation with jitter."""
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=60.0, jitter=True)

        # With jitter, delays should be within ±25% of base delay
        for attempt in range(5):
            delay = backoff.calculate(attempt)
            base_delay = min(1.0 * (2.0**attempt), 60.0)
            assert 0.75 * base_delay <= delay <= 1.25 * base_delay

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=10.0, jitter=False)

        assert backoff.calculate(10) == 10.0  # Should be capped at 10
        assert backoff.calculate(100) == 10.0  # Should still be capped

    def test_custom_parameters(self):
        """Test with custom base and multiplier."""
        backoff = ExponentialBackoff(base=0.5, multiplier=3.0, max_delay=20.0, jitter=False)

        assert backoff.calculate(0) == 0.5
        assert backoff.calculate(1) == 1.5  # 0.5 * 3
        assert backoff.calculate(2) == 4.5  # 0.5 * 9
        assert backoff.calculate(3) == 13.5  # 0.5 * 27


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""

    def test_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        @retry_sync(max_attempts=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_success_after_retries(self):
        """Test successful execution after failures."""
        call_count = 0

        @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_all_attempts_fail(self):
        """Test failure after all retry attempts exhausted."""

        @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            always_fails()

        assert "failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, ValueError)

    def test_unexpected_exception_no_retry(self):
        """Test that unexpected exceptions are not retried."""
        call_count = 0

        @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def raises_unexpected():
            nonlocal call_count
            call_count += 1
            raise TypeError("Unexpected")

        with pytest.raises(TypeError):
            raises_unexpected()

        assert call_count == 1  # Should not retry

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        retry_attempts = []

        def on_retry(exc, attempt):
            retry_attempts.append((str(exc), attempt))

        call_count = 0

        @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.01, on_retry=on_retry)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert len(retry_attempts) == 2
        assert retry_attempts[0] == ("Attempt 1", 0)
        assert retry_attempts[1] == ("Attempt 2", 1)


class TestRetryAsyncDecorator:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Test successful async execution on first attempt."""
        call_count = 0

        @retry_async(max_attempts=3)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_function()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test successful async execution after failures."""
        call_count = 0

        @retry_async(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = await eventually_successful()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self):
        """Test async failure after all retry attempts exhausted."""

        @retry_async(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        async def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            await always_fails()

        assert "failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, ValueError)

    @pytest.mark.asyncio
    async def test_unexpected_exception_no_retry(self):
        """Test that unexpected async exceptions are not retried."""
        call_count = 0

        @retry_async(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        async def raises_unexpected():
            nonlocal call_count
            call_count += 1
            raise TypeError("Unexpected")

        with pytest.raises(TypeError):
            await raises_unexpected()

        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test that async on_retry callback is called."""
        retry_attempts = []

        def on_retry(exc, attempt):
            retry_attempts.append((str(exc), attempt))

        call_count = 0

        @retry_async(max_attempts=3, exceptions=(ValueError,), base_delay=0.01, on_retry=on_retry)
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"

        result = await eventually_successful()
        assert result == "success"
        assert len(retry_attempts) == 2
        assert retry_attempts[0] == ("Attempt 1", 0)
        assert retry_attempts[1] == ("Attempt 2", 1)


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_success_first_attempt(self):
        """Test successful execution on first attempt."""
        retry_ctx = RetryContext(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)

        attempt_count = 0
        for attempt in retry_ctx:
            with attempt:
                attempt_count += 1
                result = "success"
                break

        assert result == "success"
        assert attempt_count == 1

    def test_success_after_retries(self):
        """Test successful execution after failures."""
        retry_ctx = RetryContext(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)

        attempt_count = 0
        for attempt in retry_ctx:
            with attempt:
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError("Not yet")
                result = "success"
                break

        assert result == "success"
        assert attempt_count == 3

    def test_all_attempts_fail(self):
        """Test failure after all retry attempts exhausted."""
        retry_ctx = RetryContext(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)

        with pytest.raises(RetryError) as exc_info:
            for attempt in retry_ctx:
                with attempt:
                    raise ValueError("Always fails")

        assert "failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, ValueError)

    def test_unexpected_exception_propagates(self):
        """Test that unexpected exceptions propagate immediately."""
        retry_ctx = RetryContext(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)

        attempt_count = 0
        with pytest.raises(TypeError):
            for attempt in retry_ctx:
                with attempt:
                    attempt_count += 1
                    raise TypeError("Unexpected")

        assert attempt_count == 1  # Should not retry


class TestRetryTiming:
    """Tests for retry timing behavior."""

    def test_sync_retry_timing(self):
        """Test that sync retries respect exponential backoff timing."""
        call_times = []

        @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.1, jitter=False)
        def timed_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not yet")
            return "success"

        result = timed_function()
        assert result == "success"
        assert len(call_times) == 3

        # Check delays between calls (should be ~0.1s, ~0.2s)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 <= delay1 <= 0.15  # ~0.1s with tolerance
        assert 0.18 <= delay2 <= 0.25  # ~0.2s with tolerance

    @pytest.mark.asyncio
    async def test_async_retry_timing(self):
        """Test that async retries respect exponential backoff timing."""
        call_times = []

        @retry_async(max_attempts=3, exceptions=(ValueError,), base_delay=0.1, jitter=False)
        async def timed_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not yet")
            return "success"

        result = await timed_function()
        assert result == "success"
        assert len(call_times) == 3

        # Check delays between calls
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 <= delay1 <= 0.15
        assert 0.18 <= delay2 <= 0.25
