"""
Retry utilities with exponential backoff for resilient API calls.

Provides decorators and utilities for handling transient failures in:
- External API calls (Binance, etc.)
- Network operations
- Rate-limited services
- Temporary service unavailability
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter.

    Implements exponential backoff with optional jitter to prevent
    thundering herd problems when multiple clients retry simultaneously.

    Args:
        base: Base delay in seconds (default: 1.0)
        multiplier: Exponential growth factor (default: 2.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        jitter: Add random jitter to delays (default: True)

    Example:
        >>> backoff = ExponentialBackoff(base=1.0, multiplier=2.0)
        >>> backoff.calculate(attempt=0)  # First retry
        1.0
        >>> backoff.calculate(attempt=1)  # Second retry
        2.0
        >>> backoff.calculate(attempt=2)  # Third retry
        4.0
    """

    def __init__(
        self,
        base: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.base = base
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter

    def calculate(self, attempt: int) -> float:
        """
        Calculate delay for given retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(self.base * (self.multiplier**attempt), self.max_delay)

        if self.jitter:
            # Add up to ±25% jitter
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)


def retry_sync(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for synchronous functions with exponential backoff retry.

    Args:
        max_attempts: Maximum number of attempts (includes initial call)
        exceptions: Tuple of exception types to catch and retry
        base_delay: Base delay in seconds
        multiplier: Exponential growth factor
        max_delay: Maximum delay cap in seconds
        jitter: Add random jitter to delays
        on_retry: Optional callback called on each retry (exception, attempt)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_sync(max_attempts=3, exceptions=(ValueError,))
        ... def fetch_data():
        ...     return api.get("/data")
    """
    backoff = ExponentialBackoff(base_delay, multiplier, max_delay, jitter)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Log the retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for " f"{func.__name__}: {e}"
                    )

                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Don't sleep after last attempt
                    if attempt < max_attempts - 1:
                        delay = backoff.calculate(attempt)
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)

            # All attempts exhausted
            error_msg = (
                f"{func.__name__} failed after {max_attempts} attempts. "
                f"Last error: {last_exception}"
            )
            logger.error(error_msg)
            raise RetryError(error_msg, last_exception)  # type: ignore

        return wrapper

    return decorator


def retry_async(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for async functions with exponential backoff retry.

    Args:
        max_attempts: Maximum number of attempts (includes initial call)
        exceptions: Tuple of exception types to catch and retry
        base_delay: Base delay in seconds
        multiplier: Exponential growth factor
        max_delay: Maximum delay cap in seconds
        jitter: Add random jitter to delays
        on_retry: Optional callback called on each retry (exception, attempt)

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @retry_async(max_attempts=3, exceptions=(aiohttp.ClientError,))
        ... async def fetch_data():
        ...     async with session.get("/data") as resp:
        ...         return await resp.json()
    """
    backoff = ExponentialBackoff(base_delay, multiplier, max_delay, jitter)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Log the retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for " f"{func.__name__}: {e}"
                    )

                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)

                    # Don't sleep after last attempt
                    if attempt < max_attempts - 1:
                        delay = backoff.calculate(attempt)
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)

            # All attempts exhausted
            error_msg = (
                f"{func.__name__} failed after {max_attempts} attempts. "
                f"Last error: {last_exception}"
            )
            logger.error(error_msg)
            raise RetryError(error_msg, last_exception)  # type: ignore

        return wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic without decorators.

    Useful when you need retry logic for a code block rather than
    a function, or when you need more control over the retry flow.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exception types to catch and retry
        base_delay: Base delay in seconds
        multiplier: Exponential growth factor
        max_delay: Maximum delay cap in seconds
        jitter: Add random jitter to delays

    Example:
        >>> retry_ctx = RetryContext(max_attempts=3)
        >>> for attempt in retry_ctx:
        ...     with attempt:
        ...         result = risky_operation()
        ...         break  # Success, exit retry loop
    """

    def __init__(
        self,
        max_attempts: int = 3,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.exceptions = exceptions
        self.backoff = ExponentialBackoff(base_delay, multiplier, max_delay, jitter)
        self.current_attempt = 0
        self.last_exception: Optional[Exception] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_attempt >= self.max_attempts:
            if self.last_exception:
                error_msg = (
                    f"RetryContext failed after {self.max_attempts} attempts. "
                    f"Last error: {self.last_exception}"
                )
                raise RetryError(error_msg, self.last_exception)
            raise StopIteration

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            return True

        if exc_type in self.exceptions:
            # Caught expected exception
            self.last_exception = exc_val
            logger.warning(
                f"Attempt {self.current_attempt + 1}/{self.max_attempts} failed: " f"{exc_val}"
            )

            # Calculate delay for next attempt
            if self.current_attempt < self.max_attempts - 1:
                delay = self.backoff.calculate(self.current_attempt)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

            self.current_attempt += 1
            return True  # Suppress exception

        # Unexpected exception, re-raise
        return False


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example 1: Sync retry with decorator
    @retry_sync(max_attempts=3, exceptions=(ValueError,), base_delay=0.5, multiplier=2.0)
    def unstable_function(success_on_attempt: int = 3):
        """Function that fails a few times before succeeding."""
        if not hasattr(unstable_function, "attempt_count"):
            unstable_function.attempt_count = 0

        unstable_function.attempt_count += 1
        logger.info(f"Attempt {unstable_function.attempt_count}")

        if unstable_function.attempt_count < success_on_attempt:
            raise ValueError(f"Failed on attempt {unstable_function.attempt_count}")

        return "Success!"

    # Example 2: Async retry with decorator
    @retry_async(
        max_attempts=3,
        exceptions=(ConnectionError,),
        base_delay=1.0,
        multiplier=2.0,
    )
    async def unstable_async_function():
        """Async function that might fail."""
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Network error")
        return "Async success!"

    # Example 3: RetryContext usage
    def example_retry_context():
        """Example using RetryContext for manual retry control."""
        retry_ctx = RetryContext(max_attempts=3, exceptions=(ValueError,), base_delay=0.5)

        attempt_count = 0
        for attempt in retry_ctx:
            with attempt:
                attempt_count += 1
                logger.info(f"Manual retry attempt {attempt_count}")

                if attempt_count < 2:
                    raise ValueError("Not yet!")

                logger.info("Success in retry context!")
                break

    # Run examples
    print("=== Example 1: Sync Retry Decorator ===")
    try:
        result = unstable_function(success_on_attempt=2)
        print(f"Result: {result}")
    except RetryError as e:
        print(f"Failed: {e}")

    print("\n=== Example 2: Async Retry Decorator ===")

    async def run_async_example():
        try:
            result = await unstable_async_function()
            print(f"Result: {result}")
        except RetryError as e:
            print(f"Failed: {e}")

    asyncio.run(run_async_example())

    print("\n=== Example 3: RetryContext ===")
    try:
        example_retry_context()
    except RetryError as e:
        print(f"Failed: {e}")

    print("\n=== Example 4: Exponential Backoff Calculation ===")
    backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=30.0, jitter=False)
    for i in range(6):
        delay = backoff.calculate(i)
        print(f"Attempt {i}: delay = {delay:.2f}s")
