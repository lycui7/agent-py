"""Shared retry logic with exponential backoff and structured logging."""

import logging
import time
from functools import wraps

logger = logging.getLogger("agent")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator: retry on specific exceptions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds; doubles each retry.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    ConnectionError,
                    TimeoutError,
                    OSError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            e,
                            exc_info=True,
                        )
            raise last_exception

        return wrapper

    return decorator
