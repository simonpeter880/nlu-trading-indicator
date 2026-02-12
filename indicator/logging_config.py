"""
Centralized logging configuration for NLU Trading Indicators.

Provides consistent logging across all modules with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output
- Log rotation
- Structured logging (JSON format option)
- Environment-based configuration
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return result


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    json_format: bool = False,
    rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        name: Logger name (defaults to root logger)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = no file logging)
        console: Enable console logging
        json_format: Use JSON format for structured logging
        rotation: Enable log file rotation
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(
        ...     name="nlu_analyzer",
        ...     level="INFO",
        ...     log_file="logs/app.log"
        ... )
        >>> logger.info("Application started")
    """
    # Get logger
    logger = logging.getLogger(name)

    # Get log level from environment or parameter
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Get log file from environment or parameter
    if log_file is None:
        log_file = os.getenv("LOG_FILE")

    # Convert string level to logging constant
    numeric_level = getattr(logging, level, logging.INFO)
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Determine format
    if json_format:
        # JSON format for structured logging
        log_format = (
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "module": "%(module)s", '
            '"function": "%(funcName)s", "line": %(lineno)d, '
            '"message": "%(message)s"}'
        )
        date_format = "%Y-%m-%dT%H:%M:%S"
    else:
        # Human-readable format
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if json_format:
            console_formatter = logging.Formatter(log_format, date_format)
        else:
            console_formatter = ColoredFormatter(log_format, date_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation:
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            # Regular file handler
            file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Don't propagate to parent loggers
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    # Check if logging has been configured
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        # Setup default logging if not configured
        setup_logging()

    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    message: str = "An exception occurred",
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        exc: Exception to log
        message: Context message
        level: Log level (default: ERROR)

    Example:
        >>> try:
        ...     risky_operation()
        ... except ValueError as e:
        ...     log_exception(logger, e, "Failed to process data")
    """
    logger.log(level, f"{message}: {exc}", exc_info=True)


def configure_default_logging() -> None:
    """
    Configure default logging for the entire application.

    Reads configuration from environment variables:
    - LOG_LEVEL: Logging level (default: INFO)
    - LOG_FILE: Log file path (default: logs/nlu_trading.log)
    - LOG_JSON: Use JSON format (default: false)
    - LOG_CONSOLE: Enable console output (default: true)
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/nlu_trading.log")
    json_format = os.getenv("LOG_JSON", "false").lower() == "true"
    console = os.getenv("LOG_CONSOLE", "true").lower() == "true"

    # Configure root logger
    setup_logging(
        level=level,
        log_file=log_file,
        console=console,
        json_format=json_format,
    )

    # Log startup message
    logger = logging.getLogger("nlu_trading")
    logger.info("=" * 60)
    logger.info("NLU Trading Indicators - Logging Initialized")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"JSON Format: {json_format}")
    logger.info(f"Console Output: {console}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(
        name="test_logger",
        level="DEBUG",
        log_file="logs/test.log",
        console=True,
    )

    # Test all log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test exception logging
    try:
        1 / 0
    except ZeroDivisionError as e:
        log_exception(logger, e, "Math error occurred")

    print("\nLog file created at: logs/test.log")
