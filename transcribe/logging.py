from enum import IntEnum
import logging
from rich.console import Console
from rich.logging import RichHandler as BaseRichHandler
from transcribe.device import DeviceManager


class LogLevel(IntEnum):
    ERROR = 1
    WARN = 2
    INFO = 3  # Errors and info (-v)
    DEBUG = 4  # All messages (-vv)


class RichHandler(BaseRichHandler):
    """Custom Rich handler that includes device information"""

    def __init__(self, console: Console, *args, **kwargs):
        super().__init__(console=console, *args, **kwargs)
        self.device = DeviceManager.get_default_device()

    def emit(self, record):
        """Emit a log record with device information"""
        record.device = self.device
        super().emit(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance by name"""
    return logging.getLogger(name)


def initialize_logging(
    console: Console, level: LogLevel = LogLevel.ERROR
) -> logging.Logger:
    """Initialize logging with Rich handler and return root logger"""
    # Convert LogLevel to standard logging levels
    log_levels = {
        LogLevel.ERROR: logging.ERROR,
        LogLevel.WARN: logging.WARNING,
        LogLevel.INFO: logging.INFO,
        LogLevel.DEBUG: logging.DEBUG,
    }

    # Configure root logger
    root_logger = logging.getLogger("transcribe")
    root_logger.setLevel(log_levels[level])

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add Rich handler
    rich_handler = RichHandler(
        console=console,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        level=log_levels[level],
    )
    root_logger.addHandler(rich_handler)

    # Disable propagation for third-party loggers
    logging.getLogger("speechbrain").propagate = False
    logging.getLogger("whisper").propagate = False
    logging.getLogger("ffmpeg").propagate = False

    root_logger.debug("Logging initialized")
    return root_logger
