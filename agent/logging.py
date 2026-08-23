"""Error logging for catalyst discovery campaigns.

Provides dual-mode logging:
1. Structured logs (JSON) for analysis and MLflow integration
2. Human-readable console logs for debugging

Logs are written to:
- JSON: agent/journal/<campaign_id>.json (per-campaign)
- Console: prints to stderr with timestamp + error level
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class LogLevel(str, Enum):
    """Error log levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_journal_path(campaign_id: str) -> Path:
    """Get path for campaign journal file.

    Args:
        campaign_id: Unique identifier for this campaign

    Returns:
        Path to JSON journal file
    """
    journal_dir = Path("agent/journal")
    journal_dir.mkdir(parents=True, exist_ok=True)
    return journal_dir / f"{campaign_id}.json"


def get_log_file_path(campaign_id: str) -> Path:
    """Get path for human-readable log file.

    Args:
        campaign_id: Unique identifier for this campaign

    Returns:
        Path to log file (e.g., logs/campaign_<id>.log)
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"campaign_{timestamp}.log"


# ---------------------------------------------------------------------------
# Error log entry
# ---------------------------------------------------------------------------

class ErrorLogEntry:
    """Single error log entry with metadata."""

    def __init__(
        self,
        level: LogLevel,
        message: str,
        campaign_id: str,
        timestamp: Optional[datetime] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.level = level
        self.message = message
        self.campaign_id = campaign_id
        self.timestamp = timestamp or datetime.now()
        self.exception = exception
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for JSON serialization."""
        result = {
            "level": self.level.value,
            "message": self.message,
            "campaign_id": self.campaign_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }

        # Include exception info if present
        if self.exception:
            result["exception_type"] = type(self.exception).__name__
            result["exception_message"] = str(self.exception)
            if hasattr(self.exception, "__traceback__") and self.exception.__traceback__:
                import traceback
                result["exception_traceback"] = traceback.format_exception(
                    type(self.exception),
                    self.exception,
                    self.exception.__traceback__,
                )

        return result

    def to_json(self) -> str:
        """Convert entry to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Campaign logger
# ---------------------------------------------------------------------------

class CampaignLogger:
    """Logger for a single campaign run.

    Attributes:
        campaign_id: Unique identifier for this campaign
        journal_path: Path to JSON journal file
        log_file_path: Path to human-readable log file
        entries: List of ErrorLogEntry objects (in-memory buffer)
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self.journal_path = get_journal_path(campaign_id)
        self.log_file_path = get_log_file_path(campaign_id)
        self.entries: list[ErrorLogEntry] = []

    def log(
        self,
        level: LogLevel,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error/warning/info entry.

        Args:
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
            message: Human-readable log message
            exception: Optional exception object (for ERROR/CRITICAL)
            context: Optional dict of additional context data
        """
        entry = ErrorLogEntry(
            level=level,
            message=message,
            campaign_id=self.campaign_id,
            exception=exception,
            context=context,
        )
        self.entries.append(entry)

        # Print to console (stderr for errors, stdout for info)
        self._print_to_console(entry)

    def _print_to_console(self, entry: ErrorLogEntry) -> None:
        """Print log entry to console."""
        color_map = {
            LogLevel.INFO: "\033[92m",      # Green
            LogLevel.WARNING: "\033[93m",   # Yellow
            LogLevel.ERROR: "\033[91m",     # Red
            LogLevel.CRITICAL: "\033[95m",  # Magenta
        }
        reset = "\033[0m"

        color = color_map.get(entry.level, "")
        prefix = f"[{entry.timestamp.strftime('%H:%M:%S')}] [{entry.level.value}]"

        print(f"{color}{prefix}{reset} {entry.message}", file=sys.stderr if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL) else sys.stdout)

        # Print exception traceback if present
        if entry.exception and entry.level in (LogLevel.ERROR, LogLevel.CRITICAL):
            print(f"\n{color}Exception: {type(entry.exception).__name__}: {entry.message}{reset}", file=sys.stderr)
            import traceback
            traceback.print_exception(type(entry.exception), entry.exception, entry.exception.__traceback__, file=sys.stderr)

    def save_journal(self) -> None:
        """Save all entries to JSON journal file."""
        try:
            with open(self.journal_path, "w") as f:
                json.dump(
                    {
                        "campaign_id": self.campaign_id,
                        "entries": [entry.to_dict() for entry in self.entries],
                        "total_entries": len(self.entries),
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Failed to save journal: {e}", file=sys.stderr)

    def append_to_log_file(self) -> None:
        """Append all entries to human-readable log file."""
        try:
            with open(self.log_file_path, "a") as f:
                for entry in self.entries:
                    # Human-readable format
                    f.write(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{entry.level.value}] {entry.message}\n")

                    if entry.exception:
                        f.write(f"\nException: {type(entry.exception).__name__}: {str(entry.exception)}\n")
                        import traceback
                        traceback.print_exception(
                            type(entry.exception),
                            entry.exception,
                            entry.exception.__traceback__,
                            file=f,
                        )

                    f.write("\n" + "="*80 + "\n\n")
        except Exception as e:
            print(f"Failed to append to log file: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_logger(campaign_id: str) -> CampaignLogger:
    """Create a new campaign logger.

    Args:
        campaign_id: Unique identifier for this campaign

    Returns:
        Configured CampaignLogger instance
    """
    return CampaignLogger(campaign_id=campaign_id)


def log_error(
    campaign_id: str,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an error message.

    Args:
        campaign_id: Unique identifier for this campaign
        message: Human-readable error message
        exception: Optional exception object
        context: Optional dict of additional context
    """
    logger = create_logger(campaign_id)
    logger.log(LogLevel.ERROR, message, exception=exception, context=context)


def log_warning(
    campaign_id: str,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a warning message.

    Args:
        campaign_id: Unique identifier for this campaign
        message: Human-readable warning message
        exception: Optional exception object (usually not applicable)
        context: Optional dict of additional context
    """
    logger = create_logger(campaign_id)
    logger.log(LogLevel.WARNING, message, exception=exception, context=context)


def log_info(
    campaign_id: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an info message.

    Args:
        campaign_id: Unique identifier for this campaign
        message: Human-readable info message
        context: Optional dict of additional context
    """
    logger = create_logger(campaign_id)
    logger.log(LogLevel.INFO, message, context=context)


def log_critical(
    campaign_id: str,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
) -> CampaignLogger:
    """Log a critical error and return logger for cleanup.

    Args:
        campaign_id: Unique identifier for this campaign
        message: Human-readable critical message
        exception: Optional exception object
        context: Optional dict of additional context

    Returns:
        CampaignLogger instance (use save_journal() before exiting)
    """
    logger = create_logger(campaign_id)
    logger.log(LogLevel.CRITICAL, message, exception=exception, context=context)
    return logger


# ---------------------------------------------------------------------------
# Context manager for campaign lifecycle
# ---------------------------------------------------------------------------

class CampaignLoggingContext:
    """Context manager that logs errors and saves journal on exit.

    Usage:
        with create_logging_context("campaign-uuid") as ctx:
            try:
                # Campaign execution code
                pass
            except Exception as e:
                ctx.log_error(str(e))
        
        # Automatically saves journal on exit
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self.logger: Optional[CampaignLogger] = None

    def __enter__(self) -> CampaignLoggingContext:
        self.logger = create_logger(self.campaign_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.logger:
            # Save journal on exit
            self.logger.save_journal()
            self.logger.append_to_log_file()


def create_logging_context(campaign_id: str) -> CampaignLoggingContext:
    """Create a logging context manager.

    Args:
        campaign_id: Unique identifier for this campaign

    Returns:
        CampaignLoggingContext instance
    """
    return CampaignLoggingContext(campaign_id=campaign_id)


__all__ = [
    "LogLevel",
    "ErrorLogEntry",
    "CampaignLogger",
    "get_journal_path",
    "get_log_file_path",
    "create_logger",
    "log_error",
    "log_warning",
    "log_info",
    "log_critical",
    "create_logging_context",
]
