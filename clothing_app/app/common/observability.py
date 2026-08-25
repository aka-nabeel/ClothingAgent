"""Structured rotating logs for clothing-application requests and failures."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.config import AppConfig

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "clothing_app_request_id", default="-"
)
_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonLogFormatter(logging.Formatter):
    """Serialize operational events as one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "clothing-app",
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", _REQUEST_ID.get()),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(config: AppConfig) -> None:
    """Configure console and rotating JSON file logs for the clothing app."""

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level.upper())
    for handler in list(root_logger.handlers):
        if getattr(handler, "_clothing_app_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(levelname)s | clothing-app | %(name)s | %(message)s"
        )
    )
    console._clothing_app_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "clothing_app.log",
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonLogFormatter())
    file_handler._clothing_app_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)


def set_request_id(request_id: str) -> contextvars.Token[str]:
    """Attach a request ID to logs emitted during one HTTP request."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """Restore the previous request ID after request processing."""

    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    """Return the current request ID."""

    return _REQUEST_ID.get()
