"""Logging setup.

Print statements are not an observability strategy. They cannot be filtered by
severity, cannot be redirected to a file, carry no timestamp, and vanish when
the process runs as a service -- which is exactly what a gate system does.

Every module gets its logger with ``get_logger(__name__)``. The hierarchy that
produces (``alpr.detection.plate``, ``alpr.ocr.paddle``) means you can raise the
level on one noisy component without drowning in the rest.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from alpr.config.schema import LoggingConfig

_PLAIN_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, for ingestion by a log aggregator."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: LoggingConfig) -> None:
    """Install handlers on the root logger. Safe to call more than once.

    Logs go to stderr rather than stdout so that piping a script's real output
    somewhere does not mix diagnostics into the data.
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(config.level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = (
        JsonFormatter()
        if config.format == "json"
        else logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT)
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if config.file is not None:
        path = Path(config.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name``, conventionally the module's ``__name__``."""
    return logging.getLogger(name)


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run. Used by tests."""
    return _configured
