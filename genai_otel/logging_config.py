"""Centralized logging configuration"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    level: Optional[str] = None, log_dir: str = "logs", log_file_name: str = "genai_otel.log"
):
    """Configure logging for the library with configurable log level via environment variable
    and log rotation.
    """
    # Determine log level from environment variable or default to INFO
    env_log_level = os.environ.get("GENAI_OTEL_LOG_LEVEL")
    log_level_str = level or env_log_level or "INFO"
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add rotating file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=10)
    handlers.append(file_handler)

    # Set library logger
    logger = logging.getLogger("genai_otel")
    logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicates in case of multiple calls
    if logger.handlers:
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

    for handler in handlers:
        logger.addHandler(handler)

    return logger
