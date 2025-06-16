"""Utility functions for the trading advisor."""

import logging
import os
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level to use (default: INFO)
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/trading_advisor.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    # Terminal handler (RichHandler for pretty output)
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.ERROR)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    ) 