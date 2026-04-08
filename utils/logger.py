from __future__ import annotations

import sys
import os
from loguru import logger


_configured = False


def setup_logging(log_level: str = "INFO", log_file: str = "./logs/agent.log") -> None:
    global _configured
    if _configured:
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]: <12}</cyan> | "
            "<white>{message}</white>"
        ),
        colorize=True,
    )

    logger.add(
        log_file,
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{extra[agent]: <12} | {message}"
        ),
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )

    _configured = True


def get_logger(agent_name: str = "system"):
    """Return a logger bound to an agent context."""
    return logger.bind(agent=agent_name)
