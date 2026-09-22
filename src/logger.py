"""
Logging Configuration
"""

import logging
from pathlib import Path

# Create logs directory

LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Configure logging

logging.basicConfig(
    filename=LOG_DIR / "project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Logger object

logger = logging.getLogger(__name__)