"""
Central configuration — loads .env and exposes typed settings.

All modules import from here rather than reading os.environ directly,
so there is one place to change defaults or add validation.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root (two levels up from this file: config/ -> invoice-agent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Groq / LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
# llama-3.3-70b-versatile is the better choice for invoice extraction:
#   - Higher accuracy on structured extraction tasks
#   - Better instruction following for complex JSON schemas
#   - Worth the slight speed/cost increase over 8b for production use
# Switch to llama-3.1-8b-instant only when prototyping or under tight latency budgets.
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Tesseract / Poppler (Windows paths)
# ---------------------------------------------------------------------------
TESSERACT_PATH: str = os.environ.get(
    "TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
POPPLER_PATH: str | None = os.environ.get("POPPLER_PATH", None)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH: Path = PROJECT_ROOT / os.environ.get("DB_PATH", "data/extractions.db")
FEW_SHOT_PATH: Path = PROJECT_ROOT / os.environ.get(
    "FEW_SHOT_PATH", "data/few_shot_examples.json"
)
FEW_SHOT_TOP_K: int = int(os.environ.get("FEW_SHOT_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


def configure_logging() -> logging.Logger:
    """Set up root logger with a consistent format and return the root logger."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("invoice_agent")


# ---------------------------------------------------------------------------
# Startup validation — warn early if critical config is missing
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """Raise EnvironmentError if required settings are absent."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
