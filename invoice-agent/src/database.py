"""
SQLite persistence layer.

Two tables:
  extractions  — one row per PDF processed
  corrections  — user-supplied fixes linked to an extraction

SQLite is chosen over Postgres/MySQL because:
  - Zero-config setup (no server, no migrations tool needed)
  - The data volume is small (hundreds → low thousands of invoices)
  - The DB ships as a single file, easy to back up or hand off
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

_CREATE_EXTRACTIONS = """
CREATE TABLE IF NOT EXISTS extractions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_filename  TEXT    NOT NULL,
    extracted_json TEXT   NOT NULL,
    confidence    REAL    NOT NULL DEFAULT 0.0,
    used_ocr      INTEGER NOT NULL DEFAULT 0,  -- 0 = False, 1 = True
    model         TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at    TEXT    NOT NULL
)
"""

_CREATE_CORRECTIONS = """
CREATE TABLE IF NOT EXISTS corrections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_id  INTEGER NOT NULL REFERENCES extractions(id),
    corrected_json TEXT    NOT NULL,
    user_notes     TEXT,
    created_at     TEXT    NOT NULL
)
"""


def _get_db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on clean exit."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.execute(_CREATE_EXTRACTIONS)
        conn.execute(_CREATE_CORRECTIONS)
    logger.info("Database initialised at %s", DB_PATH)


def save_extraction(
    pdf_filename: str,
    extracted_json: dict,
    confidence: float,
    used_ocr: bool,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> int:
    """
    Persist one extraction result.

    Returns:
        The new row's primary key (extraction_id).
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO extractions
                (pdf_filename, extracted_json, confidence, used_ocr,
                 model, prompt_tokens, completion_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pdf_filename,
                json.dumps(extracted_json),
                confidence,
                int(used_ocr),
                model,
                prompt_tokens,
                completion_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row_id: int = cursor.lastrowid  # type: ignore[assignment]

    logger.info("Saved extraction id=%d for %s", row_id, pdf_filename)
    return row_id


def save_correction(
    extraction_id: int,
    corrected_json: dict,
    user_notes: str | None = None,
) -> int:
    """
    Persist a user correction linked to an existing extraction.

    Returns:
        The new correction row's primary key.
    """
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO corrections (extraction_id, corrected_json, user_notes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                extraction_id,
                json.dumps(corrected_json),
                user_notes,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row_id = cursor.lastrowid  # type: ignore[assignment]

    logger.info("Saved correction id=%d for extraction_id=%d", row_id, extraction_id)
    return row_id


def get_extraction(extraction_id: int) -> dict | None:
    """Fetch a single extraction row by id, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM extractions WHERE id = ?", (extraction_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["extracted_json"] = json.loads(result["extracted_json"])
    return result


def get_stats() -> dict:
    """
    Return aggregate statistics for the `stats` CLI command.

    Includes total extractions, total corrections, and average confidence.
    """
    with _connect() as conn:
        total_extractions = conn.execute(
            "SELECT COUNT(*) FROM extractions"
        ).fetchone()[0]
        total_corrections = conn.execute(
            "SELECT COUNT(*) FROM corrections"
        ).fetchone()[0]
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM extractions"
        ).fetchone()[0]
        ocr_count = conn.execute(
            "SELECT COUNT(*) FROM extractions WHERE used_ocr = 1"
        ).fetchone()[0]

    return {
        "total_extractions": total_extractions,
        "total_corrections": total_corrections,
        "avg_confidence": round(avg_confidence or 0.0, 3),
        "ocr_extractions": ocr_count,
        "digital_extractions": total_extractions - ocr_count,
    }
