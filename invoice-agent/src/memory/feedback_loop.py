"""
Layer 1+2 integration: Feedback Loop — correction processor with analytics.

When a user submits a correction this module:
  1. Saves to SQLite (audit trail)
  2. Adds to ChromaDB corrections collection (high-priority few-shot memory)
  3. Tracks which fields were wrong (analytics)
  4. Triggers pattern extraction if the threshold is reached (Layer 2)

The field-level analytics answer the question: "which fields does the model
get wrong most often?" This guides where to focus corrections and helps
evaluate whether the learning system is actually improving over time.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from src.database import get_extraction, save_correction
from src.memory.vector_store import VectorStore
from src.memory.pattern_library import (
    extract_patterns_from_corrections,
    should_extract_patterns,
)

logger = logging.getLogger(__name__)


def submit_correction(
    extraction_id: int,
    corrected_json: dict,
    user_notes: str | None = None,
) -> dict:
    """
    End-to-end correction handler.

    Args:
        extraction_id: SQLite id of the extraction being corrected.
        corrected_json: The user-verified correct extraction dict.
        user_notes: Optional free-text explanation of what was wrong.

    Returns:
        Summary dict with changed_fields, pattern_extraction_triggered.

    Raises:
        ValueError: If extraction_id doesn't exist in the database.
    """
    original = get_extraction(extraction_id)
    if original is None:
        raise ValueError(f"No extraction found with id={extraction_id}")

    original_json = original["extracted_json"]

    # 1. Persist to SQLite
    save_correction(
        extraction_id=extraction_id,
        corrected_json=corrected_json,
        user_notes=user_notes,
    )

    # 2. Add to ChromaDB corrections collection (high priority)
    # We use the stored JSON as a text proxy since the raw PDF text
    # isn't kept in SQLite. This still gives useful semantic signal
    # because invoice JSON contains vendor names, amounts, and structure.
    store = VectorStore()
    text_proxy = json.dumps(original_json, ensure_ascii=False)
    store.add_correction(
        pdf_text=text_proxy,
        corrected_json=corrected_json,
        extraction_id=extraction_id,
    )

    # 3. Analyse which fields changed
    changed = _diff_fields(original_json, corrected_json)
    logger.info(
        "Correction for extraction_id=%d — changed fields: %s",
        extraction_id,
        changed or "none",
    )

    # 4. Trigger pattern extraction if threshold reached
    patterns_triggered = False
    if should_extract_patterns():
        logger.info("Correction threshold reached — running pattern extraction")
        new_rules = extract_patterns_from_corrections()
        patterns_triggered = True
        logger.info("Pattern extraction added/updated %d rules", len(new_rules))

    return {
        "extraction_id": extraction_id,
        "changed_fields": changed,
        "pattern_extraction_triggered": patterns_triggered,
    }


def get_field_accuracy_report() -> dict:
    """
    Analyse all corrections to find which fields are wrong most often.

    Returns a dict sorted by error frequency — most-corrected field first.
    Useful for understanding where the model needs the most improvement
    and for deciding which corrections to prioritise submitting.
    """
    try:
        import sqlite3
        from config.settings import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            """
            SELECT c.corrected_json, e.extracted_json
            FROM corrections c
            JOIN extractions e ON c.extraction_id = e.id
            ORDER BY c.created_at DESC
            """
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.error("Could not load corrections for accuracy report: %s", exc)
        return {}

    if not rows:
        return {}

    field_errors: Counter = Counter()
    total_corrections = len(rows)

    for corrected_raw, original_raw in rows:
        corrected = json.loads(corrected_raw)
        original = json.loads(original_raw)
        changed = _diff_fields(original, corrected)
        for field in changed:
            field_errors[field] += 1

    return {
        "total_corrections": total_corrections,
        "field_error_counts": dict(field_errors.most_common()),
        "most_problematic_field": field_errors.most_common(1)[0][0] if field_errors else None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _diff_fields(original: dict, corrected: dict) -> list[str]:
    """
    Return a list of top-level field names that differ between two dicts.

    Nested dicts (vendor, customer) are compared as a whole for simplicity —
    any change inside them registers as a change to the parent key.
    """
    changed: list[str] = []
    all_keys = set(original.keys()) | set(corrected.keys())

    for key in all_keys:
        if key in ("confidence_score", "raw_extraction_metadata", "validation_warnings"):
            continue  # agent-generated fields — ignore in diffs
        if original.get(key) != corrected.get(key):
            changed.append(key)

    return sorted(changed)
