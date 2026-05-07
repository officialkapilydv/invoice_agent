"""
Bulk-convert all 27 existing extractions to SimplifiedInvoice format.

For each extraction:
  - If a correction exists → use corrected JSON (gold-standard)
  - Otherwise            → use original extracted JSON

No LLM calls are made. All data comes from SQLite.
Output is written to results_simple/<pdf_stem>.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
import traceback
from pathlib import Path

# Make sure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.schema import Invoice, simplify_invoice

DB_PATH = Path("data/extractions.db")
OUT_DIR = Path("results_simple")


def _stem(pdf_filename: str) -> str:
    """Turn 'Cargil.pdf' into 'Cargil'."""
    return Path(pdf_filename).stem


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Load all extractions
    extractions = conn.execute(
        "SELECT id, pdf_filename, extracted_json FROM extractions ORDER BY id"
    ).fetchall()

    # Build lookup: extraction_id → latest corrected_json
    # If multiple corrections exist for one extraction, take the most recent (MAX id)
    corrections_rows = conn.execute(
        "SELECT extraction_id, corrected_json "
        "FROM corrections "
        "WHERE id IN (SELECT MAX(id) FROM corrections GROUP BY extraction_id)"
    ).fetchall()
    corrections: dict[int, dict] = {
        row["extraction_id"]: json.loads(row["corrected_json"])
        for row in corrections_rows
    }

    conn.close()

    used_correction: list[str] = []
    used_original: list[str] = []
    failed: list[tuple[str, str]] = []

    for row in extractions:
        ext_id: int = row["id"]
        pdf_filename: str = row["pdf_filename"]
        stem = _stem(pdf_filename)
        out_path = OUT_DIR / (stem + ".json")

        source_label = "correction" if ext_id in corrections else "original"

        try:
            raw: dict = (
                corrections[ext_id]
                if ext_id in corrections
                else json.loads(row["extracted_json"])
            )
            invoice = Invoice.model_validate(raw)
            simple = simplify_invoice(invoice)
            out_path.write_text(
                json.dumps(simple.model_dump(mode="json"), indent=2, default=str),
                encoding="utf-8",
            )
            if source_label == "correction":
                used_correction.append(pdf_filename)
            else:
                used_original.append(pdf_filename)

        except Exception as exc:
            failed.append((pdf_filename, f"{type(exc).__name__}: {exc}"))
            # Print traceback for debugging without aborting the loop
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total = len(extractions)
    print(f"\n{'='*60}")
    print(f"  Bulk SimplifiedInvoice conversion complete")
    print(f"{'='*60}")
    print(f"  Total processed : {total}")
    print(f"  Used corrections: {len(used_correction)}")
    for name in used_correction:
        print(f"    + {name}")
    print(f"  Used originals  : {len(used_original)}")
    print(f"  Failed          : {len(failed)}")
    for name, reason in failed:
        print(f"    ! {name}  ->  {reason}")
    print(f"\n  Output folder   : {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
