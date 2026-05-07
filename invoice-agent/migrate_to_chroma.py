"""
One-time migration: load all existing SQLite extractions into ChromaDB.

Run once after upgrading to the vector store backend:
  python migrate_to_chroma.py

Safe to re-run — ChromaDB upserts won't create duplicates.
"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import configure_logging, DB_PATH
from src.memory.vector_store import VectorStore

configure_logging()

conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute(
    "SELECT id, pdf_filename, extracted_json FROM extractions ORDER BY id"
).fetchall()
conn.close()

if not rows:
    print("No extractions in DB yet — nothing to migrate.")
    sys.exit(0)

store = VectorStore()
print(f"Migrating {len(rows)} extractions into ChromaDB...")

for eid, filename, raw_json in rows:
    extracted = json.loads(raw_json)
    # Use the stored JSON as a text proxy — the original PDF text isn't in SQLite.
    # This is less ideal than the real PDF text but still provides useful signal
    # for invoice number / vendor name similarity matching.
    text_proxy = json.dumps(extracted, ensure_ascii=False)
    store.add_extraction(text_proxy, extracted, eid)
    print(f"  [{eid:2d}] {filename}")

print(f"\nDone. ChromaDB now has {store.get_stats()['vector_extractions']} extractions.")
