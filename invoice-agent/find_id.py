"""Quick helper to find extraction IDs by filename."""
import sqlite3
import sys

search = sys.argv[1] if len(sys.argv) > 1 else ""

conn = sqlite3.connect("data/extractions.db")
rows = conn.execute(
    "SELECT id, pdf_filename FROM extractions WHERE pdf_filename LIKE ?",
    (f"%{search}%",),
).fetchall()
conn.close()

if not rows:
    print(f"No extractions found matching '{search}'")
else:
    print(f"\nFound {len(rows)} match(es):")
    for r in rows:
        print(f"  ID: {r[0]:3d}  →  {r[1]}")