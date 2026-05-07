"""
Debug extraction: shows every layer of the learning system firing before
the LLM call, then runs the full extraction and prints the result.

Usage:
    python debug_extraction.py <pdf_path>
    python debug_extraction.py data/sample_invoices/Sullivan.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import configure_logging
from src.pdf_extractor import extract_text
from src.memory.vector_store import VectorStore
from src.memory.pattern_library import get_relevant_rules, load_rules
from src.prompts import build_user_prompt

configure_logging()

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_invoices/Sullivan.pdf"

DIVIDER = "=" * 70

# ── Step 1: Extract PDF text ────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print(f"  PDF: {pdf_path}")
print(DIVIDER)

extraction = extract_text(pdf_path)
print(f"  OCR used   : {extraction.used_ocr}")
print(f"  Page count : {extraction.page_count}")
print(f"  Text length: {len(extraction.text)} chars")
print(f"\n  --- First 500 chars of extracted text ---")
print(extraction.text[:500])

# ── Step 2: Layer 2 — Pattern Library ──────────────────────────────────────
print(f"\n{DIVIDER}")
print("  LAYER 2: Pattern Library — rule matching")
print(DIVIDER)

all_rules = load_rules()
print(f"  Total rules in library: {len(all_rules)}")

matched_rules = get_relevant_rules(extraction.text)
print(f"  Rules matched for this PDF: {len(matched_rules)}")

if matched_rules:
    for i, rule in enumerate(matched_rules, 1):
        print(f"\n  [{i}] {rule}")
else:
    print("  (no rules matched — either library is empty or no triggers found in text)")

# ── Step 3: Layer 1 — ChromaDB few-shot retrieval ──────────────────────────
print(f"\n{DIVIDER}")
print("  LAYER 1: ChromaDB — semantic similar example retrieval")
print(DIVIDER)

store = VectorStore()
similar = store.find_similar(extraction.text, k=3)
print(f"  Examples retrieved: {len(similar)}")

for i, ex in enumerate(similar, 1):
    source = ex.get("source", "unknown")
    eid    = ex.get("extraction_id", "?")
    inv_no = ex.get("output_json", {}).get("invoice_number", "?")
    vendor = (ex.get("output_json", {}).get("vendor") or {}).get("name", "?")
    tag    = "[CORRECTION - gold standard]" if source == "correction" else "[past extraction]"
    print(f"\n  [{i}] {tag}")
    print(f"       extraction_id : {eid}")
    print(f"       invoice_number: {inv_no}")
    print(f"       vendor        : {vendor}")

# ── Step 4: Build actual prompt ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  PROMPT: What gets sent to Groq (first 2500 chars of user turn)")
print(DIVIDER)

prompt = build_user_prompt(extraction.text, similar, matched_rules)
print(prompt[:2500])
if len(prompt) > 2500:
    print(f"\n  ... [{len(prompt) - 2500} more chars] ...")

print(f"\n  Total prompt length: {len(prompt)} chars")

# ── Step 5: Full extraction ─────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  GROQ: Running full extraction...")
print(DIVIDER)

confirm = input("\n  Send to Groq API now? [y/N]: ").strip().lower()
if confirm != "y":
    print("  Skipped. Re-run and type 'y' to call the API.")
    sys.exit(0)

from src.database import init_db
from src.agent import InvoiceAgent

init_db()
agent = InvoiceAgent()
result = agent.run(pdf_path)

inv = result.invoice
print(f"\n  invoice_number  : {inv.invoice_number}")
print(f"  invoice_date    : {inv.invoice_date}")
print(f"  vendor          : {(inv.vendor.name if inv.vendor else None)}")
print(f"  grand_total     : {inv.grand_total}")
print(f"  confidence      : {inv.confidence_score}")
print(f"  few_shot_sources: {result.few_shot_sources}")
print(f"  rules_injected  : {inv.raw_extraction_metadata.get('rules_injected', 0)}")

if inv.validation_warnings:
    print(f"\n  Validation warnings:")
    for w in inv.validation_warnings:
        print(f"    [{w.field}] {w.message}")

print(f"\n  Full JSON:")
print(json.dumps(inv.model_dump(mode="json"), indent=2, default=str))
