import sys, json
sys.path.insert(0, '.')
from config.settings import configure_logging
from src.database import init_db
from src.agent import InvoiceAgent

configure_logging()
init_db()

agent = InvoiceAgent()
result = agent.run('data/sample_invoices/Sullivan.pdf')
inv = result.invoice
meta = inv.raw_extraction_metadata

print()
print("EXTRACTION RESULT")
print("=" * 60)
print(f"invoice_number  : {inv.invoice_number}")
print(f"invoice_date    : {inv.invoice_date}")
print(f"due_date        : {inv.due_date}")
print(f"vendor          : {inv.vendor.name}")
print(f"grand_total     : {inv.grand_total}")
print(f"confidence      : {inv.confidence_score}")
print(f"few_shot_sources: {result.few_shot_sources}")
print(f"rules_injected  : {meta.get('rules_injected', 0)}")
print(f"latency_ms      : {meta.get('latency_ms')}")
print()

print("LINE ITEMS:")
for li in inv.line_items:
    desc = (li.description or "")[:55]
    print(f"  {desc:<55} | qty={li.quantity} | total={li.total}")

if inv.validation_warnings:
    print()
    print("WARNINGS:")
    for w in inv.validation_warnings:
        print(f"  [{w.field}] {w.message}")
