"""
Demo: simplify_invoice on Purina_corrected.json

Shows item number extraction from embedded description strings
and the cleaned description output with unit prices.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.schema import Invoice, simplify_invoice

data = json.loads(Path("corrections/Purina_corrected.json").read_text(encoding="utf-8"))
invoice = Invoice.model_validate(data)
simple = simplify_invoice(invoice)

print(f"\nInvoice : {simple.invoice_number}")
print(f"Vendor  : {simple.vendor_name}")
print(f"Customer: {simple.customer_name}")
print(f"Total   : ${simple.grand_total:,.2f} {invoice.currency}")
print(f"\n{'#':<6} {'Item #':<25} {'Qty':>5} {'Unit Price':>11}  Description")
print("-" * 90)
for i, item in enumerate(simple.line_items, 1):
    item_num = item.item_number or "(none)"
    price_str = f"${item.unit_price:,.2f}" if item.unit_price is not None else "    —"
    desc = item.description[:45]
    print(f"{i:<6} {item_num:<25} {item.quantity:>5.0f} {price_str:>11}  {desc}")

print(f"\n{len(simple.line_items)} line items shown  (items with null qty are skipped)")
