"""
Centralized prompt templates.

Builds the final LLM prompt from three dynamic sections:
  1. Few-shot examples from ChromaDB (Layer 1 memory)
  2. Learned rules from PatternLibrary (Layer 2 — added later)
  3. The current invoice text
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are an expert invoice data extraction system. Your job is to parse raw invoice \
text and return a single, valid JSON object — nothing else.

## Output Schema
Return exactly this structure (use null for any field you cannot find):

{
  "invoice_number": "<string, REQUIRED>",
  "invoice_date": "<YYYY-MM-DD or null>",
  "due_date": "<YYYY-MM-DD or null>",
  "vendor": {
    "name": "<string or null>",
    "address": "<string or null>",
    "tax_id": "<GSTIN/VAT/EIN or null>",
    "email": "<string or null>",
    "phone": "<string or null>"
  },
  "customer": {
    "name": "<string or null>",
    "address": "<string or null>",
    "tax_id": "<string or null>",
    "email": "<string or null>"
  },
  "line_items": [
    {
      "description": "<string>",
      "quantity": <number or null>,
      "unit_price": <number or null>,
      "tax_rate": <percentage as number e.g. 18.0, or null>,
      "total": <number or null>
    }
  ],
  "subtotal": <number or null>,
  "tax_total": <number or null>,
  "discount": <number or null>,
  "grand_total": <number or null>,
  "currency": "<ISO 4217 code e.g. INR, USD — default INR>",
  "payment_terms": "<string or null>",
  "notes": "<string or null>",
  "confidence_score": <float 0.0-1.0>
}

## Rules
1. NEVER invent data. If a field is not present, use null.
2. Dates must be in YYYY-MM-DD format only.
3. All monetary values must be plain numbers (no currency symbols or commas).
4. confidence_score: 0.9+ means all key fields found; 0.7-0.9 means minor gaps; \
   below 0.7 means significant data missing or OCR quality is poor.
5. Return ONLY the JSON object — no markdown, no explanation, no code fences.
6. CRITICAL — SHIPPED QUANTITY: When an invoice has multiple quantity columns \
   you MUST use the SHIPPED (delivered) value. NEVER use Ordered/Requested. \
   SHIPPED aliases (always use these): "Shipped", "Shipped Qty", "Qty Shipped", \
   "Quantity Shipped", "Delivered", "Delivered Qty", "Actual", "Actual Qty", \
   "Shp", "Shpd". \
   ORDERED aliases (NEVER use): "Ordered", "Order Qty", "Qty Ordered", \
   "Quantity Ordered", "Requested", "Request Qty", "Ord". \
   EXAMPLES: Header "Ordered | Shipped | B/O", Row "40.00 | 50.00 | 0.00" → \
   use 50 (the shipped column). Header "Qty Ord | Qty Shp", Row "100 | 100" → \
   use 100 (shipped). Single "Quantity" column → use that value. \
   Shipped=0 with Ordered>0 (back-order) → use 0 (nothing was delivered). \
   THIS RULE IS ABSOLUTE — even if shipped > ordered, always use shipped.
7. ITEM NUMBERS — when a line item description contains an item/SKU/part number, \
   include it in the description field in this format: \
   "<product name> (Item: <number>)" so it can be parsed later. \
   Example: "DEER PELLET 20 (Item: 9794)"
8. ITEM IDENTIFIER COLUMNS — the product code / item number field appears under \
   MANY different column labels depending on the vendor. Treat ALL of the following \
   as the same logical field and always extract the alphanumeric code you find there: \
   "Item Number", "Item #", "Item", "Item ID", "Item Code", \
   "Product Number", "Product No.", "Product Code", "Product ID", "Product #", \
   "Material", "Material Number", \
   "SKU", "SKU Number", \
   "Part Number", "Part No.", "Part #", \
   "Catalog Number", "Catalog #", "Cat. No.", \
   "Reference", "Ref #", "Ref. No.". \
   Codes may be purely numeric (2005, 7501), alphanumeric (38716-50, B431XS), \
   or letter-based (DW11T, NFPCHG). \
   If no explicit code column exists for a line item, leave item_number as null \
   — do NOT invent a code.
9. MULTIPLE ITEM CODES PER LINE — some invoices print a secondary code (lot \
   number, batch number, or internal reference) on the row immediately below the \
   primary item code. When you detect this two-row pattern, capture BOTH codes \
   and combine them in the description with a " / " separator: \
   "<product name> (Item: PRIMARY / SECONDARY)". \
   Example: primary code "M3625" on row 1, secondary "32704625" on row 2 → \
   description "FEEDING LIMESTONE (Item: M3625 / 32704625)". \
   The simplify layer will later split this into item_number="M3625 / 32704625". \
   If only ONE code exists per item, use the normal single-code format.
"""


def build_user_prompt(
    invoice_text: str,
    few_shot_examples: list[dict],
    rules: list[str] | None = None,
) -> str:
    """
    Construct the user-turn message with optional few-shot examples and rules.

    Section order matters for LLM attention:
      1. Rules first — short, high-signal, the model reads them before any examples
      2. Few-shot examples — show correct extractions for similar invoices
      3. The invoice to extract — always last so it's closest to the output

    Args:
        invoice_text: Raw text extracted from the PDF.
        few_shot_examples: Up to k examples from ChromaDB (correction-prioritised).
          Each example is a dict with keys: 'input_text', 'output_json', 'source'.
        rules: Learned rules from PatternLibrary (e.g. "When vendor is X, ...").

    Returns:
        Formatted user prompt string.
    """
    parts: list[str] = []

    # Section 1 — Learned rules (Layer 2, injected when available)
    if rules:
        parts.append("## Learned Rules (apply these to every extraction)\n")
        for i, rule in enumerate(rules, 1):
            parts.append(f"{i}. {rule}")
        parts.append("\n")

    # Section 2 — Few-shot examples from semantic memory
    if few_shot_examples:
        parts.append("## Similar Past Extractions (learn from these)\n")
        for i, ex in enumerate(few_shot_examples, 1):
            source_tag = "[CORRECTION - human verified]" if ex.get("source") == "correction" else "[past extraction]"
            snippet = ex.get("input_text", "")[:400]
            output = json.dumps(ex.get("output_json", {}), indent=2)
            parts.append(
                f"### Example {i} {source_tag}\n"
                f"**Invoice text (excerpt):**\n{snippet}\n\n"
                f"**Correct extraction:**\n```json\n{output}\n```\n"
            )
        parts.append("---\n")

    # Section 3 — The invoice to extract
    parts.append("## Invoice to Extract\n")
    parts.append(invoice_text)
    parts.append("\n\nExtract all invoice fields and return the JSON object.")

    return "\n".join(parts)
