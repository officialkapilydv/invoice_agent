"""
Pydantic v2 models for invoice data.

Design choice: all fields except invoice_number are Optional with None defaults.
This is intentional — invoices in the wild are inconsistent, and forcing required
fields causes the whole extraction to fail when one field is absent. Instead we
capture what we can and surface missing fields via the validator module.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class VendorInfo(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None        # GSTIN / VAT / EIN etc.
    email: str | None = None
    phone: str | None = None


class CustomerInfo(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None
    email: str | None = None


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    tax_rate: float | None = None    # percentage, e.g. 18.0 for 18%
    total: float | None = None


class ValidationWarning(BaseModel):
    field: str
    message: str


class Invoice(BaseModel):
    # Core identifiers
    invoice_number: str = Field(..., description="Invoice / bill number — required")
    invoice_date: date | None = None
    due_date: date | None = None

    # Parties
    vendor: VendorInfo = Field(default_factory=VendorInfo)
    customer: CustomerInfo = Field(default_factory=CustomerInfo)

    # Line items
    line_items: list[LineItem] = Field(default_factory=list)

    # Financials
    subtotal: float | None = None
    tax_total: float | None = None
    discount: float | None = Field(default=None, description="Discount amount (not %)")
    grand_total: float | None = None
    currency: str = Field(default="INR")

    # Metadata
    payment_terms: str | None = None
    notes: str | None = None

    # Agent-generated quality signals
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the extraction (0–1)",
    )
    validation_warnings: list[ValidationWarning] = Field(default_factory=list)
    raw_extraction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Debug info: OCR flag, token usage, model used, etc.",
    )

    @field_validator("invoice_number", mode="before")
    @classmethod
    def coerce_invoice_number(cls, v: Any) -> str:
        """Accept int invoice numbers (common in LLM output) by converting to str."""
        if v is None:
            raise ValueError("invoice_number is required")
        return str(v).strip()

    @field_validator("currency", mode="before")
    @classmethod
    def upper_currency(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip().upper()
        return "INR"

    @model_validator(mode="before")
    @classmethod
    def coerce_floats(cls, data: Any) -> Any:
        """Convert string numbers like '1,234.50' to floats before validation."""
        if not isinstance(data, dict):
            return data
        float_fields = {"subtotal", "tax_total", "discount", "grand_total"}
        for field in float_fields:
            val = data.get(field)
            if isinstance(val, str):
                try:
                    data[field] = float(val.replace(",", "").replace("₹", "").strip())
                except ValueError:
                    data[field] = None
        return data


# ---------------------------------------------------------------------------
# Simplified output schema
# ---------------------------------------------------------------------------

# Regex patterns to detect embedded item numbers in description strings.
# Priority order matters — more specific patterns first.
# e.g. "DEER PELLET 20 (Item: 9794, Pack: 50#)" → item_number="9794"
_ITEM_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bItem(?:\s+(?:No|Number|#|ID))?[:\s]+([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\bSKU[:\s]+([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\bPart(?:\s+(?:No|Number|#))?[:\s]+([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\bProduct(?:\s+(?:No|Number|#|Code))?[:\s]+([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\bP/?N[:\s]+([A-Z0-9\-]+)", re.IGNORECASE),
    re.compile(r"\bUPC[:\s]+([0-9]+)", re.IGNORECASE),
    re.compile(r"#([A-Z0-9]{4,})\b"),   # bare hash prefix e.g. "#90210"
]

# Characters/tokens that indicate extra metadata appended after the real
# description (parenthetical blocks, serial numbers, etc.)
_DESCRIPTION_NOISE_PATTERN = re.compile(
    r"\s*[\(\[].*?[\)\]]"       # anything inside ( ) or [ ]
    r"|\s*,\s*(?:Pack|Unit|Serial|Tracking|Lot)[:\s].*$",
    re.IGNORECASE | re.DOTALL,
)


class SimplifiedLineItem(BaseModel):
    """
    Minimal line item for downstream systems that don't need full detail.

    item_number is extracted from the description string when the LLM has
    embedded it in parentheses (a common pattern in corrected extractions).
    """
    description: str
    item_number: str | None = None
    quantity: float
    unit_price: float | None = None


class SimplifiedInvoice(BaseModel):
    """
    Flat, minimal invoice view — useful for spreadsheet export, quick
    review, or feeding into downstream inventory/accounting systems that
    don't need vendor addresses, tax IDs, or agent metadata.

    Always derived from a full Invoice so nothing is re-extracted —
    the full pipeline (validator, ChromaDB, pattern library) ran first.
    """
    invoice_number: str
    invoice_date: date | None = None
    vendor_name: str | None = None
    customer_name: str | None = None
    line_items: list[SimplifiedLineItem]
    grand_total: float | None = None


def simplify_invoice(invoice: Invoice) -> SimplifiedInvoice:
    """
    Convert a fully-extracted Invoice into a SimplifiedInvoice.

    The full Invoice is untouched — this is a read-only projection.
    All learning-system side-effects (ChromaDB writes, rule matching)
    have already happened before this function is called.

    Line item logic:
      - item_number is parsed from the description string using a set of
        regex patterns covering Item:, SKU:, Part No:, Product No:, etc.
      - description is cleaned of the extracted item number and any
        parenthetical noise blocks to keep it human-readable.
      - quantity falls through from invoice.line_items[i].quantity.
        The system prompt already instructs the LLM to prefer SHIPPED
        quantity when both ordered and shipped columns exist, so no
        additional logic is needed here.

    Args:
        invoice: A fully validated Invoice object.

    Returns:
        SimplifiedInvoice with only the fields defined above.
    """
    simple_items: list[SimplifiedLineItem] = []

    for item in invoice.line_items:
        raw_desc = item.description or ""
        item_number: str | None = None

        # Try each pattern in priority order, stop at first match
        for pattern in _ITEM_NUMBER_PATTERNS:
            m = pattern.search(raw_desc)
            if m:
                item_number = m.group(1).strip()
                break

        # Clean description: remove parenthetical blocks and trailing noise
        clean_desc = _DESCRIPTION_NOISE_PATTERN.sub("", raw_desc).strip()
        # If cleaning left nothing useful, fall back to the raw description
        if len(clean_desc) < 3:
            clean_desc = raw_desc.strip()

        qty = item.quantity
        if qty is None:
            # Skip items with no quantity — they're usually header/subtotal rows
            # the LLM accidentally captured as line items
            continue

        simple_items.append(
            SimplifiedLineItem(
                description=clean_desc,
                item_number=item_number,
                quantity=qty,
                unit_price=item.unit_price,
            )
        )

    return SimplifiedInvoice(
        invoice_number=invoice.invoice_number,
        invoice_date=invoice.invoice_date,
        vendor_name=invoice.vendor.name if invoice.vendor else None,
        customer_name=invoice.customer.name if invoice.customer else None,
        line_items=simple_items,
        grand_total=invoice.grand_total,
    )
