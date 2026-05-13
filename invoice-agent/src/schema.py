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
# Capture group uses [^),]+ so it stops at ) or , — this naturally handles
# both single codes ("Item: 9794") and multi-code slash format
# ("Item: M3625 / 32704625") without needing separate patterns.
# e.g. "DEER PELLET 20 (Item: 9794, Pack: 50#)"   → item_number="9794"
# e.g. "LIMESTONE (Item: M3625 / 32704625, Unit:…)" → item_number="M3625 / 32704625"
_ITEM_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bItem(?:\s+(?:No|Number|#|ID|Code))?[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bSKU[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bPart(?:\s+(?:No|Number|#))?[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bProduct(?:\s+(?:No|Number|#|Code))?[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bMaterial(?:\s+(?:No|Number))?[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bP/?N[:\s]+([^),]+)", re.IGNORECASE),
    re.compile(r"\bUPC[:\s]+([0-9\s/]+)", re.IGNORECASE),
    re.compile(r"#([A-Z0-9]{4,})\b"),   # bare hash prefix e.g. "#90210"
]

# Characters/tokens that indicate extra metadata appended after the real
# description (parenthetical blocks, serial numbers, etc.)
_DESCRIPTION_NOISE_PATTERN = re.compile(
    r"\s*[\(\[].*?[\)\]]"       # anything inside ( ) or [ ]
    r"|\s*,\s*(?:Pack|Unit|Serial|Tracking|Lot)[:\s].*$",
    re.IGNORECASE | re.DOTALL,
)

# Freight / shipping / handling detection
# Exact set: description IS just this word (case-insensitive)
_FREIGHT_EXACT: frozenset[str] = frozenset(
    {"freight", "shipping", "fuel", "handling", "delivery"}
)
# Substring set: description CONTAINS one of these phrases
_FREIGHT_SUBSTRINGS: tuple[str, ...] = (
    "freight",
    "shipping",
    "fuel charge",
    "fuel surcharge",
    "handling fee",
    "delivery charge",
    "transport charge",
    "shipping & handling",
    "ship charge",
    "freight charge",
    "shipping cost",
    "freight cost",
)


def _is_freight_or_shipping(description: str | None) -> bool:
    """Return True if the line item is a freight/shipping/fuel/handling charge."""
    if not description:
        return False
    desc_lower = description.lower().strip()
    if desc_lower in _FREIGHT_EXACT:
        return True
    return any(phrase in desc_lower for phrase in _FREIGHT_SUBSTRINGS)


def _extract_item_number(raw_desc: str) -> tuple[str, str | None]:
    """
    Parse embedded item number from a description string and return
    (cleaned_description, item_number).

    Handles both single codes ("Item: 9794") and multi-code slash format
    ("Item: M3625 / 32704625"). The [^),]+ capture group stops at ) or ,,
    so trailing noise like ", Unit: 50LB" is never included. Trailing
    whitespace from the capture is stripped.

    Tries each pattern in _ITEM_NUMBER_PATTERNS in priority order.
    Parenthetical noise blocks are stripped from the description regardless
    of whether an item number was found.
    """
    item_number: str | None = None
    for pattern in _ITEM_NUMBER_PATTERNS:
        m = pattern.search(raw_desc)
        if m:
            item_number = m.group(1).strip()
            if not item_number:         # guard against whitespace-only capture
                item_number = None
            break

    clean_desc = _DESCRIPTION_NOISE_PATTERN.sub("", raw_desc).strip()
    if len(clean_desc) < 3:
        clean_desc = raw_desc.strip()

    return clean_desc, item_number


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

    Key transformations applied here (not in the agent):
      1. Freight / shipping / fuel / handling items are dropped entirely.
      2. grand_total is CALCULATED as sum(quantity × unit_price) for the
         remaining product items — the invoice's original grand_total is
         ignored. This guarantees math self-consistency and removes freight
         pollution, giving a pure product cost figure.
      3. item_number is parsed from description strings via regex.
      4. Descriptions are cleaned of parenthetical noise.
      5. Items with null quantity are skipped (header/subtotal rows).
    """
    products: list[SimplifiedLineItem] = []
    grand_total = 0.0

    for item in invoice.line_items:
        raw_desc = item.description or ""

        if _is_freight_or_shipping(raw_desc):
            continue

        if item.quantity is None:
            continue

        clean_desc, item_number = _extract_item_number(raw_desc)
        qty = float(item.quantity)
        price = float(item.unit_price) if item.unit_price is not None else 0.0
        grand_total += qty * price

        products.append(
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
        line_items=products,
        grand_total=round(grand_total, 2) if products else None,
    )
