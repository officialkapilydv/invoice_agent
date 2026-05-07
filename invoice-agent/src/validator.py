"""
Post-extraction validation.

Philosophy: validation should surface problems, not hide them behind hard
failures. Every check appends a warning to the Invoice object rather than
raising an exception — the caller decides what to do with warnings.

Checks performed:
  1. grand_total reconciliation: subtotal + tax_total - discount ≈ grand_total
  2. line_item sum: sum(item.total) ≈ subtotal
  3. Date sanity: due_date >= invoice_date
  4. Required-field presence: warn if invoice_number looks obviously wrong
"""

from __future__ import annotations

import logging
from datetime import date

from src.schema import Invoice, ValidationWarning

logger = logging.getLogger(__name__)

# Monetary tolerance in whatever currency the invoice uses.
# ₹1 covers typical floating-point rounding in Indian invoices.
_MONEY_TOLERANCE = 1.0


def validate(invoice: Invoice) -> Invoice:
    """
    Run all validation checks on an extracted Invoice, appending warnings.

    Mutates the invoice in place (adds to validation_warnings) and returns it.
    Does NOT raise — callers should inspect validation_warnings.

    Args:
        invoice: Parsed and schema-validated Invoice object.

    Returns:
        The same invoice with warnings populated.
    """
    invoice.validation_warnings.clear()

    _check_grand_total(invoice)
    _check_line_item_sum(invoice)
    _check_date_order(invoice)
    _check_invoice_number(invoice)

    if invoice.validation_warnings:
        logger.warning(
            "Invoice %s has %d validation warning(s)",
            invoice.invoice_number,
            len(invoice.validation_warnings),
        )
    else:
        logger.info("Invoice %s passed all validation checks", invoice.invoice_number)

    return invoice


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _warn(invoice: Invoice, field: str, message: str) -> None:
    invoice.validation_warnings.append(
        ValidationWarning(field=field, message=message)
    )


def _check_grand_total(invoice: Invoice) -> None:
    """subtotal + tax_total - discount ≈ grand_total."""
    if any(
        v is None
        for v in [invoice.subtotal, invoice.tax_total, invoice.grand_total]
    ):
        return   # not enough data to validate

    discount = invoice.discount or 0.0
    expected = (invoice.subtotal or 0.0) + (invoice.tax_total or 0.0) - discount
    actual = invoice.grand_total or 0.0
    diff = abs(expected - actual)

    if diff > _MONEY_TOLERANCE:
        _warn(
            invoice,
            "grand_total",
            f"grand_total={actual} does not match "
            f"subtotal({invoice.subtotal}) + tax({invoice.tax_total}) "
            f"- discount({discount}) = {expected:.2f} "
            f"(difference: {diff:.2f})",
        )


def _check_line_item_sum(invoice: Invoice) -> None:
    """Sum of line_item.total values should approximate subtotal."""
    if not invoice.line_items or invoice.subtotal is None:
        return

    item_totals = [
        item.total for item in invoice.line_items if item.total is not None
    ]
    if not item_totals:
        return

    items_sum = sum(item_totals)
    diff = abs(items_sum - (invoice.subtotal or 0.0))

    if diff > _MONEY_TOLERANCE:
        _warn(
            invoice,
            "subtotal",
            f"Sum of line items ({items_sum:.2f}) differs from "
            f"subtotal ({invoice.subtotal}) by {diff:.2f}",
        )


def _check_date_order(invoice: Invoice) -> None:
    """due_date must be >= invoice_date when both are present."""
    if invoice.invoice_date is None or invoice.due_date is None:
        return

    if invoice.due_date < invoice.invoice_date:
        _warn(
            invoice,
            "due_date",
            f"due_date ({invoice.due_date}) is before invoice_date "
            f"({invoice.invoice_date})",
        )


def _check_invoice_number(invoice: Invoice) -> None:
    """Flag suspiciously generic invoice numbers that suggest extraction failure."""
    number = invoice.invoice_number.strip()
    generic_values = {"n/a", "na", "null", "none", "invoice", "inv"}
    if number.lower() in generic_values or len(number) < 2:
        _warn(
            invoice,
            "invoice_number",
            f"invoice_number '{number}' looks invalid — verify manually",
        )
