"""
Tests for the invoice extraction pipeline.

These tests mock the Groq API call and the PDF extractor so they run fast
and offline (no API key required, no real PDFs needed).

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.schema import Invoice, LineItem, VendorInfo, CustomerInfo
from src.validator import validate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INVOICE_DICT = {
    "invoice_number": "INV-TEST-001",
    "invoice_date": "2024-06-01",
    "due_date": "2024-06-15",
    "vendor": {
        "name": "Test Vendor Pvt Ltd",
        "address": "123 Test Road, Delhi 110001",
        "tax_id": "07AAAAA0000A1Z5",
        "email": "vendor@test.com",
        "phone": "9999999999",
    },
    "customer": {
        "name": "Sample Customer Corp",
        "address": "456 Sample Lane, Mumbai",
        "tax_id": None,
        "email": "customer@sample.com",
    },
    "line_items": [
        {
            "description": "Consulting Services",
            "quantity": 10,
            "unit_price": 1000.0,
            "tax_rate": 18.0,
            "total": 10000.0,   # pre-tax line total; tax is captured in tax_total
        }
    ],
    "subtotal": 10000.0,
    "tax_total": 1800.0,
    "discount": None,
    "grand_total": 11800.0,
    "currency": "INR",
    "payment_terms": "Net 30",
    "notes": None,
    "confidence_score": 0.92,
}


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestInvoiceSchema:
    def test_valid_invoice_parses(self) -> None:
        invoice = Invoice.model_validate(SAMPLE_INVOICE_DICT)
        assert invoice.invoice_number == "INV-TEST-001"
        assert invoice.currency == "INR"
        assert len(invoice.line_items) == 1

    def test_invoice_number_required(self) -> None:
        """invoice_number=None must raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Invoice.model_validate({**SAMPLE_INVOICE_DICT, "invoice_number": None})

    def test_integer_invoice_number_coerced(self) -> None:
        """LLMs sometimes return invoice_number as an integer — must be coerced."""
        invoice = Invoice.model_validate({**SAMPLE_INVOICE_DICT, "invoice_number": 12345})
        assert invoice.invoice_number == "12345"

    def test_string_amount_coercion(self) -> None:
        """Amounts like '1,800.00' should be coerced to float."""
        data = {**SAMPLE_INVOICE_DICT, "tax_total": "1,800.00"}
        invoice = Invoice.model_validate(data)
        assert invoice.tax_total == 1800.0

    def test_currency_uppercased(self) -> None:
        invoice = Invoice.model_validate({**SAMPLE_INVOICE_DICT, "currency": "inr"})
        assert invoice.currency == "INR"

    def test_optional_fields_default_none(self) -> None:
        minimal = {"invoice_number": "MIN-001"}
        invoice = Invoice.model_validate(minimal)
        assert invoice.due_date is None
        assert invoice.discount is None
        assert invoice.grand_total is None


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def _make_invoice(self, **overrides) -> Invoice:
        data = {**SAMPLE_INVOICE_DICT, **overrides}
        return Invoice.model_validate(data)

    def test_no_warnings_for_valid_invoice(self) -> None:
        invoice = self._make_invoice()
        result = validate(invoice)
        assert result.validation_warnings == []

    def test_grand_total_mismatch_warning(self) -> None:
        invoice = self._make_invoice(grand_total=99999.0)
        result = validate(invoice)
        fields = [w.field for w in result.validation_warnings]
        assert "grand_total" in fields

    def test_due_date_before_invoice_date_warning(self) -> None:
        invoice = self._make_invoice(
            invoice_date="2024-06-15",
            due_date="2024-06-01",    # reversed
        )
        result = validate(invoice)
        fields = [w.field for w in result.validation_warnings]
        assert "due_date" in fields

    def test_line_item_sum_mismatch_warning(self) -> None:
        invoice = self._make_invoice(subtotal=99999.0)
        result = validate(invoice)
        fields = [w.field for w in result.validation_warnings]
        assert "subtotal" in fields

    def test_suspicious_invoice_number_warning(self) -> None:
        invoice = self._make_invoice(invoice_number="N/A")
        result = validate(invoice)
        fields = [w.field for w in result.validation_warnings]
        assert "invoice_number" in fields


# ---------------------------------------------------------------------------
# Agent integration test (mocked)
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    @patch("src.agent.VectorStore")       # mock ChromaDB — no disk access in tests
    @patch("src.agent.extract_text")
    @patch("src.agent.GroqClient")
    def test_agent_run_returns_invoice(
        self,
        mock_groq_cls: MagicMock,
        mock_extract: MagicMock,
        mock_vector_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline run with all external calls mocked."""
        from src.pdf_extractor import ExtractionResult
        from src.llm_client import LLMResponse

        # Mock PDF extraction
        mock_extract.return_value = ExtractionResult(
            text="Invoice text here...",
            used_ocr=False,
            page_count=1,
            source_path="test.pdf",
        )

        # Mock vector store — returns no examples (cold start)
        mock_store = MagicMock()
        mock_store.find_similar.return_value = []
        mock_vector_store_cls.return_value = mock_store

        # Mock Groq response
        mock_llm = MagicMock()
        mock_llm.extract_invoice.return_value = LLMResponse(
            raw_json=SAMPLE_INVOICE_DICT,
            model="llama-3.3-70b-versatile",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            latency_ms=310.0,
        )
        mock_groq_cls.return_value = mock_llm

        from src.agent import InvoiceAgent
        agent = InvoiceAgent()
        result = agent.run("test.pdf")

        assert result.invoice.invoice_number == "INV-TEST-001"
        assert result.invoice.grand_total == 11800.0
        assert result.invoice.raw_extraction_metadata["used_ocr"] is False
        assert result.invoice.raw_extraction_metadata["latency_ms"] == 310.0


# ---------------------------------------------------------------------------
# Feedback / few-shot tests
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_save_and_load_example(self, tmp_path: Path) -> None:
        """Saved examples should round-trip through JSON correctly."""
        import src.feedback as fb
        from unittest.mock import patch

        example_path = tmp_path / "few_shot_examples.json"

        with patch.object(fb, "FEW_SHOT_PATH", example_path):
            fb.save_example("Some invoice text", {"invoice_number": "FB-001"})
            loaded = fb.load_examples()

        assert len(loaded) == 1
        assert loaded[0]["output_json"]["invoice_number"] == "FB-001"

    def test_get_similar_returns_most_relevant(self, tmp_path: Path) -> None:
        """TF-IDF similarity should rank relevant examples above unrelated ones."""
        import src.feedback as fb
        from unittest.mock import patch

        example_path = tmp_path / "few_shot_examples.json"

        examples = [
            {"input_text": "Consulting invoice for software services Bangalore", "output_json": {"invoice_number": "A"}},
            {"input_text": "Grocery store purchase fruits vegetables", "output_json": {"invoice_number": "B"}},
            {"input_text": "Software development monthly retainer invoice", "output_json": {"invoice_number": "C"}},
            {"input_text": "Hotel accommodation booking Chennai", "output_json": {"invoice_number": "D"}},
        ]

        with patch.object(fb, "FEW_SHOT_PATH", example_path):
            import json
            example_path.write_text(json.dumps(examples), encoding="utf-8")

            with patch.object(fb, "FEW_SHOT_TOP_K", 2):
                results = fb.get_similar_examples("Software consulting invoice services")

        result_ids = {r["output_json"]["invoice_number"] for r in results}
        # Software-related examples should rank higher than grocery/hotel
        assert "A" in result_ids or "C" in result_ids


# ---------------------------------------------------------------------------
# SimplifiedInvoice tests
# ---------------------------------------------------------------------------

class TestSimplifiedInvoice:
    def _make_invoice(self, **overrides) -> Invoice:
        from tests.test_extraction import SAMPLE_INVOICE_DICT
        data = {**SAMPLE_INVOICE_DICT, **overrides}
        return Invoice.model_validate(data)

    def test_basic_conversion(self) -> None:
        from src.schema import simplify_invoice
        invoice = Invoice.model_validate(SAMPLE_INVOICE_DICT)
        simple = simplify_invoice(invoice)

        assert simple.invoice_number == "INV-TEST-001"
        assert simple.vendor_name == "Test Vendor Pvt Ltd"
        assert simple.customer_name == "Sample Customer Corp"
        assert simple.grand_total == 10000.0   # calculated: 10 qty × $1000 unit_price
        assert len(simple.line_items) == 1
        assert simple.line_items[0].unit_price == 1000.0

    def test_item_number_extracted_from_description(self) -> None:
        """Item: / SKU: / Part No: patterns should be pulled into item_number."""
        from src.schema import simplify_invoice, LineItem
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "DEER PELLET 20 (Item: 9794, Pack: 50#)",
                 "quantity": 5.0, "unit_price": 20.0, "tax_rate": None, "total": 100.0},
                {"description": "Blue Halter (SKU: BH-2201)",
                 "quantity": 2.0, "unit_price": 15.0, "tax_rate": None, "total": 30.0},
                {"description": "Wire Reel (Part No: WR-440)",
                 "quantity": 1.0, "unit_price": 45.0, "tax_rate": None, "total": 45.0},
            ],
        }
        invoice = Invoice.model_validate(data)
        simple = simplify_invoice(invoice)

        assert simple.line_items[0].item_number == "9794"
        assert simple.line_items[1].item_number == "BH-2201"
        assert simple.line_items[2].item_number == "WR-440"

    def test_description_cleaned_of_noise(self) -> None:
        """Parenthetical noise should be stripped from description."""
        from src.schema import simplify_invoice
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "DEER PELLET 20 (Item: 9794, Pack: 50#)",
                 "quantity": 5.0, "unit_price": 20.0, "tax_rate": None, "total": 100.0},
            ],
        }
        invoice = Invoice.model_validate(data)
        simple = simplify_invoice(invoice)

        assert simple.line_items[0].description == "DEER PELLET 20"

    def test_items_without_quantity_skipped(self) -> None:
        """Line items with null quantity should be dropped (header/subtotal rows)."""
        from src.schema import simplify_invoice
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "Real item", "quantity": 3.0,
                 "unit_price": 10.0, "tax_rate": None, "total": 30.0},
                {"description": "Subtotal header row", "quantity": None,
                 "unit_price": None, "tax_rate": None, "total": None},
            ],
        }
        invoice = Invoice.model_validate(data)
        simple = simplify_invoice(invoice)

        assert len(simple.line_items) == 1
        assert simple.line_items[0].description == "Real item"
        assert simple.line_items[0].unit_price == 10.0

    def test_unit_price_copied_from_full_invoice(self) -> None:
        """unit_price on SimplifiedLineItem should match the source LineItem."""
        from src.schema import simplify_invoice
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "Widget A", "quantity": 4.0,
                 "unit_price": 25.50, "tax_rate": None, "total": 102.0},
                {"description": "Widget B (SKU: WB-99)", "quantity": 2.0,
                 "unit_price": 8.75, "tax_rate": None, "total": 17.50},
            ],
        }
        invoice = Invoice.model_validate(data)
        simple = simplify_invoice(invoice)

        assert simple.line_items[0].unit_price == 25.50
        assert simple.line_items[1].unit_price == 8.75


# ---------------------------------------------------------------------------
# Freight removal + calculated grand_total tests
# ---------------------------------------------------------------------------

class TestFreightRemovalAndCalculatedTotal:
    def _make_invoice(self, line_items: list, grand_total: float | None = None) -> Invoice:
        data = {**SAMPLE_INVOICE_DICT, "line_items": line_items, "grand_total": grand_total}
        return Invoice.model_validate(data)

    def test_freight_filtered_out(self) -> None:
        """A FREIGHT line item must not appear in simplified line_items."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(line_items=[
            {"description": "Widget A", "quantity": 5.0, "unit_price": 10.0, "tax_rate": None, "total": 50.0},
            {"description": "FREIGHT",  "quantity": 1.0, "unit_price": 15.0, "tax_rate": None, "total": 15.0},
        ])
        simple = simplify_invoice(invoice)
        assert len(simple.line_items) == 1
        assert simple.line_items[0].description == "Widget A"

    def test_grand_total_calculated_from_items(self) -> None:
        """grand_total = sum(qty × unit_price); invoice.grand_total is ignored."""
        from src.schema import simplify_invoice
        # invoice says $60 but product is qty=10 @ $5 → real total is $50
        invoice = self._make_invoice(
            line_items=[{"description": "Item A", "quantity": 10.0,
                         "unit_price": 5.0, "tax_rate": None, "total": 50.0}],
            grand_total=60.0,
        )
        simple = simplify_invoice(invoice)
        assert simple.grand_total == 50.0

    def test_kpy_pdf_scenario(self) -> None:
        """KPY scenario: LUBE AID + FREIGHT → only LUBE AID, grand_total=33.59."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(
            line_items=[
                {"description": "LUBE AID; 50LB (Item: SP16087)",
                 "quantity": 1.0, "unit_price": 33.59, "tax_rate": None, "total": 33.59},
                {"description": "FREIGHT",
                 "quantity": 1.0, "unit_price": 0.75, "tax_rate": None, "total": 0.75},
            ],
            grand_total=34.34,
        )
        simple = simplify_invoice(invoice)
        assert len(simple.line_items) == 1
        assert simple.line_items[0].item_number == "SP16087"
        assert simple.grand_total == 33.59   # 1 × 33.59, NOT the invoice's 34.34

    def test_cargil_scenario_with_discounts(self) -> None:
        """Calculated total ignores a discount-adjusted grand_total from the PDF."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(
            line_items=[
                {"description": "Product A", "quantity": 40.0, "unit_price": 32.87, "tax_rate": None, "total": 1314.80},
                {"description": "Product B", "quantity": 10.0, "unit_price": 15.90, "tax_rate": None, "total": 159.00},
                {"description": "Product C", "quantity": 80.0, "unit_price": 16.20, "tax_rate": None, "total": 1296.00},
            ],
            grand_total=2000.0,   # simulated discount-adjusted total from PDF
        )
        simple = simplify_invoice(invoice)
        # (40×32.87) + (10×15.90) + (80×16.20) = 1314.80 + 159.00 + 1296.00 = 2769.80
        assert simple.grand_total == 2769.80
        assert simple.grand_total != 2000.0  # PDF total must be ignored

    def test_multiple_freight_types_all_removed(self) -> None:
        """FREIGHT, FUEL CHARGE, and HANDLING FEE are all filtered out."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(line_items=[
            {"description": "Real Product",  "quantity": 2.0, "unit_price": 50.0, "tax_rate": None, "total": 100.0},
            {"description": "FREIGHT",        "quantity": 1.0, "unit_price": 10.0, "tax_rate": None, "total": 10.0},
            {"description": "FUEL CHARGE",    "quantity": 1.0, "unit_price":  5.0, "tax_rate": None, "total":  5.0},
            {"description": "HANDLING FEE",   "quantity": 1.0, "unit_price":  3.0, "tax_rate": None, "total":  3.0},
        ])
        simple = simplify_invoice(invoice)
        assert len(simple.line_items) == 1
        assert simple.line_items[0].description == "Real Product"
        assert simple.grand_total == 100.0   # 2 × 50, no freight

    def test_only_freight_returns_empty(self) -> None:
        """All-freight invoice → empty line_items and grand_total=None."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(line_items=[
            {"description": "FREIGHT",     "quantity": 1.0, "unit_price": 20.0, "tax_rate": None, "total": 20.0},
            {"description": "FUEL CHARGE", "quantity": 1.0, "unit_price":  5.0, "tax_rate": None, "total":  5.0},
        ])
        simple = simplify_invoice(invoice)
        assert simple.line_items == []
        assert simple.grand_total is None

    def test_case_insensitive_freight_detection(self) -> None:
        """'FREIGHT', 'Freight', and 'freight' are all filtered."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(line_items=[
            {"description": "Product X", "quantity": 3.0, "unit_price": 10.0, "tax_rate": None, "total": 30.0},
            {"description": "FREIGHT",   "quantity": 1.0, "unit_price":  5.0, "tax_rate": None, "total":  5.0},
            {"description": "Freight",   "quantity": 1.0, "unit_price":  5.0, "tax_rate": None, "total":  5.0},
            {"description": "freight",   "quantity": 1.0, "unit_price":  5.0, "tax_rate": None, "total":  5.0},
        ])
        simple = simplify_invoice(invoice)
        assert len(simple.line_items) == 1
        assert simple.grand_total == 30.0   # 3 × 10

    def test_decimal_quantity_calculated_correctly(self) -> None:
        """Decimal quantities (e.g. 12.2 tons) produce the correct grand_total."""
        from src.schema import simplify_invoice
        invoice = self._make_invoice(line_items=[
            {"description": "Bulk Product", "quantity": 12.2, "unit_price": 3.0, "tax_rate": None, "total": 36.6},
        ])
        simple = simplify_invoice(invoice)
        assert simple.grand_total == 36.6   # 12.2 × 3.0
        assert simple.line_items[0].quantity == 12.2

    def test_no_item_number_stays_none(self) -> None:
        """Plain descriptions with no item number patterns → item_number is None."""
        from src.schema import simplify_invoice
        data = {
            **SAMPLE_INVOICE_DICT,
            "line_items": [
                {"description": "Consulting Services",
                 "quantity": 10.0, "unit_price": 1000.0, "tax_rate": 18.0, "total": 10000.0},
            ],
        }
        invoice = Invoice.model_validate(data)
        simple = simplify_invoice(invoice)

        assert simple.line_items[0].item_number is None
        assert simple.line_items[0].description == "Consulting Services"


# ---------------------------------------------------------------------------
# Strict shipped quantity + multi-code item_number tests
# ---------------------------------------------------------------------------

class TestStrictShippedAndMultiCode:
    """Tests for shipped-quantity enforcement and slash-separated item codes."""

    def _inv(self, line_items: list) -> Invoice:
        return Invoice.model_validate(
            {**SAMPLE_INVOICE_DICT, "line_items": line_items, "grand_total": None}
        )

    def test_shipped_quantity_passed_through(self) -> None:
        """Quantity on SimplifiedLineItem matches whatever the agent extracted."""
        from src.schema import simplify_invoice
        invoice = self._inv([
            {"description": "PRODUCT (Item: TEST123)",
             "quantity": 50.0, "unit_price": 10.0, "tax_rate": None, "total": 500.0},
        ])
        simple = simplify_invoice(invoice)
        assert simple.line_items[0].quantity == 50.0

    def test_multi_code_item_number_with_slash(self) -> None:
        """Primary / secondary codes combined with ' / ' are preserved in item_number."""
        from src.schema import simplify_invoice
        invoice = self._inv([
            {"description": "FEEDING LIMESTONE (Item: M3625 / 32704625)",
             "quantity": 50.0, "unit_price": 6.0, "tax_rate": None, "total": 300.0},
        ])
        simple = simplify_invoice(invoice)
        assert simple.line_items[0].item_number == "M3625 / 32704625"

    def test_single_code_unchanged(self) -> None:
        """Single item code works as before — no slash added."""
        from src.schema import simplify_invoice
        invoice = self._inv([
            {"description": "PRODUCT (Item: ABC123)",
             "quantity": 1.0, "unit_price": 100.0, "tax_rate": None, "total": 100.0},
        ])
        simple = simplify_invoice(invoice)
        assert simple.line_items[0].item_number == "ABC123"

    def test_multi_code_with_extra_info_after_comma(self) -> None:
        """Trailing metadata after the codes (e.g. ', Unit: 50LB') is not included."""
        from src.schema import simplify_invoice
        invoice = self._inv([
            {"description": "PRODUCT (Item: M3625 / 32704625, Unit: 50LB)",
             "quantity": 50.0, "unit_price": 6.0, "tax_rate": None, "total": 300.0},
        ])
        simple = simplify_invoice(invoice)
        assert simple.line_items[0].item_number == "M3625 / 32704625"

    def test_kyp_pdf_scenario(self) -> None:
        """KYP.pdf: 4 items with combined primary/secondary codes, freight excluded."""
        from src.schema import simplify_invoice
        invoice = self._inv([
            {"description": "FEEDING LIMESTONE - COARSE; 50LB (Item: M3625 / 32704625)",
             "quantity": 50.0, "unit_price": 6.00, "tax_rate": None, "total": 300.00},
            {"description": "PLAIN SOY OIL MINI BULK 2,000LB (Item: FA5760 / 11113514)",
             "quantity": 1.0,  "unit_price": 1960.00, "tax_rate": None, "total": 1960.00},
            {"description": "EMPTY TOTE DEPOSIT-CHARGE (Item: FA5001 / 123)",
             "quantity": 1.0,  "unit_price": 225.00, "tax_rate": None, "total": 225.00},
            {"description": "LYSINE - NON-DOMESTIC - 98.5%; 25 KG (Item: V8654 / 202502101)",
             "quantity": 3.0,  "unit_price": 54.57, "tax_rate": None, "total": 163.71},
            {"description": "FREIGHT",
             "quantity": 1.0,  "unit_price": 104.16, "tax_rate": None, "total": 104.16},
        ])
        simple = simplify_invoice(invoice)

        assert len(simple.line_items) == 4          # freight excluded
        assert simple.line_items[0].item_number == "M3625 / 32704625"
        assert simple.line_items[0].quantity    == 50.0
        assert simple.line_items[1].item_number == "FA5760 / 11113514"
        assert simple.line_items[2].item_number == "FA5001 / 123"
        assert simple.line_items[3].item_number == "V8654 / 202502101"
        # grand_total = (50×6) + (1×1960) + (1×225) + (3×54.57) = 2648.71
        assert simple.grand_total == 2648.71
