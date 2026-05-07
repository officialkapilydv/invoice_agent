"""
Main agent orchestrator.

Coordinates the full extraction pipeline:
  PDF → text → vector memory (few-shot) → LLM → schema validation → business validation → result

The memory layer is now ChromaDB-backed (semantic similarity) instead of
TF-IDF, which means "similar past invoice" is measured by meaning, not word overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from src.llm_client import GroqClient, LLMResponse
from src.memory.pattern_library import get_relevant_rules
from src.memory.vector_store import VectorStore
from src.pdf_extractor import ExtractionResult, extract_text
from src.schema import Invoice
from src.validator import validate

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    invoice: Invoice
    extraction: ExtractionResult
    llm_response: LLMResponse
    schema_errors: list[str]
    few_shot_sources: list[str]    # "correction" or "extraction" — for transparency


class InvoiceAgent:
    """
    End-to-end invoice extraction agent with semantic memory.

    The vector store is initialised once per agent instance (not per call)
    so the embedding model is loaded only once — important for batch runs.
    """

    def __init__(self) -> None:
        self._llm = GroqClient()
        self._memory = VectorStore()

    def run(self, pdf_path: str, rules: list[str] | None = None) -> AgentResult:
        """
        Extract, validate, and return structured invoice data from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            rules: Optional learned rules from PatternLibrary to inject into prompt.

        Returns:
            AgentResult with the Invoice and full metadata.
        """
        logger.info("=== Starting extraction for: %s ===", pdf_path)

        # 1. PDF → text
        extraction = extract_text(pdf_path)

        # 2. Layer 1 — semantic retrieval (corrections first, then extractions)
        similar_examples = self._memory.find_similar(extraction.text)
        sources = [ex["source"] for ex in similar_examples]
        logger.info(
            "Injecting %d few-shot examples: %s",
            len(similar_examples),
            sources or "none",
        )

        # 3. Layer 2 — pattern library rules (auto-generated from corrections)
        #    Caller can pass explicit rules; otherwise we fetch from the library.
        active_rules = rules if rules is not None else get_relevant_rules(extraction.text)
        if active_rules:
            logger.info("Injecting %d learned rules into prompt", len(active_rules))

        # 4. LLM call with both layers injected
        llm_response = self._llm.extract_invoice(
            invoice_text=extraction.text,
            few_shot_examples=similar_examples,
            rules=active_rules,
        )

        # 5. Schema coercion
        schema_errors: list[str] = []
        raw = llm_response.raw_json

        try:
            invoice = Invoice.model_validate(raw)
        except ValidationError as exc:
            schema_errors = [str(e) for e in exc.errors()]
            logger.warning(
                "Schema validation had %d error(s) — attempting partial parse",
                len(schema_errors),
            )
            cleaned = _strip_invalid_fields(raw, exc)
            try:
                invoice = Invoice.model_validate(cleaned)
            except ValidationError:
                invoice = Invoice(invoice_number="PARSE_ERROR")
                invoice.confidence_score = 0.0

        # 6. Debug metadata
        invoice.raw_extraction_metadata = {
            "used_ocr": extraction.used_ocr,
            "page_count": extraction.page_count,
            "model": llm_response.model,
            "prompt_tokens": llm_response.prompt_tokens,
            "completion_tokens": llm_response.completion_tokens,
            "latency_ms": round(llm_response.latency_ms, 1),
            "few_shot_count": len(similar_examples),
            "few_shot_sources": sources,
            "rules_injected": len(active_rules),
        }

        # 7. Business validation
        invoice = validate(invoice)

        logger.info(
            "=== Done: %s | confidence=%.2f | warnings=%d ===",
            invoice.invoice_number,
            invoice.confidence_score,
            len(invoice.validation_warnings),
        )

        return AgentResult(
            invoice=invoice,
            extraction=extraction,
            llm_response=llm_response,
            schema_errors=schema_errors,
            few_shot_sources=sources,
        )


def _strip_invalid_fields(raw: dict, exc: ValidationError) -> dict:
    bad_fields = {str(err["loc"][0]) for err in exc.errors() if err.get("loc")}
    cleaned = {k: v for k, v in raw.items() if k not in bad_fields}
    logger.debug("Stripped invalid fields: %s", bad_fields)
    return cleaned
