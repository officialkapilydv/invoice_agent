"""
Groq API wrapper with retry logic and token tracking.

Why Groq over OpenAI for this use case?
  - Groq runs on custom LPU hardware — typical latency is 200-500 ms vs 2-5 s
  - llama-3.3-70b-versatile on Groq is competitive with GPT-4o-mini on
    structured extraction tasks at a fraction of the cost
  - JSON mode is supported, which eliminates markdown-wrapper parsing bugs
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from groq import Groq, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config.settings import GROQ_API_KEY, GROQ_MODEL
from src.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Retry on transient network/rate-limit errors; stop after 4 attempts.
# Exponential backoff: 2s → 4s → 8s → give up.
_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)


@dataclass
class LLMResponse:
    raw_json: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float = 0.0


class GroqClient:
    """Thin wrapper around the Groq SDK focused on invoice extraction."""

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not configured.")
        self._client = Groq(api_key=GROQ_API_KEY)
        self._model = GROQ_MODEL

    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def extract_invoice(
        self,
        invoice_text: str,
        few_shot_examples: list[dict] | None = None,
        rules: list[str] | None = None,
    ) -> LLMResponse:
        """
        Send invoice text to Groq and return a structured extraction.

        Args:
            invoice_text: Raw text from the PDF extractor.
            few_shot_examples: Semantically similar past examples from ChromaDB.
            rules: Learned rules from PatternLibrary to inject into the prompt.
        """
        import time

        examples = few_shot_examples or []
        rule_list = rules or []
        user_content = build_user_prompt(invoice_text, examples, rule_list)

        logger.info(
            "Calling Groq model=%s, few_shot=%d, rules=%d",
            self._model, len(examples), len(rule_list),
        )

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,   # low temperature = more deterministic extraction
            max_tokens=4096,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_text = response.choices[0].message.content or ""

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            # Should not happen with JSON mode enabled, but guard anyway.
            logger.error("JSON decode failed. Raw output:\n%s", raw_text[:500])
            raise ValueError(f"Groq returned non-JSON output: {exc}") from exc

        usage = response.usage
        logger.info(
            "Groq response received in %.0f ms | tokens: prompt=%d completion=%d",
            elapsed_ms,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return LLMResponse(
            raw_json=parsed,
            model=self._model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=elapsed_ms,
        )
