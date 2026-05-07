"""
Layer 2: Pattern Library — auto-generated rules from corrections.

HOW META-LEARNING WORKS:
------------------------
When a user corrects an extraction, they're implicitly telling the agent
"you made this mistake — here's the truth." After enough corrections, patterns
emerge: maybe the agent always misreads a certain vendor's invoice number, or
always mistakes USD for INR on invoices with a specific layout.

This module detects those patterns automatically by sending correction history
to Groq and asking it to reason about what rules would prevent those mistakes.
The rules are stored as natural language and injected into every future prompt.

This is "meta-learning" — the model learning about its own failure modes,
not just memorising correct examples.

WHY natural language rules (not code):
  Code rules would be brittle ("if vendor == 'Bluebonnet' then field = 'Order #'").
  Natural language rules are interpreted by the LLM at inference time, so they
  apply flexibly to variations we haven't seen yet.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import FEW_SHOT_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

_RULES_PATH = PROJECT_ROOT / "data" / "learned_rules.json"

# How many corrections to accumulate before triggering pattern extraction
PATTERN_EXTRACTION_THRESHOLD = 10

# How many recent corrections to analyse per pattern extraction run
CORRECTIONS_WINDOW = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _empty_rules_store() -> dict:
    return {"rules": [], "last_extracted_at": None, "total_extractions_run": 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rules() -> list[dict]:
    """
    Load all stored rules from disk.

    Each rule dict has:
      trigger       — keyword/phrase that makes the rule relevant
      rule          — the natural language rule to inject
      confidence    — 0.0–1.0, how reliably this pattern holds
      support_count — how many corrections supported this rule
      created_at    — ISO timestamp
    """
    if not _RULES_PATH.exists():
        return []
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        return data.get("rules", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load rules: %s", exc)
        return []


def get_relevant_rules(pdf_text: str, max_rules: int = 5) -> list[str]:
    """
    Return rules whose trigger keywords appear in the invoice text.

    Trigger matching is case-insensitive keyword search — deliberately simple.
    We don't want complex matching logic here; the LLM will interpret the rules
    intelligently at inference time.

    Args:
        pdf_text: Raw text from the PDF extractor.
        max_rules: Cap to avoid bloating the prompt.

    Returns:
        List of rule strings ready to inject into the prompt.
    """
    rules = load_rules()
    if not rules:
        return []

    text_lower = pdf_text.lower()
    matched: list[dict] = []

    for rule in rules:
        trigger = rule.get("trigger", "").lower()
        # Match if trigger is empty (global rule) or keyword found in text
        if not trigger or trigger in text_lower:
            matched.append(rule)

    # Sort by confidence descending, take top N
    matched.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    result = [r["rule"] for r in matched[:max_rules]]

    if result:
        logger.debug("Injecting %d learned rules into prompt", len(result))

    return result


def should_extract_patterns() -> bool:
    """
    Return True if enough new corrections have accumulated since the last
    pattern extraction run to warrant running it again.
    """
    try:
        import sqlite3
        from config.settings import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        total_corrections = conn.execute(
            "SELECT COUNT(*) FROM corrections"
        ).fetchone()[0]
        conn.close()

        data = _load_store()
        already_processed = data.get("total_extractions_run", 0)
        new_since_last = total_corrections - already_processed

        return new_since_last >= PATTERN_EXTRACTION_THRESHOLD

    except Exception as exc:
        logger.warning("Could not check pattern extraction threshold: %s", exc)
        return False


def extract_patterns_from_corrections() -> list[dict]:
    """
    Use Groq to analyse recent corrections and generate new rules.

    This is the meta-learning step:
      1. Load the CORRECTIONS_WINDOW most recent corrections from SQLite
      2. Send them to Groq with a prompt asking it to identify patterns
      3. Parse the returned rules and merge into learned_rules.json

    New rules with the same trigger as existing ones bump the support_count
    rather than creating duplicates.

    Returns:
        List of newly added/updated rule dicts.
    """
    corrections = _load_recent_corrections(CORRECTIONS_WINDOW)
    if not corrections:
        logger.info("No corrections available for pattern extraction")
        return []

    logger.info(
        "Running pattern extraction on %d corrections...", len(corrections)
    )

    prompt = _build_pattern_prompt(corrections)
    raw_rules = _call_groq_for_patterns(prompt)

    if not raw_rules:
        return []

    updated = _merge_rules(raw_rules)
    _update_run_count()

    logger.info(
        "Pattern extraction complete — %d rules now in library", len(load_rules())
    )
    return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    if not _RULES_PATH.exists():
        return _empty_rules_store()
    try:
        return json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_rules_store()


def _save_store(data: dict) -> None:
    _RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RULES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_recent_corrections(n: int) -> list[dict]:
    """
    Load the N most recent corrections paired with their original extractions.
    Returns list of {original_json, corrected_json, notes}.
    """
    try:
        import sqlite3
        from config.settings import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            """
            SELECT c.corrected_json, c.user_notes, e.extracted_json
            FROM corrections c
            JOIN extractions e ON c.extraction_id = e.id
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        conn.close()

        return [
            {
                "original": json.loads(row[2]),
                "corrected": json.loads(row[0]),
                "notes": row[1] or "",
            }
            for row in rows
        ]
    except Exception as exc:
        logger.error("Could not load corrections for pattern extraction: %s", exc)
        return []


def _build_pattern_prompt(corrections: list[dict]) -> str:
    """Build the meta-learning prompt sent to Groq."""
    examples_text = ""
    for i, c in enumerate(corrections, 1):
        diff = _summarise_diff(c["original"], c["corrected"])
        examples_text += f"\n### Correction {i}\n{diff}\n"
        if c["notes"]:
            examples_text += f"User note: {c['notes']}\n"

    return f"""You are analysing invoice extraction mistakes to find patterns.

Below are {len(corrections)} corrections where a human fixed an AI's mistakes.
Each shows what the AI extracted (WRONG) vs what the human corrected it to (RIGHT).

{examples_text}

Your task: identify 3-7 ACTIONABLE rules that would prevent these mistakes.

Return a JSON array. Each rule object must have exactly these fields:
{{
  "trigger": "<keyword or phrase that signals this rule applies, e.g. 'Bluebonnet' or 'Tax Invoice' — use empty string for global rules>",
  "rule": "<a single clear instruction for the extraction model, e.g. 'When vendor name contains Bluebonnet, the invoice number is in the field labeled Order # not Invoice #'>",
  "confidence": <float 0.5-1.0 based on how consistently the pattern appears>,
  "support_count": <how many of the corrections above support this rule>
}}

Return ONLY the JSON array. No explanation, no markdown.
"""


def _summarise_diff(original: dict, corrected: dict) -> str:
    """
    Return a compact human-readable diff of fields that changed.
    Keeps the prompt concise — we don't want to send full JSON blobs.
    """
    lines = []
    all_keys = set(original.keys()) | set(corrected.keys())
    scalar_keys = {
        k for k in all_keys
        if not isinstance(original.get(k), (dict, list))
        and not isinstance(corrected.get(k), (dict, list))
    }
    for key in sorted(scalar_keys):
        o_val = original.get(key)
        c_val = corrected.get(key)
        if o_val != c_val:
            lines.append(f"  {key}: WRONG={o_val!r} → RIGHT={c_val!r}")

    # Summarise vendor/customer name changes (most signal-rich)
    for nested in ("vendor", "customer"):
        o_nested = original.get(nested) or {}
        c_nested = corrected.get(nested) or {}
        if isinstance(o_nested, dict) and isinstance(c_nested, dict):
            if o_nested.get("name") != c_nested.get("name"):
                lines.append(
                    f"  {nested}.name: WRONG={o_nested.get('name')!r} "
                    f"→ RIGHT={c_nested.get('name')!r}"
                )

    return "\n".join(lines) if lines else "  (no scalar field changes)"


def _call_groq_for_patterns(prompt: str) -> list[dict]:
    """Call Groq with the meta-learning prompt and parse the rule array."""
    try:
        from groq import Groq
        from config.settings import GROQ_API_KEY, GROQ_MODEL

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)

        # Groq might wrap the array in a key — handle both formats
        if isinstance(parsed, list):
            rules = parsed
        elif isinstance(parsed, dict):
            # Find the first list value
            rules = next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
        else:
            rules = []

        logger.info("Groq proposed %d new rules", len(rules))
        return rules

    except Exception as exc:
        logger.error("Pattern extraction LLM call failed: %s", exc)
        return []


def _merge_rules(new_rules: list[dict]) -> list[dict]:
    """
    Merge new rules into the store.
    If a rule with the same trigger already exists, increment support_count.
    Otherwise append as a new rule.
    """
    store = _load_store()
    existing = store.get("rules", [])
    existing_triggers = {r["trigger"].lower(): i for i, r in enumerate(existing)}
    updated: list[dict] = []

    for rule in new_rules:
        trigger = rule.get("trigger", "").lower()
        rule["created_at"] = datetime.now(timezone.utc).isoformat()

        if trigger in existing_triggers:
            idx = existing_triggers[trigger]
            existing[idx]["support_count"] = (
                existing[idx].get("support_count", 1) + rule.get("support_count", 1)
            )
            existing[idx]["confidence"] = max(
                existing[idx].get("confidence", 0.5),
                rule.get("confidence", 0.5),
            )
            updated.append(existing[idx])
        else:
            existing.append(rule)
            updated.append(rule)

    store["rules"] = existing
    store["last_extracted_at"] = datetime.now(timezone.utc).isoformat()
    _save_store(store)
    return updated


def _update_run_count() -> None:
    """Record how many corrections existed when we last ran pattern extraction."""
    try:
        import sqlite3
        from config.settings import DB_PATH

        conn = sqlite3.connect(str(DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        conn.close()

        store = _load_store()
        store["total_extractions_run"] = total
        _save_store(store)
    except Exception as exc:
        logger.warning("Could not update run count: %s", exc)
