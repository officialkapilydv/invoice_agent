"""
Feedback / continuous in-context learning module.

How it works (RAG-style few-shot learning):
---------------------------------------------------------------------------
When a user corrects an extraction:
  1. The corrected JSON + the original PDF text are appended to
     `data/few_shot_examples.json`.
  2. On the next extraction, this module retrieves the TOP-K most similar
     past corrections by comparing the new invoice text against stored texts
     using TF-IDF cosine similarity.
  3. The top matches are injected into the LLM prompt as few-shot examples.

Why this is NOT "true fine-tuning":
---------------------------------------------------------------------------
True fine-tuning updates the model's weights so it permanently encodes the
new knowledge. What we're doing here is retrieval-augmented prompting —
we're giving the model examples at inference time. The model forgets them
the moment the conversation ends.

The practical difference:
  - Fine-tuning: expensive (GPU hours), slow (days), but fully baked in
  - RAG few-shot (this): free, instant, but uses up context window tokens

For most invoice extraction workloads with <10k unique invoice formats, the
RAG approach delivers ~80% of fine-tuning accuracy improvements with 0 cost.
See README.md for upgrade paths (LoRA on HuggingFace, Together.ai, etc.).
---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import FEW_SHOT_PATH, FEW_SHOT_TOP_K
from src.database import save_correction

logger = logging.getLogger(__name__)


def load_examples() -> list[dict]:
    """
    Load all stored few-shot examples from disk.

    Returns an empty list if the file doesn't exist yet.
    Each example has: input_text (str), output_json (dict), created_at (str).
    """
    path = Path(FEW_SHOT_PATH)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load few-shot examples: %s", exc)
        return []


def save_example(input_text: str, corrected_json: dict) -> None:
    """
    Append a new correction to the few-shot examples file.

    Args:
        input_text: The raw PDF text for which the correction was made.
        corrected_json: The user-verified correct extraction dict.
    """
    from datetime import datetime, timezone

    path = Path(FEW_SHOT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples()
    examples.append(
        {
            "input_text": input_text,
            "output_json": corrected_json,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved correction to few_shot_examples.json (total: %d)", len(examples)
    )


def get_similar_examples(query_text: str) -> list[dict]:
    """
    Retrieve the TOP-K most similar past corrections for a new invoice text.

    Uses TF-IDF + cosine similarity — no embeddings API needed, runs locally.
    Falls back to the most-recent examples if there are too few for TF-IDF.

    Args:
        query_text: The raw PDF text of the invoice being extracted.

    Returns:
        List of up to FEW_SHOT_TOP_K example dicts (input_text, output_json).
    """
    examples = load_examples()
    if not examples:
        return []

    if len(examples) <= FEW_SHOT_TOP_K:
        # Not enough examples to rank — return all of them
        return examples[-FEW_SHOT_TOP_K:]

    stored_texts = [ex["input_text"] for ex in examples]

    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        # Fit on stored texts so vocabulary is from the corpus, then transform query
        tfidf_matrix = vectorizer.fit_transform(stored_texts)
        query_vec = vectorizer.transform([query_text])

        scores: np.ndarray = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = scores.argsort()[-FEW_SHOT_TOP_K:][::-1]  # descending order

        selected = [examples[i] for i in top_indices]
        logger.debug(
            "Retrieved %d few-shot examples (top scores: %s)",
            len(selected),
            [round(float(scores[i]), 3) for i in top_indices],
        )
        return selected

    except Exception as exc:
        # TF-IDF can fail with very short texts; fall back to most-recent
        logger.warning("TF-IDF similarity failed (%s) — using most-recent examples", exc)
        return examples[-FEW_SHOT_TOP_K:]


def submit_correction(
    extraction_id: int,
    original_text: str,
    corrected_json: dict,
    user_notes: str | None = None,
) -> None:
    """
    Handle a user correction end-to-end:
      1. Persist to SQLite (for audit trail and stats)
      2. Append to few_shot_examples.json (for future prompt injection)

    Args:
        extraction_id: The DB id of the original extraction being corrected.
        original_text: Raw PDF text that was used for the original extraction.
        corrected_json: The user-verified correct extraction.
        user_notes: Optional free-text note from the user.
    """
    save_correction(
        extraction_id=extraction_id,
        corrected_json=corrected_json,
        user_notes=user_notes,
    )
    save_example(input_text=original_text, corrected_json=corrected_json)
    logger.info(
        "Correction submitted for extraction_id=%d — will be used in future prompts",
        extraction_id,
    )
