"""
Layer 1: Semantic vector memory using ChromaDB.

WHY ChromaDB over our previous TF-IDF approach:
  TF-IDF compares word overlap. Two invoices from different vendors that use
  different terminology but have the same STRUCTURE look unrelated to TF-IDF.
  ChromaDB converts text into dense embedding vectors that capture meaning,
  so "tax invoice Mumbai" and "GST bill Maharashtra" are correctly treated
  as semantically similar.

WHY two collections:
  Not all examples are equal. A user correction is a gold-standard example —
  the human explicitly said "the model was wrong, here is the truth." We store
  corrections in a separate, high-priority collection so `find_similar` always
  returns corrections before regular extractions. This mimics how a teacher
  would emphasize "remember you made this mistake before."

Storage: ChromaDB persists to `data/chroma_db/` automatically. No server needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
_COLLECTION_EXTRACTIONS = "extractions"
_COLLECTION_CORRECTIONS = "corrections"


def _get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client, creating the directory if needed."""
    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_CHROMA_PATH))


def _embedding_fn() -> embedding_functions.EmbeddingFunction:
    """
    ChromaDB's default ONNX-backed embedder (all-MiniLM-L6-v2).
    Runs locally, no API call, no PyTorch needed.
    First call downloads the ~23 MB model to the ChromaDB cache.
    """
    return embedding_functions.DefaultEmbeddingFunction()


class VectorStore:
    """
    Semantic memory for the invoice agent.

    Maintains two ChromaDB collections:
      - corrections  (high priority: human-verified gold examples)
      - extractions  (lower priority: all successful agent outputs)

    On `find_similar`, corrections are always returned first up to k,
    only filling remaining slots from extractions.
    """

    def __init__(self) -> None:
        client = _get_client()
        ef = _embedding_fn()
        self._corrections = client.get_or_create_collection(
            name=_COLLECTION_CORRECTIONS,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._extractions = client.get_or_create_collection(
            name=_COLLECTION_EXTRACTIONS,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "VectorStore ready — corrections: %d, extractions: %d",
            self._corrections.count(),
            self._extractions.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_extraction(
        self,
        pdf_text: str,
        extracted_json: dict,
        extraction_id: int,
    ) -> None:
        """
        Store a successful extraction so future similar invoices can learn from it.

        Uses the SQLite extraction_id as the ChromaDB document id so the two
        stores stay in sync and we can cross-reference them.
        """
        doc_id = f"ext_{extraction_id}"
        # Truncate to 8k chars — ChromaDB embeds the document text, and very
        # long texts slow down embedding without adding retrieval value.
        self._extractions.upsert(
            ids=[doc_id],
            documents=[pdf_text[:8000]],
            metadatas=[{
                "extraction_id": extraction_id,
                "invoice_number": extracted_json.get("invoice_number", ""),
                "type": "extraction",
            }],
        )
        logger.debug("Added extraction id=%d to vector store", extraction_id)

    def add_correction(
        self,
        pdf_text: str,
        corrected_json: dict,
        extraction_id: int,
    ) -> None:
        """
        Store a human correction with HIGH priority.

        Also upserts into the extractions collection to replace the original
        (incorrect) embedding, so future semantic searches don't keep finding
        the wrong version.
        """
        doc_id = f"cor_{extraction_id}"
        metadata = {
            "extraction_id": extraction_id,
            "invoice_number": corrected_json.get("invoice_number", ""),
            "type": "correction",
        }
        self._corrections.upsert(
            ids=[doc_id],
            documents=[pdf_text[:8000]],
            metadatas=[metadata],
        )
        # Replace the original extraction with the corrected version
        self._extractions.upsert(
            ids=[f"ext_{extraction_id}"],
            documents=[pdf_text[:8000]],
            metadatas=[{**metadata, "type": "corrected_extraction"}],
        )
        logger.info(
            "Added correction id=%d to vector store (high priority)", extraction_id
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def find_similar(self, pdf_text: str, k: int = 3) -> list[dict]:
        """
        Retrieve the k most semantically similar past examples.

        Strategy:
          1. Query corrections collection first (gold standard)
          2. Fill remaining slots from the extractions collection,
             excluding any ids already returned by corrections
          3. Return full example dicts: {input_text, output_json, source}

        WHY corrections first: A correction means the model was wrong and a
        human fixed it. That correction carries far more signal than a regular
        extraction where we don't know if the model was right or just lucky.
        """
        results: list[dict] = []
        used_ids: set[str] = set()

        # Step 1 — corrections (gold standard)
        if self._corrections.count() > 0:
            n_corrections = min(k, self._corrections.count())
            cor_results = self._corrections.query(
                query_texts=[pdf_text[:8000]],
                n_results=n_corrections,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(
                cor_results["documents"][0], cor_results["metadatas"][0]
            ):
                eid = meta.get("extraction_id")
                used_ids.add(f"ext_{eid}")
                results.append({
                    "input_text": doc,
                    "output_json": _load_corrected_json(eid),
                    "source": "correction",
                    "extraction_id": eid,
                })

        # Step 2 — fill remaining slots from general extractions
        remaining = k - len(results)
        if remaining > 0 and self._extractions.count() > 0:
            n_ext = min(remaining + len(used_ids), self._extractions.count())
            ext_results = self._extractions.query(
                query_texts=[pdf_text[:8000]],
                n_results=n_ext,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(
                ext_results["documents"][0], ext_results["metadatas"][0]
            ):
                eid = meta.get("extraction_id")
                doc_id = f"ext_{eid}"
                if doc_id in used_ids:
                    continue
                results.append({
                    "input_text": doc,
                    "output_json": _load_extraction_json(eid),
                    "source": "extraction",
                    "extraction_id": eid,
                })
                used_ids.add(doc_id)
                if len(results) >= k:
                    break

        logger.debug(
            "find_similar returned %d examples (%d corrections, %d extractions)",
            len(results),
            sum(1 for r in results if r["source"] == "correction"),
            sum(1 for r in results if r["source"] != "correction"),
        )
        return results[:k]

    def get_stats(self) -> dict:
        """Return counts for the stats CLI command."""
        return {
            "vector_extractions": self._extractions.count(),
            "vector_corrections": self._corrections.count(),
        }


# ------------------------------------------------------------------
# Helpers — load JSON from SQLite to pair with embeddings
# ------------------------------------------------------------------

def _load_corrected_json(extraction_id: int | None) -> dict:
    """Fetch the latest corrected JSON for an extraction from SQLite."""
    if extraction_id is None:
        return {}
    try:
        import sqlite3
        from config.settings import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT corrected_json FROM corrections WHERE extraction_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (extraction_id,),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else {}
    except Exception as exc:
        logger.warning("Could not load corrected JSON for id=%s: %s", extraction_id, exc)
        return {}


def _load_extraction_json(extraction_id: int | None) -> dict:
    """Fetch the extracted JSON for a given extraction_id from SQLite."""
    if extraction_id is None:
        return {}
    try:
        import sqlite3
        from config.settings import DB_PATH
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT extracted_json FROM extractions WHERE id=?",
            (extraction_id,),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else {}
    except Exception as exc:
        logger.warning("Could not load extraction JSON for id=%s: %s", extraction_id, exc)
        return {}
