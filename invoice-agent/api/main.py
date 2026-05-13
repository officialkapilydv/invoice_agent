"""
FastAPI application — thin wrapper over the existing invoice extraction agent.

Run with:  python -m uvicorn api.main:app --reload
UI at:     http://localhost:8000

All heavy lifting is done by src/ — this module only handles HTTP concerns
(routing, request parsing, response shaping) and file I/O for uploads.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root (invoice-agent/) is on sys.path when uvicorn runs
# from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import configure_logging, validate_config
from src.database import get_extraction, get_stats, init_db, save_extraction
from src.memory.feedback_loop import get_field_accuracy_report, submit_correction
from src.memory.pattern_library import load_rules
from src.memory.vector_store import VectorStore
from src.schema import simplify_invoice

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_STATIC_DIR   = Path(__file__).resolve().parent / "static"
_UPLOADS_DIR  = _PROJECT_ROOT / "data" / "uploads"
_RESULTS_DIR  = _PROJECT_ROOT / "results"
_SIMPLE_DIR   = _PROJECT_ROOT / "results_simple"

for _d in (_UPLOADS_DIR, _RESULTS_DIR, _SIMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Invoice Extraction Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    init_db()
    logger.info("API started — uploads: %s", _UPLOADS_DIR)


# Serve static files (JS, CSS, etc.) — must be mounted BEFORE the root route
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _vs_stats() -> dict:
    try:
        return VectorStore().get_stats()
    except Exception:
        return {"vector_extractions": "n/a", "vector_corrections": "n/a"}


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def api_stats() -> dict:
    """Aggregate statistics: extractions, corrections, rules, confidence."""
    db      = get_stats()
    accuracy = get_field_accuracy_report()
    rules   = load_rules()
    vs      = _vs_stats()

    return {
        "total_extractions":    db["total_extractions"],
        "total_corrections":    db["total_corrections"],
        "vector_db_extractions": vs["vector_extractions"],
        "vector_db_corrections": vs["vector_corrections"],
        "learned_rules":        len(rules),
        "avg_confidence":       db["avg_confidence"],
        "field_error_counts":   accuracy.get("field_error_counts", {}),
    }


# ---------------------------------------------------------------------------
# GET /api/rules
# ---------------------------------------------------------------------------

@app.get("/api/rules")
async def api_rules() -> list:
    """Return all learned rules from the pattern library."""
    return load_rules()


# ---------------------------------------------------------------------------
# GET /api/extractions
# ---------------------------------------------------------------------------

@app.get("/api/extractions")
async def api_list_extractions() -> list:
    """Lightweight list of all extractions (id, filename, date, confidence)."""
    import sqlite3
    from config.settings import DB_PATH

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, pdf_filename, confidence, used_ocr, created_at "
            "FROM extractions ORDER BY id DESC"
        ).fetchall()

    return [
        {
            "id":         r["id"],
            "filename":   r["pdf_filename"],
            "confidence": r["confidence"],
            "used_ocr":   bool(r["used_ocr"]),
            "date":       r["created_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /api/extractions/{id}
# ---------------------------------------------------------------------------

@app.get("/api/extractions/{extraction_id}")
async def api_get_extraction(
    extraction_id: int,
    simple: bool = Query(default=False),
) -> dict:
    """Fetch a single extraction by id. Use ?simple=true for simplified view."""
    from src.schema import Invoice

    row = get_extraction(extraction_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Extraction {extraction_id} not found")

    if simple:
        invoice = Invoice.model_validate(row["extracted_json"])
        return simplify_invoice(invoice).model_dump(mode="json")

    return row["extracted_json"]


# ---------------------------------------------------------------------------
# GET /api/uploads
# ---------------------------------------------------------------------------

@app.get("/api/uploads")
async def api_uploads() -> list:
    """List files in data/uploads/ (PDFs uploaded via the UI)."""
    files = sorted(_UPLOADS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size_kb":  round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        for f in files
    ]


# ---------------------------------------------------------------------------
# POST /api/extract
# ---------------------------------------------------------------------------

@app.post("/api/extract")
async def api_extract(
    file: UploadFile = File(...),
    simple: bool = Query(default=False),
) -> dict:
    """
    Upload a PDF, run the extraction pipeline, and return structured JSON.

    The PDF is saved to data/uploads/<timestamp>_<filename>.pdf.
    Full JSON is always written to results/<stem>.json.
    Simplified JSON is written to results_simple/<stem>.json when simple=true.
    The extraction is always saved to SQLite + ChromaDB.
    """
    from src.agent import InvoiceAgent

    validate_config()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file with timestamp prefix
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).name          # strip any path components
    upload_filename = f"{ts}_{safe_name}"
    upload_path = _UPLOADS_DIR / upload_filename

    content = await file.read()
    upload_path.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", upload_path, len(content))

    # Run extraction
    try:
        agent = InvoiceAgent()
        result = agent.run(str(upload_path))
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", upload_filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    invoice      = result.invoice
    invoice_dict = invoice.model_dump(mode="json")
    stem         = Path(safe_name).stem

    # Persist to SQLite
    extraction_id = save_extraction(
        pdf_filename=safe_name,
        extracted_json=invoice_dict,
        confidence=invoice.confidence_score,
        used_ocr=result.extraction.used_ocr,
        model=result.llm_response.model,
        prompt_tokens=result.llm_response.prompt_tokens,
        completion_tokens=result.llm_response.completion_tokens,
    )

    # Write full JSON result file
    full_json_path = _RESULTS_DIR / f"{stem}.json"
    full_json_path.write_text(
        json.dumps(invoice_dict, indent=2, default=str), encoding="utf-8"
    )

    # Write simplified JSON if requested
    simple_json_path: Path | None = None
    simplified_dict: dict | None  = None
    if simple:
        simplified     = simplify_invoice(invoice)
        simplified_dict = simplified.model_dump(mode="json")
        simple_json_path = _SIMPLE_DIR / f"{stem}.json"
        simple_json_path.write_text(
            json.dumps(simplified_dict, indent=2, default=str), encoding="utf-8"
        )

    # Build response
    saved_paths: dict = {
        "pdf":            str(upload_path.relative_to(_PROJECT_ROOT)),
        "full_json":      str(full_json_path.relative_to(_PROJECT_ROOT)),
        "database":       f"extractions.db (id={extraction_id})",
        "vector_memory":  "ChromaDB (extractions collection)",
    }
    if simple_json_path:
        saved_paths["simple_json"] = str(simple_json_path.relative_to(_PROJECT_ROOT))

    return {
        "extraction_id": extraction_id,
        "saved_paths":   saved_paths,
        "invoice":       simplified_dict if simple else invoice_dict,
        "validation_warnings": [
            {"field": w.field, "message": w.message}
            for w in invoice.validation_warnings
        ],
        "confidence": invoice.confidence_score,
    }


# ---------------------------------------------------------------------------
# POST /api/correct/{extraction_id}
# ---------------------------------------------------------------------------

class CorrectionRequest(BaseModel):
    corrected_json: dict
    notes: str | None = None


@app.post("/api/correct/{extraction_id}")
async def api_correct(extraction_id: int, body: CorrectionRequest) -> dict:
    """Submit a human correction for an existing extraction."""
    try:
        result = submit_correction(
            extraction_id=extraction_id,
            corrected_json=body.corrected_json,
            user_notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "success":                    True,
        "changed_fields":             result.get("changed_fields", []),
        "pattern_extraction_triggered": result.get("pattern_extraction_triggered", False),
    }
