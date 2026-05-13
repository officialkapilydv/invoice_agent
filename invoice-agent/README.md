# Invoice AI Agent

A production-ready Python agent that extracts structured JSON from invoice PDFs using Groq's LLM API, learns from human corrections, and handles both digital and scanned documents.

---

## Why These Tech Choices

| Choice | Reason |
|--------|--------|
| **Groq** over OpenAI | 5-10× faster inference via LPU hardware; critical for high-volume invoice processing |
| **llama-3.3-70b-versatile** over 8b | Better structured extraction accuracy; latency difference is <300 ms — worth it for correctness |
| **pdfplumber** + OCR fallback | Digital extraction is lossless and 20× faster; OCR only fires when text layer is absent or too short |
| **ChromaDB** for few-shot memory | Semantic similarity retrieval finds the most relevant past corrections at inference time |
| **Pattern Library** for learned rules | Auto-generates generalised extraction rules from corrections; rules persist across restarts |
| **Pydantic v2** | Schema enforcement with partial parsing — never crashes on incomplete LLM output |
| **FastAPI** for the web UI | Thin async wrapper; zero changes to CLI or extraction pipeline |

---

## Setup

### 1. Prerequisites

**Python 3.10+** required.

**Tesseract OCR (Windows)**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer — default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Update `TESSERACT_PATH` in your `.env` if you installed elsewhere

**Poppler (Windows — required by pdf2image)**
1. Download the latest release from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to a folder, e.g. `C:\poppler\`
3. Set `POPPLER_PATH=C:\poppler\Library\bin` in your `.env`

**Verify Tesseract works:**
```
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

### 2. Clone and Install

```bash
cd invoice-agent
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
# Edit .env with your Groq API key and paths
```

Get a free Groq API key at: https://console.groq.com/keys

---

## Running the CLI

```bash
# Extract from a PDF (full Invoice output)
python main.py extract data/sample_invoices/your_invoice.pdf

# Extract and get simplified output (product items only, no freight)
python main.py extract data/sample_invoices/your_invoice.pdf --simple

# Extract and save to database + results files
python main.py extract data/sample_invoices/your_invoice.pdf --save

# Both flags can be combined
python main.py extract data/sample_invoices/your_invoice.pdf --simple --save

# Submit a human correction (triggers learning)
python main.py correct <extraction_id> corrections/your_corrected.json

# View statistics and learned rules
python main.py stats

# Look up an extraction ID by filename
python find_id.py <filename_stem>
```

---

## Running the Web UI

```bash
uvicorn api.main:app --reload
# Open http://127.0.0.1:8000 in your browser
```

**Features:**
- Drag-and-drop PDF upload with live extraction results
- Toggle between full and simplified output
- Syntax-highlighted JSON viewer
- Dashboard with extraction stats, correction count, and learned rules
- Saved-path grid showing where results were written

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/extract` | Upload PDF, run agent, return JSON |
| `GET` | `/api/extractions` | List all past extractions |
| `GET` | `/api/extractions/{id}` | Single extraction (`?simple=true` for simplified view) |
| `POST` | `/api/correct/{id}` | Submit a correction for an extraction |
| `GET` | `/api/stats` | Dashboard stats |
| `GET` | `/api/rules` | Current learned rules |
| `GET` | `/api/uploads` | Files in the uploads folder |

---

## Output Formats

### Full Invoice (`Invoice`)

All fields extracted from the PDF: vendor/customer info, address, tax ID, phone, all line items with tax rates, subtotal, tax, discounts, grand total, payment terms, notes, confidence score, and validation warnings.

### Simplified Invoice (`SimplifiedInvoice`)

Minimal view for downstream systems (inventory, accounting):

```json
{
  "invoice_number": "280886",
  "invoice_date": "2025-01-13",
  "vendor_name": "BRYANT GRAIN COMPANY, INC.",
  "customer_name": "LAST CHANCE FEED & MORE",
  "line_items": [
    {
      "description": "CORN CHOPS",
      "item_number": "2005",
      "quantity": 120.0,
      "unit_price": 8.75
    }
  ],
  "grand_total": 5860.60
}
```

Key differences from the full format:
- **Freight / shipping / fuel / handling items are excluded** — only product lines appear
- **`grand_total` is recalculated** as `sum(quantity × unit_price)` for product items only — the PDF's grand total is ignored to guarantee math self-consistency
- **`item_number`** is parsed from the description string (e.g. `"(Item: 2005)"`) into its own field
- Addresses, tax IDs, payment terms, notes, warnings, and metadata are omitted

**Bulk conversion** — regenerate all simplified files from the database in one step:

```bash
python convert_all_to_simple.py
```

---

## How the Learning System Works

The agent has two complementary learning layers that activate without any manual configuration after you submit corrections.

### Layer 1 — ChromaDB Semantic Memory

When you submit a correction with `python main.py correct`:

1. The corrected JSON is embedded and stored in ChromaDB
2. On the next extraction, the agent queries ChromaDB for the 3 most semantically similar past invoices
3. Corrections are prioritised over raw extractions in retrieval
4. Those examples are injected into the LLM prompt as few-shot demonstrations: *"Here are 3 similar invoices — extract the new one consistently"*

### Layer 2 — Pattern Library (Learned Rules)

After each correction, the agent compares the original extraction to the corrected version and asks the LLM to generate a generalised rule explaining the mistake. Rules are stored in `data/learned_rules.json` and injected into every future prompt. Current rules cover:

- **Shipped vs. Ordered quantity** — always use the shipped column, never ordered
- **Multi-code item numbers** — capture primary + secondary codes as `PRIMARY / SECONDARY`
- **Item identifier column synonyms** — recognise Product Number, Material, SKU, Part Number, Catalog Number, etc.
- **Freight as a separate line item** — always include shipping/freight in line_items
- **Vendor name from logo/footer** — verify against full PDF text, not just header
- **Truncated descriptions** — restore missing pack sizes and product details
- **Vendor-specific rules** — e.g. Bryant Grain invoice number parsing, ADM pallet + ACH discounts

Rules accumulate over time. You can also edit `data/learned_rules.json` directly to add or tune rules manually.

### What This Is NOT (True Fine-Tuning)

True fine-tuning updates model weights permanently. This system shows the model examples and rules at inference time — when the call ends the model "forgets" them, but they are re-injected on the next call from the database. For most invoice workloads with a few hundred unique vendors, this approach gives ~80% of fine-tuning accuracy at 0 training cost and instant deployment.

---

## Extraction Rules (System Prompt)

The system prompt enforces several hard rules on every extraction:

1. Never invent data — use `null` for absent fields
2. Dates always in `YYYY-MM-DD` format
3. Monetary values as plain numbers (no symbols or commas)
4. **Shipped quantity** — when Ordered and Shipped columns both exist, always use Shipped (aliases: Shp, Delivered, Actual). Never use Ordered/Ord/Requested
5. **Item numbers in description** — embed in format `"<name> (Item: <code>)"` so the simplify layer can parse them
6. **16 column label synonyms** for item identifiers (Item #, Product Number, Material, SKU, Part No., etc.)
7. **Multi-code items** — when a secondary code appears on the next row, combine as `(Item: PRIMARY / SECONDARY)`

---

## Project Structure

```
invoice-agent/
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app — 7 endpoints
│   └── static/
│       ├── index.html            # Single-page UI
│       ├── style.css
│       └── script.js
│
├── config/
│   └── settings.py               # All config — reads from .env
│
├── src/
│   ├── agent.py                  # Orchestrator — calls all modules
│   ├── pdf_extractor.py          # pdfplumber → OCR fallback
│   ├── llm_client.py             # Groq SDK wrapper + retry
│   ├── schema.py                 # Invoice, SimplifiedInvoice, simplify_invoice()
│   ├── prompts.py                # System prompt + dynamic user prompt builder
│   ├── validator.py              # Math, date, and field checks
│   ├── database.py               # SQLite: extractions + corrections tables
│   └── memory/
│       ├── vector_store.py       # ChromaDB embeddings + retrieval
│       ├── pattern_library.py    # Rule generation + storage
│       └── feedback_loop.py      # Wires corrections → ChromaDB + rules
│
├── data/
│   ├── chroma_db/                # Vector embeddings (auto-managed)
│   ├── extractions.db            # SQLite (auto-created on first run)
│   ├── learned_rules.json        # Accumulated extraction rules
│   ├── uploads/                  # PDFs uploaded via web UI
│   └── sample_invoices/          # Test PDFs
│
├── corrections/                  # Gold-standard human-corrected JSON files
│
├── results/                      # Full Invoice JSON (one file per extraction)
│
├── results_simple/               # SimplifiedInvoice JSON (one file per extraction)
│
├── tests/
│   └── test_extraction.py        # 33 tests — no API key or PDF needed
│
├── main.py                       # CLI (click)
├── convert_all_to_simple.py      # Bulk convert all extractions to simplified format
├── find_id.py                    # Look up extraction ID by filename
├── demo_simplify.py              # SimplifiedInvoice demonstration
├── batch_review.py               # Quality triage across all extractions
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pytest tests/ -v
```

33 tests — all mock the Groq API and PDF extractor. No real PDF or API key needed. Runs in <5 seconds.

Test coverage includes:
- Basic field extraction and Pydantic validation
- OCR fallback path
- Few-shot example injection
- ChromaDB correction retrieval
- Pattern Library rule generation
- Freight / shipping / fuel / handling removal from simplified output
- Grand total recalculation from product items only
- Multi-code item number extraction (`PRIMARY / SECONDARY`)
- Shipped vs. Ordered quantity enforcement
- Item number column label synonyms (Product Number, Material, SKU, etc.)

---

## Troubleshooting

**`TesseractNotFoundError`** — Tesseract not installed or wrong path in `.env`

**`PDFInfoNotInstalledError`** — Poppler missing; set `POPPLER_PATH` in `.env`

**`GROQ_API_KEY is not set`** — Copy `.env.example` to `.env` and add your key

**OCR text is garbled** — Increase scan DPI (edit `dpi=300` in `pdf_extractor.py`) or pre-process the image (deskew, denoise)

**LLM returns wrong fields** — Submit a correction via `python main.py correct <id> corrected.json`; the agent learns from it immediately and generates a rule to prevent the same mistake on future invoices

**Port 8000 already in use** — Kill the existing process or start uvicorn on a different port: `uvicorn api.main:app --port 8001`
