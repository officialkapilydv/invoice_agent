# Invoice AI Agent

A production-ready Python agent that extracts structured JSON from invoice PDFs using Groq's LLM API, learns from corrections, and handles both digital and scanned documents.

---

## Why These Tech Choices

| Choice | Reason |
|--------|--------|
| **Groq** over OpenAI | 5-10× faster inference via LPU hardware; critical for high-volume invoice processing |
| **llama-3.3-70b-versatile** over 8b | Better structured extraction accuracy; the latency difference is <300ms — worth it for correctness |
| **pdfplumber** + OCR fallback | Digital extraction is lossless and 20× faster; OCR only fires when needed |
| **Pydantic v2** | Schema enforcement with partial parsing — never crashes on incomplete LLM output |
| **RAG few-shot** over fine-tuning | See "How Learning Works" section below |

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

### 4. Run

```bash
# Extract from a PDF
python main.py extract data/sample_invoices/your_invoice.pdf

# Extract and save to database
python main.py extract data/sample_invoices/your_invoice.pdf --save

# Submit a correction (for learning)
python main.py correct 1 corrected.json

# View statistics
python main.py stats
```

---

## How the "Learning" Works

### What it is (RAG-style few-shot learning)

When you correct an extraction using `python main.py correct`:

1. The corrected JSON is saved to `data/few_shot_examples.json`
2. On the **next** extraction, the agent finds the 3 most similar past corrections using TF-IDF cosine similarity
3. Those examples are injected directly into the LLM prompt: *"Here are 3 similar invoices where corrections were made — extract the new one consistently"*

The model sees the right patterns at inference time. It never sees wrong patterns again.

### What it is NOT (true fine-tuning)

True fine-tuning would update the model's weights permanently — the knowledge would be baked in. What we're doing is showing the model examples at runtime. When the conversation ends, the model "forgets" them. The next call must re-inject the examples from our file.

**Practical impact:**
- RAG few-shot: ~80% of fine-tuning accuracy gain, 0 training cost, instant deployment
- Fine-tuning: higher ceiling, but requires GPU hours, training pipelines, and model management

For most invoice workloads with a few hundred unique vendors, few-shot is the right call.

### Upgrading to True Fine-Tuning (Roadmap)

When you've collected 200+ corrections and need higher accuracy:

| Option | When to Use |
|--------|-------------|
| **HuggingFace + LoRA** | Best control; use `peft` library with `llama-3` base; needs a GPU (Colab works) |
| **Together.ai fine-tuning** | Easiest cloud option; supports Llama models; pay-per-token |
| **OpenAI fine-tuning** | Most reliable API; best for GPT-3.5-turbo; data stays on OpenAI servers |
| **LayoutLMv3 / Donut** | Multimodal — reads PDF layout + text together; best for complex table invoices |

**Note:** Groq does not currently offer fine-tuning. You would fine-tune on HuggingFace/Together.ai and then serve via their API (or self-host).

---

## Project Structure

```
invoice-agent/
├── config/settings.py        # All config in one place — imports from .env
├── src/
│   ├── agent.py              # Orchestrator — calls all other modules
│   ├── pdf_extractor.py      # pdfplumber → OCR fallback
│   ├── llm_client.py         # Groq SDK wrapper + retry logic
│   ├── schema.py             # Pydantic Invoice model
│   ├── prompts.py            # System prompt + dynamic user prompt builder
│   ├── validator.py          # Math checks, date checks, warnings
│   ├── feedback.py           # Few-shot example store + TF-IDF retrieval
│   └── database.py           # SQLite: extractions + corrections
├── data/
│   ├── few_shot_examples.json  # Grows as you submit corrections
│   └── extractions.db          # Auto-created on first run
├── tests/test_extraction.py    # pytest suite (no API key needed)
└── main.py                     # CLI (click)
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock the Groq API and PDF extractor — no real PDF or API key needed. They run in <2 seconds.

---

## Troubleshooting

**`TesseractNotFoundError`** — Tesseract not installed or wrong path in `.env`

**`PDFInfoNotInstalledError`** — Poppler missing; set `POPPLER_PATH` in `.env`

**`GROQ_API_KEY is not set`** — Copy `.env.example` to `.env` and add your key

**OCR text is garbled** — Increase scan DPI (edit `dpi=300` in `pdf_extractor.py`) or pre-process the image (deskew, denoise)

**LLM returns wrong fields** — Submit a correction via `python main.py correct <id> corrected.json` and the agent will learn from it



## PROJECT FINAL STATE:

📁 invoice-agent/
├── 📁 src/                          (Production code)
│   ├── agent.py                     (Self-improving agent)
│   ├── schema.py                    (Invoice + SimplifiedInvoice)
│   ├── prompts.py                   (With shipped qty rule)
│   ├── memory/
│   │   ├── vector_store.py          (ChromaDB)
│   │   ├── pattern_library.py       (7 auto-rules)
│   │   └── feedback_loop.py
│   └── ... (other modules)
│
├── 📁 data/
│   ├── chroma_db/                   (Vector embeddings)
│   ├── extractions.db               (SQLite)
│   └── learned_rules.json           (7 auto-generated rules)
│
├── 📁 corrections/                   (10 gold-standard files)
│   └── 10 corrected JSON files
│
├── 📁 results/                       (Original 27 extractions)
│   └── 27 full-format JSON files
│
├── 📁 results_simple/                ← NEW! Just created!
│   └── 27 simplified JSON files     (per business requirement)
│
├── 📁 tests/                         (20 tests, all passing)
│
├── 📜 main.py                        (CLI with --simple flag)
├── 📜 batch_review.py                (Quality triage tool)
├── 📜 find_id.py                     (ID lookup helper)
├── 📜 demo_simplify.py               (Demonstration)
├── 📜 convert_all_to_simple.py       ← NEW! Bulk converter
├── 📜 migrate_to_chroma.py
└── 📜 README.md



# Invoice Extraction Agent — Complete Commands Reference

> A complete reference guide for setup, extraction, correction workflows, testing, monitoring, and utilities used in the Invoice Extraction Agent project.

---

# 📚 Table of Contents

- [Setup & Installation](#1-setup--installation)
- [Server Commands](#2-server-commands)
- [CLI Extraction Commands](#3-cli-extraction)
- [Correction Commands](#4-correction-commands)
- [Stats & Monitoring](#5-stats--monitoring)
- [Helper Scripts](#6-helper-scripts)
- [Testing](#7-testing)
- [Database & ChromaDB](#8-database--chromadb)
- [Useful Utilities](#9-useful-utilities)

---

# 1. Setup & Installation

## 📂 Project Directory

```powershell
cd C:\Users\lenovo\Desktop\PDF_AiAgent\invoice-agent
```

## 📦 Install Dependencies

```powershell
pip install -r requirements.txt

pip install fastapi "uvicorn[standard]" python-multipart --prefer-binary
```

---

# 2. Server Commands

## 🚀 Start API + UI Server

```powershell
python -m uvicorn api.main:app --port 8000
```

### Development Mode

```powershell
python -m uvicorn api.main:app --port 8000 --reload
```

### Local Network Access

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🌐 Access URLs

| Service | URL |
|---|---|
| Local UI | `http://localhost:8000` |
| API Docs | `http://localhost:8000/docs` |
| Stats Endpoint | `http://localhost:8000/api/stats` |
| Rules Endpoint | `http://localhost:8000/api/rules` |

---

# 3. CLI Extraction

## 📄 Single PDF Extraction

```powershell
python main.py extract data/sample_invoices/Cargil.pdf
```

### Simplified Output

```powershell
python main.py extract data/sample_invoices/Cargil.pdf --simple
```

### Save Output

```powershell
python main.py extract data/sample_invoices/Cargil.pdf --save
```

### Simplified + Save

```powershell
python main.py extract data/sample_invoices/Cargil.pdf --simple --save
```

---

## 📦 Batch Extraction

```powershell
python main.py batch data/sample_invoices
```

### Batch with Simplified Output

```powershell
python main.py batch data/sample_invoices --simple
```

### Batch with Save

```powershell
python main.py batch data/sample_invoices --save
```

---

# 4. Correction Commands

## ✏️ Submit Correction

```powershell
python main.py correct <extraction_id> corrections/<filename>_corrected.json
```

### Correction with Notes

```powershell
python main.py correct 24 corrections/Bryant_corrected.json --notes "Fixed item codes from PRODUCT NUMBER column"
```

## 🔍 Find Extraction ID

```powershell
python find_id.py Bryant
```

---

# 5. Stats & Monitoring

## 📊 View System Stats

```powershell
python main.py stats
```

### View Learned Rules

```powershell
type data/learned_rules.json
```

---

# 6. Helper Scripts

```powershell
python find_id.py <search_term>

python batch_review.py

python migrate_to_chroma.py

python convert_all_to_simple.py

python demo_simplify.py

python debug_extraction.py
```

---

# 7. Testing

## 🧪 Run Tests

```powershell
python -m pytest tests/ -v
```

### Cleaner Errors

```powershell
python -m pytest tests/ -v --tb=short
```

### Run Specific Test

```powershell
python -m pytest tests/test_extraction.py::TestSimplifiedInvoice::test_basic_conversion -v
```

---

# 8. Database & ChromaDB

## 🗄️ SQLite Inspection

```powershell
python -c "import sqlite3; conn = sqlite3.connect('data/extractions.db'); print('Extractions:', conn.execute('SELECT COUNT(*) FROM extractions').fetchone()[0]); conn.close()"
```

## 🧠 ChromaDB Inspection

```powershell
python -c "import sys; sys.path.insert(0, '.'); from src.memory.vector_store import VectorStore; vs = VectorStore(); print('Stats:', vs.get_stats())"
```

---

# 9. Useful Utilities

## 📂 File Operations

```powershell
Get-ChildItem data/sample_invoices/ -Filter "*.pdf" -Name

Get-ChildItem data/uploads/ -Filter "*.pdf" -Name

(Get-ChildItem results/ -Filter "*.json").Count
```

---

# 💡 Quick Cheat Sheet

```powershell
# Start server
python -m uvicorn api.main:app --port 8000 --reload

# Check stats
python main.py stats

# Extract PDF
python main.py extract data/sample_invoices/<file>.pdf --simple

# Find extraction ID
python find_id.py <name>

# Submit correction
python main.py correct <id> corrections/<file>_corrected.json --notes "..."

# Run tests
python -m pytest tests/ -v --tb=short

# View rules
type data/learned_rules.json

# Bulk refresh simplified outputs
python convert_all_to_simple.py
```
