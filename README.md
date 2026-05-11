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


📊 Complete File Structure After UI Build

invoice-agent/
├── api/                              ← NEW
│   ├── main.py                       ← FastAPI app
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── data/
│   ├── uploads/                      ← NEW! User-uploaded PDFs
│   │   └── 20260507_113445_Bryant.pdf
│   ├── sample_invoices/              ← Original 27 PDFs (untouched)
│   ├── chroma_db/                    ← Vector memory (untouched)
│   ├── extractions.db                ← SQLite (gets new entries)
│   └── learned_rules.json            ← Pattern Library (untouched)
│
├── results/                          ← Full JSON outputs
│   ├── Bryant.json                   ← UI uploads + CLI extractions
│   ├── Cargil.json
│   └── ... (27+ files, growing)
│
├── results_simple/                   ← Simplified JSON outputs
│   ├── Bryant.json
│   ├── Cargil.json
│   └── ... (27+ files, growing)
│
├── corrections/                      ← Gold standard corrections
│   └── 10 corrected JSON files
│
├── src/                              ← UNTOUCHED!
├── tests/                            ← UNTOUCHED!
├── main.py                           ← CLI (still works)
└── ... (other files)



📋 **COMPLETE COMMANDS REFERENCE — Invoice Extraction Agent**
Bahut accha idea, bhai! 🎯 Yeh production engineering ka core habit hai — documentation maintain karna.
Chaliye saari commands organize karke deta hu jo aapne iss project mein use ki hain.

🎯 Quick Navigation

Setup & Installation
Server Commands (API + UI)
CLI Extraction Commands
Correction Commands
Stats & Monitoring
Helper Scripts
Testing
Database & ChromaDB
Useful Utilities


1. Setup & Installation
Project Directory
powershellcd C:\Users\lenovo\Desktop\PDF_AiAgent\invoice-agent
Install Dependencies
powershell# All dependencies
pip install -r requirements.txt

# Manually install API dependencies (already done)
pip install fastapi "uvicorn[standard]" python-multipart --prefer-binary

2. Server Commands
🚀 Start API + UI Server
powershell# Standard mode
python -m uvicorn api.main:app --port 8000

# With auto-reload (development)
python -m uvicorn api.main:app --port 8000 --reload

# Access on local network (other devices)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
🌐 Access URLs
Local:           http://localhost:8000
API Docs:        http://localhost:8000/docs  (FastAPI auto-generated!)
Stats Endpoint:  http://localhost:8000/api/stats
Rules:           http://localhost:8000/api/rules
🛑 Stop Server
powershell# In the terminal where server is running:
Ctrl + C

# Or kill all uvicorn processes:
Get-Process | Where-Object { $_.ProcessName -like "*uvicorn*" } | Stop-Process -Force

3. CLI Extraction
📄 Single PDF Extraction
powershell# Basic extraction (full format)
python main.py extract data/sample_invoices/Cargil.pdf

# Simplified output (only essential fields)
python main.py extract data/sample_invoices/Cargil.pdf --simple

# Save JSON to results/ folder
python main.py extract data/sample_invoices/Cargil.pdf --save

# Combine both
python main.py extract data/sample_invoices/Cargil.pdf --simple --save

# Files with spaces in name (use quotes)
python main.py extract "data/sample_invoices/Lyssy & Eckel, Inc..pdf" --simple
python main.py extract "data/sample_invoices/Two Bulls.pdf" --simple
python main.py extract "data/sample_invoices/L&H Branding Irons.pdf" --simple
📦 Batch Extraction (Multiple PDFs)
powershell# Process entire folder
python main.py batch data/sample_invoices

# With simplified output
python main.py batch data/sample_invoices --simple

# Save all to disk
python main.py batch data/sample_invoices --save

# Combined
python main.py batch data/sample_invoices --simple --save

4. Correction Commands
✏️ Submit Correction
powershell# Basic correction
python main.py correct <extraction_id> corrections/<filename>_corrected.json

# With notes (recommended!)
python main.py correct 24 corrections/Bryant_corrected.json --notes "Fixed item codes from PRODUCT NUMBER column"

# Examples of what we used:
python main.py correct 1 corrections/Bryant_corrected.json --notes "Tabular OCR fix"
python main.py correct 5 corrections/Cargil_corrected.json --notes "Math hallucination fix"
python main.py correct 18 corrections/Purina_corrected.json --notes "Vendor name fix"
🔍 Find Extraction ID
powershell# Find ID by filename
python find_id.py Bryant
python find_id.py Cargil
python find_id.py "Lyssy"

5. Stats & Monitoring
📊 View System Stats
powershell# Full stats (extractions, corrections, rules, error rates)
python main.py stats
Output shows:

Total extractions, corrections
Vector DB counts
Learned rules count
Most-corrected fields table
All learned rules with confidence

🔍 Check Specific Data
powershell# View learned rules
type data/learned_rules.json

# Check rules count
python -c "import json; rules = json.loads(open('data/learned_rules.json').read())['rules']; print('Total rules:', len(rules))"

# Better formatted view
python -c "import json; rules = json.loads(open('data/learned_rules.json').read())['rules']; [print(f'{i+1}.', r['rule'][:80]) for i, r in enumerate(rules)]"

6. Helper Scripts
🔧 Built-in Helper Scripts
powershell# Find extraction IDs by filename
python find_id.py <search_term>

# Triage all extractions for quality issues
python batch_review.py

# One-time migration (already done — don't re-run unless needed)
python migrate_to_chroma.py

# Convert all to simplified format
python convert_all_to_simple.py

# Demo simplification on Purina
python demo_simplify.py

# Pre-flight check (rules + corrections that will inject)
python debug_extraction.py
📁 What Each Script Does
ScriptPurposefind_id.pyLookup extraction ID by filenamebatch_review.pyQuality triage of all 27 extractionsconvert_all_to_simple.pyBulk convert to simplified formatdemo_simplify.pyDemo simplification on Purinamigrate_to_chroma.pyOne-time migration (already done)debug_extraction.pyPre-flight check before extraction

7. Testing
🧪 Run Tests
powershell# Run all tests with verbose output
python -m pytest tests/ -v

# Run with short tracebacks (cleaner errors)
python -m pytest tests/ -v --tb=short

# Run only last 15 lines (summary)
python -m pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 15

# Run specific test class
python -m pytest tests/test_extraction.py::TestSimplifiedInvoice -v

# Run specific test
python -m pytest tests/test_extraction.py::TestSimplifiedInvoice::test_basic_conversion -v
Expected Output
======================== 20 passed, 1 warning in 4.21s ========================

8. Database & ChromaDB
🗄️ SQLite Database Inspection
powershell# Quick stats from SQLite
python -c "import sqlite3; conn = sqlite3.connect('data/extractions.db'); print('Extractions:', conn.execute('SELECT COUNT(*) FROM extractions').fetchone()[0]); print('Corrections:', conn.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]); conn.close()"

# List all extractions with IDs and filenames
python -c "import sqlite3; conn = sqlite3.connect('data/extractions.db'); [print(r[0], r[1], 'conf=' + str(r[2])) for r in conn.execute('SELECT id, pdf_filename, confidence FROM extractions ORDER BY id').fetchall()]; conn.close()"

# Check table schema
python -c "import sqlite3; conn = sqlite3.connect('data/extractions.db'); [print(r) for r in conn.execute('PRAGMA table_info(extractions)').fetchall()]; conn.close()"

# View specific extraction (replace 24 with your ID)
python -c "import sqlite3, json; conn = sqlite3.connect('data/extractions.db'); row = conn.execute('SELECT extracted_json FROM extractions WHERE id=24').fetchone(); print(json.dumps(json.loads(row[0]), indent=2)) if row else print('Not found'); conn.close()"
🧠 ChromaDB Inspection
powershell# Check ChromaDB stats
python -c "import sys; sys.path.insert(0, '.'); from src.memory.vector_store import VectorStore; vs = VectorStore(); print('Stats:', vs.get_stats())"

9. Useful Utilities
📂 File Operations
powershell# List all PDFs in sample folder
Get-ChildItem data/sample_invoices/ -Filter "*.pdf" -Name

# List uploaded PDFs
Get-ChildItem data/uploads/ -Filter "*.pdf" -Name

# Count files in results/
(Get-ChildItem results/ -Filter "*.json").Count

# Count files in results_simple/
(Get-ChildItem results_simple/ -Filter "*.json").Count

# List all corrections
Get-ChildItem corrections/ -Filter "*.json" -Name
🔍 View JSON Files
powershell# View a result file
type results/Bryant.json

# View a corrected file
type corrections/Bryant_corrected.json

# Pretty-print JSON
python -c "import json; print(json.dumps(json.load(open('results/Bryant.json')), indent=2))"

# View specific field
python -c "import json; data = json.load(open('results/Bryant.json')); print(data.get('invoice_number'), '|', data.get('grand_total'))"
📊 API Testing (curl-style with PowerShell)
powershell# Get stats
Invoke-RestMethod -Uri "http://localhost:8000/api/stats" | ConvertTo-Json

# Get rules
Invoke-RestMethod -Uri "http://localhost:8000/api/rules" | ConvertTo-Json -Depth 3

# List all extractions
Invoke-RestMethod -Uri "http://localhost:8000/api/extractions" | ConvertTo-Json -Depth 2

# Get specific extraction (full)
Invoke-RestMethod -Uri "http://localhost:8000/api/extractions/24" | ConvertTo-Json -Depth 4

# Get specific extraction (simplified)
Invoke-RestMethod -Uri "http://localhost:8000/api/extractions/24?simple=true" | ConvertTo-Json -Depth 4

# Get list of uploaded files
Invoke-RestMethod -Uri "http://localhost:8000/api/uploads" | ConvertTo-Json

🔥 Most-Used Commands (Daily Workflow)
Bhai, yeh top 10 commands aap daily use karoge:
powershell# 1. Start server
python -m uvicorn api.main:app --port 8000 --reload

# 2. Check stats
python main.py stats

# 3. Extract single PDF (full)
python main.py extract data/sample_invoices/<file>.pdf

# 4. Extract single PDF (simplified)
python main.py extract data/sample_invoices/<file>.pdf --simple

# 5. Find extraction ID
python find_id.py <filename>

# 6. Submit correction
python main.py correct <id> corrections/<file>_corrected.json --notes "..."

# 7. Run tests
python -m pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 10

# 8. View learned rules
type data/learned_rules.json

# 9. Bulk convert to simplified
python convert_all_to_simple.py

# 10. Check uploaded files
Get-ChildItem data/uploads/ -Filter "*.pdf" -Name

📁 Important File Paths Reference
PROJECT ROOT: C:\Users\lenovo\Desktop\PDF_AiAgent\invoice-agent\

📁 Source Code:
   src/agent.py                  - Main agent
   src/schema.py                 - Pydantic models
   src/prompts.py                - LLM prompts
   src/validator.py              - Math validation
   src/database.py               - SQLite
   src/memory/vector_store.py    - ChromaDB
   src/memory/pattern_library.py - Auto-rules
   src/memory/feedback_loop.py   - Corrections handler

📁 API + UI:
   api/main.py                   - FastAPI app
   api/static/index.html         - Main UI
   api/static/style.css          - Styles
   api/static/script.js          - Frontend logic

📁 Data:
   data/extractions.db           - SQLite database
   data/chroma_db/               - Vector embeddings
   data/learned_rules.json       - Pattern Library
   data/sample_invoices/         - 27 original PDFs
   data/uploads/                 - User-uploaded PDFs

📁 Outputs:
   results/                      - Full JSON files
   results_simple/               - Simplified JSON files
   corrections/                  - 10 gold-standard corrections

📁 Tests:
   tests/test_extraction.py      - 20 test cases

📜 Helper Scripts:
   main.py                       - CLI entrypoint
   find_id.py                    - ID lookup
   batch_review.py               - Quality triage
   convert_all_to_simple.py      - Bulk converter
   demo_simplify.py              - Demo script
   migrate_to_chroma.py          - One-time migration
   debug_extraction.py           - Pre-flight check

🎯 Common Workflows
Workflow 1: Add New Invoice & Get Data
powershell# Option A: Via Web UI (Recommended)
# 1. Start server
python -m uvicorn api.main:app --port 8000 --reload

# 2. Open browser
# Navigate to http://localhost:8000

# 3. Drag-drop PDF, toggle simplified, click Extract

# Option B: Via CLI
python main.py extract data/sample_invoices/NewInvoice.pdf --simple --save
Workflow 2: Correct an Extraction
powershell# 1. Find extraction ID
python find_id.py Bryant

# 2. View current extraction
python -c "import sqlite3, json; conn = sqlite3.connect('data/extractions.db'); row = conn.execute('SELECT extracted_json FROM extractions WHERE id=24').fetchone(); print(json.dumps(json.loads(row[0]), indent=2)); conn.close()"

# 3. Create/edit corrections/Bryant_corrected.json

# 4. Submit correction
python main.py correct 24 corrections/Bryant_corrected.json --notes "Fixed OCR errors and item codes"

# 5. Verify
python main.py stats
Workflow 3: After Schema Changes
powershell# 1. Run tests
python -m pytest tests/ -v

# 2. Bulk refresh simplified outputs
python convert_all_to_simple.py

# 3. Verify stats unchanged
python main.py stats
Workflow 4: Debug Issue
powershell# 1. Pre-flight check
python debug_extraction.py

# 2. Run quality review
python batch_review.py

# 3. View specific extraction
python -c "import sqlite3, json; conn = sqlite3.connect('data/extractions.db'); row = conn.execute('SELECT extracted_json FROM extractions WHERE id=24').fetchone(); print(json.dumps(json.loads(row[0]), indent=2))"

📝 Save This Reference!
Bhai, iss reference ko save karo:
Option 1: Save as Markdown File
powershell# Create reference doc in project
notepad COMMANDS.md
# Paste this entire reference
Option 2: Save in OneNote/Notion
Copy-paste in your notes app for quick access.
Option 3: Add to README
powershellnotepad README.md
# Add a "Commands Reference" section

🌟 Pro Tips
1. Use PowerShell History
↑ Arrow key — previous command
↓ Arrow key — next command
Ctrl+R — search history
2. Tab Completion
python main.py ext<TAB>  → completes to "extract"
data/sa<TAB>             → completes to "data/sample_invoices/"
3. Multiple Terminals
- Terminal 1: API server (don't close!)
- Terminal 2: CLI commands
- Terminal 3: Testing/inspection
4. PowerShell vs Command Prompt
PowerShell: More powerful (use this!)
Command Prompt: Limited

Always use PowerShell.
5. File Path Tricks
powershell# Use Tab for completion
python main.py extract data/sample_invoices/<TAB>

# Use forward slashes (Python doesn't care)
data/sample_invoices/Cargil.pdf

# Use quotes for spaces
"data/sample_invoices/Two Bulls.pdf"

💡 Quick Cheat Sheet (Most Common)
powershell# ╔══════════════════════════════════════════════════╗
# ║         DAILY WORKFLOW CHEAT SHEET                ║
# ╚══════════════════════════════════════════════════╝

# Start server
python -m uvicorn api.main:app --port 8000 --reload

# Check stats
python main.py stats

# Extract PDF (simplified)
python main.py extract data/sample_invoices/<file>.pdf --simple

# Find ID
python find_id.py <name>

# Submit correction  
python main.py correct <id> corrections/<file>_corrected.json --notes "..."

# Run tests
python -m pytest tests/ -v --tb=short

# View rules
type data/learned_rules.json

# Bulk refresh
python convert_all_to_simple.py
