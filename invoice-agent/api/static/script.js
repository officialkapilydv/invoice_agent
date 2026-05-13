'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let currentFile = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const selectedFile  = document.getElementById('selected-file');
const simpleToggle  = document.getElementById('simple-toggle');
const extractBtn    = document.getElementById('extract-btn');
const statusBar     = document.getElementById('status-bar');
const resultSection = document.getElementById('result-section');
const resultTitle   = document.getElementById('result-title');
const confidenceBadge = document.getElementById('confidence-badge');
const jsonOutput    = document.getElementById('json-output');
const warningsSection = document.getElementById('warnings-section');
const warningsList  = document.getElementById('warnings-list');
const savedPaths    = document.getElementById('saved-paths');

// ─── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const data = await fetch('/api/stats').then(r => r.json());
    document.getElementById('stat-extractions').textContent = data.total_extractions;
    document.getElementById('stat-corrections').textContent = data.total_corrections;
    document.getElementById('stat-rules').textContent       = data.learned_rules;
    const pct = data.avg_confidence ? (data.avg_confidence * 100).toFixed(1) + '%' : '—';
    document.getElementById('stat-confidence').textContent  = pct;
  } catch (e) {
    console.error('Stats load failed:', e);
  }
}

// ─── JSON syntax highlighting ─────────────────────────────────────────────────
function syntaxHighlight(json) {
  const escaped = json
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return escaped.replace(
    /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    match => {
      let cls = 'json-number';
      if (/^"/.test(match))        cls = /:$/.test(match) ? 'json-key' : 'json-string';
      else if (/true|false/.test(match)) cls = 'json-boolean';
      else if (/null/.test(match)) cls = 'json-null';
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

// ─── Status bar ───────────────────────────────────────────────────────────────
function setStatus(type, message) {
  statusBar.className = `status-bar status-${type}`;
  if (type === 'loading') {
    statusBar.innerHTML = `<span class="spinner"></span>${message}`;
  } else {
    statusBar.textContent = message;
  }
}

// ─── Drag & drop ──────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    setStatus('error', 'Only PDF files are accepted.');
    return;
  }
  currentFile = file;
  selectedFile.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
  extractBtn.disabled = false;
  setStatus('idle', 'Ready — click Extract to process.');
  resultSection.style.display = 'none';
}

// ─── Extract ──────────────────────────────────────────────────────────────────
extractBtn.addEventListener('click', () => {
  if (currentFile) doExtract(currentFile);
});

async function doExtract(file) {
  const simple = simpleToggle.checked;
  setStatus('loading', `Extracting ${file.name} …`);
  resultSection.style.display = 'none';
  extractBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const url = '/api/extract' + (simple ? '?simple=true' : '');
    const resp = await fetch(url, { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || `Server error ${resp.status}`);
    }

    const confPct = Math.round(data.confidence * 100);
    setStatus('success',
      `Done  |  extraction #${data.extraction_id}  |  confidence ${confPct}%`
      + (data.validation_warnings.length ? `  |  ${data.validation_warnings.length} warning(s)` : '')
    );

    renderResult(data, simple);
    await loadStats();

  } catch (e) {
    setStatus('error', 'Error: ' + e.message);
  } finally {
    extractBtn.disabled = false;
  }
}

// ─── Render result ────────────────────────────────────────────────────────────
function renderResult(data, simple) {
  const inv = data.invoice;

  // Title
  const invoiceNum = inv.invoice_number || '—';
  const vendor     = inv.vendor_name || (inv.vendor && inv.vendor.name) || '—';
  resultTitle.textContent = `Invoice ${invoiceNum}  —  ${vendor}`;

  // Confidence badge
  const conf = Math.round(data.confidence * 100);
  confidenceBadge.textContent = conf + '%';
  confidenceBadge.className = 'confidence-badge';
  if (conf < 70)      confidenceBadge.classList.add('low');
  else if (conf < 90) confidenceBadge.classList.add('warn');

  // JSON
  jsonOutput.innerHTML = syntaxHighlight(JSON.stringify(inv, null, 2));

  // Warnings
  if (data.validation_warnings && data.validation_warnings.length > 0) {
    warningsList.innerHTML = data.validation_warnings
      .map(w => `<li><strong>${w.field}:</strong> ${w.message}</li>`)
      .join('');
    warningsSection.style.display = '';
  } else {
    warningsSection.style.display = 'none';
  }

  // Saved paths
  const paths = data.saved_paths || {};
  const labels = { pdf: 'PDF upload', full_json: 'Full JSON', simple_json: 'Simple JSON',
                   database: 'Database', vector_memory: 'Vector DB' };
  savedPaths.innerHTML = Object.entries(paths)
    .map(([k, v]) => `
      <div class="path-row">
        <span class="path-label">${labels[k] || k}</span>
        <span class="path-value">${v}</span>
      </div>`)
    .join('');

  resultSection.style.display = '';
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
loadStats();
setStatus('idle', 'Ready — drag a PDF onto the upload area or click Choose File.');
