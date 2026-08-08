from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.rag_engine import build_index_from_pdf, answer_question, has_index
from app.config import GOOGLE_API_KEY

app = FastAPI(title="KT RAG Agent")


class AskRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>KT RAG Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; }
  :root {
    --bg: radial-gradient(circle at 25% 15%, #1e3a5f, #0f1f3d 55%, #060d1f 100%);
    --text-main: white;
    --text-sub: rgba(255,255,255,0.6);
    --card-bg: rgba(255,255,255,0.05);
    --card-border: rgba(212,175,55,0.2);
    --input-bg: rgba(255,255,255,0.04);
    --input-border: rgba(255,255,255,0.15);
    --input-text: white;
    --placeholder: rgba(255,255,255,0.35);
    --answer-bg: rgba(212,175,55,0.08);
    --answer-text: rgba(255,255,255,0.85);
  }
  body[data-theme="light"] {
    --bg: linear-gradient(135deg, #fdf6e3, #fff9ed 60%, #fdf0d5 100%);
    --text-main: #2a2110;
    --text-sub: #6b5d3f;
    --card-bg: white;
    --card-border: rgba(212,175,55,0.4);
    --input-bg: #faf6ea;
    --input-border: rgba(212,175,55,0.35);
    --input-text: #2a2110;
    --placeholder: #b0a17a;
    --answer-bg: #fdf3d8;
    --answer-text: #4a3c1e;
  }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    max-width: 820px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    background: var(--bg);
    min-height: 100vh;
    color: var(--text-main);
    transition: background 0.3s ease, color 0.3s ease;
  }
  .top-bar { display: flex; justify-content: flex-end; margin-bottom: 8px; }
  .theme-toggle {
    background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.35);
    color: #d4af37;
    padding: 7px 14px;
    border-radius: 999px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
  }
  .layout { display: flex; gap: 20px; align-items: flex-start; position: relative; }
  .glow {
    position: absolute; width: 220px; height: 220px;
    background: rgba(212,175,55,0.08); border-radius: 50%;
    top: -80px; right: -60px; filter: blur(10px); pointer-events: none;
  }
  .mascot { flex-shrink: 0; margin-top: 60px; }
  .content { flex: 1; min-width: 0; }
  .badge {
    display: inline-block;
    background: rgba(212,175,55,0.15);
    color: #d4af37;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
    border: 1px solid rgba(212,175,55,0.35);
  }
  h1 { font-size: 30px; color: var(--text-main); margin: 0 0 6px; font-weight: 600; }
  .subtitle { color: var(--text-sub); font-size: 14px; margin-bottom: 26px; }
  .card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
  }
  .step-label { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; margin: 0 0 14px; color: var(--text-main); }
  .step-num {
    background: rgba(212,175,55,0.85);
    color: #0f1f3d;
    width: 24px; height: 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
  }
  input[type=file] { display: block; margin: 6px 0 14px; font-size: 14px; color: var(--text-sub); }
  button.action-btn {
    background: linear-gradient(135deg, #d4af37, #b8860b);
    color: #0f1f3d;
    border: none;
    padding: 10px 18px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 700;
  }
  button.action-btn:hover { filter: brightness(1.08); }
  #uploadStatus, #answerBox {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 10px;
    background: var(--answer-bg);
    border-left: 3px solid #d4af37;
    display: none;
    white-space: pre-wrap;
    font-size: 13px;
    color: var(--answer-text);
    line-height: 1.5;
  }
  input[type=text] {
    width: 100%;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--input-border);
    font-size: 14px;
    background: var(--input-bg);
    color: var(--input-text);
    outline: none;
  }
  input[type=text]::placeholder { color: var(--placeholder); }
  input[type=text]:focus { border-color: #d4af37; }
  .row { display: flex; gap: 8px; }
  .row input { flex: 1; }
  @media (max-width: 640px) {
    .layout { flex-direction: column; align-items: center; }
    .mascot { margin-top: 0; }
  }
</style>
</head>
<body data-theme="dark">
  <div class="top-bar">
    <button class="theme-toggle" id="themeBtn" onclick="toggleTheme()">Light mode</button>
  </div>

  <div class="layout">
    <div class="glow"></div>
    <svg class="mascot" width="110" height="150" viewBox="0 0 110 150">
      <ellipse cx="55" cy="140" rx="32" ry="6" fill="rgba(0,0,0,0.25)"/>
      <rect x="30" y="55" width="50" height="55" rx="14" fill="#d4af37"/>
      <rect x="38" y="63" width="34" height="30" rx="8" fill="#0f1f3d"/>
      <circle cx="55" cy="35" r="26" fill="#d4af37"/>
      <circle cx="46" cy="32" r="5" fill="#0f1f3d"/>
      <circle cx="64" cy="32" r="5" fill="#0f1f3d"/>
      <circle cx="47" cy="30" r="1.6" fill="white"/>
      <circle cx="65" cy="30" r="1.6" fill="white"/>
      <path d="M46 44 Q55 50 64 44" stroke="#0f1f3d" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <rect x="10" y="9" width="8" height="16" rx="4" fill="#d4af37"/>
      <rect x="92" y="9" width="8" height="16" rx="4" fill="#d4af37"/>
      <circle cx="14" cy="8" r="4" fill="#f5e08c"/>
      <circle cx="96" cy="8" r="4" fill="#f5e08c"/>
      <rect x="8" y="65" width="14" height="34" rx="7" fill="#d4af37"/>
      <rect x="88" y="65" width="14" height="34" rx="7" fill="#d4af37"/>
      <rect x="35" y="110" width="16" height="26" rx="6" fill="#b8860b"/>
      <rect x="59" y="110" width="16" height="26" rx="6" fill="#b8860b"/>
    </svg>

    <div class="content">
      <span class="badge">KNOWLEDGE TRANSFER</span>
      <h1>KT RAG Agent</h1>
      <p class="subtitle">Upload a knowledge-transfer PDF, then ask questions grounded only in that document.</p>

      <div class="card">
        <p class="step-label"><span class="step-num">1</span> Upload PDF</p>
        <input type="file" id="pdfFile" accept="application/pdf">
        <br>
        <button class="action-btn" onclick="uploadPDF()">Upload &amp; Index</button>
        <div id="uploadStatus"></div>
      </div>

      <div class="card">
        <p class="step-label"><span class="step-num">2</span> Ask a Question</p>
        <div class="row">
          <input type="text" id="question" placeholder="e.g. What is the deployment process?">
          <button class="action-btn" onclick="askQuestion()">Ask</button>
        </div>
        <div id="answerBox"></div>
      </div>
    </div>
  </div>

<script>
function applyTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  document.getElementById('themeBtn').textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
  localStorage.setItem('kt-rag-theme', theme);
}

function toggleTheme() {
  const current = document.body.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

(function initTheme() {
  const saved = localStorage.getItem('kt-rag-theme');
  applyTheme(saved || 'dark');
})();

async function uploadPDF() {
  const fileInput = document.getElementById('pdfFile');
  const statusDiv = document.getElementById('uploadStatus');
  if (!fileInput.files.length) { alert('Choose a PDF first.'); return; }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  statusDiv.style.display = 'block';
  statusDiv.textContent = 'Uploading and indexing... please wait.';

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    statusDiv.textContent = 'Indexed "' + data.filename + '" - ' + data.chunks_indexed + ' chunks.';
  } catch (err) {
    statusDiv.textContent = 'Error: ' + err.message;
  }
}

async function askQuestion() {
  const question = document.getElementById('question').value.trim();
  const answerBox = document.getElementById('answerBox');
  if (!question) { alert('Type a question first.'); return; }

  answerBox.style.display = 'block';
  answerBox.textContent = 'Thinking...';

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    answerBox.textContent = data.answer;
  } catch (err) {
    answerBox.textContent = 'Error: ' + err.message;
  }
}
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "google_api_key_set": bool(GOOGLE_API_KEY),
        "index_ready": has_index(),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    try:
        num_chunks = build_index_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "status": "indexed",
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result
