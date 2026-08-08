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
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f7f7f9; color: #222; }
  h1 { font-size: 24px; }
  .card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  input[type=file] { margin: 10px 0; }
  button { background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 15px; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #999; cursor: not-allowed; }
  #uploadStatus, #answerBox { margin-top: 14px; padding: 12px; border-radius: 8px; background: #f0f4ff; display: none; white-space: pre-wrap; }
  input[type=text] { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 15px; box-sizing: border-box; margin-top: 6px; }
  .row { display: flex; gap: 10px; margin-top: 10px; }
  .row input { flex: 1; }
</style>
</head>
<body>
  <h1>📄 KT RAG Agent</h1>
  <p>Upload a knowledge-transfer PDF, then ask questions grounded only in that document.</p>

  <div class="card">
    <h3>1. Upload PDF</h3>
    <input type="file" id="pdfFile" accept="application/pdf">
    <br>
    <button onclick="uploadPDF()">Upload & Index</button>
    <div id="uploadStatus"></div>
  </div>

  <div class="card">
    <h3>2. Ask a Question</h3>
    <div class="row">
      <input type="text" id="question" placeholder="e.g. What is the deployment process?">
      <button onclick="askQuestion()">Ask</button>
    </div>
    <div id="answerBox"></div>
  </div>

<script>
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
    statusDiv.textContent = `✅ Indexed "${data.filename}" — ${data.chunks_indexed} chunks.`;
  } catch (err) {
    statusDiv.textContent = '❌ ' + err.message;
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
    answerBox.textContent = '❌ ' + err.message;
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
