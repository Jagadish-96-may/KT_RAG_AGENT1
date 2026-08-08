from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.rag_engine import build_index_from_pdf, answer_question, has_index
from app.config import GOOGLE_API_KEY

app = FastAPI(title="KT RAG Agent")


class AskRequest(BaseModel):
    question: str


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

