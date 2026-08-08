import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai

from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL, STORAGE_DIR
from app.pdf_utils import extract_text_from_pdf, chunk_text

genai.configure(api_key=GOOGLE_API_KEY)

INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# In-memory state (rebuilt from disk on startup if present)
_index = None
_chunks: list[str] = []
_embedding_dim = None


def _embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def _get_or_create_index(dim: int):
    global _index, _embedding_dim
    if _index is None:
        _embedding_dim = dim
        _index = faiss.IndexFlatL2(dim)
    return _index


def build_index_from_pdf(file_bytes: bytes) -> int:
    """Extract, chunk, embed, and index a PDF. Returns number of chunks indexed."""
    global _chunks, _index, _embedding_dim

    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError("No extractable text found in PDF.")

    new_chunks = chunk_text(text)
    if not new_chunks:
        raise ValueError("Text extracted but chunking produced no chunks.")

    embeddings = []
    for chunk in new_chunks:
        emb = _embed_text(chunk, task_type="retrieval_document")
        embeddings.append(emb)

    embeddings_np = np.array(embeddings, dtype="float32")
    dim = embeddings_np.shape[1]  # dynamic dimension, not hardcoded

    index = _get_or_create_index(dim)
    index.add(embeddings_np)

    _chunks.extend(new_chunks)

    _save_to_disk()

    return len(new_chunks)


def _save_to_disk():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    if _index is not None:
        faiss.write_index(_index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(_chunks, f)


def _load_from_disk():
    global _index, _chunks, _embedding_dim
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        _index = faiss.read_index(INDEX_PATH)
        _embedding_dim = _index.d
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)


def has_index() -> bool:
    return _index is not None and len(_chunks) > 0


def answer_question(question: str, top_k: int = 4) -> dict:
    """Retrieve relevant chunks and generate a grounded answer. Refuses if not found in doc."""
    if not has_index():
        raise ValueError("No document has been indexed yet. Upload a PDF first.")

    query_emb = _embed_text(question, task_type="retrieval_query")
    query_np = np.array([query_emb], dtype="float32")

    k = min(top_k, len(_chunks))
    distances, indices = _index.search(query_np, k)

    retrieved_chunks = [_chunks[i] for i in indices[0] if i != -1]
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a knowledge transfer (KT) assistant. Answer the question using ONLY the context below, which comes from an internal document.

If the answer is not contained in the context, respond exactly with: "I don't have information about that in this document."

Do not use outside knowledge. Do not guess.

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "sources_used": len(retrieved_chunks),
    }


# Attempt to load any previously saved index on module import
_load_from_disk()

