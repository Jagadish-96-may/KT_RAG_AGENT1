import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-3.1-flash-lite"
STORAGE_DIR = "storage"

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set")

