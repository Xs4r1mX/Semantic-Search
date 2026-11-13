import os
from pathlib import Path

from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

EMBEDDING_MODEL = "text-embedding-3-small"

# Local storage paths
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
DOCUMENTS_PATH = BASE_DIR / "documents"