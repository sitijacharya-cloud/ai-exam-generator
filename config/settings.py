"""Configuration settings for the RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base" / "books"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Text Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K_CONTEXTS = int(os.getenv("TOP_K_CONTEXTS", "3"))

# Relevance Threshold
# Controls how relevant retrieved content must be to use it
# Range: 0.0 to 1.0
# - 0.3-0.4: Lenient (accepts broader matches)
# - 0.5-0.6: Moderate (recommended - default)
# - 0.7-0.8: Strict (only very relevant content)
# If retrieved contexts are below this threshold, system uses LLM's general knowledge instead
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

# ChromaDB
COLLECTION_NAME = "exam_knowledge_base"