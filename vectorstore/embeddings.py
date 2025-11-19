"""Embedding creation and management"""

from langchain_openai import OpenAIEmbeddings
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages OpenAI embeddings"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info(f"Initialized embeddings with model: {EMBEDDING_MODEL}")
    
    def get_embeddings(self):
        """Return the embeddings instance"""
        return self.embeddings
    
    def test_embedding(self, text: str = "Test embedding"):
        """Test if embeddings are working"""
        try:
            result = self.embeddings.embed_query(text)
            logger.info(f"✓ Embeddings working. Dimension: {len(result)}")
            return True
        except Exception as e:
            logger.error(f"✗ Embedding test failed: {e}")
            return False
