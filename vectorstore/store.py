"""ChromaDB vector store operations"""

from typing import List
from langchain_chroma import Chroma
from langchain.schema import Document
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, persist_directory: str, embeddings, collection_name: str = "knowledge_base"):
        self.persist_directory = Path(persist_directory)
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False):
        """Create or load vector store"""
        
        # Delete existing if force recreate
        if force_recreate and self.persist_directory.exists():
            logger.info(f"Deleting existing vectorstore at {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        
        # Check if vectorstore exists
        if self.persist_directory.exists() and not force_recreate:
            logger.info(f"Loading existing vectorstore from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            logger.info(f"Creating new vectorstore with {len(documents)} documents")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name
            )
        
        logger.info(f"✓ Vectorstore ready with {self.get_count()} documents")
        return self.vectorstore
    
    def load_vectorstore(self):
        """Load existing vectorstore"""
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Vectorstore not found at {self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        logger.info(f"✓ Loaded vectorstore with {self.get_count()} documents")
        return self.vectorstore
    
    def get_count(self) -> int:
        """Get count of documents in vectorstore"""
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0
    
    # def get_statistics(self) -> dict:
    #     """Get vectorstore statistics"""
    #     if not self.vectorstore:
    #         return {}
        
    #     collection = self.vectorstore._collection
    #     count = collection.count()
        
    #     stats = {
    #         "total_vectors": count,
    #         "collection_name": self.collection_name,  # Fixed: use self.collection_name
    #         "persist_directory": str(self.persist_directory)
    #     }
        
    #     # Get sample embedding dimensions
    #     if count > 0:
    #         try:
    #             sample = collection.get(limit=1, include=["embeddings"])
    #             # Fixed: properly check if embeddings exist and have content
    #             if sample and "embeddings" in sample and sample["embeddings"] and len(sample["embeddings"]) > 0:
    #                 stats["embedding_dimensions"] = len(sample["embeddings"][0])
    #         except Exception as e:
    #             logger.warning(f"Could not get embedding dimensions: {e}")
        
    #     return stats
    def get_statistics(self) -> dict:
        """Get vectorstore statistics"""
        if not self.vectorstore:
            return {}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        stats = {
            "total_vectors": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }
        
        # Get sample embedding dimensions
        if count > 0:
            try:
                sample = collection.get(limit=1, include=["embeddings"])
                # ✅ Fixed: properly check embeddings without triggering NumPy ambiguity
                if (sample and 
                    "embeddings" in sample and 
                    sample["embeddings"] is not None and 
                    len(sample["embeddings"]) > 0 and
                    len(sample["embeddings"][0]) > 0):
                    stats["embedding_dimensions"] = len(sample["embeddings"][0])
            except Exception as e:
                logger.warning(f"Could not get embedding dimensions: {e}")
        
        return stats
    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results