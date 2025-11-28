
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles text splitting and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        if not documents:
            logger.warning("No documents to split")
            return []
        
        logger.info(f"Splitting {len(documents)} documents...")
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk IDs to metadata
        for idx, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = idx
        
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Show example
        if split_docs:
            logger.info(f"Example chunk preview:")
            logger.info(f"  Content: {split_docs[0].page_content[:200]}...")
            logger.info(f"  Metadata: {split_docs[0].metadata}")
        
        return split_docs
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "unique_sources": len(set(chunk.metadata.get('source_file', 'unknown') for chunk in chunks))
        }
