"""
Cross-Encoder Reranking for improved context relevance
Reorders retrieved documents by actual relevance to query

Place this file in: generation/reranker.py
"""

from typing import List, Tuple
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextReranker:
    """
    Reranks retrieved contexts using Cross-Encoder model
    
    Much more accurate than simple cosine similarity because:
    - Considers full query-document interaction
    - Uses bidirectional attention
    - Specifically trained for relevance ranking
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize Cross-Encoder reranker
        
        Args:
            model_name: Hugging Face model name
                Options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, 80MB)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (better, 120MB)
                - "cross-encoder/ms-marco-TinyBERT-L-6" (fastest, 60MB)
        """
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"✓ Cross-Encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 3,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: The search query/topic
            documents: List of retrieved documents
            top_k: Number of top documents to return
            return_scores: Whether to return relevance scores
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        if len(documents) <= top_k:
            logger.info(f"Only {len(documents)} documents, returning all")
            # Still score them
            pass
        
        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        # Step 1: Prepare query-document pairs
        # Cross-encoder needs [query, document] pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Step 2: Score all pairs
        # Returns relevance scores (higher = more relevant)
        try:
            scores = self.model.predict(pairs)
            logger.info(f"✓ Scored {len(scores)} documents")
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            # Fallback: return original order with dummy scores
            return [(doc, 0.5) for doc in documents[:top_k]]
        
        # Step 3: Pair documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Step 4: Sort by score (descending - highest first)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Take top-k
        top_results = doc_score_pairs[:top_k]
        
        # Log reranking results
        logger.info(f"Reranking results:")
        for i, (doc, score) in enumerate(top_results, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            logger.info(f"  [{i}] Score: {score:.4f} | {source} (Page {page})")
        
        if return_scores:
            return top_results
        else:
            return [doc for doc, score in top_results]
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Rerank and filter by minimum score threshold
        
        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Maximum number to return
            min_score: Minimum relevance score (filters low-quality results)
            
        Returns:
            List of (document, score) tuples above threshold
        """
        # Get all reranked results
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in reranked if score >= min_score]
        
        # Take top-k
        result = filtered[:top_k]
        
        logger.info(f"Reranked with threshold {min_score}: {len(result)}/{len(documents)} above threshold")
        
        return result
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Document]],
        top_k: int = 3
    ) -> List[List[Tuple[Document, float]]]:
        """
        Rerank multiple query-document sets efficiently
        
        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of results per query
            
        Returns:
            List of reranked results (one list per query)
        """
        results = []
        
        for query, documents in zip(queries, document_lists):
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test the reranker
    print("Testing Cross-Encoder Reranker...")
    
    # Create dummy documents
    test_docs = [
        Document(page_content="Machine learning is a subset of AI", metadata={"source": "doc1"}),
        Document(page_content="Python is a programming language", metadata={"source": "doc2"}),
        Document(page_content="Supervised learning uses labeled data", metadata={"source": "doc3"}),
    ]
    
    # Initialize reranker
    reranker = ContextReranker()
    
    # Test query
    query = "What is machine learning?"
    
    # Rerank
    results = reranker.rerank(query, test_docs, top_k=3)
    
    print(f"\nQuery: {query}")
    print("\nReranked Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} - {doc.page_content[:50]}...")
    
    print("\n✓ Reranker test complete!")